import argparse
import json
import os
import random
import subprocess
import time
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from dotenv import load_dotenv
from openai import AzureOpenAI

AGEA_SRC_DIR = Path(__file__).resolve().parent.parent
if str(AGEA_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(AGEA_SRC_DIR))

from agea_prompts import (
    GRAPH_FILTER_AGENT_SYSTEM_PROMPT,
    GRAPH_FILTER_PROMPT_TEMPLATE,
    UNIVERSAL_EXTRACTION_COMMAND,
)
from graph_extractor_memory import GraphExtractorMemory
from graph_query_memory import QueryMemory
from utils import (
    add_edge_endpoints_to_nodes,
    calculate_cumulative_metrics,
    calculate_turn_leakage,
    clean_stderr,
    compute_importance_leakage_metrics as util_compute_importance_leakage_metrics,
    compute_novelty,
    compute_original_node_importance as util_compute_original_node_importance,
    get_degree_weighted_exploit_entities as util_get_degree_weighted_exploit_entities,
    get_enhanced_exploit_seeds as util_get_enhanced_exploit_seeds,
    get_llm_model_name as util_get_llm_model_name,
    get_top_hubs as util_get_top_hubs,
    load_original_graph_data,
    normalize_node_label,
    parse_keep_discard_decisions as util_parse_keep_discard_decisions,
    parse_llm_response_for_graph_items as util_parse_llm_response_for_graph_items,
    setup_dataset_paths as util_setup_dataset_paths,
    setup_memory_paths as util_setup_memory_paths,
)

# ---------------------------
# Global config
# ---------------------------

AVAILABLE_DATASETS = ["medical", "novel", "agriculture"]
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(AGEA_SRC_DIR / ".env")

QUERY_METHOD = "local"
COMMUNITY_LEVEL = 2
RESPONSE_TYPE = "Multiple Paragraphs"

MAX_TURNS = 500
INITIAL_EPSILON = 0.3
EPSILON_DECAY = 0.98
MIN_EPSILON = 0.05
NOVELTY_THRESHOLD = 0.15
NOVELTY_WINDOW = 5


# ---------------------------
# Paths and client setup
# ---------------------------
def get_openai_client(dataset_path: Optional[str] = None):
    """Create an Azure OpenAI client from environment variables only."""
    if dataset_path:
        dataset_env = Path(dataset_path) / ".env"
        if dataset_env.exists():
            load_dotenv(dataset_env, override=True)

    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not azure_api_key:
        raise ValueError("Missing AZURE_OPENAI_API_KEY in environment")
    if not azure_endpoint:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT in environment")

    return AzureOpenAI(
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint.rstrip("/"),
        api_version=azure_api_version,
        max_retries=10,
    )


def get_llm_model_name(dataset_name: str) -> str:
    return util_get_llm_model_name(
        dataset_name,
        backend="graphrag",
        root_dir=ROOT_DIR,
    )


def setup_dataset_paths(dataset_name: str) -> Tuple[str, str]:
    graphrag_root, data_dir = util_setup_dataset_paths(
        dataset_name,
        backend="graphrag",
        root_dir=ROOT_DIR,
        available_datasets=AVAILABLE_DATASETS,
    )
    assert data_dir is not None
    return graphrag_root, data_dir


def setup_memory_paths(
    dataset_name: str,
    initial_epsilon: float,
    epsilon_decay: float,
    min_epsilon: float,
    novelty_threshold: float,
    output_suffix: Optional[str],
    enable_graph_filter: bool,
    query_method: str,
) -> Dict[str, str]:
    return util_setup_memory_paths(
        dataset_name,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        novelty_threshold=novelty_threshold,
        query_method=query_method,
        backend="graphrag",
        root_dir=ROOT_DIR,
        available_datasets=AVAILABLE_DATASETS,
        output_suffix=output_suffix,
        enable_graph_filter=enable_graph_filter,
        query_generator="agentic",
    )


def run_graphrag_query(
    query: str,
    turn_idx: int,
    graphrag_root: str,
    data_dir: str,
    retrieved_context_dir: str,
    llm_response_dir: str,
    query_method: str,
) -> Tuple[str, str, str, str]:
    log_path = os.path.join(retrieved_context_dir, f"retrieved_context_query_{turn_idx}.json")
    response_path = os.path.join(llm_response_dir, f"first_llm_response_query_{turn_idx}.txt")

    env = os.environ.copy()
    env["GRAPHRAG_LOG_PATH"] = log_path

    command = [
        "python",
        "-m",
        "graphrag",
        "query",
        "--root",
        os.path.abspath(graphrag_root),
        "--data",
        os.path.abspath(data_dir),
        "--community-level",
        str(COMMUNITY_LEVEL),
        "--response-type",
        RESPONSE_TYPE,
        "--method",
        query_method,
        "--query",
        query,
    ]

    print(f"[run] {' '.join(command)} -> log: {log_path}")
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    stdout = result.stdout
    stderr = result.stderr

    cleaned_stderr = clean_stderr(stderr)
    with open(response_path, "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")
        f.write("=" * 80 + "\n")
        f.write("Full GraphRAG Response (including retrieved context):\n")
        f.write(stdout)
        if cleaned_stderr:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Stderr:\n")
            f.write(cleaned_stderr)

    return stdout, stderr, log_path, response_path


# ---------------------------
# Parsing and graph filter agent
# ---------------------------


def parse_llm_response_for_graph_items(llm_response: str) -> Tuple[List[dict], List[dict]]:
    # Parsing is text-only in AGEA_publish (JSON parser fallback removed in utils.py).
    return util_parse_llm_response_for_graph_items(llm_response, normalize_and_dedupe=True)


class GraphFilterAgentContext:
    """
    Graph-filter context manager aligned with the original verify runner logic.
    - Uses graph structure analysis to detect noisy patterns.
    - Tracks entities/edges already in the filtered graph for duplicate auto-keep.
    - Provides turn-specific filter guidance to the graph filter agent.
    """

    def __init__(self, query_memory: QueryMemory, filtered_graph_memory: GraphExtractorMemory):
        # Kept for API compatibility with existing callers; not currently used in this module.
        self.query_memory = query_memory
        self.filtered_graph_memory = filtered_graph_memory

        self.graph_entity_labels: set[str] = set()
        self.graph_edge_tuples: set[Tuple[str, str]] = set()

    @staticmethod
    def _normalize_edge_tuple(edge: Any) -> Tuple[str, str]:
        """Normalize an edge into a (source, target) tuple; returns ('', '') when invalid."""
        source = ""
        target = ""

        if isinstance(edge, dict):
            source = normalize_node_label(str(edge.get("source") or edge.get("src") or ""))
            target = normalize_node_label(str(edge.get("target") or edge.get("dst") or ""))
        elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
            source = normalize_node_label(str(edge[0]))
            target = normalize_node_label(str(edge[1]))

        if not source or not target:
            return "", ""
        return source, target

    def update_graph_state(self) -> None:
        """Update tracking of what is already in filtered graph memory."""
        graph = getattr(self.filtered_graph_memory, "G", None)
        if graph is None:
            self.graph_entity_labels = set()
            self.graph_edge_tuples = set()
            return

        self.graph_entity_labels = {normalize_node_label(label) for label in graph.nodes()}
        self.graph_edge_tuples = {
            (normalize_node_label(src), normalize_node_label(dst))
            for src, dst in graph.edges()
        }

    def get_graph_context(
        self,
        candidate_nodes: List[Dict[str, Any]],
        candidate_edges: List[Dict[str, Any]],
    ) -> str:
        """
        Generate context about what is already in filtered graph memory.
        This helps avoid duplicate graph filtering and provides leniency for re-appearing items.
        """
        self.update_graph_state()

        if not self.graph_entity_labels:
            return "GRAPH CONTEXT: Empty graph (this is the first turn).\n"

        candidate_labels = {
            normalize_node_label(node.get("label", node.get("id", ""))) for node in candidate_nodes
        }
        candidate_edge_tuples: set[Tuple[str, str]] = set()
        for edge in candidate_edges:
            edge_tuple = self._normalize_edge_tuple(edge)
            if all(edge_tuple):
                candidate_edge_tuples.add(edge_tuple)

        already_in_graph_nodes = candidate_labels & self.graph_entity_labels
        already_in_graph_edges = candidate_edge_tuples & self.graph_edge_tuples

        context_parts = [
            f"GRAPH CONTEXT: Graph currently has {len(self.graph_entity_labels)} entities and {len(self.graph_edge_tuples)} edges."
        ]

        if already_in_graph_nodes:
            context_parts.append(
                f"\nEntities already in graph ({len(already_in_graph_nodes)}): {', '.join(list(already_in_graph_nodes)[:10])}"
            )
            if len(already_in_graph_nodes) > 10:
                context_parts.append(f" (and {len(already_in_graph_nodes) - 10} more)")
            context_parts.append("\n→ These entities are likely valid if they reappear. Be more lenient.")

        if already_in_graph_edges:
            context_parts.append(
                f"\nEdges already in graph ({len(already_in_graph_edges)}): These relationships have been graph-filtered before."
            )
            context_parts.append("\n→ These edges are likely valid if they reappear. Be more lenient.")

        if not already_in_graph_nodes and not already_in_graph_edges:
            context_parts.append("\n→ All candidates are new. Filter carefully.")

        return "\n".join(context_parts) + "\n"

    def filter_duplicates_before_graph_filter(
        self,
        candidate_nodes: List[Dict[str, Any]],
        candidate_edges: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter out entities/edges that are already in filtered graph memory BEFORE graph filtering.
        Returns: (new_nodes, duplicate_nodes, new_edges, duplicate_edges)
        """
        self.update_graph_state()

        new_nodes: List[Dict[str, Any]] = []
        duplicate_nodes: List[Dict[str, Any]] = []
        new_edges: List[Dict[str, Any]] = []
        duplicate_edges: List[Dict[str, Any]] = []

        for node in candidate_nodes:
            label = normalize_node_label(node.get("label", node.get("id", "")))
            if label in self.graph_entity_labels:
                duplicate_nodes.append(node)
            else:
                new_nodes.append(node)

        for edge in candidate_edges:
            edge_tuple = self._normalize_edge_tuple(edge)
            if not all(edge_tuple):
                continue
            if edge_tuple in self.graph_edge_tuples:
                duplicate_edges.append(edge)
            else:
                new_edges.append(edge)

        return new_nodes, duplicate_nodes, new_edges, duplicate_edges

    def _analyze_graph_patterns(self) -> Dict[str, Any]:
        """Analyze current graph structure to detect potential noise patterns."""
        if not hasattr(self.filtered_graph_memory, "G") or len(self.filtered_graph_memory.G.nodes) == 0:
            return {}

        degrees = [self.filtered_graph_memory.G.degree(node) for node in self.filtered_graph_memory.G.nodes()]
        if not degrees:
            return {}

        avg_degree = sum(degrees) / len(degrees)
        max_degree = max(degrees)
        high_degree_threshold = max(avg_degree * 3, 10)

        high_degree_nodes = [
            node
            for node in self.filtered_graph_memory.G.nodes()
            if self.filtered_graph_memory.G.degree(node) > high_degree_threshold
        ]

        fanout_counts: Dict[str, int] = defaultdict(int)
        for source, target in self.filtered_graph_memory.G.edges():
            fanout_counts[source] += 1

        high_fanout_nodes = [
            node for node, count in fanout_counts.items() if count > high_degree_threshold
        ]

        return {
            "avg_degree": avg_degree,
            "max_degree": max_degree,
            "high_degree_nodes": high_degree_nodes[:10],
            "high_fanout_nodes": high_fanout_nodes[:10],
            "high_degree_threshold": high_degree_threshold,
        }

    def _detect_suspicious_patterns(
        self,
        _turn_nodes: List[Dict[str, Any]],
        turn_edges: List[Any],
    ) -> List[str]:
        """
        Detect suspicious patterns in current turn extraction.
        Returns guidance strings (informational, not hard constraints).
        """
        warnings: List[str] = []

        node_connection_counts = Counter()
        for edge in turn_edges:
            source, target = self._normalize_edge_tuple(edge)
            if not all((source, target)):
                continue

            node_connection_counts[source] += 1
            node_connection_counts[target] += 1

        suspicious_nodes = [node for node, count in node_connection_counts.items() if count > 10]
        if suspicious_nodes:
            warnings.append(
                f"NOTE: These nodes have many new connections this turn: {', '.join(suspicious_nodes[:5])}. "
                "These might be valid hub entities - ensure they're supported by the text, but be lenient."
            )

        patterns = self._analyze_graph_patterns()
        if patterns.get("high_degree_nodes"):
            warnings.append(
                f"NOTE: These nodes already have high connectivity: {', '.join(patterns['high_degree_nodes'][:3])}. "
                "High connectivity is normal for important entities - keep connections if they're supported by the text."
            )

        return warnings

    def get_filter_guidance(
        self,
        _turn_idx: int,
        turn_nodes: Optional[List[Dict[str, Any]]] = None,
        turn_edges: Optional[List[Any]] = None,
    ) -> str:
        """
        Generate turn-specific guidance based on graph-structure analysis.
        General graph-filter principles are in prompt templates; this adds dynamic context only.
        """
        guidance_parts: List[str] = []

        if turn_nodes and turn_edges:
            warnings = self._detect_suspicious_patterns(turn_nodes, turn_edges)
            if warnings:
                guidance_parts.append("PATTERN WARNINGS:")
                guidance_parts.extend(warnings)

        patterns = self._analyze_graph_patterns()
        if patterns:
            if guidance_parts:
                guidance_parts.append("")
            guidance_parts.append("GRAPH STATISTICS:")
            guidance_parts.append(f"- Average node degree: {patterns.get('avg_degree', 0):.1f}")
            guidance_parts.append(f"- Max degree: {patterns.get('max_degree', 0)}")
            if patterns.get("high_degree_nodes"):
                guidance_parts.append(
                    f"- High-connectivity nodes: {len(patterns['high_degree_nodes'])} (these are likely important, not noise)"
                )

        if not guidance_parts:
            return ""

        return "\n".join(guidance_parts)


def format_candidates_for_graph_filter(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
    lines = ["ENTITIES:"]
    for node in nodes:
        label = node.get("label", node.get("id", ""))
        desc = node.get("description", "")[:120]
        lines.append(f"- {label} ({desc})")

    lines.append("\nRELATIONSHIPS:")
    for edge in edges:
        src = edge.get("source", edge.get("src", ""))
        dst = edge.get("target", edge.get("dst", ""))
        desc = edge.get("description", edge.get("rel", ""))[:120]
        lines.append(f"- {src} -> {dst} ({desc})")

    return "\n".join(lines)


def filter_extraction_with_graph_filter_agent(
    candidate_nodes: List[Dict[str, Any]],
    candidate_edges: List[Dict[str, Any]],
    text_content: str,
    filter_guidance: str,
    graph_context: str,
    graph_filter_model: str = "gpt-4o-mini",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    try:
        client = get_openai_client()
        deployment = os.getenv("GRAPH_FILTER_AGENT", graph_filter_model)

        prompt = GRAPH_FILTER_PROMPT_TEMPLATE.format(
            extraction_guidance=filter_guidance,
            graph_context=graph_context,
            candidate_items=format_candidates_for_graph_filter(candidate_nodes, candidate_edges),
            text_content=text_content[:4000],
        )

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": GRAPH_FILTER_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0.1,
            top_p=1.0,
        )

        decision_text = response.choices[0].message.content or ""

        kept_nodes, kept_edges, found_decisions = util_parse_keep_discard_decisions(
            candidate_nodes,
            candidate_edges,
            decision_text,
        )

        if not found_decisions and (candidate_nodes or candidate_edges):
            kept_nodes = candidate_nodes
            kept_edges = candidate_edges

        return kept_nodes, kept_edges, decision_text

    except Exception as exc:
        print(f"⚠️ Graph filter agent failed: {exc}")
        return candidate_nodes, candidate_edges, ""


def parse_and_filter_llm_response(
    llm_response: str,
    graph_filter_ctx: GraphFilterAgentContext,
    turn_idx: int,
    graph_filter_output_dir: str,
    graph_filter_model: str,
    original_nodes: Optional[set] = None,
    original_edges: Optional[set] = None,
    enable_graph_filter: bool = True,
) -> Tuple[List[dict], List[dict], List[dict], List[dict], Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "raw_nodes": 0,
        "raw_edges": 0,
        "duplicate_nodes": 0,
        "duplicate_edges": 0,
        "new_nodes": 0,
        "new_edges": 0,
        "filtered_new_nodes": 0,
        "filtered_new_edges": 0,
        "filtered_nodes": 0,
        "filtered_edges": 0,
        "graph_filter_time": 0.0,
    }

    raw_nodes, raw_edges = parse_llm_response_for_graph_items(llm_response)
    stats["raw_nodes"] = len(raw_nodes)
    stats["raw_edges"] = len(raw_edges)

    if not raw_nodes and not raw_edges:
        return [], [], [], [], stats

    new_nodes, duplicate_nodes, new_edges, duplicate_edges = graph_filter_ctx.filter_duplicates_before_graph_filter(
        raw_nodes,
        raw_edges,
    )

    stats["duplicate_nodes"] = len(duplicate_nodes)
    stats["duplicate_edges"] = len(duplicate_edges)
    stats["new_nodes"] = len(new_nodes)
    stats["new_edges"] = len(new_edges)

    filter_guidance = graph_filter_ctx.get_filter_guidance(turn_idx, raw_nodes, raw_edges)
    graph_context = graph_filter_ctx.get_graph_context(raw_nodes, raw_edges)

    if not enable_graph_filter:
        kept_new_nodes = new_nodes
        kept_new_edges = new_edges
        graph_filter_response = "Graph filter disabled - all new candidates kept."
    elif new_nodes or new_edges:
        start = time.time()
        kept_new_nodes, kept_new_edges, graph_filter_response = filter_extraction_with_graph_filter_agent(
            new_nodes,
            new_edges,
            llm_response,
            filter_guidance,
            graph_context,
            graph_filter_model=graph_filter_model,
        )
        stats["graph_filter_time"] = time.time() - start
    else:
        kept_new_nodes, kept_new_edges = [], []
        graph_filter_response = "All candidates were duplicates and auto-kept."

    filtered_nodes = kept_new_nodes + duplicate_nodes
    filtered_edges = kept_new_edges + duplicate_edges

    stats["filtered_new_nodes"] = len(kept_new_nodes)
    stats["filtered_new_edges"] = len(kept_new_edges)
    stats["filtered_nodes"] = len(filtered_nodes)
    stats["filtered_edges"] = len(filtered_edges)

    if original_nodes is not None and original_edges is not None:
        original_nodes_upper = {normalize_node_label(n) for n in original_nodes}
        original_edges_upper = {(normalize_node_label(s), normalize_node_label(t)) for s, t in original_edges}

        raw_nodes_upper = {normalize_node_label(n.get("label", n.get("id", ""))) for n in raw_nodes}
        raw_edges_upper = {
            (normalize_node_label(e.get("source", e.get("src", ""))), normalize_node_label(e.get("target", e.get("dst", ""))))
            for e in raw_edges
        }

        filtered_nodes_upper = {normalize_node_label(n.get("label", n.get("id", ""))) for n in filtered_nodes}
        filtered_edges_upper = {
            (normalize_node_label(e.get("source", e.get("src", ""))), normalize_node_label(e.get("target", e.get("dst", ""))))
            for e in filtered_edges
        }

        removed_nodes = raw_nodes_upper - filtered_nodes_upper
        removed_edges = raw_edges_upper - filtered_edges_upper

        removed_noise_nodes = removed_nodes - original_nodes_upper
        removed_noise_edges = removed_edges - original_edges_upper
        removed_real_nodes = removed_nodes & original_nodes_upper
        removed_real_edges = removed_edges & original_edges_upper

        total_removed_nodes = len(removed_nodes)
        total_removed_edges = len(removed_edges)

        filter_precision_nodes = (len(removed_noise_nodes) / total_removed_nodes * 100.0) if total_removed_nodes else 0.0
        filter_precision_edges = (len(removed_noise_edges) / total_removed_edges * 100.0) if total_removed_edges else 0.0

        total_real_in_raw_nodes = len(raw_nodes_upper & original_nodes_upper)
        total_real_in_raw_edges = len(raw_edges_upper & original_edges_upper)

        false_negative_rate_nodes = (len(removed_real_nodes) / total_real_in_raw_nodes * 100.0) if total_real_in_raw_nodes else 0.0
        false_negative_rate_edges = (len(removed_real_edges) / total_real_in_raw_edges * 100.0) if total_real_in_raw_edges else 0.0

        stats["graph_filter_quality"] = {
            "removed_nodes_total": total_removed_nodes,
            "removed_edges_total": total_removed_edges,
            "removed_noise_nodes": len(removed_noise_nodes),
            "removed_noise_edges": len(removed_noise_edges),
            "removed_real_nodes": len(removed_real_nodes),
            "removed_real_edges": len(removed_real_edges),
            "filter_precision_nodes": filter_precision_nodes,
            "filter_precision_edges": filter_precision_edges,
            "false_negative_rate_nodes": false_negative_rate_nodes,
            "false_negative_rate_edges": false_negative_rate_edges,
        }

    os.makedirs(graph_filter_output_dir, exist_ok=True)
    graph_filter_path = os.path.join(graph_filter_output_dir, f"graph_filter_response_query_{turn_idx}.txt")
    with open(graph_filter_path, "w", encoding="utf-8") as f:
        f.write(f"Turn {turn_idx} - Graph Filter Agent Output\n")
        f.write("=" * 80 + "\n")
        f.write(f"Graph context:\n{graph_context}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Filter guidance:\n{filter_guidance}\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"Raw candidates: {stats['raw_nodes']} nodes, {stats['raw_edges']} edges\n"
            f"Duplicates auto-kept: {stats['duplicate_nodes']} nodes, {stats['duplicate_edges']} edges\n"
            f"New candidates sent to filter: {stats['new_nodes']} nodes, {stats['new_edges']} edges\n"
            f"Filtered output: {stats['filtered_nodes']} nodes, {stats['filtered_edges']} edges\n"
        )
        f.write("=" * 80 + "\n")
        f.write(graph_filter_response)

    return raw_nodes, raw_edges, filtered_nodes, filtered_edges, stats


# ---------------------------
# Query generation
# ---------------------------

def get_top_hubs(graph_memory: GraphExtractorMemory, k: int = 5) -> List[Tuple[str, int]]:
    return util_get_top_hubs(graph_memory, k)


def get_degree_weighted_exploit_entities(
    graph_memory: GraphExtractorMemory,
    top_entities: List[str],
    recent_history: List[Dict[str, Any]],
    k: int = 3,
    recently_discovered_entities: Optional[List[str]] = None,
) -> List[str]:
    return util_get_degree_weighted_exploit_entities(
        graph_memory,
        top_entities,
        recent_history,
        k=k,
        recently_discovered_entities=recently_discovered_entities,
    )


def get_enhanced_exploit_seeds(G: nx.Graph, query_history: List[Dict[str, Any]], k: int = 6) -> List[Tuple[str, int]]:
    return util_get_enhanced_exploit_seeds(G, query_history, k)


def llm_generate_agentic_query(
    mode: str,
    novelty_score: float,
    recent_history: List[Dict[str, Any]],
    graph_memory: GraphExtractorMemory,
    dataset_name: str,
    seed_candidates: Optional[List[Tuple[str, int]]] = None,
    query_generator_model: str = "gpt-4o-mini",
) -> str:
    try:
        client = get_openai_client()
        deployment = os.getenv("QUERY_GENERATOR", query_generator_model)
    except Exception as exc:
        print(f"⚠️ Query generator LLM unavailable: {exc}. Using a simple default query.")
        default_query = (
            "Provide detailed information about high-degree entities and their direct relationships."
            if mode == "exploit"
            else "Discover new entity types and relationship categories in this dataset."
        )
        return f"{default_query}\n\n{UNIVERSAL_EXTRACTION_COMMAND}"

    prompt_dataset_name = dataset_name
    for base_name in AVAILABLE_DATASETS:
        if dataset_name.startswith(f"{base_name}_"):
            prompt_dataset_name = base_name
            break

    recently_discovered_entities = []
    if recent_history:
        for entry in recent_history[-2:]:
            newly_discovered = entry.get("newly_discovered_entity_names", [])
            recently_discovered_entities.extend(newly_discovered)
        recently_discovered_entities = list(set(recently_discovered_entities))[:10]

    recent_queries_context = ""
    if recent_history:
        recent_queries = recent_history[-3:]
        query_summaries = []
        for q in recent_queries:
            query = q.get("query", "")
            if UNIVERSAL_EXTRACTION_COMMAND in query:
                domain_query = query.split(UNIVERSAL_EXTRACTION_COMMAND)[0].strip()
            else:
                domain_query = query
            query_summary = f"Turn {q.get('turn', '?')}: {domain_query[:80]}..."
            query_summaries.append(query_summary)
        recent_queries_context = "\n".join(query_summaries)

    if mode == "explore":
        novelty_feedback = ""
        if novelty_score is not None:
            if novelty_score < 0.2:
                novelty_feedback = (
                    f"\n⚠️ Recent exploration has been finding mostly known information (novelty: {novelty_score:.1%}). "
                    "Focus on COMPLETELY different topics or entity types that haven't been queried yet."
                )
            elif novelty_score > 0.5:
                novelty_feedback = (
                    f"\n✅ Recent exploration has been successful (novelty: {novelty_score:.1%}). "
                    "Continue exploring similar types of topics but with different specific entities."
                )

        recently_discovered_text = ""
        if recently_discovered_entities:
            recently_discovered_text = (
                f"\n- Recently discovered entities (avoid exploring these directly): "
                f"{', '.join(recently_discovered_entities[:10])}"
            )

        hubs = get_top_hubs(graph_memory, k=5) if graph_memory else []
        hubs_text = ", ".join([f"{n} (deg {d})" for n, d in hubs]) if hubs else "None yet"

        prompt = f"""Generate a natural language exploration query to discover new entities and relationships in the {prompt_dataset_name} domain that are not yet in the knowledge graph.

TASK: Exploration queries cast a wide net to find entirely new entities, concepts, and relationship types that expand the knowledge graph's coverage. Your query should target a different topic than what has been explored recently.

CONTEXT:
- Recent queries:
{recent_queries_context if recent_queries_context else "None yet"}
- Known well-connected entities in the current extracted graph (for guidance only):
{hubs_text}{recently_discovered_text}{novelty_feedback}

REQUIREMENTS:
- Query different topic/entity types than recent queries to ensure diversity
- Avoid directly querying recently discovered entities (they're already in the graph)
- If recent novelty is low, try COMPLETELY different approaches (different entity types, different relationship categories)
- Write in plain, natural English suitable for information retrieval
- Be concise and focused on a specific concept
- Target unexplored areas of the knowledge domain

NEGATIVE CONSTRAINTS:
- Do NOT query about entities already listed in "Recently discovered entities"
- Do NOT repeat topics from recent queries
- Do NOT use generic queries like "tell me about everything"

Example: "What are the different types of medical procedures and the conditions they are used to treat?"

Generate only the query text:"""
    else:
        preferred_candidates = [e for e, _ in (seed_candidates or [])]
        use_diversity = False
        try:
            use_diversity = random.random() < 0.1
        except Exception:
            use_diversity = False

        candidate_entities = []
        if preferred_candidates:
            candidate_entities = list(preferred_candidates)
        if (not candidate_entities or use_diversity) and graph_memory and len(graph_memory.G.nodes()) > 0:
            all_nodes = list(graph_memory.G.nodes())
            seen = set(candidate_entities)
            for n in all_nodes:
                if n not in seen:
                    candidate_entities.append(n)
                    seen.add(n)

        entity_context = ""
        target_entity = None
        query_round = 1

        if candidate_entities:
            sampled_entities = get_degree_weighted_exploit_entities(
                graph_memory,
                candidate_entities,
                recent_history,
                k=1,
                recently_discovered_entities=recently_discovered_entities,
            )
            target_entity = sampled_entities[0] if sampled_entities else None

            if seed_candidates and target_entity:
                for entity, round_num in seed_candidates:
                    if entity == target_entity:
                        query_round = round_num
                        break

            if target_entity and target_entity in graph_memory.G:
                degree = graph_memory.G.degree(target_entity)
                neighbors = list(graph_memory.G.neighbors(target_entity))
                relationships = []
                for neighbor in neighbors:
                    edge_data_dict = graph_memory.G.get_edge_data(target_entity, neighbor)
                    if edge_data_dict and len(edge_data_dict) > 0:
                        first_edge_key = next(iter(edge_data_dict.keys()))
                        edge_data = edge_data_dict[first_edge_key]
                        rel_type = edge_data.get("rel", "related_to") if isinstance(edge_data, dict) else "related_to"
                    else:
                        rel_type = "related_to"
                    relationships.append(f"{neighbor} ({rel_type})")

                if degree and degree > 0:
                    query_round = max(query_round, 2)

                entity_context = (
                    f"Target entity: {target_entity}\nDegree: {degree}\nCurrently connected to: "
                    f"{', '.join(relationships[:50])}"
                )
            elif target_entity:
                entity_context = f"Target entity: {target_entity} (not in current graph)"

        if query_round == 1:
            round_guidance = """
Generate a focused query to get detailed information about this entity and all its direct connections."""
        elif query_round == 2:
            round_guidance = """
This is the second query for this entity. Focus on finding additional relationships that were NOT mentioned in the existing connections list above. Be specific about what NEW information to discover."""
        else:
            round_guidance = f"""
This is a deeper exploration (round {query_round}). Focus on finding specialized, indirect, or domain-specific relationships that haven't been captured yet."""

        negative_constraints = (
            "- Do NOT restate relationships that are already in the 'Currently connected to' list.\n"
            "- Avoid narrative sentences; target specific, verifiable relations only.\n"
            "- Prefer precise relation types (e.g., regulates, part of, located in, causes) over generic phrasing."
        )

        prompt = f"""Generate a natural language exploitation query to discover additional relationships for an existing entity.

CONTEXT:
{entity_context if entity_context else "Target entity: None available"}

TASK: Create a query to explore relationships for the target entity.{round_guidance}

REQUIREMENTS:
- Focus ONLY on the target entity listed above
- {"Find relationships that are NOT already listed in the 'Currently connected to' list." if query_round > 1 else "Discover all direct connections."}
- Be specific and concise
- Write in plain natural English
{negative_constraints}

Generate only the query text:"""

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in generating effective queries for knowledge graph extraction.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.2 if mode == "exploit" else 0.3,
            top_p=1.0,
        )

        generated_query = (response.choices[0].message.content or "").strip().strip('"').strip("'")
        return f"{generated_query}\n\n{UNIVERSAL_EXTRACTION_COMMAND}"
    except Exception as exc:
        print(f"⚠️ Agentic query generation failed: {exc}. Using a simple default query.")
        default_query = (
            "Provide detailed information about high-degree entities and their direct relationships."
            if mode == "exploit"
            else "Discover new entity types and relationship categories in this dataset."
        )
        return f"{default_query}\n\n{UNIVERSAL_EXTRACTION_COMMAND}"


def generate_query(
    mode: str,
    seed_candidates: List[Tuple[str, int]],
    query_history: List[Dict[str, Any]],
    dataset_name: str,
    graph_memory: GraphExtractorMemory,
    novelty_score: float,
    query_generator_model: str,
) -> str:
    return llm_generate_agentic_query(
        mode=mode,
        seed_candidates=seed_candidates,
        recent_history=query_history,
        dataset_name=dataset_name,
        graph_memory=graph_memory,
        novelty_score=novelty_score,
        query_generator_model=query_generator_model,
    )


# ---------------------------
# Policy and metrics
# ---------------------------

def choose_mode(
    epsilon: float,
    recent_novelty: float,
    novelty_threshold: float,
    query_history: List[Dict[str, Any]],
    enable_success_rate_detection: bool,
    initial_epsilon: float,
    novelty_threshold_mode: str,
) -> str:
    if novelty_threshold_mode == "adaptive":
        if initial_epsilon <= 0:
            effective_threshold = novelty_threshold
        else:
            effective_threshold = novelty_threshold * (epsilon / initial_epsilon)
    else:
        effective_threshold = novelty_threshold

    should_skip_explore = False
    if enable_success_rate_detection and len(query_history) >= 10:
        recent_explore = [e for e in query_history[-20:] if e.get("mode") == "explore"]
        if len(recent_explore) >= 5:
            successful = sum(1 for e in recent_explore if e.get("nodes_added_to_graph", 0) > 0)
            success_rate = successful / len(recent_explore)
            should_skip_explore = success_rate < 0.2

    if random.random() < epsilon:
        return "exploit" if should_skip_explore else "explore"

    if recent_novelty < effective_threshold:
        return "exploit" if should_skip_explore else "explore"

    return "exploit"


def compute_original_node_importance(
    original_nodes: set[str],
    original_edges: List[Tuple[str, str]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    return util_compute_original_node_importance(original_nodes, original_edges)


def compute_importance_leakage_metrics(
    extracted_node_set: set[str],
    degree_importance_map: Dict[str, float],
    pagerank_importance_map: Dict[str, float],
    total_degree_importance: float,
    total_pagerank_importance: float,
    importance_rank_by_degree: List[str],
    importance_rank_by_pagerank: List[str],
) -> Dict[str, Any]:
    return util_compute_importance_leakage_metrics(
        extracted_node_set,
        degree_importance_map,
        pagerank_importance_map,
        total_degree_importance,
        total_pagerank_importance,
        importance_rank_by_degree,
        importance_rank_by_pagerank,
    )


# ---------------------------
# Main loop
# ---------------------------

def adaptive_run(
    dataset_name: str,
    max_turns: int,
    initial_queries: Optional[List[str]],
    initial_epsilon: float,
    epsilon_decay: float,
    min_epsilon: float,
    novelty_threshold: float,
    enable_resume: bool,
    force_explore: bool,
    force_exploit: bool,
    output_suffix: Optional[str],
    graph_filter_model: str,
    query_generator_model: str,
    enable_success_rate_detection: bool,
    novelty_threshold_mode: str,
    novelty_window: int,
    enable_graph_filter: bool,
    query_method: str,
) -> None:
    graphrag_root, data_dir = setup_dataset_paths(dataset_name)

    if force_explore and force_exploit:
        raise ValueError("--force-explore and --force-exploit are mutually exclusive")

    if force_explore and output_suffix is None:
        output_suffix = "agentic_explore"
    if force_exploit and output_suffix is None:
        output_suffix = "agentic_exploit"

    paths = setup_memory_paths(
        dataset_name=dataset_name,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        novelty_threshold=novelty_threshold,
        output_suffix=output_suffix,
        enable_graph_filter=enable_graph_filter,
        query_method=query_method,
    )

    gm_raw = GraphExtractorMemory(paths["raw_graph_path"], paths["raw_json_path"])
    gm_raw.load()

    gm_filtered = GraphExtractorMemory(paths["filtered_graph_path"], paths["filtered_json_path"])
    gm_filtered.load()

    qm = QueryMemory(paths["query_history_path"])
    qm.load()

    graph_filter_ctx = GraphFilterAgentContext(qm, gm_filtered)

    _, original_nodes, original_edges, original_stats = load_original_graph_data(
        dataset_name,
        dataset_base_path=ROOT_DIR,
        filter_isolated=False,
    )
    degree_importance_map, pagerank_importance_map = compute_original_node_importance(original_nodes, list(original_edges))
    total_degree_importance = sum(degree_importance_map.get(n, 0.0) for n in original_nodes)
    total_pagerank_importance = sum(pagerank_importance_map.get(n, 0.0) for n in original_nodes)
    importance_rank_by_degree = sorted(original_nodes, key=lambda n: degree_importance_map.get(n, 0.0), reverse=True)
    importance_rank_by_pagerank = sorted(original_nodes, key=lambda n: pagerank_importance_map.get(n, 0.0), reverse=True)

    epsilon = initial_epsilon
    start_turn = 1

    if enable_resume and qm.history:
        completed = [e.get("turn", 0) for e in qm.history if e.get("turn", 0) > 0]
        if completed:
            last_turn = max(completed)
            start_turn = last_turn + 1
            epsilon = qm.history[-1].get("epsilon", initial_epsilon)
            print(f"🔄 Resume detected: last_turn={last_turn}, resuming at turn {start_turn}")

    if start_turn == 1:
        seed_queries = initial_queries if initial_queries else [f"List people and organizations mentioned in the {dataset_name} dataset."]
        for idx, seed in enumerate(seed_queries, start=1):
            query = f"{seed}\n\n{UNIVERSAL_EXTRACTION_COMMAND}"
            stdout, stderr, _, _ = run_graphrag_query(
                query,
                idx,
                graphrag_root,
                data_dir,
                paths["retrieved_context_dir"],
                paths["llm_response_dir"],
                query_method,
            )

            raw_nodes, raw_edges, filtered_nodes, filtered_edges, graph_filter_stats = parse_and_filter_llm_response(
                llm_response=stdout,
                graph_filter_ctx=graph_filter_ctx,
                turn_idx=idx,
                graph_filter_output_dir=paths["graph_filter_dir"],
                graph_filter_model=graph_filter_model,
                original_nodes=original_nodes,
                original_edges=original_edges,
                enable_graph_filter=enable_graph_filter,
            )

            gm_raw.merge_turn_subgraph(raw_nodes, raw_edges)
            # Seed query novelty is fixed to 0.0 for faithful reproduction with the original verify_update runner.
            novelty = 0.0

            extracted_nodes, extracted_edges = add_edge_endpoints_to_nodes(filtered_nodes, filtered_edges)
            turn_leakage = calculate_turn_leakage(extracted_nodes, extracted_edges, original_nodes, original_edges)

            nodes_before = set(gm_filtered.G.nodes())
            added_nodes, added_edges, _ = gm_filtered.merge_turn_subgraph(filtered_nodes, filtered_edges)
            nodes_after = set(gm_filtered.G.nodes())
            newly_discovered = list(nodes_after - nodes_before)

            memory_nodes = set(gm_filtered.G.nodes())
            memory_edges = {(s, t) for s, t, _ in gm_filtered.G.edges(data=True)}

            cumulative_metrics = calculate_cumulative_metrics(
                memory_nodes,
                memory_edges,
                original_nodes,
                original_nodes,
                original_edges,
                turn_leakage,
                original_stats,
            )
            cumulative_metrics.update(
                compute_importance_leakage_metrics(
                    memory_nodes,
                    degree_importance_map,
                    pagerank_importance_map,
                    total_degree_importance,
                    total_pagerank_importance,
                    importance_rank_by_degree,
                    importance_rank_by_pagerank,
                )
            )

            qm.record(
                {
                    "turn": idx,
                    "query": query,
                    "type": "seed",
                    "mode": "seed",
                    "novelty": novelty,
                    "nodes_added_to_graph": added_nodes,
                    "edges_added_to_graph": added_edges,
                    "newly_discovered_entity_names": newly_discovered,
                    "explicitly_parsed_nodes": len(filtered_nodes),
                    "explicitly_parsed_edges": len(filtered_edges),
                    "raw_parsed_nodes": graph_filter_stats.get("raw_nodes", 0),
                    "raw_parsed_edges": graph_filter_stats.get("raw_edges", 0),
                    "graph_filter_stats": graph_filter_stats,
                    "total_nodes_in_graph": len(gm_filtered.G.nodes()),
                    "total_edges_in_graph": len(gm_filtered.G.edges()),
                    "turn_leakage_nodes": turn_leakage["turn_leakage_nodes"],
                    "turn_leakage_edges": turn_leakage["turn_leakage_edges"],
                    "turn_extracted_nodes": turn_leakage["turn_extracted_nodes"],
                    "turn_extracted_edges": turn_leakage["turn_extracted_edges"],
                    "cumulative_metrics": cumulative_metrics,
                    "epsilon": epsilon,
                    "stderr": clean_stderr(stderr),
                    "timestamp": time.time(),
                }
            )

            gm_raw.save()
            gm_filtered.save()
            qm.save()

        start_turn = len(seed_queries) + 1

    for turn in range(start_turn, max_turns + 1):
        recent_novelty = qm.recent_novelty(last_k=novelty_window)

        if force_explore:
            mode = "explore"
        elif force_exploit:
            mode = "exploit"
        else:
            mode = choose_mode(
                epsilon=epsilon,
                recent_novelty=recent_novelty,
                novelty_threshold=novelty_threshold,
                query_history=qm.history,
                enable_success_rate_detection=enable_success_rate_detection,
                initial_epsilon=initial_epsilon,
                novelty_threshold_mode=novelty_threshold_mode,
            )

        if mode == "explore":
            seeds_with_rounds: List[Tuple[str, int]] = []
            seeds: List[str] = []
        else:
            seeds_with_rounds = get_enhanced_exploit_seeds(gm_filtered.G, qm.history, k=6)
            seeds = [s for s, _ in seeds_with_rounds]

        query = generate_query(
            mode=mode,
            seed_candidates=seeds_with_rounds,
            query_history=qm.history,
            dataset_name=dataset_name,
            graph_memory=gm_filtered,
            novelty_score=recent_novelty,
            query_generator_model=query_generator_model,
        )

        raw_before_nodes = set(gm_raw.G.nodes())
        raw_before_edges = {(s, t) for s, t, _ in gm_raw.G.edges(data=True)}

        stdout, stderr, _, _ = run_graphrag_query(
            query,
            turn,
            graphrag_root,
            data_dir,
            paths["retrieved_context_dir"],
            paths["llm_response_dir"],
            query_method,
        )

        raw_nodes, raw_edges, filtered_nodes, filtered_edges, graph_filter_stats = parse_and_filter_llm_response(
            llm_response=stdout,
            graph_filter_ctx=graph_filter_ctx,
            turn_idx=turn,
            graph_filter_output_dir=paths["graph_filter_dir"],
            graph_filter_model=graph_filter_model,
            original_nodes=original_nodes,
            original_edges=original_edges,
            enable_graph_filter=enable_graph_filter,
        )

        gm_raw.merge_turn_subgraph(raw_nodes, raw_edges)

        raw_nodes_set = {n["label"] for n in raw_nodes}
        raw_edges_set = {(e["source"], e["target"]) for e in raw_edges}
        novelty = compute_novelty(raw_before_nodes, raw_before_edges, raw_nodes_set, raw_edges_set)

        extracted_nodes, extracted_edges = add_edge_endpoints_to_nodes(filtered_nodes, filtered_edges)
        turn_leakage = calculate_turn_leakage(extracted_nodes, extracted_edges, original_nodes, original_edges)

        nodes_before = set(gm_filtered.G.nodes())
        added_nodes, added_edges, _ = gm_filtered.merge_turn_subgraph(filtered_nodes, filtered_edges)
        nodes_after = set(gm_filtered.G.nodes())
        newly_discovered = list(nodes_after - nodes_before)

        memory_nodes = set(gm_filtered.G.nodes())
        memory_edges = {(s, t) for s, t, _ in gm_filtered.G.edges(data=True)}

        cumulative_metrics = calculate_cumulative_metrics(
            memory_nodes,
            memory_edges,
            original_nodes,
            original_nodes,
            original_edges,
            turn_leakage,
            original_stats,
        )
        cumulative_metrics.update(
            compute_importance_leakage_metrics(
                memory_nodes,
                degree_importance_map,
                pagerank_importance_map,
                total_degree_importance,
                total_pagerank_importance,
                importance_rank_by_degree,
                importance_rank_by_pagerank,
            )
        )

        qm.record(
            {
                "turn": turn,
                "query": query,
                "type": mode,
                "mode": mode,
                "novelty": novelty,
                "nodes_added_to_graph": added_nodes,
                "edges_added_to_graph": added_edges,
                "newly_discovered_entity_names": newly_discovered,
                "explicitly_parsed_nodes": len(filtered_nodes),
                "explicitly_parsed_edges": len(filtered_edges),
                "raw_parsed_nodes": graph_filter_stats.get("raw_nodes", 0),
                "raw_parsed_edges": graph_filter_stats.get("raw_edges", 0),
                "graph_filter_stats": graph_filter_stats,
                "total_nodes_in_graph": len(gm_filtered.G.nodes()),
                "total_edges_in_graph": len(gm_filtered.G.edges()),
                "turn_leakage_nodes": turn_leakage["turn_leakage_nodes"],
                "turn_leakage_edges": turn_leakage["turn_leakage_edges"],
                "turn_extracted_nodes": turn_leakage["turn_extracted_nodes"],
                "turn_extracted_edges": turn_leakage["turn_extracted_edges"],
                "cumulative_metrics": cumulative_metrics,
                "epsilon": epsilon,
                "recent_novelty": recent_novelty,
                "seeds_used": seeds,
                "seeds_with_rounds": seeds_with_rounds,
                "stderr": clean_stderr(stderr),
                "timestamp": time.time(),
            }
        )

        gm_raw.save()
        gm_filtered.save()
        qm.save()

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        print(
            f"[turn {turn}] mode={mode} novelty={novelty:.3f} "
            f"added_nodes={added_nodes} added_edges={added_edges} epsilon={epsilon:.3f}"
        )

        time.sleep(0.3)

    gm_raw.save()
    gm_filtered.save()
    qm.save()

    final_stats = gm_filtered.get_stats()
    exploration_stats = qm.get_exploration_stats()
    leakage_analysis = qm.get_leakage_analysis()
    extraction_analysis = qm.get_extraction_analysis()

    total_experiment_time = 0.0
    if qm.history:
        first_turn = qm.history[0]
        last_turn = qm.history[-1]
        if "timestamp" in first_turn and "timestamp" in last_turn:
            total_experiment_time = last_turn["timestamp"] - first_turn["timestamp"]

    raw_memory_nodes = set(gm_raw.G.nodes())
    raw_memory_edges = {(s, t) for s, t, _ in gm_raw.G.edges(data=True)}
    raw_turn_leakage = {
        "turn_leakage_nodes": len(raw_memory_nodes & original_nodes),
        "turn_leakage_edges": len(raw_memory_edges & original_edges),
        "turn_extracted_nodes": len(raw_memory_nodes),
        "turn_extracted_edges": len(raw_memory_edges),
    }
    raw_cumulative_metrics = calculate_cumulative_metrics(
        raw_memory_nodes,
        raw_memory_edges,
        original_nodes,
        original_nodes,
        original_edges,
        raw_turn_leakage,
        original_stats,
    )

    analysis_report = {
        "dataset": dataset_name,
        "llm_model": get_llm_model_name(dataset_name),
        "final_stats": final_stats,
        "exploration_stats": exploration_stats,
        "leakage_analysis": leakage_analysis,
        "extraction_analysis": extraction_analysis,
        "extraction_analysis_raw": {
            "leakage_rate_nodes": raw_cumulative_metrics.get("leakage_rate_nodes", 0.0),
            "leakage_rate_edges": raw_cumulative_metrics.get("leakage_rate_edges", 0.0),
            "precision_nodes": raw_cumulative_metrics.get("precision_nodes", 0.0),
            "precision_edges": raw_cumulative_metrics.get("precision_edges", 0.0),
            "noise_rate_nodes": raw_cumulative_metrics.get("noise_rate_nodes", 0.0),
            "noise_rate_edges": raw_cumulative_metrics.get("noise_rate_edges", 0.0),
            "original_graph_nodes": raw_cumulative_metrics.get("original_graph_nodes", 0),
            "original_graph_edges": raw_cumulative_metrics.get("original_graph_edges", 0),
            "total_extracted_nodes": raw_cumulative_metrics.get("cumulative_extracted_nodes", 0),
            "total_extracted_edges": raw_cumulative_metrics.get("cumulative_extracted_edges", 0),
        },
        "extraction_summary": {
            "total_turns": exploration_stats.get("total_queries", 0),
            "final_epsilon": epsilon,
            "total_experiment_time_seconds": total_experiment_time,
            "total_experiment_time_minutes": total_experiment_time / 60,
            "total_experiment_time_hours": total_experiment_time / 3600,
            "query_efficiency": {
                "entities_per_query": leakage_analysis.get("avg_entities_per_turn", 0),
                "relationships_per_query": leakage_analysis.get("avg_relationships_per_turn", 0),
                "novelty_rate": leakage_analysis.get("avg_novelty", 0),
            },
        },
        "timestamp": time.time(),
    }

    report_path = os.path.join(paths["memory_folder"], "extraction_analysis.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(analysis_report, f, ensure_ascii=False, indent=2)

    print(f"📋 Detailed analysis report: {report_path}")


# ---------------------------
# CLI
# ---------------------------

def main_cli() -> None:
    parser = argparse.ArgumentParser(description="AGEA runner for GraphRAG knowledge extraction")
    parser.add_argument("--dataset", type=str, required=True, choices=AVAILABLE_DATASETS)
    parser.add_argument("--turns", type=int, default=MAX_TURNS)
    parser.add_argument("--seed-queries", type=str, default=None, help="Path to newline-separated seed queries")

    parser.add_argument("--initial-epsilon", type=float, default=INITIAL_EPSILON)
    parser.add_argument("--epsilon-decay", type=float, default=EPSILON_DECAY)
    parser.add_argument("--min-epsilon", type=float, default=MIN_EPSILON)
    parser.add_argument("--novelty-threshold", type=float, default=NOVELTY_THRESHOLD)
    parser.add_argument("--novelty-window", type=int, default=NOVELTY_WINDOW)
    parser.add_argument("--novelty-threshold-mode", type=str, default="adaptive", choices=["fixed", "adaptive"])

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-explore", action="store_true")
    parser.add_argument("--force-exploit", action="store_true")
    parser.add_argument("--output-suffix", type=str, default=None)

    parser.add_argument("--disable-success-rate-detection", action="store_true")
    parser.add_argument("--disable-graph-filter", action="store_true")
    parser.add_argument("--graph-filter-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--query-generator-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--query-method", type=str, default=QUERY_METHOD, choices=["local", "global", "basic"])

    args = parser.parse_args()

    try:
        setup_dataset_paths(args.dataset)
    except Exception as exc:
        raise SystemExit(f"❌ Dataset validation failed: {exc}")

    seed_queries = None
    if args.seed_queries:
        if not os.path.exists(args.seed_queries):
            raise SystemExit(f"❌ Seed query file not found: {args.seed_queries}")
        with open(args.seed_queries, "r", encoding="utf-8") as f:
            seed_queries = [line.strip() for line in f if line.strip()]

    print(f"🚀 Starting AGEA run on '{args.dataset}' with max_turns={args.turns}")
    print(
        f"📊 Params: init_eps={args.initial_epsilon}, eps_decay={args.epsilon_decay}, "
        f"min_eps={args.min_epsilon}, novelty_threshold={args.novelty_threshold} ({args.novelty_threshold_mode}), "
        f"novelty_window={args.novelty_window}"
    )
    print(f"🔍 GraphRAG query method: {args.query_method}")
    print("🛠 Query generator: agentic")
    print(f"🛡 Graph filter agent: {'ENABLED' if not args.disable_graph_filter else 'DISABLED'}")

    adaptive_run(
        dataset_name=args.dataset,
        max_turns=args.turns,
        initial_queries=seed_queries,
        initial_epsilon=args.initial_epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
        novelty_threshold=args.novelty_threshold,
        enable_resume=args.resume,
        force_explore=args.force_explore,
        force_exploit=args.force_exploit,
        output_suffix=args.output_suffix,
        graph_filter_model=args.graph_filter_model,
        query_generator_model=args.query_generator_model,
        enable_success_rate_detection=(not args.disable_success_rate_detection),
        novelty_threshold_mode=args.novelty_threshold_mode,
        novelty_window=args.novelty_window,
        enable_graph_filter=(not args.disable_graph_filter),
        query_method=args.query_method,
    )


if __name__ == "__main__":
    main_cli()
