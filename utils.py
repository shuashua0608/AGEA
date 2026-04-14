# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Unified utility functions for adaptive query extraction with advanced LLM parsing capabilities."""

import os
import re
import math
import random
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

import networkx as nx
import numpy as np
import yaml

def normalize_node_label(label: str) -> str:
    """Normalize node labels for consistent comparison."""
    if not label:
        return ""
    return label.strip().upper()


def get_dataset_path(
    dataset_name: str,
    *,
    backend: str = "graphrag",
    root_dir: Optional[str | Path] = None,
    available_datasets: Optional[List[str]] = None,
) -> str:
    """Resolve dataset directory for GraphRAG/LightRAG backends."""
    if not dataset_name or "/" in dataset_name or "\\" in dataset_name:
        raise ValueError(
            f"Invalid dataset name: '{dataset_name}'. Dataset name cannot be empty or contain path separators."
        )

    if backend == "graphrag" and available_datasets and dataset_name not in available_datasets:
        raise ValueError(f"Dataset '{dataset_name}' not available. Choose from: {available_datasets}")

    base = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parent
    if backend == "lightrag":
        return str(base / "graphs" / dataset_name)
    return str(base / dataset_name)


def get_llm_model_name(
    dataset_name: str,
    *,
    backend: str = "graphrag",
    root_dir: Optional[str | Path] = None,
    settings_file: str = "settings.yaml",
    query_llm_deployment: Optional[str] = None,
) -> str:
    """Get sanitized model name used for output folder naming."""
    if backend == "lightrag":
        model_name = query_llm_deployment or "unknown"
        return model_name.replace(".", "_").replace("/", "_").replace("-", "_")

    dataset_path = get_dataset_path(
        dataset_name,
        backend="graphrag",
        root_dir=root_dir,
    )
    settings_path = Path(dataset_path) / settings_file
    if not settings_path.exists():
        return "unknown_model"

    try:
        with settings_path.open("r", encoding="utf-8") as f:
            settings = yaml.safe_load(f)
        model_name = settings.get("models", {}).get("default_chat_model", {}).get("model", "unknown")
        return str(model_name).replace(".", "_").replace("/", "_").replace("-", "_")
    except Exception:
        return "unknown_model"


def setup_dataset_paths(
    dataset_name: str,
    *,
    backend: str = "graphrag",
    root_dir: Optional[str | Path] = None,
    available_datasets: Optional[List[str]] = None,
) -> Tuple[str, Optional[str]]:
    """Validate dataset paths and return backend-specific path tuple."""
    dataset_path = os.path.abspath(
        get_dataset_path(
            dataset_name,
            backend=backend,
            root_dir=root_dir,
            available_datasets=available_datasets,
        )
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    if backend == "graphrag":
        data_dir = os.path.join(dataset_path, "output")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset output directory not found: {data_dir}")
        return dataset_path, data_dir

    return dataset_path, None


def setup_memory_paths(
    dataset_name: str,
    *,
    initial_epsilon: float,
    epsilon_decay: float,
    min_epsilon: float,
    novelty_threshold: float,
    query_method: str,
    backend: str = "graphrag",
    root_dir: Optional[str | Path] = None,
    available_datasets: Optional[List[str]] = None,
    output_suffix: Optional[str] = None,
    enable_graph_filter: bool = True,
    query_generator: str = "agentic",
    settings_file: str = "settings.yaml",
    query_llm_deployment: Optional[str] = None,
) -> Dict[str, str]:
    """Build standardized memory/output paths for AGEA runs."""
    dataset_root = get_dataset_path(
        dataset_name,
        backend=backend,
        root_dir=root_dir,
        available_datasets=available_datasets,
    )
    llm_model_name = get_llm_model_name(
        dataset_name,
        backend=backend,
        root_dir=root_dir,
        settings_file=settings_file,
        query_llm_deployment=query_llm_deployment,
    )

    filter_suffix = "graph_filter" if enable_graph_filter else "no_graph_filter"
    param_suffix = (
        f"init_eps_{initial_epsilon}_eps_decay_{epsilon_decay}_"
        f"min_eps_{min_epsilon}_thres_{novelty_threshold}_{filter_suffix}"
    )
    folder_name = output_suffix if output_suffix is not None else ("agentic" if backend == "graphrag" else query_generator)
    memory_folder = os.path.join(dataset_root, param_suffix, llm_model_name, query_method, folder_name)
    os.makedirs(memory_folder, exist_ok=True)

    turn_log_dir = os.path.join(memory_folder, "turn_logs")
    retrieved_context_dir = os.path.join(turn_log_dir, "retrieved_contexts")
    llm_response_dir = os.path.join(turn_log_dir, "llm_response")
    graph_filter_dir = os.path.join(turn_log_dir, "graph_filter")
    os.makedirs(turn_log_dir, exist_ok=True)
    os.makedirs(retrieved_context_dir, exist_ok=True)
    os.makedirs(llm_response_dir, exist_ok=True)
    os.makedirs(graph_filter_dir, exist_ok=True)

    raw_graph_name = "extracted_graph_raw.graphml" if backend == "graphrag" else "extracted_graph_regex.graphml"
    raw_json_name = "extracted_graph_raw.json" if backend == "graphrag" else "extracted_graph_regex.json"

    return {
        "memory_folder": memory_folder,
        "filtered_graph_path": os.path.join(memory_folder, "extracted_graph.graphml"),
        "filtered_json_path": os.path.join(memory_folder, "extracted_graph.json"),
        "raw_graph_path": os.path.join(memory_folder, raw_graph_name),
        "raw_json_path": os.path.join(memory_folder, raw_json_name),
        "query_history_path": os.path.join(memory_folder, "query_history.json"),
        "turn_log_dir": turn_log_dir,
        "retrieved_context_dir": retrieved_context_dir,
        "llm_response_dir": llm_response_dir,
        "graph_filter_dir": graph_filter_dir,
    }


def parse_llm_response_for_graph_items(
    llm_response: str,
    *,
    normalize_and_dedupe: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """Parse response into node/edge candidates using shared parser utilities."""
    nodes = parse_llm_response_for_entities(llm_response)
    edges = parse_llm_response_for_relationships(llm_response, nodes)

    if not normalize_and_dedupe:
        return nodes, edges

    def uniq_nodes_by_label(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for item in items:
            label = normalize_node_label(item.get("label", ""))
            if label and label not in seen:
                seen.add(label)
                new_item = dict(item)
                new_item["label"] = label
                out.append(new_item)
        return out

    def uniq_edges_by_triple(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for item in items:
            source = normalize_node_label(item.get("source", ""))
            target = normalize_node_label(item.get("target", ""))
            rel = item.get("rel", "related_to")
            key = (source, target, rel)
            if source and target and key not in seen:
                seen.add(key)
                new_item = dict(item)
                new_item["source"] = source
                new_item["target"] = target
                out.append(new_item)
        return out

    return uniq_nodes_by_label(nodes), uniq_edges_by_triple(edges)


def parse_keep_discard_decisions(
    candidate_nodes: List[Dict[str, Any]],
    candidate_edges: List[Dict[str, Any]],
    decision_text: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """
    Parse KEEP/DISCARD decisions using the exact split-based logic from the
    original verify_update runner.

    Returns:
        (kept_nodes, kept_edges, found_decision_markers)
    """
    kept_nodes: List[Dict[str, Any]] = []
    kept_edges: List[Dict[str, Any]] = []

    lines = decision_text.split('\n')

    node_map = {node.get('label', node.get('id', '')): node for node in candidate_nodes}
    edge_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for edge in candidate_edges:
        source = edge.get('source', edge.get('src', ''))
        target = edge.get('target', edge.get('dst', ''))
        edge_map[(source, target)] = edge

    found_decision_markers = False

    for line in lines:
        line_upper = line.upper()
        if 'ENTITY:' in line_upper and ('->' in line):
            parts = line.split('->')
            if len(parts) >= 2:
                found_decision_markers = True
                entity_name = parts[0].replace('ENTITY:', '').strip()
                decision = parts[-1].strip().upper()
                if decision == 'KEEP' and entity_name in node_map:
                    kept_nodes.append(node_map[entity_name])
        elif 'RELATIONSHIP:' in line_upper and ('->' in line):
            parts = line.split('->')
            if len(parts) >= 3:
                found_decision_markers = True
                source = parts[0].replace('RELATIONSHIP:', '').strip()
                target = parts[1].strip()
                decision = parts[-1].strip().upper()
                if decision == 'KEEP' and (source, target) in edge_map:
                    kept_edges.append(edge_map[(source, target)])

    return kept_nodes, kept_edges, found_decision_markers


def is_valid_entity(node: str) -> bool:
    """
    Check if a node is a valid entity for exploitation.
    Restored to match the original verify_update runner exactly.
    """
    generic_terms = {
        "SUMMARY",
        "ORGANIZATIONS",
        "ROLES",
        "OVERVIEW",
        "ENTITY NAME",
        "ENTITY",
        "RELATIONSHIPS",
        "ENTITIES",
        "REPORTS",
        "INFORMATION",
        "DATA",
        "CONTENT",
        "UNKNOWN",
        "N/A",
        "NULL",
        "EMPTY",
    }
    if len(node.strip()) < 2:
        return False
    if (
        node.startswith("ENTITIES (")
        or node.startswith("RELATIONSHIPS (")
        or node.startswith("REPORTS (")
        or (node.startswith("[") and node.endswith("]"))
    ):
        return False
    if node.upper().strip() in generic_terms:
        return False
    return True


def get_top_hubs(graph_memory: Any, k: int = 5) -> List[Tuple[str, int]]:
    if not graph_memory or not hasattr(graph_memory, "G"):
        return []
    nodes_obj = graph_memory.G.nodes
    nodes = list(nodes_obj() if callable(nodes_obj) else nodes_obj)
    if not nodes:
        return []
    degrees = [(n, graph_memory.G.degree(n)) for n in nodes]
    degrees.sort(key=lambda x: x[1], reverse=True)
    return degrees[:k]


def get_degree_weighted_exploit_entities(
    graph_memory: Any,
    top_entities: List[str],
    recent_history: List[Dict[str, Any]],
    k: int = 3,
    recently_discovered_entities: Optional[List[str]] = None,
) -> List[str]:
    if not top_entities:
        return []

    recent_entities_set = set(recently_discovered_entities or [])

    entity_query_counts: Dict[str, int] = {}
    for entry in recent_history:
        for seed in entry.get("seeds_used", []):
            entity_query_counts[seed] = entity_query_counts.get(seed, 0) + 1

    entities, weights = [], []
    for entity in top_entities:
        degree = graph_memory.G.degree(entity) if entity in graph_memory.G else 0
        weight = max(math.log(degree + 1), 1.0)
        if entity in recent_entities_set:
            weight *= 1.2

        penalty = 1.0 / (1.0 + entity_query_counts.get(entity, 0) * 0.05)
        final_weight = max(weight * penalty, 0.01)
        entities.append(entity)
        weights.append(final_weight)

    total_weight = sum(weights)
    if total_weight <= 0:
        return random.sample(entities, min(k, len(entities)))

    probabilities = [w / total_weight for w in weights]
    sampled_indices = np.random.choice(
        len(entities),
        size=min(k, len(entities)),
        replace=False,
        p=probabilities,
    )
    return [entities[i] for i in sampled_indices]


def get_enhanced_exploit_seeds(
    G: nx.Graph,
    query_history: List[Dict[str, Any]],
    k: int = 6,
    super_hub_max_queries: int = 10,
    major_hub_max_queries: int = 5,
    medium_hub_max_queries: int = 3,
) -> List[Tuple[str, int]]:
    valid_entities = [node for node in G.nodes() if is_valid_entity(node)]

    entity_query_counts: Dict[str, int] = {}
    for entry in query_history:
        for seed in entry.get("seeds_used", []):
            entity_query_counts[seed] = entity_query_counts.get(seed, 0) + 1

    node_degrees = sorted([(e, G.degree(e)) for e in valid_entities], key=lambda x: x[1], reverse=True)
    candidates = []
    for entity, degree in node_degrees:
        query_count = entity_query_counts.get(entity, 0)
        if degree >= 100:
            max_queries = super_hub_max_queries
        elif degree >= 50:
            max_queries = major_hub_max_queries
        elif degree >= 20:
            max_queries = medium_hub_max_queries
        else:
            max_queries = 1

        if query_count < max_queries:
            priority = degree / (query_count + 1)
            candidates.append((entity, priority, query_count + 1))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [(entity, round_num) for entity, _, round_num in candidates[:k]]


def compute_original_node_importance(
    original_nodes: set[str],
    original_edges: List[Tuple[str, str]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    graph = nx.DiGraph()
    for node in original_nodes:
        if node:
            graph.add_node(node)
    for src, dst in original_edges:
        if src and dst:
            graph.add_edge(src, dst)

    undirected = graph.to_undirected()
    degree_importance = {n: float(undirected.degree(n)) for n in undirected.nodes()}
    try:
        pagerank_importance = nx.pagerank(undirected, alpha=0.85) if undirected.number_of_nodes() > 0 else {}
    except Exception:
        pagerank_importance = {n: 0.0 for n in undirected.nodes()}
    return degree_importance, pagerank_importance


def summarize_topk_coverage(leaked_nodes: set[str], ranked_nodes: List[str], k_values: List[int]) -> Dict[str, int]:
    out = {}
    for k in k_values:
        out[f"top{k}_important_nodes_leaked"] = len(set(ranked_nodes[:k]) & leaked_nodes)
    return out


def compute_importance_leakage_metrics(
    extracted_node_set: set[str],
    degree_importance_map: Dict[str, float],
    pagerank_importance_map: Dict[str, float],
    total_degree_importance: float,
    total_pagerank_importance: float,
    importance_rank_by_degree: List[str],
    importance_rank_by_pagerank: List[str],
) -> Dict[str, Any]:
    leaked_degree_sum = sum(degree_importance_map.get(n, 0.0) for n in extracted_node_set)
    leaked_pagerank_sum = sum(pagerank_importance_map.get(n, 0.0) for n in extracted_node_set)

    degree_rate = (leaked_degree_sum / total_degree_importance * 100.0) if total_degree_importance else 0.0
    pagerank_rate = (leaked_pagerank_sum / total_pagerank_importance * 100.0) if total_pagerank_importance else 0.0

    k_values = [5, 10, 20]
    degree_cov = summarize_topk_coverage(extracted_node_set, importance_rank_by_degree, k_values)
    pagerank_cov = summarize_topk_coverage(extracted_node_set, importance_rank_by_pagerank, k_values)

    metrics = {
        "importance_leakage_rate_nodes_degree": degree_rate,
        "importance_leakage_rate_nodes_pagerank": pagerank_rate,
    }
    for k in k_values:
        metrics[f"top{k}_by_degree_leaked"] = degree_cov.get(f"top{k}_important_nodes_leaked", 0)
        metrics[f"top{k}_by_pagerank_leaked"] = pagerank_cov.get(f"top{k}_important_nodes_leaked", 0)

    return metrics


def create_entity_alias_map(original_entities: set) -> Dict[str, str]:
    
    alias_map = {}
    
    for entity in original_entities:
        # Handle (abbreviation) pattern: "Full Name (ABBR)" -> "ABBR"
        if '(' in entity and ')' in entity:
            paren_start = entity.find('(')
            paren_end = entity.rfind(')')
            if paren_start < paren_end:
                abbreviation = entity[paren_start+1:paren_end].strip()
                if abbreviation and len(abbreviation) <= 10:  # Reasonable abbreviation length
                    alias_map[abbreviation] = entity
        
        # Handle reverse mapping: "ABBR" -> "Full Name (ABBR)" if it exists
        # Extract potential abbreviations from the entity name
        words = entity.split()
        for word in words:
            # If word is all caps and 2-10 characters, it might be an abbreviation
            if (word.isupper() and 2 <= len(word) <= 10 and 
                word.isalpha() and word not in ['AND', 'THE', 'OF', 'IN', 'TO', 'FOR']):
                # Check if this abbreviation could map to this entity
                if any(word in entity for word in [word]):
                    alias_map[word] = entity
    
    return alias_map


def resolve_entity_alias(entity_name: str, alias_map: Dict[str, str], original_entities: set) -> str:
   
    normalized_name = normalize_node_label(entity_name)
    
    # Direct match
    if normalized_name in original_entities:
        return normalized_name
    
    # Check alias map
    if normalized_name in alias_map:
        canonical = alias_map[normalized_name]
        if canonical in original_entities:
            return canonical
    
    # Return normalized name as fallback
    return normalized_name


def load_original_graph_data(
    dataset_name: str,
    filter_isolated: bool = True,
    dataset_base_path: str = None,
    graph_backend: str = "auto",
) -> Tuple[set, set, set, dict]:
   
    if graph_backend not in {"auto", "graphrag", "lightrag"}:
        raise ValueError(f"Unsupported graph_backend '{graph_backend}'. Use 'auto', 'graphrag', or 'lightrag'.")

    script_dir = os.path.dirname(__file__)
    if dataset_base_path is None:
        candidate_dataset_paths = [
            os.path.join(script_dir, dataset_name),
            os.path.join(script_dir, "graphs", dataset_name),
        ]
    else:
        candidate_dataset_paths = [os.path.join(dataset_base_path, dataset_name)]

    def _has_graphrag_files(path: str) -> bool:
        return os.path.exists(os.path.join(path, "output", "entities.parquet")) and os.path.exists(
            os.path.join(path, "output", "relationships.parquet")
        )

    def _has_lightrag_files(path: str) -> bool:
        return os.path.exists(os.path.join(path, "graph_chunk_entity_relation.graphml"))

    def _load_graphrag(path: str) -> Tuple[set, set, set, dict]:
        entities_file = os.path.join(path, "output", "entities.parquet")
        relationships_file = os.path.join(path, "output", "relationships.parquet")

        print(f"📊 Loading original GraphRAG data from {dataset_name} dataset...")

        entities_df = pd.read_parquet(entities_file)
        total_entities = len(entities_df)
        original_node_labels = set(entities_df["title"].tolist())

        if "degree" in entities_df.columns:
            isolated_entities = len(entities_df[entities_df["degree"] == 0])
            connected_entities = total_entities - isolated_entities
            if filter_isolated:
                connected_entities_df = entities_df[entities_df["degree"] > 0]
                filtered_node_labels = set(connected_entities_df["title"].tolist())
                print(f"   Loaded {len(filtered_node_labels)} connected entities (filtered out {isolated_entities} isolated)")
                print(f"   Total original entities (including isolated): {len(original_node_labels)}")
            else:
                filtered_node_labels = original_node_labels
                print(f"   Loaded {len(original_node_labels)} entities (including {isolated_entities} isolated)")
        else:
            isolated_entities = 0
            connected_entities = total_entities
            filtered_node_labels = original_node_labels
            print(f"   Loaded {len(original_node_labels)} entities (no degree information available)")

        relationships_df = pd.read_parquet(relationships_file)
        original_edge_keys = set()
        for _, row in relationships_df.iterrows():
            original_edge_keys.add((row["source"], row["target"]))
        print(f"   Loaded {len(original_edge_keys)} original relationships")

        stats = {
            "original_graph_nodes": len(original_node_labels),
            "original_graph_edges": len(original_edge_keys),
            "isolated_nodes": isolated_entities,
            "connected_nodes": connected_entities,
        }
        return filtered_node_labels, original_node_labels, original_edge_keys, stats

    def _load_lightrag(path: str) -> Tuple[set, set, set, dict]:
       
        graphml_file = os.path.join(path, "graph_chunk_entity_relation.graphml")
        print(f"📊 Loading original LightRAG data from {dataset_name} dataset...")
        print(f"   GraphML file: {graphml_file}")

        try:
            graph = nx.read_graphml(graphml_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load GraphML file: {e}")

        all_nodes = list(graph.nodes())
        original_node_labels = {
            normalize_node_label(str(node)) for node in all_nodes if normalize_node_label(str(node))
        }

        node_degrees = {}
        for node in all_nodes:
            normalized_node = normalize_node_label(str(node))
            if normalized_node:
                node_degrees[normalized_node] = graph.degree(node)

        isolated_nodes = {node for node, degree in node_degrees.items() if degree == 0}
        connected_nodes = original_node_labels - isolated_nodes

        if filter_isolated:
            filtered_node_labels = connected_nodes
            print(f"   Loaded {len(filtered_node_labels)} connected entities (filtered out {len(isolated_nodes)} isolated)")
            print(f"   Total original entities (including isolated): {len(original_node_labels)}")
        else:
            filtered_node_labels = original_node_labels
            print(f"   Loaded {len(original_node_labels)} entities (including {len(isolated_nodes)} isolated)")

        # LightRAG graph is undirected; normalize each edge to lexicographic order.
        original_edge_keys = set()
        for source, target in graph.edges():
            s = normalize_node_label(str(source))
            t = normalize_node_label(str(target))
            if not s or not t:
                continue
            original_edge_keys.add((s, t) if s <= t else (t, s))

        print(f"   Loaded {len(original_edge_keys)} original relationships (undirected)")

        stats = {
            "original_graph_nodes": len(original_node_labels),
            "original_graph_edges": len(original_edge_keys),
            "isolated_nodes": len(isolated_nodes),
            "connected_nodes": len(connected_nodes),
        }
        return filtered_node_labels, original_node_labels, original_edge_keys, stats

    graphrag_path = next((path for path in candidate_dataset_paths if _has_graphrag_files(path)), None)
    lightrag_path = next((path for path in candidate_dataset_paths if _has_lightrag_files(path)), None)

    if graph_backend == "graphrag":
        if not graphrag_path:
            raise FileNotFoundError(
                f"GraphRAG files not found for dataset '{dataset_name}'. Tried: {candidate_dataset_paths}"
            )
        return _load_graphrag(graphrag_path)

    if graph_backend == "lightrag":
        if not lightrag_path:
            raise FileNotFoundError(
                f"LightRAG GraphML not found for dataset '{dataset_name}'. Tried: {candidate_dataset_paths}"
            )
        return _load_lightrag(lightrag_path)

    # auto: prefer GraphRAG when both exist to preserve existing behavior
    if graphrag_path:
        return _load_graphrag(graphrag_path)
    if lightrag_path:
        return _load_lightrag(lightrag_path)

    raise FileNotFoundError(f"No supported graph files found for dataset '{dataset_name}'. Tried: {candidate_dataset_paths}")


def clean_stderr(stderr: str) -> str:
   
    if not stderr:
        return ""
    
    # Check if stderr contains error (case-insensitive)
    has_error = 'error' in stderr.lower()
    
    # If it contains error, truncate to first 100 characters
    if has_error:
        truncated = stderr[:100]
        if len(stderr) > 100:
            truncated += f"... [truncated, original length: {len(stderr)} chars]"
        return truncated
    
    # Otherwise, clean as before (remove deprecation warnings)
    lines = stderr.strip().split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip common deprecation warnings
        if any(skip_phrase in line.lower() for skip_phrase in [
            'model config based on fnllm is deprecated',
            'please use chat or embedding instead',
            'will be removed in graphrag v3'
        ]):
            continue
            
        if line:  # Only add non-empty lines
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def extract_actual_llm_response(full_response: str) -> str:
   
    if not full_response:
        return ""
    if not isinstance(full_response, str):
        return ""
    
    # Look for the separator line that indicates the start of the actual LLM response
    separator_patterns = [
        r'={80,}\s*\n\s*LLM Response:\s*\n',
        r'LLM Response:\s*\n',
        r'Response:\s*\n',
        r'={50,}\s*\n'
    ]
    
    for pattern in separator_patterns:
        match = re.search(pattern, full_response, re.IGNORECASE)
        if match:
            return full_response[match.end():].strip()
    
    # If no separator found, return the full response
    return full_response


# ---------------------------
# Entity and Relationship Parsing Functions
# ---------------------------

def parse_llm_response_for_entities(llm_response: str, use_advanced_parsing: bool = True) -> List[Dict[str, Any]]:
  
    entities = []
    
    if not llm_response or not isinstance(llm_response, str):
        return entities
    
    # Extract only the actual LLM response part (after the separator)
    actual_response = extract_actual_llm_response(llm_response)

    # Text parsing only
    if use_advanced_parsing:
        new_format_entities = parse_structured_entities(actual_response)
        if new_format_entities:
            return new_format_entities

        current_format_entities = parse_current_format_entities(actual_response)
        if current_format_entities:
            return current_format_entities
    else:
        # Use basic parsing (backward compatibility)
        structured_entities = parse_structured_entities_basic(actual_response)
        if structured_entities:
            return structured_entities
    
    # No valid parsing format found - return empty list
    return []


def parse_current_format_entities(llm_response: str) -> List[Dict[str, Any]]:
    """
    Parse entities using improved regex patterns that handle various LLM output formats.
    """
    entities = []
    
    # Extract actual LLM response (after separator)
    actual_response = extract_actual_llm_response(llm_response)
    
    # Use the improved parsing approach
    parsed_data = parse_entities_and_relationships_improved(actual_response)
    
    for node in parsed_data['nodes']:
        entity_name = node['id']
        description = node['description']
        
        # Skip if name is too short or generic
        if (len(entity_name) > 2 and 
            not entity_name.lower() in ['summary', 'organizations', 'roles', 'overview', 'entity name'] and
            not entity_name.startswith('[') and not entity_name.endswith(']')):
            
            entities.append({
                'id': entity_name,
                'label': entity_name,
                'type': 'extracted',
                'description': description,
                'frequency': 1,
                'degree': 0,
                'content': f"ENTITY: {entity_name}\nDescription: {description}"
            })
    
    return entities


def _dedupe_entities_by_label(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique_entities = []
    seen = set()
    for entity in entities:
        label = entity.get("label")
        if label and label not in seen:
            unique_entities.append(entity)
            seen.add(label)
    return unique_entities


def _extract_description_from_entity_content(content: str) -> str:
    desc_patterns = [
        r'^\s*Description:\s*([^\n]+(?:\n(?!(?:\s*Description:|Relationships:|\d+\.))[^\n]+)*)',
        r'-\s*Description:\s*([^\n]+(?:\n(?!--|\n###|\n##|\n#|\n####)[^\n]+)*)',
        r'-\s*Description;\s*([^\n]+(?:\n(?!--|\n###|\n##|\n#|\n####)[^\n]+)*)',
        r'Description:?\s*([^\n]+(?:\n(?!--|\n###|\n##|\n#|\n####)[^\n]+)*)',
    ]
    for pattern in desc_patterns:
        desc_match = re.search(pattern, content, re.DOTALL)
        if desc_match:
            return re.sub(r'\[Data:.*?\]', '', desc_match.group(1)).strip()
    return ""


def _parse_legacy_numbered_entities(
    llm_response: str,
    *,
    include_content: bool,
    include_header_format: bool,
) -> List[Dict[str, Any]]:
    patterns = [
        r'(\d+)\.\s*([^\n]+)\n(.*?)(?=\n\d+\.|\n###|\n##|\n#|\Z)',
        r'(\d+)\.\s*\*\*([^*]+)\*\*(.*?)(?=\n\d+\.|\n###|\n##|\n#|\Z)',
    ]
    if include_header_format:
        patterns.append(
            r'####\s*(\d+)\.\s*\*\*(?:Entity:\s*)?([^*]+?)(?:\s*\(ID:\s*\d+\))?\*\*(.*?)(?=\n####|\n###|\n##|\n#|\Z)'
        )

    entities = []
    for pattern in patterns:
        for _, name, content in re.findall(pattern, llm_response, re.DOTALL):
            name_clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', name.strip())
            name_clean = re.sub(r'\*([^*]+)\*', r'\1', name_clean).strip()
            description_clean = _extract_description_from_entity_content(content)

            if (
                len(name_clean) > 2
                and name_clean.lower() not in ['summary', 'organizations', 'roles', 'overview', 'entity name']
                and not name_clean.startswith('[')
                and not name_clean.endswith(']')
            ):
                entity = {
                    'id': name_clean,
                    'label': name_clean,
                    'type': 'extracted',
                    'description': description_clean,
                    'frequency': 1,
                    'degree': 0,
                }
                if include_content:
                    entity['content'] = content
                entities.append(entity)

    return _dedupe_entities_by_label(entities)


def parse_structured_entities(llm_response: str) -> List[Dict[str, Any]]:
    
    entities = []

    pattern1 = r'(?:#{1,3}\s*)?(?:\*{1,2})?(?:ENTITY|Entity)(?:\*{1,2})?(?:\s*:)?\s*([^\n]+)\n(?:\*{1,2})?Description(?:\*{1,2})?(?:\s*:)?\s*(.*?)(?=\n(?:#{1,3}\s*)?(?:\*{1,2})?(?:ENTITY|Entity)(?:\*{1,2})?(?:\s*:)?|\n\d+\.|\n###|\n##|\n#|\Z)'
    matches1 = re.findall(pattern1, llm_response, re.DOTALL | re.IGNORECASE)

    pattern1b = r'\*\*(?:ENTITY|Entity)\s*:\s*([^*\n]+)\*\*\s*\n(?:\*{1,2})?Description(?:\*{1,2})?(?:\s*:)?\s*(.*?)(?=\n\*\*(?:ENTITY|Entity)\s*:\s*[^*]+\*\*|\n\*\*[^*]+\*\*|\n(?:#{1,3}\s*)?(?:\*{1,2})?(?:ENTITY|Entity)(?:\*{1,2})?(?:\s*:)?|\n\d+\.|\n###|\n##|\n#|\Z)'
    matches1b = re.findall(pattern1b, llm_response, re.DOTALL | re.IGNORECASE)

    pattern1c = r'\*\*([^*\n]+)\*\*\s*\n(?:\*{1,2})?Description(?:\*{1,2})?(?:\s*:)?\s*(.*?)(?=\n\*\*(?:ENTITY|Entity)\s*:\s*[^*]+\*\*|\n\*\*[^*]+\*\*|\n(?:#{1,3}\s*)?(?:\*{1,2})?(?:ENTITY|Entity)(?:\*{1,2})?(?:\s*:)?|\n\d+\.|\n###|\n##|\n#|\Z)'
    matches1c = re.findall(pattern1c, llm_response, re.DOTALL | re.IGNORECASE)
    matches1c_filtered = [
        (name, desc)
        for name, desc in matches1c
        if not name.strip().upper().startswith('ENTITY:')
    ]

    invalid_exact = {'summary', 'organizations', 'roles', 'overview', 'entity name', 'entity', 'entity:'}
    invalid_prefixes = ('entity:', 'source:', 'target:', 'description:')

    def _append_clean_entity(entity_name: str, description: str) -> None:
        entity_name = re.sub(r'\*\*$', '', entity_name.strip())
        entity_name = re.sub(r'\*+', '', entity_name).strip()
        if entity_name.startswith('[') and entity_name.endswith(']'):
            entity_name = entity_name[1:-1].strip()

        description = re.sub(r'^\s*(?:\*\*)?Description(?:\*\*)?\s*:\s*', '', description.strip(), flags=re.IGNORECASE)
        description = re.sub(r'^\*\*\s*', '', description)
        description = re.sub(r'\s*\*\*$', '', description)
        description = re.sub(r'^\*\s*', '', description)
        description = re.sub(r'\s*\*$', '', description)
        description = re.sub(r'\[Data:.*?\]', '', description).strip()

        lowered = entity_name.lower()
        if (
            len(entity_name) <= 2
            or lowered in invalid_exact
            or any(lowered.startswith(prefix) for prefix in invalid_prefixes)
        ):
            return

        entities.append({
            'id': entity_name,
            'label': entity_name,
            'type': 'extracted',
            'description': description,
            'frequency': 1,
            'degree': 0,
            'content': f"ENTITY: {entity_name}\nDescription: {description}",
        })

    for entity_name, description in matches1:
        _append_clean_entity(entity_name, description)
    for entity_name, description in matches1b:
        _append_clean_entity(entity_name, description)
    for entity_name, description in matches1c_filtered:
        _append_clean_entity(entity_name, description)

    entities.extend(
        _parse_legacy_numbered_entities(
            llm_response,
            include_content=True,
            include_header_format=False,
        )
    )

    return _dedupe_entities_by_label(entities)


def parse_structured_entities_basic(llm_response: str) -> List[Dict[str, Any]]:
    """Backward-compatible parser for legacy numbered entity sections."""
    return _parse_legacy_numbered_entities(
        llm_response,
        include_content=False,
        include_header_format=True,
    )


def parse_llm_response_for_relationships(llm_response: str, entities: List[Dict], use_advanced_parsing: bool = True) -> List[Dict[str, Any]]:
    
    relationships = []
    
    if not llm_response:
        return relationships
    
    # Extract only the actual LLM response part (after the separator)
    actual_response = extract_actual_llm_response(llm_response)

    # Text parsing only
    if use_advanced_parsing:
        new_format_relationships = parse_structured_relationships(actual_response)
        if new_format_relationships:
            return new_format_relationships

        current_format_relationships = parse_current_format_relationships(actual_response)
        if current_format_relationships:
            return current_format_relationships
    else:
        # Use basic parsing (backward compatibility)
        structured_relationships = parse_structured_relationships_basic(actual_response, entities)
        if structured_relationships:
            return structured_relationships
    
    # No valid parsing format found - return empty list
    return []


def _strip_non_ascii(text: str) -> str:
    """
    Remove non-ASCII characters (commonly stray CJK glyphs from certain LLMs)
    while preserving standard whitespace.
    """
    if not isinstance(text, str):
        return text
    allowed = []
    for ch in text:
        code = ord(ch)
        if ch in ("\n", "\r", "\t") or 32 <= code <= 126:
            allowed.append(ch)
    return "".join(allowed)


def parse_entities_and_relationships_improved(text: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Improved parsing function based on robust regex patterns.
    Handles various LLM output formats with markdown formatting.
    """
    # Improved regex patterns - handle both [ENTITY] and ENTITY formats
    ENTITY_HDR_RE = re.compile(r'(?mi)^(?:#{0,6}\s*)?ENTITY:\s*\[?(.+?)\]?\s*$')
    DESC_RE = re.compile(
        r'(?is)(?:\*\*)?\s*Description\s*:(?:\*\*)?\s*(.+?)(?=\n(?:\*\*)?\s*Relationships\s*:(?:\*\*)?|\n(?:#{0,6}\s*)?ENTITY:|\Z)'
    )
    RELS_BLOCK_RE = re.compile(
        r'(?is)(?:\*\*)?\s*Relationships\s*:(?:\*\*)?\s*(.+?)(?=\n(?:#{0,6}\s*)?ENTITY:|\Z)'
    )
    REL_TRIPLE_RE = re.compile(
        r'(?is)^\s*-\s*(?:\*\*)?Source(?:\*\*)?\s*:\s*(.+?)\s*$\s*^\s*(?:\*\*)?Target(?:\*\*)?\s*:\s*(.+?)\s*$\s*^\s*(?:\*\*)?Description(?:\*\*)?\s*:\s*(.+?)(?=^\s*-\s*(?:\*\*)?Source(?:\*\*)?\s*:|\Z)',
        re.MULTILINE
    )

    def _clean(s: str) -> str:
        # Trim and remove surrounding bold markers and square brackets if present
        s = _strip_non_ascii(s).strip()
        # Remove bold markers from anywhere in the string, not just start/end
        s = s.replace("**", "")
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        return _strip_non_ascii(s)

    nodes: Dict[str, str] = {}  # entity_name -> description (may be "")
    edges: List[Dict[str, str]] = []

    # Find all entity headers and their spans
    entity_spans: List[Tuple[str, int, int]] = []
    matches = list(ENTITY_HDR_RE.finditer(text))
    for i, m in enumerate(matches):
        name = _clean(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entity_spans.append((name, start, end))

    # If no explicit ENTITY headers found, treat whole text as one block (best-effort)
    if not entity_spans:
        entity_spans = [("UNKNOWN_ENTITY", 0, len(text))]

    # Parse each entity block
    for name, start, end in entity_spans:
        block = text[start:end]

        # Description (optional)
        desc_match = DESC_RE.search(block)
        desc = _clean(desc_match.group(1)) if desc_match else ""
        # Prefer first non-empty description if entity repeats
        if name not in nodes or (not nodes[name] and desc):
            nodes[name] = desc

        # Relationships block (optional)
        rels_block_match = RELS_BLOCK_RE.search(block)
        if rels_block_match:
            rels_block = rels_block_match.group(1)
            for sm, tm, dm in REL_TRIPLE_RE.findall(rels_block):
                src = _clean(sm)
                tgt = _clean(tm)
                rdesc = _clean(dm)

                # Record edge
                edges.append({"src": src, "target": tgt, "description": rdesc})

                # Ensure both endpoints exist as nodes
                if src not in nodes:
                    nodes[src] = ""
                if tgt not in nodes:
                    nodes[tgt] = ""

    # Materialize nodes list (stable order: insertion order as seen)
    nodes_list = [{"id": k, "description": v} for k, v in nodes.items()]

    return {"nodes": nodes_list, "edges": edges}


def parse_current_format_relationships(llm_response: str) -> List[Dict[str, Any]]:
    """
    Parse relationships using improved regex patterns that handle various LLM output formats.
    """
    relationships = []
    
    # Extract actual LLM response (after separator)
    actual_response = extract_actual_llm_response(llm_response)
    
    # Use the improved parsing approach
    parsed_data = parse_entities_and_relationships_improved(actual_response)
    
    # Convert to the expected format
    for edge in parsed_data['edges']:
        source = edge['src']
        target = edge['target']
        description = edge['description']
        
        # Basic validation
        if (source and target and description and 
            source != target and len(description) > 5):
            
            relationships.append({
                'source': source,
                'target': target,
                'rel': 'related_to',
                'description': description,
                'weight': 1.0,
                'type': 'extracted'
            })
    
    return relationships


def parse_structured_relationships(llm_response: str) -> List[Dict[str, Any]]:
    """
    Parse relationships from the new clean structured format using improved regex patterns.
    
    Handles format where:
    - Only 'Source:' is bulleted and 'Target:'/'Description:' are indented on following lines, OR
    - All three are bulleted
    """
    relationships = []
    
    # Parse relationships from the full response text (not just entity content)
    # Find all relationship sections for each entity block
    # Pattern 1: Matches "ENTITY: ..." or "**ENTITY: ..." format
    entity_blocks1 = re.finditer(
        r'(?:#{1,3}\s*)?(?:\*\*)?(?:ENTITY|Entity):\s*([^\n]+)\n.*?(?=\n(?:#{1,3}\s*)?(?:\*\*)?(?:ENTITY|Entity):|\n\d+\.|\n###|\n##|\n#|\Z)',
        llm_response,
        re.DOTALL
    )
    
    # Pattern 2: Matches "**ENTITY: ENTITY_NAME**" format (markdown bold with ENTITY: inside)
    entity_blocks2 = re.finditer(
        r'\*\*(?:ENTITY|Entity)\s*:\s*([^*\n]+)\*\*\n.*?(?=\n\*\*(?:ENTITY|Entity)\s*:\s*[^*]+\*\*|\n\*\*[^*]+\*\*|\n(?:#{1,3}\s*)?(?:\*\*)?(?:ENTITY|Entity):|\n\d+\.|\n###|\n##|\n#|\Z)',
        llm_response,
        re.DOTALL | re.IGNORECASE
    )
    
    # Pattern 2b: Matches "**ENTITY_NAME**" format (markdown bold without "ENTITY:" prefix)
    entity_blocks2b = re.finditer(
        r'\*\*([^*\n]+)\*\*\n.*?(?=\n\*\*(?:ENTITY|Entity)\s*:\s*[^*]+\*\*|\n\*\*[^*]+\*\*|\n(?:#{1,3}\s*)?(?:\*\*)?(?:ENTITY|Entity):|\n\d+\.|\n###|\n##|\n#|\Z)',
        llm_response,
        re.DOTALL
    )
    
    # Process all formats
    all_blocks = []
    for block_match in entity_blocks1:
        entity_block = block_match.group(0)
        entity_name = block_match.group(1).strip()
        all_blocks.append((entity_block, entity_name))
    
    for block_match in entity_blocks2:
        entity_block = block_match.group(0)
        entity_name = block_match.group(1).strip()
        # Skip if this is actually an ENTITY: format (already handled by pattern1)
        if not entity_name.upper().startswith('ENTITY:'):
            all_blocks.append((entity_block, entity_name))
    
    for block_match in entity_blocks2b:
        entity_block = block_match.group(0)
        entity_name = block_match.group(1).strip()
        # Skip if this starts with "ENTITY:" (already handled)
        if not entity_name.upper().startswith('ENTITY:'):
            all_blocks.append((entity_block, entity_name))
    
    for entity_block, entity_name in all_blocks:
        # Clean entity name
        entity_name = re.sub(r'\*+', '', entity_name).strip()
        if entity_name.startswith('[') and entity_name.endswith(']'):
            entity_name = entity_name[1:-1].strip()
            
        # Find the relationships section for this entity block
        # Stop at the next entity, numbered item, or separator line (---)
        rel_section_match = re.search(r'(?:\*\*)?Relationships(?:\*\*)?:\s*(.*?)(?=\n\*\*[^*]+\*\*|\n(?:#{1,3}\s*)?(?:\*\*)?(?:ENTITY|Entity):|\n\d+\.|\n###|\n##|\n#|\n---|\Z)', entity_block, re.DOTALL)
        if not rel_section_match:
            continue
            
        rel_section = rel_section_match.group(1)
        
        # Use improved parsing logic that handles both formats
        parsed_rels = _parse_relationships_block_improved(rel_section)
        
        for rel in parsed_rels:
            source_clean = rel['src'].strip()
            target_clean = rel['target'].strip()
            description_clean = rel['description'].strip()
            
            # Clean up description
            description_clean = re.sub(r'\[Data:.*?\]', '', description_clean).strip()
            
            if (source_clean != target_clean and
                len(target_clean.strip()) >= 1 and
                len(source_clean.strip()) >= 1 and
                len(description_clean.strip()) > 10):  # Ensure description has meaningful content
                
                relationships.append({
                    'source': source_clean,
                    'target': target_clean,
                    'rel': 'related_to',
                    'weight': 1.0,
                    'description': description_clean,
                    'entity_context': entity_name  # Track which entity this came from
                })
    
    # Deduplicate relationships (same source and target pair)
    # Use a set to track unique relationship keys
    seen_rels = set()
    unique_relationships = []
    for rel in relationships:
        # Create a key from source, target (keep direction)
        key = (rel['source'], rel['target'])
        if key not in seen_rels:
            seen_rels.add(key)
            unique_relationships.append(rel)
    
    return unique_relationships


def _parse_relationships_block_improved(block: str) -> List[Dict[str, str]]:
   
    # Regex patterns for each line type
    # Pattern: (optional whitespace) (optional '-') Source: (value)
    # Must match the full line to ensure Source/Target/Description are on separate lines
    SRC_LINE_RE = re.compile(r'^\s*(?:-\s*)?Source\s*:\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)
    TGT_LINE_RE = re.compile(r'^\s*(?:-\s*)?Target\s*:\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)
    DESC_LINE_RE = re.compile(r'^\s*(?:-\s*)?Description\s*:\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)
    
    def _clean(s: str) -> str:
        s = _strip_non_ascii(s).strip()
        if s.startswith("**") and s.endswith("**"):
            s = s[2:-2].strip()
        # collapse internal whitespace but preserve paragraphs
        s = re.sub(r'[ \t]+', ' ', s)
        return _strip_non_ascii(s)
    
    def _norm_entity(name: str) -> str:
        return re.sub(r'\s+', ' ', _clean(name))
    
    lines = block.strip().splitlines()
    edges = []
    cur = {"src": None, "target": None, "description": []}
    
    def _flush():
        if cur["src"] and cur["target"]:
            edges.append({
                "src": _norm_entity(cur["src"]),
                "target": _norm_entity(cur["target"]),
                "description": _clean("\n".join(cur["description"]).strip())
            })
    
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            # allow blank lines inside description
            if cur["description"] and cur["description"][-1] != "":
                cur["description"].append("")
            continue
        
        m_src = SRC_LINE_RE.match(line)
        if m_src:
            # starting a new triple → flush previous (if any)
            if cur["src"] or cur["target"] or cur["description"]:
                _flush()
            cur = {"src": m_src.group(1), "target": None, "description": []}
            continue
        
        m_tgt = TGT_LINE_RE.match(line)
        if m_tgt:
            cur["target"] = m_tgt.group(1)
            continue
        
        m_desc = DESC_LINE_RE.match(line)
        if m_desc:
            # Description may span multiple lines; start with this line's payload
            cur["description"] = [m_desc.group(1)]
            continue
        
        # If it's a continuation line (indented text), treat it as part of description (when started)
        if cur["description"]:
            cur["description"].append(line.strip())
    
    # flush last
    _flush()
    
    # Deduplicate exact triples
    seen = set()
    uniq = []
    for e in edges:
        key = (e["src"], e["target"], e["description"])
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    return uniq


def parse_structured_relationships_basic(llm_response: str, entities: List[Dict]) -> List[Dict[str, Any]]:
    """
    Parse relationships from the basic numbered format (backward compatibility).
    
    Expected format:
    Relationships:
      - Source: [Entity A]
      - Target: [Entity B]
      - Description: [Relationship description]
    """
    relationships = []
    
    # Create entity lookup for validation
    entity_names = {entity['label'].upper() for entity in entities}
    
    # Format 1: Complete Source/Target/Description triplets
    pattern1 = r'-\s*Source:\s*([^\n]+)\s*\n.*?-\s*Target:\s*([^\n]+)\s*\n.*?-\s*Description:\s*([^\n]+)'
    matches1 = re.findall(pattern1, llm_response, re.DOTALL)
    
    for source, target, description in matches1:
        source_clean = source.strip()
        target_clean = target.strip()
        description_clean = description.strip()
        
        # Clean up description
        description_clean = re.sub(r'\[Data:.*?\]', '', description_clean).strip()
        
        # Validate entities exist and are different
        if (source_clean.upper() in entity_names and 
            target_clean.upper() in entity_names and
            source_clean != target_clean):
            
            relationships.append({
                'source': source_clean,
                'target': target_clean,
                'rel': 'related_to',
                'weight': 1.0,
                'description': description_clean
            })
    
    # Format 2: Source/Description only (extract target from description context)
    pattern2 = r'-\s*Source:\s*([^\n]+)\s*\n.*?-\s*Description:\s*([^\n]+)'
    matches2 = re.findall(pattern2, llm_response, re.DOTALL)
    
    for source, description in matches2:
        source_clean = source.strip()
        description_clean = re.sub(r'\[Data:.*?\]', '', description).strip()
        
        # Try to extract target entity from description
        target_candidates = []
        for entity_name in entity_names:
            if entity_name.lower() in description_clean.lower() and entity_name != source_clean.upper():
                target_candidates.append(entity_name)
        
        # Use the most specific/longest match
        if target_candidates:
            target_clean = max(target_candidates, key=len)
            
            relationships.append({
                'source': source_clean,
                'target': target_clean,
                'rel': 'related_to',
                'weight': 1.0,
                'description': description_clean
            })
    
    # Format 3: Inline relationship descriptions
    pattern3 = r'-\s*([^.]+?)\s*\[Data:.*?Relationships.*?\]\.'
    matches3 = re.findall(pattern3, llm_response, re.DOTALL)
    
    for description in matches3:
        description_clean = description.strip()
        
        # Try to extract entity pairs from description
        for entity_name in entity_names:
            if entity_name.lower() in description_clean.lower():
                # Look for other entities in the same description
                other_entities = [e for e in entity_names if e != entity_name and e.lower() in description_clean.lower()]
                if other_entities:
                    target = max(other_entities, key=len)  # Use longest match
                    relationships.append({
                        'source': entity_name,
                        'target': target,
                        'rel': 'related_to',
                        'weight': 1.0,
                        'description': description_clean
                    })
                    break
    
    # Remove duplicates
    unique_relationships = []
    seen = set()
    for rel in relationships:
        key = (rel['source'], rel['target'])
        if key not in seen and key[::-1] not in seen:  # Also check reverse
            unique_relationships.append(rel)
            seen.add(key)
    
    return unique_relationships


# ---------------------------
# Evaluation Functions
# ---------------------------

def add_edge_endpoints_to_nodes(parsed_nodes: List[Dict], parsed_edges: List[Dict]) -> Tuple[set, set]:
    """
    Step 2: Add edge endpoints to nodes list to ensure full extraction.
    
    Since LLM answers can have edge endpoints that aren't explicitly presented as nodes,
    we add all edge endpoints to the nodes list to ensure complete extraction.
    
    Args:
        parsed_nodes: List of nodes parsed from LLM response
        parsed_edges: List of edges parsed from LLM response
        
    Returns:
        Tuple of (extracted_nodes_set, extracted_edges_set)
    """
    # Start with explicitly parsed nodes
    extracted_nodes = {node['label'] for node in parsed_nodes}
    
    # Add all edge endpoints to ensure full extraction
    for edge in parsed_edges:
        if edge.get('source'):
            extracted_nodes.add(edge['source'])
        if edge.get('target'):
            extracted_nodes.add(edge['target'])
    
    # Convert edges to tuples for consistent comparison
    extracted_edges = {(edge['source'], edge['target']) for edge in parsed_edges 
                      if edge.get('source') and edge.get('target')}
    
    return extracted_nodes, extracted_edges


def _resolve_nodes_and_edges_with_aliases(
    nodes: set,
    edges: set,
    original_nodes: set,
    use_alias_mapping: bool,
) -> Tuple[set, set]:
    if not use_alias_mapping:
        return nodes, edges
    alias_map = create_entity_alias_map(original_nodes)
    resolved_nodes = {resolve_entity_alias(node, alias_map, original_nodes) for node in nodes}
    resolved_edges = {
        (
            resolve_entity_alias(source, alias_map, original_nodes),
            resolve_entity_alias(target, alias_map, original_nodes),
        )
        for source, target in edges
    }
    return resolved_nodes, resolved_edges


def _edge_key(source: str, target: str, edge_direction_agnostic: bool) -> Tuple[str, str]:
    s_upper = source.upper()
    t_upper = target.upper()
    if edge_direction_agnostic and s_upper > t_upper:
        return t_upper, s_upper
    return s_upper, t_upper


def _build_original_edge_lookup(
    original_edges: set,
    edge_direction_agnostic: bool,
) -> Dict[Tuple[str, str], Tuple[str, str]]:
    return {
        _edge_key(source, target, edge_direction_agnostic): (source, target)
        for source, target in original_edges
    }


def calculate_turn_leakage(
    extracted_nodes: set,
    extracted_edges: set,
    original_nodes: set,
    original_edges: set,
    use_alias_mapping: bool = True,
    edge_direction_agnostic: bool = False,
) -> Dict[str, int]:
    """
    Calculate strict (direction-aware) turn leakage against the original graph.

    Returns node/edge overlaps and extracted counts for the current turn.
    """
    final_nodes, final_edges = _resolve_nodes_and_edges_with_aliases(
        extracted_nodes,
        extracted_edges,
        original_nodes,
        use_alias_mapping,
    )

    original_nodes_upper = {n.upper() for n in original_nodes}
    turn_leakage_nodes = sum(1 for node in final_nodes if node.upper() in original_nodes_upper)

    original_edge_keys = set(_build_original_edge_lookup(original_edges, edge_direction_agnostic).keys())
    turn_leakage_edges = sum(
        1 for source, target in final_edges if _edge_key(source, target, edge_direction_agnostic) in original_edge_keys
    )

    return {
        'turn_leakage_nodes': turn_leakage_nodes,
        'turn_leakage_edges': turn_leakage_edges,
        'turn_extracted_nodes': len(final_nodes),
        'turn_extracted_edges': len(final_edges),
    }


def compute_novelty(cumulative_nodes: set, cumulative_edges: set, cur_nodes: set, cur_edges: set) -> float:
    # Handle empty cases
    if not cur_nodes and not cur_edges:
        return 0.0
    
    # Calculate node novelty (unidirectional - exact match)
    node_intersection = len(cumulative_nodes & cur_nodes)
    node_novelty = 1.0 - (node_intersection / len(cur_nodes)) if cur_nodes else 0.0
    
    cumulative_edges_upper = {}
    for s, t in cumulative_edges:
        key_forward = (s.upper(), t.upper())
        key_reverse = (t.upper(), s.upper())
        cumulative_edges_upper[key_forward] = (s, t)
        cumulative_edges_upper[key_reverse] = (s, t)
    
    matching_edges_bidirectional = set()
    for source, target in cur_edges:
        source_upper = source.upper()
        target_upper = target.upper()
        key_forward = (source_upper, target_upper)
        key_reverse = (target_upper, source_upper)
        if key_forward in cumulative_edges_upper or key_reverse in cumulative_edges_upper:
            matching_edges_bidirectional.add((source, target))
    
    edge_intersection = len(matching_edges_bidirectional)
    edge_novelty = 1.0 - (edge_intersection / len(cur_edges)) if cur_edges else 0.0
    
    # Weighted average based on the number of items in each category
    # This ensures the result is between 0-1
    total_items = len(cur_nodes) + len(cur_edges)
    if total_items == 0:
        return 0.0
    
    # Weighted average: each category contributes proportionally to its size
    weighted_novelty = (node_novelty * len(cur_nodes) + edge_novelty * len(cur_edges)) / total_items
    
    return weighted_novelty


def calculate_cumulative_metrics(
    memory_nodes: set,
    memory_edges: set,
    filtered_nodes: set,
    original_nodes: set,
    original_edges: set,
    turn_leakage: Dict[str, int],
    original_stats: dict = None,
    use_alias_mapping: bool = True,
    edge_direction_agnostic: bool = False,
) -> Dict[str, float]:
    """
    Calculate cumulative strict metrics (direction-aware edges):
    leakage, precision, and noise rates for the filtered graph memory.

    Note: `filtered_nodes` is kept in the signature for compatibility.
    """
    del filtered_nodes

    final_memory_nodes, final_memory_edges = _resolve_nodes_and_edges_with_aliases(
        memory_nodes,
        memory_edges,
        original_nodes,
        use_alias_mapping,
    )

    original_nodes_upper = {n.upper(): n for n in original_nodes}
    matched_original_nodes = {
        original_nodes_upper[node.upper()]
        for node in final_memory_nodes
        if node.upper() in original_nodes_upper
    }

    original_edges_upper = _build_original_edge_lookup(original_edges, edge_direction_agnostic)
    matched_original_edges = set()
    matching_edges = set()
    for source, target in final_memory_edges:
        key = _edge_key(source, target, edge_direction_agnostic)
        if key in original_edges_upper:
            matching_edges.add((source, target))
            matched_original_edges.add(original_edges_upper[key])

    leakage_rate_nodes = (len(matched_original_nodes) / len(original_nodes) * 100.0) if original_nodes else 0.0
    leakage_rate_edges = (len(matched_original_edges) / len(original_edges) * 100.0) if original_edges else 0.0

    precision_nodes = (len(matched_original_nodes) / len(final_memory_nodes) * 100.0) if final_memory_nodes else 0.0
    precision_edges = (len(matching_edges) / len(final_memory_edges) * 100.0) if final_memory_edges else 0.0

    noise_nodes = len(final_memory_nodes) - len(matched_original_nodes)
    noise_edges = len(final_memory_edges) - len(matching_edges)
    noise_rate_nodes = (noise_nodes / len(final_memory_nodes) * 100.0) if final_memory_nodes else 0.0
    noise_rate_edges = (noise_edges / len(final_memory_edges) * 100.0) if final_memory_edges else 0.0

    metrics = {
        'leakage_rate_nodes': leakage_rate_nodes,
        'leakage_rate_edges': leakage_rate_edges,
        'precision_nodes': precision_nodes,
        'precision_edges': precision_edges,
        'noise_rate_nodes': noise_rate_nodes,
        'noise_rate_edges': noise_rate_edges,
        'cumulative_extracted_nodes': len(memory_nodes),
        'cumulative_extracted_edges': len(memory_edges),
        'cumulative_true_nodes': len(matched_original_nodes),
        'cumulative_true_edges': len(matched_original_edges),
        'cumulative_noise_nodes': noise_nodes,
        'cumulative_noise_edges': noise_edges,
        'turn_leakage_nodes': turn_leakage['turn_leakage_nodes'],
        'turn_leakage_edges': turn_leakage['turn_leakage_edges'],
    }

    if original_stats:
        metrics['original_graph_nodes'] = original_stats.get('original_graph_nodes', len(original_nodes))
        metrics['original_graph_edges'] = original_stats.get('original_graph_edges', len(original_edges))
    else:
        metrics['original_graph_nodes'] = len(original_nodes)
        metrics['original_graph_edges'] = len(original_edges)

    return metrics
