# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Graph extractor memory module for storing and managing extracted knowledge graphs."""

import os
import json
import networkx as nx
from typing import List, Dict, Any, Tuple

from utils import normalize_node_label


class GraphExtractorMemory:
    """Memory module for storing extracted knowledge graphs from LLM responses."""
    
    def __init__(self, extractor_graph_path: str, extractor_json_path: str):
        self.G = nx.MultiDiGraph()  # allow multiple edges if needed
        self.extractor_graph_path = extractor_graph_path
        self.extractor_json_path = extractor_json_path

    def load(self):
        """Load existing graph from JSON file."""
        if os.path.exists(self.extractor_json_path):
            try:
                with open(self.extractor_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._load_from_dict(data)
                print(f"[info] loaded memory from {self.extractor_json_path}")
            except Exception as e:
                print(f"[warn] could not load existing extractor memory: {e}")

    def _load_from_dict(self, data: dict):
        """Load graph from dictionary data."""
        self.G = nx.node_link_graph(data)  # recover MultiDiGraph

    def save(self):
        """Save graph to both GraphML and JSON formats."""
        # save as GraphML and JSON (node-link)
        try:
            nx.write_graphml(self.G, self.extractor_graph_path)
        except Exception as e:
            print(f"[warn] could not write GraphML: {e}")
        try:
            data = nx.node_link_data(self.G)
            with open(self.extractor_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[info] saved extractor memory to {self.extractor_json_path}")
        except Exception as e:
            print(f"[warn] could not write extractor JSON: {e}")

    def merge_turn_subgraph(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        """
        Merge nodes and edges into memory graph.
        
        Args:
            nodes: list of dicts like {'id': 'xxx', 'label': 'Name', ...}
            edges: list of dicts like {'source': 'idA', 'target': 'idB', 'rel': 'works_at', ...}

        Returns:
            (num_new_nodes, num_new_edges, num_total_items_in_turn)
        """
        added_nodes = 0
        added_edges = 0

        current_turn_node_ids = set()
        current_turn_edge_ids = set()

        # merge nodes
        for n in nodes:
            label = normalize_node_label(n.get('label', n.get('id', '')))
            node_id = label  # for now use label as id
            current_turn_node_ids.add(node_id)
            if node_id not in self.G:
                self.G.add_node(node_id, **{k: v for k, v in n.items() if k not in ('id', 'label')})
                added_nodes += 1
            else:
                # update metadata if any new attrs
                for k, v in n.items():
                    if k not in ('id', 'label') and (k not in self.G.nodes[node_id] or self.G.nodes[node_id][k] != v):
                        self.G.nodes[node_id][k] = v

        # merge edges
        for e in edges:
            src = normalize_node_label(e.get('source') or e.get('src') or e.get('s') or '')
            dst = normalize_node_label(e.get('target') or e.get('dst') or e.get('t') or '')
            rel = e.get('rel') or e.get('relation') or e.get('predicate') or e.get('type') or e.get('label') or 'related_to'
            edge_key = (src, dst, rel)
            current_turn_edge_ids.add(edge_key)
            # add nodes if missing
            if src and src not in self.G:
                self.G.add_node(src)
                added_nodes += 1
            if dst and dst not in self.G:
                self.G.add_node(dst)
                added_nodes += 1
            # check if same edge exists with same attributes
            exists = False
            for _, edge_dst, _, data in self.G.edges(src, data=True, keys=True):
                if edge_dst == dst and data.get('rel', data.get('relation')) == rel:
                    exists = True
                    break
            if not exists:
                self.G.add_edge(src, dst, **{k: v for k, v in e.items() if k not in ('source', 'target', 'src', 'dst')})
                added_edges += 1

        # update node degrees after adding edges
        self._update_node_degrees()

        total_items = len(current_turn_node_ids) + len(current_turn_edge_ids)
        # return newly added (w.r.t memory, not w.r.t last turn) and other stats
        return added_nodes, added_edges, total_items

    def _update_node_degrees(self):
        """Update the degree attribute for all nodes based on actual graph structure."""
        for node in self.G.nodes():
            # Calculate total degree (in + out edges)
            degree = self.G.degree(node)
            self.G.nodes[node]['degree'] = degree

    def get_stats(self):
        """Get basic statistics about the stored graph."""
        return {
            "num_nodes": self.G.number_of_nodes(),
            "num_edges": self.G.number_of_edges()
        }
