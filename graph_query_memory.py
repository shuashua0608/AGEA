# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Query memory module for tracking query history and extraction analytics."""

import json
import os
from typing import Any, Dict

class QueryMemory:
    """Memory module for tracking query history and extraction analytics."""
    
    def __init__(self, query_history_path: str):
        # list of dicts with comprehensive tracking
        self.history = []
        self.query_history_path = query_history_path

    def load(self):
        """Load query history from JSON file."""
        if os.path.exists(self.query_history_path):
            with open(self.query_history_path, 'r', encoding='utf-8') as f:
                self.history = json.load(f)

    def save(self):
        """Save query history to JSON file."""
        with open(self.query_history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def record(self, rec: Dict[str, Any]):
        """Record a query turn with associated metrics."""
        self.history.append(rec)

    def recent_novelty(self, last_k: int = 10) -> float:
        """Calculate average novelty over the last k turns."""
        vals = [h.get('novelty', 0.0) for h in self.history[-last_k:]]
        return sum(vals) / len(vals) if vals else 0.0
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get statistics about exploration vs exploitation patterns."""
        if not self.history:
            return {"explore_count": 0, "exploit_count": 0, "explore_ratio": 0.0}
        
        explore_count = sum(1 for h in self.history if h.get('type') == 'explore')
        exploit_count = sum(1 for h in self.history if h.get('type') == 'exploit')
        total = len(self.history)
        
        return {
            "explore_count": explore_count,
            "exploit_count": exploit_count,
            "explore_ratio": explore_count / total if total > 0 else 0.0,
            "total_queries": total
        }
    
    def get_leakage_analysis(self) -> Dict[str, Any]:
        """Analyze information leakage patterns using simplified 4-step metrics."""
        if not self.history:
            return {"avg_novelty": 0.0, "total_new_entities": 0, "total_new_relationships": 0}
        
        total_new_entities = sum(h.get('nodes_added_to_graph', 0) for h in self.history)
        total_new_relationships = sum(h.get('edges_added_to_graph', 0) for h in self.history)
        avg_novelty = sum(h.get('novelty', 0.0) for h in self.history) / len(self.history)
        
        return {
            "avg_novelty": avg_novelty,
            "total_new_entities": total_new_entities,
            "total_new_relationships": total_new_relationships,
            "avg_entities_per_turn": total_new_entities / len(self.history) if self.history else 0,
            "avg_relationships_per_turn": total_new_relationships / len(self.history) if self.history else 0
        }
    
    def get_extraction_analysis(self) -> Dict[str, Any]:
        """Analyze comprehensive extraction metrics using simplified 4-step approach.
        
        Note: All graphs are now connected (no isolated nodes), so only one set of metrics is used.
        """
        if not self.history:
            return {
                "leakage_rate_nodes": 0.0,
                "leakage_rate_edges": 0.0,
                "precision_nodes": 0.0,
                "precision_edges": 0.0,
                "noise_rate_nodes": 0.0, "noise_rate_edges": 0.0,
                "original_graph_nodes": 0,
                "original_graph_edges": 0,
                "total_extracted_nodes": 0, "total_extracted_edges": 0
            }
        
        # Get the last entry's cumulative metrics (which represents final progress)
        last_entry = self.history[-1]
        cumulative_metrics = last_entry.get('cumulative_metrics', {})
        
        return {
            "leakage_rate_nodes": cumulative_metrics.get('leakage_rate_nodes', 0.0),
            "leakage_rate_edges": cumulative_metrics.get('leakage_rate_edges', 0.0),
            "precision_nodes": cumulative_metrics.get('precision_nodes', 0.0),
            "precision_edges": cumulative_metrics.get('precision_edges', 0.0),
            "noise_rate_nodes": cumulative_metrics.get('noise_rate_nodes', 0.0),
            "noise_rate_edges": cumulative_metrics.get('noise_rate_edges', 0.0),
            "original_graph_nodes": cumulative_metrics.get('original_graph_nodes', 0),
            "original_graph_edges": cumulative_metrics.get('original_graph_edges', 0),
            "total_extracted_nodes": cumulative_metrics.get('cumulative_extracted_nodes', 0),
            "total_extracted_edges": cumulative_metrics.get('cumulative_extracted_edges', 0)
        }
