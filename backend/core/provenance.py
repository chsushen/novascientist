"""NovaScientist Research Provenance & Lineage Tracker.

Maintains explicit parent-child traceability relationships across:
Research Question -> Source -> Claim -> Methodology -> Experiment -> Result -> Conclusion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ProvenanceNode:
    """Single discrete entity in the scientific provenance lineage graph."""
    node_id: str
    node_type: str  # 'question', 'source', 'claim', 'methodology', 'experiment', 'result', 'conclusion'
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_ids: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProvenanceTracker:
    """Coordinates graph lineage creation, traversal, and audit reporting."""

    def __init__(self, task_id: str = "task_001") -> None:
        self.task_id = task_id
        self.nodes: Dict[str, ProvenanceNode] = {}
        self.edges: List[Dict[str, str]] = []

    def record_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_ids: Optional[List[str]] = None,
    ) -> ProvenanceNode:
        """Record a new scientific entity in the lineage graph."""
        p_ids = parent_ids or []
        node = ProvenanceNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            metadata=metadata or {},
            parent_ids=p_ids,
        )
        self.nodes[node_id] = node
        for p in p_ids:
            self.edges.append({"source": p, "target": node_id})
        return node

    def trace_lineage(self, node_id: str) -> List[ProvenanceNode]:
        """Backtrack the complete lineage path from conclusion back to initial question."""
        if node_id not in self.nodes:
            return []
        
        path: List[ProvenanceNode] = []
        visited = set()
        queue = [node_id]

        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            node = self.nodes.get(curr)
            if node:
                path.append(node)
                queue.extend(node.parent_ids)

        return path

    def export_graph(self) -> Dict[str, Any]:
        """Export complete lineage graph in machine-readable JSON format."""
        return {
            "task_id": self.task_id,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": self.edges,
        }
