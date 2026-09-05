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
        relation: Optional[str] = None,
    ) -> ProvenanceNode:
        """Record a new scientific entity in the lineage graph with explicit deduplication."""
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
            edge: Dict[str, str] = {
                "source": p,
                "target": node_id,
                "relation": relation or f"{node_type}_lineage",
            }
            if edge not in self.edges:
                self.edges.append(edge)
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

    def validate_graph_integrity(self, contract: Optional[Any] = None) -> Dict[str, Any]:
        """Audit the provenance DAG integrity against contract specifications."""
        audit = validate_complete_provenance(self)
        violations = []
        if not audit.get("passed", False):
            if not audit.get("every_experiment_has_result", True):
                violations.append("Provenance DAG contains experiment runs without downstream result nodes.")
            if not audit.get("statistical_critic_present", True):
                violations.append("Provenance DAG missing statistical critic evaluation node.")
            if not audit.get("review_present", True):
                violations.append("Provenance DAG missing scientific peer review node.")
            if not audit.get("revision_present", True):
                violations.append("Provenance DAG missing manuscript revision cycle node.")
            if audit.get("orphan_nodes"):
                violations.append(f"Provenance DAG contains {len(audit['orphan_nodes'])} orphan nodes.")
            if audit.get("missing_edges"):
                violations.append(f"Provenance DAG contains {len(audit['missing_edges'])} broken edge references.")
        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "audit": audit,
        }


def validate_complete_provenance(
    graph_or_tracker: Dict[str, Any] | ProvenanceTracker,
    expected_num_methods: int = 4,
    expected_num_seeds: int = 5,
) -> Dict[str, Any]:
    """Forensically audit a provenance graph for completeness, structural integrity,

    and zero missing or fabricated execution entities.
    """
    if isinstance(graph_or_tracker, ProvenanceTracker):
        graph = graph_or_tracker.export_graph()
    else:
        graph = graph_or_tracker

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    duplicate_experiments: List[str] = []
    seen_ids = set()

    for n in nodes:
        n_id = n.get("node_id")
        if n_id in seen_ids:
            duplicate_experiments.append(n_id)
        seen_ids.add(n_id)
        nodes_by_id[n_id] = n

    # 1. Experiment Node Coverage
    exp_nodes = [
        n for n in nodes
        if n.get("node_type") in ("experiment", "seed_run")
    ]
    exp_runs_traced = len(exp_nodes)
    exp_runs_expected = expected_num_methods * expected_num_seeds

    # Check method and seed pairs
    seen_runs = set()
    missing_experiments: List[str] = []
    for en in exp_nodes:
        m = en.get("metadata", {}).get("method_id") or en.get("metadata", {}).get("method") or en.get("node_id")
        s = en.get("metadata", {}).get("seed")
        key = f"{m}_seed_{s}"
        if key in seen_runs:
            duplicate_experiments.append(en.get("node_id"))
        seen_runs.add(key)

    # 2. Result Lineage Check: Every experiment node has a downstream result node
    res_nodes = [n for n in nodes if n.get("node_type") == "result"]
    exp_with_results = set()
    for rn in res_nodes:
        for p in rn.get("parent_ids", []):
            if p in nodes_by_id and nodes_by_id[p].get("node_type") in ("experiment", "seed_run"):
                exp_with_results.add(p)
    # If no separate result nodes but experiment nodes contain results directly
    missing_results = [
        en.get("node_id") for en in exp_nodes
        if en.get("node_id") not in exp_with_results and not res_nodes
    ]
    every_experiment_has_result = (len(exp_with_results) == len(exp_nodes)) if res_nodes else True

    # 3. Statistical Critic & Meta Analysis Node
    stat_critic_node = next(
        (n for n in nodes if n.get("node_type") in ("statistical_critic", "stat_critic")),
        None
    )
    stat_critic_present = stat_critic_node is not None
    stat_input_ids = (
        stat_critic_node.get("metadata", {}).get("input_experiment_ids", [])
        if stat_critic_node else []
    )
    all_exp_ids = {en.get("node_id") for en in exp_nodes}
    stat_critic_covers_all_exp = (
        bool(all_exp_ids.issubset(set(stat_input_ids)))
        if (stat_critic_present and stat_input_ids)
        else stat_critic_present
    )

    meta_analysis_node = next(
        (n for n in nodes if n.get("node_type") in ("meta_analysis", "statistical_analysis")),
        None
    )
    meta_analysis_present = meta_analysis_node is not None

    # 4. Review & Revision Lineage
    review_node = next(
        (n for n in nodes if n.get("node_type") in ("scientific_review", "review_findings", "review", "review_verdict")),
        None
    )
    review_present = review_node is not None

    revision_node = next(
        (n for n in nodes if n.get("node_type") in ("revision", "revision_cycle")),
        None
    )
    revision_present = revision_node is not None

    # 5. Publication / Output Deliverable Node
    publication_node = next(
        (n for n in nodes if n.get("node_type") in ("publication", "deliverable", "conclusion")),
        None
    )
    publication_present = publication_node is not None

    # 6. Orphan Node Detection (nodes with no parents or no children in workflow DAG)
    all_parents = {p for n in nodes for p in n.get("parent_ids", [])}
    orphan_nodes: List[str] = []
    root_types = {"question", "plan"}
    leaf_or_reference_types = {
        "publication", "deliverable", "conclusion", "benchmark_eval",
        "source", "claim", "doi_verification",
    }

    for n in nodes:
        n_id = n.get("node_id")
        n_type = n.get("node_type")
        has_parents = len(n.get("parent_ids", [])) > 0
        has_children = n_id in all_parents

        if not has_parents and not has_children:
            orphan_nodes.append(n_id)
            continue

        if n_type in root_types:
            if not has_children:
                orphan_nodes.append(n_id)
        elif n_type in leaf_or_reference_types:
            if not has_parents:
                orphan_nodes.append(n_id)
        else:
            # Execution pipeline entity must have both upstream inputs and downstream consumers
            if not has_parents or not has_children:
                orphan_nodes.append(n_id)

    # 7. DAG Edge and Parent ID Validity
    missing_edges: List[str] = []
    for e in edges:
        if e.get("source") not in nodes_by_id or e.get("target") not in nodes_by_id:
            missing_edges.append(f"{e.get('source')}->{e.get('target')}")

    for n in nodes:
        for p in n.get("parent_ids", []):
            if p not in nodes_by_id:
                missing_edges.append(f"parent:{p}->node:{n.get('node_id')}")

    passed = (
        exp_runs_traced >= exp_runs_expected
        and len(duplicate_experiments) == 0
        and every_experiment_has_result
        and stat_critic_present
        and stat_critic_covers_all_exp
        and review_present
        and revision_present
        and publication_present
        and len(orphan_nodes) == 0
        and len(missing_edges) == 0
    )

    return {
        "passed": passed,
        "experiment_runs_expected": exp_runs_expected,
        "experiment_runs_traced": exp_runs_traced,
        "missing_experiments": missing_experiments,
        "duplicate_experiments": duplicate_experiments,
        "every_experiment_has_result": every_experiment_has_result,
        "statistical_critic_present": stat_critic_present,
        "statistical_critic_covers_all_exp": stat_critic_covers_all_exp,
        "meta_analysis_present": meta_analysis_present,
        "review_present": review_present,
        "revision_present": revision_present,
        "publication_present": publication_present,
        "orphan_nodes": orphan_nodes,
        "missing_edges": missing_edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }
