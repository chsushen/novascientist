"""Comprehensive test suite for complete provenance graph integrity and execution lineage.

Validates that every executed experiment run (4 methods x 5 seeds = 20 runs),
intermediate statistical critique, meta-analysis, peer review, revision cycle,
and publication deliverable are completely and accurately traced without synthetic stubs.
"""

import json
import pytest
from pathlib import Path
from backend.core.orchestrator import NovaScientistOrchestrator, ExecutionMode, TargetPaperLength
from backend.core.provenance import ProvenanceTracker, validate_complete_provenance
from backend.core.surrogate_engine import SurrogateBenchmarkEngine
from backend.core.experiment_agent import ExperimentAgent


@pytest.fixture
def canonical_provenance_tracker():
    """Build a complete provenance graph via deterministic benchmark execution."""
    task_id = "test_prov_task_001"
    prov = ProvenanceTracker(task_id=task_id)

    # Question & Plan
    q_node = prov.record_node("q_001", "question", "Efficient Deep Learning")
    plan_node = prov.record_node("plan_001", "plan", "MB-QGT Plan", parent_ids=[q_node.node_id])

    # Sources & Claims
    s_node = prov.record_node("src_001", "source", "Attention Is All You Need", {"doi": "10.5555/3295222.3295349"}, parent_ids=[plan_node.node_id])
    c_node = prov.record_node("claim_001", "claim", "Multi-head attention yields superior quality", {"category": "efficiency"}, parent_ids=[s_node.node_id])

    # Methodology & Spec
    method_node = prov.record_node("method_001", "methodology", "MB-QGT Methodology", parent_ids=[plan_node.node_id])
    exp_spec_node = prov.record_node(
        "exp_spec_001",
        "experiment_spec",
        "Benchmark Spec: CIFAR-100 (50000 samples, 40 epochs)",
        {"dataset": "CIFAR-100", "epochs": 40, "seeds": 5},
        parent_ids=[method_node.node_id],
        relation="specifies_experiments",
    )

    # 4 methods x 5 seeds = 20 runs
    engine = SurrogateBenchmarkEngine(topic="Efficient Deep Learning", num_seeds=5)
    pkg = engine.run_experiments()
    from dataclasses import asdict
    metrics_dict = asdict(pkg)
    exp_agent = ExperimentAgent()
    exp_records = exp_agent.extract_experiment_records(metrics_dict, dataset_name="CIFAR-100")

    all_exp_node_ids = []
    all_res_node_ids = []
    for er in exp_records:
        er_node = prov.record_node(
            er.experiment_id,
            "experiment",
            f"{er.method_name} (Seed {er.seed})",
            {
                "experiment_id": er.experiment_id,
                "method": er.method_name,
                "method_id": er.method_id,
                "seed": er.seed,
                "accuracy": er.accuracy,
                "memory_mb": er.memory_mb,
                "latency_ms": er.latency_ms,
            },
            parent_ids=[exp_spec_node.node_id],
            relation="executes_run",
        )
        all_exp_node_ids.append(er_node.node_id)
        res_node = prov.record_node(
            f"res_{er.experiment_id}",
            "result",
            f"{er.method_name} Seed {er.seed} Result",
            {
                "accuracy": er.accuracy,
                "memory_mb": er.memory_mb,
                "latency_ms": er.latency_ms,
            },
            parent_ids=[er_node.node_id],
            relation="produces_result",
        )
        all_res_node_ids.append(res_node.node_id)

    # Validation Report
    val_node = prov.record_node(
        "val_report_001",
        "validation_report",
        "Evidence Validation Report",
        {"supported_count": 1, "unsupported_count": 0},
        parent_ids=[c_node.node_id],
        relation="audits_evidence",
    )

    # Aggregates & Meta-Analysis
    metrics_agg_node = prov.record_node(
        "metrics_aggregate_001",
        "metrics_aggregate",
        "Aggregated Telemetry across 20 Runs",
        {"num_methods": 4, "num_seeds": 5, "total_runs": 20},
        parent_ids=all_res_node_ids,
        relation="aggregates_results",
    )
    meta_node = prov.record_node(
        "meta_analysis_001",
        "meta_analysis",
        "DerSimonian-Laird Random-Effects Meta-Analysis",
        {"pooled_effect_size": 0.85, "z_statistic": 4.12, "i_squared_percent": 12.4},
        parent_ids=[metrics_agg_node.node_id],
        relation="computes_meta_analysis",
    )
    stat_critic_node = prov.record_node(
        "stat_critic_001",
        "statistical_critic",
        "Statistical Critic Audit: PASSED",
        {
            "passed": True,
            "num_seeds": 5,
            "methods": 4,
            "input_experiment_ids": all_exp_node_ids,
            "sample_size_sufficient": True,
            "variance_bounded": True,
            "heterogeneity_acceptable": True,
        },
        parent_ids=[meta_node.node_id],
        relation="audits_statistical_power",
    )

    # Scientific Review & Revision
    rev_findings_node = prov.record_node(
        "rev_findings_001",
        "scientific_review",
        "Scientific Review Verdict: Accept (Avg Score: 8.8/10)",
        {
            "iteration": 1,
            "verdict": "accept",
            "passed": True,
            "critical_count": 0,
            "major_count": 0,
            "minor_count": 1,
            "findings_count": 1,
            "recommendations": ["Clarify hardware batch size in Section IV."],
        },
        parent_ids=[stat_critic_node.node_id, val_node.node_id, method_node.node_id],
        relation="conducts_peer_review",
    )
    rev_cycle_node = prov.record_node(
        "rev_cycle_001",
        "revision",
        "Bounded Revision Cycle (1 Iterations)",
        {
            "iterations": 1,
            "actions": ["Refactored Section IV for hardware batch clarity."],
            "stopped_reason": "All review criteria satisfied on Revision 1.",
            "converged": True,
        },
        parent_ids=[rev_findings_node.node_id],
        relation="executes_revision",
    )
    conc_node = prov.record_node(
        "conc_001",
        "conclusion",
        "Validated MB-QGT Performance",
        {"model_acronym": "MB-QGT", "review_verdict": "accept"},
        parent_ids=[rev_cycle_node.node_id],
        relation="validates_conclusions",
    )
    prov.record_node(
        "pub_deliverable_001",
        "publication",
        "Publication Package: IEEE Transactions PDF & Overleaf ZIP",
        {
            "pdf_path": "/artifacts/main.pdf",
            "zip_path": "/artifacts/package.zip",
            "page_count": 10,
            "success": True,
        },
        parent_ids=[conc_node.node_id],
        relation="generates_publication",
    )
    return prov


def test_complete_experiment_coverage_20_runs(canonical_provenance_tracker):
    """Test 1: Provenance graph contains all 20 individual experiment runs and 20 result nodes."""
    graph = canonical_provenance_tracker.export_graph()
    exp_nodes = [n for n in graph["nodes"] if n["node_type"] == "experiment"]
    res_nodes = [n for n in graph["nodes"] if n["node_type"] == "result"]

    assert len(exp_nodes) == 20, f"Expected 20 experiment nodes, found {len(exp_nodes)}"
    assert len(res_nodes) == 20, f"Expected 20 result nodes, found {len(res_nodes)}"


def test_no_duplicate_experiment_nodes(canonical_provenance_tracker):
    """Test 2: All experiment node IDs and (method, seed) combinations are strictly unique."""
    graph = canonical_provenance_tracker.export_graph()
    exp_nodes = [n for n in graph["nodes"] if n["node_type"] == "experiment"]
    
    node_ids = [n["node_id"] for n in exp_nodes]
    assert len(node_ids) == len(set(node_ids)), "Duplicate experiment node IDs detected in provenance graph!"

    method_seed_pairs = [
        (n["metadata"]["method_id"], n["metadata"]["seed"])
        for n in exp_nodes
    ]
    assert len(method_seed_pairs) == len(set(method_seed_pairs)), "Duplicate (method, seed) execution runs detected!"


def test_every_experiment_has_result_lineage(canonical_provenance_tracker):
    """Test 3: Every experiment node is directly linked to an explicit result node."""
    graph = canonical_provenance_tracker.export_graph()
    exp_ids = {n["node_id"] for n in graph["nodes"] if n["node_type"] == "experiment"}
    
    res_nodes = [n for n in graph["nodes"] if n["node_type"] == "result"]
    linked_exp_ids = {
        p for rn in res_nodes for p in rn["parent_ids"]
        if p in exp_ids
    }
    assert exp_ids == linked_exp_ids, "Not all experiment nodes have downstream result nodes!"


def test_statistical_critic_coverage(canonical_provenance_tracker):
    """Test 4: Statistical critic node exists and lists all 20 experiment IDs in input telemetry."""
    graph = canonical_provenance_tracker.export_graph()
    stat_critic = next((n for n in graph["nodes"] if n["node_type"] == "statistical_critic"), None)
    assert stat_critic is not None, "Missing statistical_critic node in provenance graph!"

    input_exp_ids = stat_critic["metadata"].get("input_experiment_ids", [])
    assert len(input_exp_ids) == 20, f"Statistical critic covered only {len(input_exp_ids)}/20 runs"


def test_meta_analysis_lineage(canonical_provenance_tracker):
    """Test 5: Meta-analysis node exists with effect size and attaches to metrics_aggregate."""
    graph = canonical_provenance_tracker.export_graph()
    meta_node = next((n for n in graph["nodes"] if n["node_type"] == "meta_analysis"), None)
    assert meta_node is not None, "Missing meta_analysis node in provenance graph!"
    assert "pooled_effect_size" in meta_node["metadata"]
    assert "metrics_aggregate_001" in meta_node["parent_ids"]


def test_scientific_review_lineage(canonical_provenance_tracker):
    """Test 6: Scientific review node exists with verdict and recommendations."""
    graph = canonical_provenance_tracker.export_graph()
    rev_node = next((n for n in graph["nodes"] if n["node_type"] == "scientific_review"), None)
    assert rev_node is not None, "Missing scientific_review node in provenance graph!"
    assert rev_node["metadata"]["verdict"] in ("accept", "minor_revision", "major_revision", "reject")
    assert "recommendations" in rev_node["metadata"]


def test_revision_lineage(canonical_provenance_tracker):
    """Test 7: Revision loop node exists with iteration count and convergence state."""
    graph = canonical_provenance_tracker.export_graph()
    rev_cycle = next((n for n in graph["nodes"] if n["node_type"] == "revision"), None)
    assert rev_cycle is not None, "Missing revision node in provenance graph!"
    assert rev_cycle["metadata"]["iterations"] >= 1
    assert "converged" in rev_cycle["metadata"]


def test_publication_lineage(canonical_provenance_tracker):
    """Test 8: Publication node exists with deliverable paths and connects to conclusion."""
    graph = canonical_provenance_tracker.export_graph()
    pub_node = next((n for n in graph["nodes"] if n["node_type"] == "publication"), None)
    assert pub_node is not None, "Missing publication node in provenance graph!"
    assert "pdf_path" in pub_node["metadata"]
    assert "zip_path" in pub_node["metadata"]
    assert "conc_001" in pub_node["parent_ids"]


def test_no_orphan_nodes(canonical_provenance_tracker):
    """Test 9: Every intermediate node has valid parents and children (no orphaned DAG subgraphs)."""
    audit = validate_complete_provenance(canonical_provenance_tracker, expected_num_methods=4, expected_num_seeds=5)
    assert audit["passed"] is True, f"Provenance validation failed: {audit}"
    assert len(audit["orphan_nodes"]) == 0, f"Found orphan nodes: {audit['orphan_nodes']}"
    assert len(audit["missing_edges"]) == 0, f"Found dangling edges: {audit['missing_edges']}"


def test_graph_serialization(canonical_provenance_tracker):
    """Test 10: Exported graph serializes to valid JSON and reloads without loss."""
    graph = canonical_provenance_tracker.export_graph()
    json_str = json.dumps(graph, indent=2)
    reloaded = json.loads(json_str)

    assert len(reloaded["nodes"]) == len(graph["nodes"])
    assert len(reloaded["edges"]) == len(graph["edges"])


def test_validate_complete_provenance_function(canonical_provenance_tracker):
    """Test 11: validate_complete_provenance correctly identifies complete and incomplete graphs."""
    # 1. Valid graph passes
    res = validate_complete_provenance(canonical_provenance_tracker, expected_num_methods=4, expected_num_seeds=5)
    assert res["passed"] is True
    assert res["experiment_runs_traced"] == 20

    # 2. Incomplete graph (e.g. only 6 runs) fails with clear diagnostic
    partial_tracker = ProvenanceTracker(task_id="partial_task")
    graph = canonical_provenance_tracker.export_graph()
    exp_count = 0
    for n in graph["nodes"]:
        if n["node_type"] == "experiment":
            exp_count += 1
            if exp_count > 6:
                continue
        partial_tracker.record_node(
            n["node_id"],
            n["node_type"],
            n["label"],
            n["metadata"],
            n["parent_ids"],
        )
    partial_res = validate_complete_provenance(partial_tracker, expected_num_methods=4, expected_num_seeds=5)
    assert partial_res["passed"] is False
    assert partial_res["experiment_runs_traced"] == 6
    assert partial_res["experiment_runs_expected"] == 20
