"""Tests for Canonical Demo Artifact and Lineage Provenance Integrity.

Verifies that run_summary.json and provenance_graph.json maintain 100% structural
consistency, complete 5-seed empirical tracing, zero dangling DAG references, and
verifiable metric grounding.
"""

import json
from pathlib import Path
import pytest


CANONICAL_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "demo" / "run_canonical_01"
RUN_SUMMARY_PATH = CANONICAL_DIR / "run_summary.json"
PROVENANCE_GRAPH_PATH = CANONICAL_DIR / "provenance_graph.json"
README_PATH = CANONICAL_DIR / "README.md"


def test_canonical_files_exist():
    """Verify physical presence of canonical demo files."""
    assert RUN_SUMMARY_PATH.is_file(), f"Missing {RUN_SUMMARY_PATH}"
    assert PROVENANCE_GRAPH_PATH.is_file(), f"Missing {PROVENANCE_GRAPH_PATH}"
    assert README_PATH.is_file(), f"Missing {README_PATH}"


def test_run_summary_metric_structure_and_values():
    """Verify run_summary.json contains valid material claims and non-zero metrics."""
    with open(RUN_SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["run_id"] == "run_canonical_01"
    assert summary["num_seeds"] == 5
    assert len(summary["seeds"]) == 5
    assert summary["seeds"] == [42, 179, 316, 453, 590]

    metrics = summary["metrics"]
    assert metrics["proposed_accuracy"] > metrics["dense_accuracy"]
    assert metrics["proposed_accuracy"] == pytest.approx(88.35, abs=0.1)
    assert metrics["dense_accuracy"] == pytest.approx(81.94, abs=0.1)
    assert metrics["accuracy_gain_pct"] == pytest.approx(6.41, abs=0.1)
    assert metrics["memory_reduction_pct"] >= 80.0
    assert metrics["latency_speedup"] >= 4.0
    assert metrics["meta_effect_size"] > 0
    assert metrics["z_statistic"] > 10.0
    assert metrics["p_value_z"] < 0.001

    integrity = summary["research_integrity"]
    assert integrity["verified_doi_rate"] == 1.0
    assert integrity["unsupported_claim_rate"] == 0.0
    assert integrity["sample_size_sufficient"] is True
    assert integrity["single_seed_fabrication"] is False


def test_provenance_graph_dag_integrity():
    """Verify provenance_graph.json forms a valid, closed DAG with no dangling nodes."""
    with open(PROVENANCE_GRAPH_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    assert len(nodes) >= 20, f"Expected at least 20 nodes, got {len(nodes)}"
    assert len(edges) >= 20, f"Expected at least 20 edges, got {len(edges)}"

    node_ids = {n["node_id"] for n in nodes}
    assert len(node_ids) == len(nodes), "Duplicate node_id found in provenance graph"

    # Verify parent references exist
    for n in nodes:
        for pid in n.get("parent_ids", []):
            assert pid in node_ids, f"Dangling parent_id {pid} in node {n['node_id']}"

    # Verify edge endpoints exist
    for e in edges:
        assert e["source"] in node_ids, f"Dangling edge source {e['source']}"
        assert e["target"] in node_ids, f"Dangling edge target {e['target']}"
        assert "relation" in e, f"Missing relation in edge {e}"


def test_provenance_graph_five_seed_traceability():
    """Verify all 5 individual seed runs for both proposed and baseline exist in graph."""
    with open(PROVENANCE_GRAPH_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = {n["node_id"]: n for n in graph.get("nodes", [])}

    expected_seeds = [42, 179, 316, 453, 590]
    for seed in expected_seeds:
        proposed_node_id = f"seed_run_mbqgt_{seed}"
        assert proposed_node_id in nodes, f"Missing proposed seed run {proposed_node_id}"
        assert nodes[proposed_node_id]["metadata"]["seed"] == seed
        assert nodes[proposed_node_id]["metadata"]["final_accuracy"] > 0.85

        dense_node_id = f"seed_run_dense_{seed}"
        assert dense_node_id in nodes, f"Missing dense seed run {dense_node_id}"
        assert nodes[dense_node_id]["metadata"]["seed"] == seed


def test_provenance_graph_complete_scientific_lineage():
    """Verify presence of key scientific pipeline entities in graph."""
    with open(PROVENANCE_GRAPH_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)

    node_types = {n["node_type"] for n in graph.get("nodes", [])}
    required_types = {
        "question",
        "plan",
        "source",
        "doi_verification",
        "claim",
        "validation_report",
        "methodology",
        "experiment_spec",
        "seed_run",
        "metrics_aggregate",
        "statistical_analysis",
        "statistical_critic",
        "review_findings",
        "revision_cycle",
        "review_verdict",
        "conclusion",
        "deliverable",
    }

    missing_types = required_types - node_types
    assert not missing_types, f"Provenance graph missing node types: {missing_types}"
