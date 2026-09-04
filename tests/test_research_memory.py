"""
Tests for Phase 6: Persistent Research Memory / Knowledge Graph.

Validates:
1. Atomic file-backed persistence and reload across ResearchMemory instances.
2. Graceful recovery and backup when encountering corrupted or malformed JSON files.
3. Keyword, domain, and model acronym relevance ranking in query matching.
4. ResearchMemoryItem serialization, deserialization, and schema integrity.
5. Aggregate summary calculation and Knowledge Graph topology export.
6. Central Orchestrator integration for prior memory querying and storage.
"""

import json
import pytest
from pathlib import Path

from backend.core.agentic_planner import ResearchPlan
from backend.core.evidence_agent import ClaimRecord, EvidenceBundle, SourceRecord, VerificationStatus
from backend.core.latex_assembler import AuthorProfile
from backend.core.orchestrator import NovaScientistOrchestrator, OrchestratorResult
from backend.core.research_memory import (
    ResearchMemory,
    ResearchMemoryItem,
    tokenize_text,
)


def test_tokenize_text():
    """Verify stopword removal and token extraction."""
    tokens = tokenize_text("Adaptive Quantization for Graph Neural Networks under Memory Bounds")
    assert "adaptive" in tokens
    assert "quantization" in tokens
    assert "graph" in tokens
    assert "neural" in tokens
    assert "networks" in tokens
    assert "bounds" in tokens
    assert "for" not in tokens
    assert "under" not in tokens


def test_research_memory_item_dataclass_and_serialization():
    """Verify ResearchMemoryItem serialization and deserialization."""
    item = ResearchMemoryItem(
        task_id="task_001",
        topic="Adaptive Block Quantization for Graph Transformers",
        domain="Spatial & Spatiotemporal Graph Neural Networks",
        plan_id="plan_001",
        sources_count=5,
        claims_count=8,
        top_claims=["Dynamic block scaling prevents memory bus saturation."],
        methods_evaluated=["dense_baseline", "proposed_mb_qgt"],
        proposed_acc=88.5,
        baseline_acc=80.2,
        mem_reduction_pct=76.4,
        speedup_ratio=4.12,
        review_status="passed",
        timestamp="2026-09-04T12:00:00Z",
        model_acronym="MB-QGT",
        dataset_name="METR-LA Sensor Benchmark",
        meta_effect_size=0.083,
        meta_i_squared=22.4,
        provenance_summary={"question": 1, "plan": 1, "source": 5, "claim": 8},
    )

    d = item.to_dict()
    assert d["task_id"] == "task_001"
    assert d["proposed_acc"] == 88.5
    assert d["model_acronym"] == "MB-QGT"

    reconstructed = ResearchMemoryItem.from_dict(d)
    assert reconstructed.task_id == "task_001"
    assert reconstructed.top_claims == item.top_claims
    assert reconstructed.provenance_summary == item.provenance_summary


def test_atomic_persistence_and_reload(tmp_path):
    """Verify that entries are atomically persisted to disk and reloadable by fresh instances."""
    store_file = tmp_path / "memory_store.json"
    memory_1 = ResearchMemory(store_path=store_file)

    mock_metrics = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 0.895, "mean_memory_mb": 72.0, "mean_latency_ms": 8.0},
            "dense_baseline": {"mean_accuracy": 0.802, "mean_memory_mb": 390.0, "mean_latency_ms": 34.0},
        },
        "meta_analysis": {"pooled_effect_size": 0.093, "i_squared_percent": 18.5},
        "timestamp": "2026-09-04T12:00:00Z",
    }

    item = memory_1.store_task(
        task_id="task_persistence_01",
        topic="Physics-Informed Neural Operator PDE Solver",
        domain="Physics-Informed Neural Operators & Scientific Machine Learning",
        plan_id="plan_pde_01",
        sources=[{"title": "Fourier Neural Operators"}],
        claims=[{"claim_text": "Spectral methods reduce discretization error."}],
        metrics=mock_metrics,
        review_passed=True,
        model_acronym="PINO",
        dataset_name="Navier-Stokes 2D Benchmark",
    )

    assert store_file.exists()
    assert item.proposed_acc == 89.5

    # Reload with a completely separate instance
    memory_2 = ResearchMemory(store_path=store_file)
    entry = memory_2.get_entry("task_persistence_01")
    assert entry is not None
    assert entry["topic"] == "Physics-Informed Neural Operator PDE Solver"
    assert entry["model_acronym"] == "PINO"
    assert entry["proposed_acc"] == 89.5


def test_corrupted_file_recovery(tmp_path):
    """Verify that ResearchMemory recovers gracefully from corrupted JSON without crashing."""
    corrupted_file = tmp_path / "corrupted_memory.json"
    # Write corrupt data
    with open(corrupted_file, "w", encoding="utf-8") as f:
        f.write("{ invalid json syntax !!! @@@ broken string ...")

    # Instance should load cleanly with empty dictionary and create a backup
    memory = ResearchMemory(store_path=corrupted_file)
    assert len(memory.get_all_entries()) == 0

    # Storing a new item works seamlessly
    mock_metrics = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 0.85, "mean_memory_mb": 60.0, "mean_latency_ms": 5.0},
            "dense_baseline": {"mean_accuracy": 0.78, "mean_memory_mb": 200.0, "mean_latency_ms": 20.0},
        },
    }
    memory.store_task(
        task_id="task_post_recovery",
        topic="Computer Vision Segmentation",
        domain="Medical Imaging & Computer Vision",
        plan_id="plan_rec_01",
        sources=[],
        claims=[],
        metrics=mock_metrics,
    )

    assert len(memory.get_all_entries()) == 1
    assert memory.get_entry("task_post_recovery") is not None


def test_json_list_root_recovery(tmp_path):
    """Verify that if a memory file contains a valid JSON list, it is ingested correctly."""
    list_file = tmp_path / "list_memory.json"
    raw_list = [
        {"task_id": "task_list_01", "topic": "Graph Neural Networks", "domain": "Graph Machine Learning", "proposed_acc": 85.0},
        {"task_id": "task_list_02", "topic": "Vision Transformers", "domain": "Computer Vision", "proposed_acc": 88.0},
    ]
    with open(list_file, "w", encoding="utf-8") as f:
        json.dump(raw_list, f)

    memory = ResearchMemory(store_path=list_file)
    entries = memory.get_all_entries()
    assert len(entries) == 2
    assert memory.get_entry("task_list_01") is not None
    assert memory.get_entry("task_list_02") is not None


def test_keyword_and_domain_relevance_ranking(tmp_path):
    """Verify relevance ranking when querying prior knowledge across multiple domains."""
    store_file = tmp_path / "relevance_memory.json"
    memory = ResearchMemory(store_path=store_file)

    mock_metrics = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 0.88, "mean_memory_mb": 70.0, "mean_latency_ms": 7.0},
            "dense_baseline": {"mean_accuracy": 0.80, "mean_memory_mb": 350.0, "mean_latency_ms": 30.0},
        }
    }

    # Store 3 diverse tasks
    memory.store_task(
        task_id="task_gnn",
        topic="Dynamic Quantization for Spatial Graph Neural Networks",
        domain="Spatial & Spatiotemporal Graph Neural Networks",
        plan_id="plan_gnn",
        sources=[],
        claims=[{"claim_text": "Block quantization reduces graph convolution memory bus pressure."}],
        metrics=mock_metrics,
        model_acronym="MB-QGT",
    )
    memory.store_task(
        task_id="task_pde",
        topic="Physics-Informed Neural Operator for Turbulent Flow",
        domain="Physics-Informed Neural Operators & Scientific Machine Learning",
        plan_id="plan_pde",
        sources=[],
        claims=[{"claim_text": "Fourier operators preserve energy conservation."}],
        metrics=mock_metrics,
        model_acronym="PINO",
    )
    memory.store_task(
        task_id="task_med",
        topic="Volumetric CT Slices Segmentation with Swin Transformers",
        domain="Medical Imaging & Computer Vision",
        plan_id="plan_med",
        sources=[],
        claims=[{"claim_text": "Hierarchical shifted windows improve boundary segmentation."}],
        metrics=mock_metrics,
        model_acronym="SwinUNet",
    )

    # Query 1: GNN topic & domain -> task_gnn should be rank 1
    results_gnn = memory.find_relevant_knowledge(
        topic="Adaptive Graph Neural Network Quantization under MB-QGT",
        domain="Spatial & Spatiotemporal Graph Neural Networks",
        top_k=2,
    )
    assert len(results_gnn) >= 1
    assert results_gnn[0]["task_id"] == "task_gnn"
    assert results_gnn[0]["relevance_score"] > 3.0

    # Query 2: Medical imaging topic
    typed_med = memory.query_prior_knowledge(
        topic="CT Volumetric Medical Segmentation",
        domain="Medical Imaging & Computer Vision",
        top_k=1,
    )
    assert len(typed_med) == 1
    assert isinstance(typed_med[0], ResearchMemoryItem)
    assert typed_med[0].task_id == "task_med"


def test_knowledge_graph_export_and_summary(tmp_path):
    """Verify Knowledge Graph export and statistical summary aggregation."""
    store_file = tmp_path / "kg_memory.json"
    memory = ResearchMemory(store_path=store_file)

    mock_metrics_1 = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 0.90, "mean_memory_mb": 60.0, "mean_latency_ms": 6.0},
            "dense_baseline": {"mean_accuracy": 0.80, "mean_memory_mb": 300.0, "mean_latency_ms": 30.0},
        }
    }
    mock_metrics_2 = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 0.85, "mean_memory_mb": 80.0, "mean_latency_ms": 10.0},
            "dense_baseline": {"mean_accuracy": 0.75, "mean_memory_mb": 240.0, "mean_latency_ms": 25.0},
        }
    }

    memory.store_task(
        task_id="t1",
        topic="Topic 1",
        domain="Domain A",
        plan_id="p1",
        sources=[],
        claims=[],
        metrics=mock_metrics_1,
        review_passed=True,
    )
    memory.store_task(
        task_id="t2",
        topic="Topic 2",
        domain="Domain B",
        plan_id="p2",
        sources=[],
        claims=[],
        metrics=mock_metrics_2,
        review_passed=False,
    )

    summary = memory.get_summary()
    assert summary["total_tasks"] == 2
    assert len(summary["domains_represented"]) == 2
    assert summary["avg_proposed_accuracy"] == 87.5
    assert summary["review_pass_rate"] == 0.5

    kg = memory.export_knowledge_graph()
    assert "nodes" in kg
    assert "edges" in kg
    assert kg["total_nodes"] >= 4  # 2 tasks + 2 domains + methods
    assert kg["total_edges"] >= 2  # BELONGS_TO_DOMAIN relations


def test_entry_deletion_and_clear(tmp_path):
    """Verify single entry deletion and full memory clearing."""
    store_file = tmp_path / "delete_memory.json"
    memory = ResearchMemory(store_path=store_file)

    mock_metrics = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 0.85, "mean_memory_mb": 50.0, "mean_latency_ms": 5.0},
            "dense_baseline": {"mean_accuracy": 0.80, "mean_memory_mb": 200.0, "mean_latency_ms": 20.0},
        }
    }

    memory.store_task("task_a", "Topic A", "Domain A", "pA", [], [], mock_metrics)
    memory.store_task("task_b", "Topic B", "Domain B", "pB", [], [], mock_metrics)

    assert len(memory.get_all_entries()) == 2

    # Delete task_a
    del_res = memory.delete_entry("task_a")
    assert del_res is True
    assert memory.get_entry("task_a") is None
    assert len(memory.get_all_entries()) == 1

    # Clear all
    memory.clear()
    assert len(memory.get_all_entries()) == 0

    # Reload from disk confirms empty
    memory_reloaded = ResearchMemory(store_path=store_file)
    assert len(memory_reloaded.get_all_entries()) == 0


@pytest.mark.asyncio
async def test_orchestrator_research_memory_lifecycle(tmp_path):
    """Verify Orchestrator queries prior research memory and stores completed task telemetry."""
    mem_file = tmp_path / "orchestrator_mem.json"
    custom_memory = ResearchMemory(store_path=mem_file)

    # Seed 1 prior task into memory
    custom_memory.store_task(
        task_id="task_prior_01",
        topic="Prior Traffic Sensor Modeling",
        domain="Spatial & Spatiotemporal Graph Neural Networks",
        plan_id="plan_prior_01",
        sources=[],
        claims=[{"claim_text": "Spatial adjacency graph reduces sensor error."}],
        metrics={
            "methods": {
                "proposed_mb_qgt": {"mean_accuracy": 0.88, "mean_memory_mb": 72.0, "mean_latency_ms": 8.0},
                "dense_baseline": {"mean_accuracy": 0.80, "mean_memory_mb": 390.0, "mean_latency_ms": 34.0},
            }
        },
        review_passed=True,
    )

    orchestrator = NovaScientistOrchestrator(
        output_dir=str(tmp_path / "dist"),
        memory=custom_memory,
    )

    result = await orchestrator.execute(
        topic="Traffic Sensor Modeling with Graph Neural Networks",
        author=AuthorProfile("Test Researcher", "AI Institute", "test@inst.edu"),
        target_length="4_page_conference",
        execution_mode="fast_microbenchmark",
        num_seeds=3,
        num_epochs=5,
    )

    assert isinstance(result, OrchestratorResult)
    # Orchestrator retrieved prior knowledge from memory
    assert result.prior_knowledge is not None
    assert len(result.prior_knowledge) >= 1
    assert result.prior_knowledge[0]["task_id"] == "task_prior_01"

    # Orchestrator also stored the newly completed task
    all_entries = custom_memory.get_all_entries()
    assert len(all_entries) == 2
