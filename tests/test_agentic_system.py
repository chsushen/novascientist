"""Unit and integration tests for NovaScientist v2.0 Agentic Architecture."""

import asyncio
import pytest
from pathlib import Path
from dataclasses import asdict

from backend.core.agentic_planner import ResearchPlan, ResearchPlannerAgent
from backend.core.evidence_agent import ClaimRecord, EvidenceBundle, LiteratureAgent, SourceRecord
from backend.core.evidence_validator import EvidenceValidationReport, EvidenceValidator, ValidatedClaim
from backend.core.experiment_agent import ExperimentAgent, ExperimentRecord, ExperimentSpec
from backend.core.methodology_agent import MethodologyAgent, MethodologySpec
from backend.core.provenance import ProvenanceNode, ProvenanceTracker
from backend.core.research_memory import ResearchMemory
from backend.core.scientific_reviewer import BoundedRevisionLoop, ReviewFinding, RevisionHistory, ScientificReviewReport, ScientificReviewerAgent
from backend.core.statistical_critic import StatisticalCriticAgent, StatisticalCritique
from backend.core.orchestrator import NovaScientistOrchestrator
from backend.core.universal_engine import ComputationalDomain, UniversalBenchmarkEngine


def test_research_planner_creation_and_validation():
    planner = ResearchPlannerAgent()
    plan = planner.create_plan(
        research_question="Physics-Informed Dynamic Neural Surrogates under Bounded Memory",
        num_seeds=5,
    )
    assert plan.is_valid is True
    assert plan.domain == ComputationalDomain.PHYSICS_SURROGATE
    assert plan.model_acronym in ["Ham-QNO", "Ada-PINN"]
    assert len(plan.objectives) >= 2
    assert len(plan.sub_questions) >= 1
    assert len(plan.experiment_plan) >= 4
    assert "plan_" in plan.plan_id
    d = plan.to_dict()
    assert d["domain"] == "physics_surrogate"


@pytest.mark.asyncio
async def test_literature_agent_evidence_and_claims():
    from tests.fixtures.literature.fixtures import MockLiteratureService
    agent = LiteratureAgent(lit_service=MockLiteratureService())
    evidence = await agent.gather_evidence("Low-Rank Dynamic Graph Attention", limit=4)
    assert evidence.total_sources_retrieved >= 2
    assert len(evidence.claims) >= 2
    for s in evidence.sources:
        assert s.doi != ""
        assert s.source_origin == "test_fixture"
        if s.retrieved_text_available:
            for c in s.claims:
                assert c.supporting_text != ""
                assert c.verification_status.value in ["grounded", "verified"]


def test_research_memory_storage_and_retrieval(tmp_path):
    mem_path = tmp_path / "test_memory.json"
    memory = ResearchMemory(store_path=mem_path)
    
    mock_metrics = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 0.912, "mean_memory_mb": 64.0, "mean_latency_ms": 3.5},
            "dense_baseline": {"mean_accuracy": 0.825, "mean_memory_mb": 280.0, "mean_latency_ms": 14.0},
        },
        "timestamp": "2026-09-04T00:00:00Z",
    }
    
    item = memory.store_task(
        task_id="task_test_01",
        topic="Volumetric CT Slices Segmentation",
        domain="Medical Imaging & Computer Vision",
        plan_id="plan_test_01",
        sources=[{"title": "Medical ViT"}],
        claims=[ClaimRecord(claim_id="c1", claim_text="Sample claim", source_id="s1", supporting_text="Sample passage text", category="empirical")],
        metrics=mock_metrics,
        review_passed=True,
    )
    assert item.proposed_acc == 91.2
    assert item.mem_reduction_pct > 70.0

    retrieved = memory.find_relevant_knowledge("Volumetric CT", "Medical Imaging & Computer Vision")
    assert len(retrieved) == 1
    assert retrieved[0]["task_id"] == "task_test_01"


def test_methodology_and_experiment_agent():
    planner = ResearchPlannerAgent()
    plan = planner.create_plan("Dynamic Graph Attention Networks", num_seeds=5)
    
    evidence = EvidenceBundle(
        topic=plan.topic_title,
        domain=plan.domain_display_name,
        sources=[SourceRecord(source_id="s1", title="GNN Paper", authors=["A. Scholar"], year=2024, doi="10.1145/123", url="https://doi.org/10.1145/123", venue="KDD")],
        claims=[ClaimRecord(claim_id="c1", claim_text="Graph memory bottleneck.", source_id="s1", supporting_text="Graph memory bottleneck passage.", category="theoretical")],
    )

    method_agent = MethodologyAgent()
    spec = method_agent.synthesize_methodology(plan, evidence)
    assert spec.model_acronym in ["MB-QGT", "Ada-GNN"]
    assert len(spec.established_facts) >= 2
    assert len(spec.hypotheses) >= 2
    assert len(spec.assumptions) >= 2

    exp_agent = ExperimentAgent()
    exp_spec = exp_agent.create_spec(spec, dataset_name="PeMS-Bay", sample_count=52116)
    assert len(exp_spec.seeds) == 5
    assert len(exp_spec.methods_to_evaluate) == 4


def test_evidence_validator_support_scoring():
    validator = EvidenceValidator()
    
    evidence = EvidenceBundle(
        topic="Graph Transformer",
        domain="Graph Systems",
        claims=[
            ClaimRecord(claim_id="c1", claim_text="Proposed method reduces working memory by over 50% during tensor evaluation.", source_id="s1", supporting_text="Proposed method reduces working memory by over 50% during tensor evaluation.", category="empirical"),
            ClaimRecord(claim_id="c2", claim_text="Quantization maintains competitive accuracy vs dense baselines.", source_id="s1", supporting_text="Quantization maintains competitive accuracy vs dense baselines.", category="empirical"),
        ]
    )
    
    mock_experiments = [
        ExperimentRecord(experiment_id="exp_001", method_id="proposed_mb_qgt", method_name="MB-QGT", seed=42, dataset="Data", accuracy=90.0, memory_mb=64.0, latency_ms=4.0, throughput=250.0, compression_ratio=4.0, runtime_sec=0.4),
        ExperimentRecord(experiment_id="exp_002", method_id="dense_baseline", method_name="Dense", seed=42, dataset="Data", accuracy=82.0, memory_mb=280.0, latency_ms=16.0, throughput=62.5, compression_ratio=1.0, runtime_sec=1.6),
    ]

    mock_metrics = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 0.90, "mean_memory_mb": 64.0},
            "dense_baseline": {"mean_accuracy": 0.82, "mean_memory_mb": 280.0},
        }
    }

    report = validator.validate_evidence(evidence, mock_experiments, mock_metrics)
    assert report.is_publishable is True
    assert report.unsupported_count == 0
    assert report.supported_count >= 1
    for c in report.claims:
        assert c.status in ["supported", "weak"]
        assert c.support_score >= 0.70


def test_provenance_tracker_lineage_graph():
    prov = ProvenanceTracker(task_id="task_100")
    q = prov.record_node("q_1", "question", "How to optimize neural surrogates?")
    p = prov.record_node("p_1", "plan", "Surrogate Plan", parent_ids=[q.node_id])
    m = prov.record_node("m_1", "methodology", "Ham-QNO", parent_ids=[p.node_id])
    e = prov.record_node("e_1", "experiment", "Seed 42", parent_ids=[m.node_id])
    c = prov.record_node("c_1", "conclusion", "Surrogate Proven Valid", parent_ids=[e.node_id])

    lineage = prov.trace_lineage("c_1")
    node_ids = [n.node_id for n in lineage]
    assert "c_1" in node_ids
    assert "e_1" in node_ids
    assert "m_1" in node_ids
    assert "p_1" in node_ids
    assert "q_1" in node_ids

    graph = prov.export_graph()
    assert graph["total_nodes"] == 5
    assert graph["total_edges"] == 4


def test_statistical_critic_and_scientific_reviewer():
    critic = StatisticalCriticAgent()
    
    mock_metrics = {
        "seeds": [42, 179, 316, 453, 590],
        "methods": {
            "proposed_mb_qgt": {"name": "Proposed MB-QGT", "std_accuracy": 0.005, "seed_results": [{"accuracy": 0.88}, {"accuracy": 0.89}, {"accuracy": 0.885}, {"accuracy": 0.887}, {"accuracy": 0.889}]},
            "dense_baseline": {"name": "Dense Baseline", "std_accuracy": 0.006, "seed_results": [{"accuracy": 0.82}, {"accuracy": 0.83}, {"accuracy": 0.825}, {"accuracy": 0.828}, {"accuracy": 0.829}]},
        },
        "meta_analysis": {
            "pooled_effect_size": 0.063,
            "z_statistic": 14.2,
            "i_squared_percent": 0.0,
        }
    }

    critique = critic.evaluate_statistics(mock_metrics)
    assert critique.passed is True
    assert critique.sample_size_sufficient is True
    assert critique.variance_bounded is True
    assert critique.meta_analysis_significant is True

    reviewer = ScientificReviewerAgent()
    mock_latex = r"\title{Quantum Tensor Networks} \section{Introduction} Deterministic evaluation with proposed_mb_qgt and limitations analysis."
    report = reviewer.review(latex_text=mock_latex, metrics_dict=mock_metrics, stat_critique=critique)
    assert report.passed is True
    assert report.overall_verdict in ["accept", "minor_revision"]


def test_bounded_revision_loop_termination():
    loop = BoundedRevisionLoop()
    mock_latex = r"\title{Test Topic} \section{Introduction} This paper completely solves the memory wall with flawless accuracy."
    mock_metrics = {
        "seeds": [42, 179, 316, 453, 590],
        "methods": {"proposed_mb_qgt": {"std_accuracy": 0.01}, "dense_baseline": {"std_accuracy": 0.01}},
        "meta_analysis": {"z_statistic": 5.5, "i_squared_percent": 0.0},
    }

    revised_latex, final_report, history = loop.run_revision_loop(
        raw_latex=mock_latex,
        metrics_dict=mock_metrics,
    )
    assert history.total_iterations <= 3
    assert "completely solves" not in revised_latex
    assert "effectively mitigates" in revised_latex


@pytest.mark.asyncio
async def test_agentic_orchestrator_full_execution(tmp_path):
    orchestrator = NovaScientistOrchestrator(output_dir=str(tmp_path / "dist"))
    result = await orchestrator.execute(
        topic="Electro-Hydrodynamic Mixing in Microfluidic Channels",
        execution_mode="Fast Microbenchmark",
        num_seeds=5,
        num_epochs=5,
    )
    assert result.success is True
    assert result.plan is not None
    assert result.evidence is not None
    assert result.methodology is not None
    assert len(result.experiment_records) >= 10
    assert result.validation_report is not None
    assert result.stat_critique is not None
    assert result.review_report is not None
    assert result.provenance_graph is not None
    assert result.revision_history is not None
    assert Path(result.pdf_path).exists()
    assert Path(result.zip_path).exists()
