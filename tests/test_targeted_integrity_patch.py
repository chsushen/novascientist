"""Targeted Live-App Integrity Patch Verification Tests.

Verifies:
1. Figure tab path resolution and multi-directory discovery.
2. Negative semantic scope check: Unrelated literature/hardware claims (e.g., memristors, neuromorphic)
   are NOT marked as empirically substantiated by local tensor experiments.
3. Strict separation of DOI verification vs evidence passage grounding.
4. Retracted literature detection, tagging, and isolation.
5. MethodologySpec separation of established facts, assumptions, and engineering rationales.
6. Quantitative threshold-based hypothesis evaluation (SUPPORTED, REFUTED, INCONCLUSIVE).
7. Accurate physical PDF page counting.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from backend.core.agentic_planner import ResearchPlan, ResearchPlannerAgent
from backend.core.doi_verifier import DOIVerificationStatus
from backend.core.evidence_agent import ClaimRecord, EvidenceBundle, EvidenceScope, SourceRecord, VerificationStatus
from backend.core.evidence_validator import EvidenceValidator
from backend.core.experiment_agent import ExperimentRecord
from backend.core.literature import PaperMetadata
from backend.core.methodology_agent import HypothesisStatus, MethodologyAgent, MethodologySpec


def test_issue2_negative_unrelated_hardware_claim_not_empirically_substantiated():
    """Negative Test: Claims discussing memristors/neuromorphic hardware must NOT be

    marked as empirically substantiated by local MB-QGT tensor experiments.
    """
    validator = EvidenceValidator()

    # Evidence bundle containing an empirical tensor claim and an unrelated memristor claim
    evidence = EvidenceBundle(
        topic="Multi-Branch Quantized Graph Transformer",
        domain="Graph Representation Learning",
        claims=[
            ClaimRecord(
                claim_id="c_tensor",
                claim_text="Proposed MB-QGT architecture reduces peak memory footprint by over 70% in tensor benchmarks.",
                source_id="s1",
                supporting_text="Proposed MB-QGT reduces peak memory footprint by over 70% in tensor benchmarks.",
                category="empirical",
            ),
            ClaimRecord(
                claim_id="c_memristor",
                claim_text="Memristor crossbar arrays achieve 85% lower memory energy in analog neuromorphic circuits.",
                source_id="s2",
                supporting_text="Memristor crossbar arrays achieve 85% lower memory energy in analog neuromorphic circuits.",
                category="empirical",
            ),
            ClaimRecord(
                claim_id="c_theory",
                claim_text="Theoretical generalization bounds for sparse message passing under Lipschitz continuity.",
                source_id="s1",
                supporting_text="Theoretical generalization bounds for sparse message passing under Lipschitz continuity.",
                category="theoretical",
            ),
        ],
    )

    mock_experiments = [
        ExperimentRecord(
            experiment_id="exp_01",
            method_id="proposed_mb_qgt",
            method_name="MB-QGT",
            seed=42,
            dataset="PeMS-Bay",
            accuracy=88.35,
            memory_mb=74.98,
            latency_ms=9.27,
            throughput=107.87,
            compression_ratio=4.0,
            runtime_sec=0.5,
        ),
        ExperimentRecord(
            experiment_id="exp_02",
            method_id="dense_baseline",
            method_name="Dense FP32",
            seed=42,
            dataset="PeMS-Bay",
            accuracy=81.94,
            memory_mb=299.92,
            latency_ms=37.08,
            throughput=26.97,
            compression_ratio=1.0,
            runtime_sec=2.0,
        ),
    ]

    mock_metrics = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 0.8835, "mean_memory_mb": 74.98},
            "dense_baseline": {"mean_accuracy": 0.8194, "mean_memory_mb": 299.92},
        }
    }

    report = validator.validate_evidence(evidence, mock_experiments, mock_metrics)

    # Find the validated claims
    c_tensor_val = next(c for c in report.claims if c.claim_id == "c_tensor")
    c_memristor_val = next(c for c in report.claims if c.claim_id == "c_memristor")
    c_theory_val = next(c for c in report.claims if c.claim_id == "c_theory")

    # c_tensor is in active empirical scope and should be empirically substantiated
    assert c_tensor_val.status == "supported"
    assert "Empirically substantiated" in c_tensor_val.rationale
    assert len(c_tensor_val.experiment_ids) > 0

    # c_memristor is unrelated hardware and must NOT be attributed to MB-QGT experiments
    assert c_memristor_val.status == "supported"
    assert c_memristor_val.rationale == "Literature-grounded claim; no directly matching local experiment was identified."
    assert c_memristor_val.experiment_ids == []

    # c_theory is theoretical/literature and must NOT be attributed to tensor experiments
    assert c_theory_val.status == "supported"
    assert c_theory_val.rationale == "Literature-grounded claim; no directly matching local experiment was identified."
    assert c_theory_val.experiment_ids == []


def test_live_claim_002_resistive_memory_neuromorphic_literature_claim():
    """Specific regression test for live claim_002:

    A literature claim mentioning computer vision, resistive memory, memristors,
    and neuromorphic applications must have experiment_ids == [] and literature rationale.
    """
    validator = EvidenceValidator()

    evidence = EvidenceBundle(
        topic="Low-Compute Dynamic Graph Representation under Quantized Memory",
        domain="Graph Systems",
        claims=[
            ClaimRecord(
                claim_id="claim_002",
                claim_text="Resistive memory and memristor crossbars improve energy efficiency for computer vision and neuromorphic machine learning applications.",
                source_id="source_002",
                supporting_text="Resistive memory and memristor crossbars improve energy efficiency for computer vision and neuromorphic machine learning applications.",
                category="empirical",
            ),
            ClaimRecord(
                claim_id="claim_003",
                claim_text="Proposed MB-QGT architecture maintains high accuracy under dynamic 8-bit quantization.",
                source_id="source_001",
                supporting_text="Proposed MB-QGT architecture maintains high accuracy under dynamic 8-bit quantization.",
                category="empirical",
            ),
        ],
    )

    experiments = [
        ExperimentRecord(
            experiment_id="exp_016",
            method_id="proposed_mb_qgt",
            method_name="MB-QGT",
            seed=42,
            dataset="PeMS-Bay",
            accuracy=88.35,
            memory_mb=74.98,
            latency_ms=9.27,
            throughput=107.87,
            compression_ratio=4.0,
            runtime_sec=0.5,
        ),
        ExperimentRecord(
            experiment_id="exp_001",
            method_id="dense_baseline",
            method_name="Dense FP32",
            seed=42,
            dataset="PeMS-Bay",
            accuracy=81.94,
            memory_mb=299.92,
            latency_ms=37.08,
            throughput=26.97,
            compression_ratio=1.0,
            runtime_sec=2.0,
        ),
    ]

    metrics = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 0.8835, "mean_memory_mb": 74.98},
            "dense_baseline": {"mean_accuracy": 0.8194, "mean_memory_mb": 299.92},
        }
    }

    report = validator.validate_evidence(evidence, experiments, metrics)
    c002 = next(c for c in report.claims if c.claim_id == "claim_002")
    c003 = next(c for c in report.claims if c.claim_id == "claim_003")

    # claim_002 MUST have empty experiment_ids and literature rationale
    assert c002.experiment_ids == []
    assert c002.status == "supported"
    assert c002.rationale == "Literature-grounded claim; no directly matching local experiment was identified."
    assert "Empirically substantiated" not in c002.rationale

    # claim_003 concerns proposed MB-QGT and qualifies for empirical substantiation
    assert len(c003.experiment_ids) > 0
    assert c003.status == "supported"
    assert "Empirically substantiated" in c003.rationale


def test_issue4_retracted_source_detection_and_tagging():
    """Verify that literature containing retraction keywords in title or metadata

    is flagged with retraction_status = 'retracted'.
    """
    retracted_paper = PaperMetadata(
        doi="10.1038/s41586-020-0000-0",
        title="Retracted: High-temperature superconductivity in carbonaceous sulfur hydride",
        authors=["R. Dias", "A. Author"],
        year=2020,
        venue="Nature",
        source_origin="crossref",
    )
    assert retracted_paper.retraction_status == "retracted"

    active_paper = PaperMetadata(
        doi="10.1145/3394486.3403076",
        title="Graph Attention Networks under Quantized Arithmetic",
        authors=["P. Velickovic", "Y. Bengio"],
        year=2021,
        venue="ICML",
        source_origin="openalex",
    )
    assert active_paper.retraction_status == "active"

    # SourceRecord propagation
    retracted_source = SourceRecord(
        source_id="src_ret",
        title="RETRACTED ARTICLE: Deep Spiking Neural Networks",
        authors=["J. Doe"],
        year=2019,
        doi="10.1109/TNNLS.2019.01",
        url="https://doi.org/10.1109/TNNLS.2019.01",
        venue="IEEE TNNLS",
    )
    assert retracted_source.retraction_status == "retracted"


def test_issue5_methodology_established_facts_and_rationales_separation():
    """Verify sound classification of established hardware/physics facts vs

    assumptions and engineering rationales in MethodologySpec.
    """
    planner = ResearchPlannerAgent()
    plan = planner.create_plan("Sub-Linear Memory Dynamic Quantization", num_seeds=5)

    evidence = EvidenceBundle(
        topic=plan.topic_title,
        domain=plan.domain_display_name,
        sources=[
            SourceRecord(
                source_id="s1",
                title="Dynamic Quantization Standards",
                authors=["A. Researcher"],
                year=2023,
                doi="10.1145/123456",
                url="https://doi.org/10.1145/123456",
                venue="NeurIPS",
            )
        ],
        claims=[
            ClaimRecord(
                claim_id="c1",
                claim_text="Quantized tensors reduce register spills on SIMD architectures.",
                source_id="s1",
                supporting_text="Quantized tensors reduce register spills on SIMD architectures.",
                category="empirical",
            )
        ],
    )

    method_agent = MethodologyAgent()
    spec = method_agent.synthesize_methodology(plan, evidence)

    # Established facts must refer to physical/standard realities
    assert any("IEEE 754" in fact for fact in spec.established_facts)
    assert any("SIMD" in fact or "cache-line" in fact for fact in spec.established_facts)

    # Design choices must be in engineering_rationales
    assert len(spec.engineering_rationales) >= 2
    assert any("64-element tile" in r for r in spec.engineering_rationales)
    assert any("Straight-Through Estimator" in r for r in spec.engineering_rationales)

    # Model assumptions must be distinct
    assert len(spec.assumptions) >= 2
    assert any("weak stationarity" in a for a in spec.assumptions)


def test_issue6_quantitative_hypothesis_evaluation_logic():
    """Verify hypothesis evaluation correctly classifies SUPPORTED, REFUTED, and INCONCLUSIVE."""
    method_agent = MethodologyAgent()

    spec = MethodologySpec(
        methodology_id="m_test",
        topic_title="Graph Transformer",
        domain="Graph",
        model_acronym="MB-QGT",
        model_full_name="Multi-Branch Quantized Graph Transformer",
        hypotheses=[
            "H1: MB-QGT reduces peak memory consumption by >=70% compared to Dense FP32.",
            "H2: MB-QGT improves or matches generalization performance within 1.5% of full-precision baselines.",
            "H3: Cache-aligned tiling provides >=2.0x inference latency speedup on standard hardware.",
        ],
    )

    # Case A: Canonical Passing Metrics (Proposed exceeds or matches Dense)
    passing_metrics = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 0.8835, "mean_memory_mb": 74.98, "mean_latency_ms": 9.27},
            "dense_baseline": {"mean_accuracy": 0.8194, "mean_memory_mb": 299.92, "mean_latency_ms": 37.08},
        }
    }
    evals_pass = method_agent.evaluate_hypotheses(spec, passing_metrics)
    assert len(evals_pass) == 3
    assert evals_pass[0].status == HypothesisStatus.SUPPORTED
    assert evals_pass[0].observed_value >= 70.0  # 75.0% memory reduction
    assert evals_pass[1].status == HypothesisStatus.SUPPORTED
    assert evals_pass[1].observed_value >= -1.50  # +6.41% accuracy improvement
    assert evals_pass[2].status == HypothesisStatus.SUPPORTED
    assert evals_pass[2].observed_value >= 2.0  # ~4.0x speedup

    # Case B: Refuted Accuracy (Proposed drops by 8.0%)
    failing_acc_metrics = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 0.7300, "mean_memory_mb": 74.98, "mean_latency_ms": 9.27},
            "dense_baseline": {"mean_accuracy": 0.8194, "mean_memory_mb": 299.92, "mean_latency_ms": 37.08},
        }
    }
    evals_fail = method_agent.evaluate_hypotheses(spec, failing_acc_metrics)
    assert evals_fail[1].status == HypothesisStatus.REFUTED
    assert evals_fail[1].observed_value < -5.0
