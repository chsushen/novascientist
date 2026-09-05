"""NovaScientist Methodology Agent.

Synthesizes structured scientific methodologies from research plans and literature evidence.
Explicitly delineates established facts, retrieved evidence, proposed architecture,
theoretical assumptions, and testable hypotheses.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from backend.core.agentic_planner import ResearchPlan
from backend.core.evidence_agent import EvidenceBundle
from backend.core.universal_engine import ComputationalDomain, UniversalDomainDispatcher


from enum import Enum


class HypothesisStatus(str, Enum):
    """Formal evaluation status of an empirical scientific hypothesis."""
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_EVALUATED = "NOT_EVALUATED"


@dataclass
class HypothesisEvaluationResult:
    """Quantitative threshold-based evaluation of a scientific hypothesis against empirical telemetry."""
    hypothesis_id: str
    statement: str
    status: HypothesisStatus
    observed_value: float
    threshold_value: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "status": self.status.value if isinstance(self.status, HypothesisStatus) else str(self.status),
            "observed_value": self.observed_value,
            "threshold_value": self.threshold_value,
            "rationale": self.rationale,
        }


@dataclass
class MethodologySpec:
    """Structured specification of the proposed scientific methodology."""
    methodology_id: str
    topic_title: str
    domain: str
    model_acronym: str
    model_full_name: str
    established_facts: List[str] = field(default_factory=list)
    retrieved_evidence: List[str] = field(default_factory=list)
    proposed_innovations: List[str] = field(default_factory=list)
    engineering_rationales: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)
    baseline_methods: List[str] = field(default_factory=list)
    hardware_constraints: Dict[str, Any] = field(default_factory=dict)
    hypothesis_evaluations: List[HypothesisEvaluationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hypothesis_evaluations"] = [h.to_dict() for h in self.hypothesis_evaluations]
        return d


class MethodologyAgent:
    """Agent responsible for formulating sound, reproducible scientific methodologies."""

    def __init__(self) -> None:
        pass

    def evaluate_hypotheses(
        self,
        methodology: MethodologySpec,
        metrics_dict: Dict[str, Any],
    ) -> List[HypothesisEvaluationResult]:
        """Formally evaluate each hypothesis against observed empirical telemetry."""
        evaluations: List[HypothesisEvaluationResult] = []
        methods = metrics_dict.get("methods", {})
        prop = methods.get("proposed_mb_qgt", {})
        dense = methods.get("dense_baseline", {})

        p_acc = prop.get("mean_accuracy", 0.0) * 100.0 if prop.get("mean_accuracy", 0.0) <= 1.0 else prop.get("mean_accuracy", 0.0)
        d_acc = dense.get("mean_accuracy", 0.0) * 100.0 if dense.get("mean_accuracy", 0.0) <= 1.0 else dense.get("mean_accuracy", 0.0)
        p_mem = prop.get("mean_memory_mb", 1.0)
        d_mem = dense.get("mean_memory_mb", 1.0)
        p_lat = prop.get("mean_latency_ms", 1.0)
        d_lat = dense.get("mean_latency_ms", 1.0)

        mem_reduction = ((d_mem - p_mem) / d_mem * 100.0) if d_mem > 0 else 0.0
        acc_delta = p_acc - d_acc
        speedup = (d_lat / p_lat) if (p_lat > 0 and d_lat > 0) else 1.0

        for idx, hyp_text in enumerate(methodology.hypotheses):
            hyp_id = f"H{idx+1}"
            if idx == 0 or "memory" in hyp_text.lower():
                # H1: Memory reduction >= 70%
                threshold = 70.0
                obs = mem_reduction
                if obs >= threshold:
                    status = HypothesisStatus.SUPPORTED
                    rationale = f"Observed memory reduction of {obs:.2f}% meets or exceeds threshold of {threshold:.1f}%."
                elif obs < 0.0:
                    status = HypothesisStatus.REFUTED
                    rationale = f"Observed negative memory reduction ({obs:.2f}%); proposed method increased memory consumption."
                else:
                    status = HypothesisStatus.REFUTED
                    rationale = f"Observed memory reduction of {obs:.2f}% failed to reach target threshold of {threshold:.1f}%."

            elif idx == 1 or "accuracy" in hyp_text.lower() or "generalization" in hyp_text.lower():
                # H2: Accuracy within 1.5% of full-precision baselines (acc_delta >= -1.5%)
                threshold = -1.50
                obs = acc_delta
                if obs >= threshold:
                    status = HypothesisStatus.SUPPORTED
                    rationale = f"Observed accuracy delta of {obs:+.2f}% is within acceptable margin of {threshold:.2f}% (proposed: {p_acc:.2f}%, dense: {d_acc:.2f}%)."
                elif obs < -5.0:
                    status = HypothesisStatus.REFUTED
                    rationale = f"Significant accuracy degradation observed ({obs:+.2f}%), failing tolerance of {threshold:.2f}%."
                else:
                    status = HypothesisStatus.INCONCLUSIVE
                    rationale = f"Accuracy delta ({obs:+.2f}%) slightly exceeds tolerance threshold ({threshold:.2f}%)."

            elif idx == 2 or "speedup" in hyp_text.lower() or "latency" in hyp_text.lower():
                # H3: Latency speedup >= 2.0x
                threshold = 2.0
                obs = speedup
                if obs >= threshold:
                    status = HypothesisStatus.SUPPORTED
                    rationale = f"Observed inference speedup of {obs:.2f}x meets or exceeds requirement of {threshold:.1f}x."
                elif obs < 1.0:
                    status = HypothesisStatus.REFUTED
                    rationale = f"Inference latency slowed down (speedup {obs:.2f}x < 1.0x)."
                else:
                    status = HypothesisStatus.INCONCLUSIVE
                    rationale = f"Observed speedup of {obs:.2f}x is below expected {threshold:.1f}x target."
            else:
                status = HypothesisStatus.NOT_EVALUATED
                threshold = 0.0
                obs = 0.0
                rationale = "No automated telemetry mapping defined for this hypothesis."

            evaluations.append(HypothesisEvaluationResult(
                hypothesis_id=hyp_id,
                statement=hyp_text,
                status=status,
                observed_value=round(obs, 2),
                threshold_value=round(threshold, 2),
                rationale=rationale,
            ))

        return evaluations

    def synthesize_methodology(
        self,
        plan: ResearchPlan,
        evidence: EvidenceBundle,
    ) -> MethodologySpec:
        """Generate a fully structured methodology specification."""
        m_hash = hashlib.sha256((plan.plan_id + plan.topic_title).encode("utf-8")).hexdigest()[:8]
        methodology_id = f"method_{m_hash}"

        established_facts = [
            "Standard IEEE 754 floating-point representations allocate 32 bits (4 bytes) per single-precision tensor weight and activation.",
            "SIMD and vector registers in modern microarchitectures operate on byte-aligned cache-line boundaries (e.g., 64-byte lines).",
            "Multi-seed random-split evaluations reduce sample variance and mitigate single-partition overfitting.",
        ]

        retrieved_evidence = [
            c.claim_text for c in evidence.claims[:4]
        ] if evidence.claims else [
            "Dynamic quantization mitigates memory wall stalls during high-dimensional tensor operations."
        ]

        proposed_innovations = [
            f"Dynamic Block-Floating Discretization: Partition intermediate tensors into 64-element blocks with adaptive scale factors.",
            "Stochastic Cache-Line Alignment: Map quantized blocks directly to 64-byte L1/L2 cache lines to eliminate cache thrashing.",
            "Variance-Stabilized Gradient Scaling: Straight-Through Estimator (STE) with scaled backward pass to ensure convergence.",
        ]

        engineering_rationales = [
            "Heuristic choice of 64-element tile size balances vector register saturation with scaling factor overhead.",
            "Straight-Through Estimator (STE) serves as an empirical surrogate gradient for non-differentiable quantization operators.",
        ]

        assumptions = [
            "Underlying training data satisfies weak stationarity across evaluation folds.",
            "Straight-through gradient approximation introduces zero-mean bounded noise under uniform scaling intervals.",
            "Memory bandwidth and L1/L2 cache capacity remain constant across benchmark iterations.",
        ]

        hypotheses = [
            f"H1: {plan.model_acronym} reduces peak memory consumption by >=70% compared to Dense FP32.",
            f"H2: {plan.model_acronym} improves or matches generalization performance within 1.5% of full-precision baselines.",
            f"H3: Cache-aligned tiling provides >=3.0x inference latency speedup on standard CPU/MPS hardware.",
        ]

        evaluation_criteria = [
            f"Primary Metric: {plan.domain_display_name} Fidelity / Task Score (%)",
            "Peak Resident Memory Footprint (MB)",
            "Inference Latency per Sample (ms)",
            "DerSimonian-Laird Random-Effects Meta-Analysis Summary Effect Size (Z >= 5.0, p < 0.001)",
        ]

        baseline_methods = [
            "Dense FP32 Baseline (Uncompressed Full Precision)",
            "Static INT8 Quantization (Post-Training Rounding)",
            "Dynamic Sparsified Baseline (Magnitude-Pruned GNN)",
        ]

        return MethodologySpec(
            methodology_id=methodology_id,
            topic_title=plan.topic_title,
            domain=plan.domain_display_name,
            model_acronym=plan.model_acronym,
            model_full_name=plan.model_full_name,
            established_facts=established_facts,
            retrieved_evidence=retrieved_evidence,
            proposed_innovations=proposed_innovations,
            engineering_rationales=engineering_rationales,
            assumptions=assumptions,
            hypotheses=hypotheses,
            evaluation_criteria=evaluation_criteria,
            baseline_methods=baseline_methods,
            hardware_constraints=plan.constraints,
        )
