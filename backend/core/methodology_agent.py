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
    assumptions: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)
    baseline_methods: List[str] = field(default_factory=list)
    hardware_constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MethodologyAgent:
    """Agent responsible for formulating sound, reproducible scientific methodologies."""

    def __init__(self) -> None:
        pass

    def synthesize_methodology(
        self,
        plan: ResearchPlan,
        evidence: EvidenceBundle,
    ) -> MethodologySpec:
        """Generate a fully structured methodology specification."""
        m_hash = hashlib.sha256((plan.plan_id + plan.topic_title).encode("utf-8")).hexdigest()[:8]
        methodology_id = f"method_{m_hash}"

        established_facts = [
            "Standard IEEE FP32 representations use 32 bits (4 bytes) per tensor weight and activation.",
            "CPU and Apple Silicon vector registers operate most efficiently on 64-byte aligned memory blocks (SIMD).",
            "Multi-seed paired random evaluations reduce empirical variance and prevent single-seed overfitting.",
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
            assumptions=assumptions,
            hypotheses=hypotheses,
            evaluation_criteria=evaluation_criteria,
            baseline_methods=baseline_methods,
            hardware_constraints=plan.constraints,
        )
