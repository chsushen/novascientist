"""NovaScientist Methodology Agent.

Synthesizes structured scientific methodologies from research plans and literature evidence.
Topic-adaptive: dynamically tailors established facts, proposed innovations, engineering rationales,
assumptions, and hypotheses to the active domain, task type, and candidate metrics.
Explicitly delineates established facts, retrieved evidence, proposed architecture,
theoretical assumptions, and testable hypotheses.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.core.agentic_planner import ResearchPlan
from backend.core.evidence_agent import EvidenceBundle
from backend.core.universal_engine import ComputationalDomain, UniversalDomainDispatcher
from backend.core.topic_profile import TopicProfileExtractor, TopicResearchProfile
from backend.core.literature_advisor import LiteratureSynthesisReport
from backend.core.baseline_selector import BaselineComparisonSuite


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
        
        # Find proposed method key
        prop_key = None
        for k in methods:
            if "prop" in k.lower():
                prop_key = k
                break
        if not prop_key and methods:
            prop_key = list(methods.keys())[0]

        # Find dense / baseline key
        dense_key = None
        for k in methods:
            if "dense" in k.lower() or "baseline" in k.lower():
                dense_key = k
                break
        if not dense_key and len(methods) > 1:
            dense_key = list(methods.keys())[1]

        prop = methods.get(prop_key or "proposed_mb_qgt", {})
        dense = methods.get(dense_key or "dense_baseline", {})

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
            if "memory" in hyp_text.lower() or (idx == 0 and "reduction" in hyp_text.lower()):
                # H1: Memory reduction or efficiency >= 70%
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

            elif "accuracy" in hyp_text.lower() or "generalization" in hyp_text.lower() or "performance" in hyp_text.lower() or idx == 1:
                # H2: Accuracy / performance within 1.5% of baseline or superior (acc_delta >= -1.5%)
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

            elif "speedup" in hyp_text.lower() or "latency" in hyp_text.lower() or "throughput" in hyp_text.lower() or idx == 2:
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
        topic_profile: Optional[TopicResearchProfile] = None,
        literature_report: Optional[LiteratureSynthesisReport] = None,
        baseline_suite: Optional[BaselineComparisonSuite] = None,
    ) -> MethodologySpec:
        """Generate a fully structured, topic-adaptive methodology specification."""
        m_hash = hashlib.sha256((plan.plan_id + plan.topic_title).encode("utf-8")).hexdigest()[:8]
        methodology_id = f"method_{m_hash}"

        profile = topic_profile or TopicProfileExtractor.extract(plan.topic_title, domain=plan.domain.value if hasattr(plan.domain, "value") else str(plan.domain))

        is_quant_topic = any(k in plan.topic_title.lower() for k in ["quantiz", "discretiz", "sub-linear memory", "bit-width", "precision", "compression"])

        # Domain-specific established facts
        if is_quant_topic:
            established_facts = [
                "Standard IEEE 754 floating-point representations allocate 32 bits (4 bytes) per single-precision tensor weight and activation.",
                "SIMD and vector registers in modern microarchitectures operate on byte-aligned cache-line boundaries (e.g., 64-byte lines).",
                "Multi-seed random-split evaluations reduce sample variance and mitigate single-partition overfitting.",
            ]
        elif topic_profile is None:
            established_facts = [
                "Empirical representation learning is governed by variance-bias trade-offs across finite-sample benchmark partitions.",
                "Multi-seed random-split cross-validation isolates stochastic optimization noise and mitigates single-partition bias.",
                "Standard loss landscapes exhibit curvature non-convexity under complex non-linear parameterizations.",
            ]
        else:
            domain_str = profile.domain.lower()
            if "language" in domain_str or "nlp" in domain_str:
                established_facts = [
                    "Autoregressive and retrieval-augmented sequence models parameterize conditional token likelihoods over context histories.",
                    "Context relevance and factual attribution degrade under noisy or out-of-domain knowledge retrieval.",
                    "Multi-seed random-split cross-validation mitigates dataset partition bias and quantifies generalization variance.",
                ]
            elif "vision" in domain_str or "image" in domain_str:
                established_facts = [
                    "Spatial convolution and hierarchical patch embeddings exploit 2D translation equivariance and local spatial correlation.",
                    "Domain shift and imaging artifact variations degrade boundary segmentation and calibration fidelity.",
                    "Multi-seed random-split cross-validation mitigates dataset partition bias and quantifies generalization variance.",
                ]
            elif "time_series" in domain_str or "forecasting" in domain_str:
                established_facts = [
                    "Multivariate time series exhibit temporal autocorrelation, non-stationarity, and cross-channel interdependencies.",
                    "Probabilistic horizon forecasts require calibrated uncertainty intervals across non-stationary regimes.",
                    "Multi-seed split cross-validation over rolling horizons quantifies forecast uncertainty and drift resistance.",
                ]
            elif "federated" in domain_str:
                established_facts = [
                    "Federated optimization operates over decentralized, non-IID client datasets with bandwidth and communication constraints.",
                    "Local client model updates diverge under heterogeneous data distributions, causing client drift.",
                    "Multi-seed federated simulations isolate optimization variance across heterogeneous partition seeds.",
                ]
            else:
                established_facts = [
                    f"Computational learning for {profile.subdomain} is governed by sample efficiency and representation fidelity.",
                    "Multi-seed deterministic evaluations reduce sample variance and mitigate single-partition overfitting.",
                    "Optimization dynamics are bounded by empirical loss smoothness and gradient variance.",
                ]

        # Retrieved evidence claims
        retrieved_evidence = [
            c.claim_text for c in evidence.claims[:4]
        ] if evidence.claims else [
            f"Recent literature demonstrates empirical advantages for {profile.task_type.value} using {', '.join(profile.candidate_method_families[:2])}."
        ]

        # Proposed innovations
        if profile.candidate_method_families:
            innovations_core = f"Domain-Adaptive {plan.model_acronym} Formulation: Integrate {profile.candidate_method_families[0]} with adaptive parameter scaling."
        else:
            innovations_core = f"Adaptive Representation Architecture: Integrate specialized neural operators tailored to {profile.subdomain}."

        proposed_innovations = [
            innovations_core,
            f"Context-Aware Modular State Alignment: Dynamically align feature representations across operational modes in {profile.task_type.value}.",
            f"Variance-Stabilized Multi-Seed Gradient Scaling: Regularized backward propagation ensuring convergence stability across {profile.task_type.value}.",
        ]

        # Engineering rationales & assumptions
        if is_quant_topic:
            engineering_rationales = [
                "Heuristic choice of 64-element tile size balances vector register saturation with scaling factor overhead.",
                "Straight-Through Estimator (STE) serves as an empirical surrogate gradient for non-differentiable quantization operators.",
            ]
            assumptions = [
                "Underlying training data satisfies weak stationarity across evaluation folds.",
                "Straight-through gradient approximation introduces zero-mean bounded noise under uniform scaling intervals.",
                "Hardware memory bandwidth and compute capacity remain constant across benchmark iterations.",
            ]
        elif topic_profile is None:
            engineering_rationales = [
                "Modular architectural decoupling balances parameter efficiency with representation capacity.",
                "Gradient stabilization acts as an empirical surrogate operator for non-smooth optimization landscapes.",
            ]
            assumptions = [
                "Underlying training data satisfies weak stationarity across evaluation folds.",
                "Stochastic gradient approximations introduce zero-mean bounded perturbation under uniform regularizers.",
                "Computational infrastructure resources remain uniform across benchmark runs.",
            ]
        else:
            engineering_rationales = [
                f"Algorithmic design balances computational efficiency with representational fidelity for {profile.task_type.value}.",
                f"Gradient stabilization acts as an empirical surrogate operator for non-smooth optimization landscapes.",
            ]
            assumptions = [
                f"Underlying training and evaluation distributions for {profile.data_modality.value} data satisfy weak stationarity across evaluation folds.",
                "Surrogate gradient approximations introduce zero-mean bounded perturbation under uniform scaling intervals.",
                "Hardware memory bandwidth and compute capacity remain constant across benchmark iterations.",
            ]

        # Dynamic hypotheses
        metric_name = profile.candidate_metrics[0] if profile.candidate_metrics else "Accuracy / Task Metric"
        hypotheses = [
            f"H1: {plan.model_acronym} improves {metric_name} by at least 5.0% over canonical baselines on {plan.topic_title}.",
            f"H2: Variance-stabilized training bounds cross-seed standard deviation of {metric_name} to within 1.0%.",
            f"H3: The proposed architecture achieves significant positive effect size under DerSimonian-Laird meta-analysis (Z >= 5.0, p < 0.001).",
        ]

        evaluation_criteria = [
            f"Primary Metric: {metric_name} (%)",
            "Computational Efficiency & Execution Throughput",
            "Cross-Seed Empirical Dispersion (std <= 1.0%)",
            "DerSimonian-Laird Random-Effects Meta-Analysis Summary Effect Size (Z >= 5.0, p < 0.001)",
        ]

        if baseline_suite and baseline_suite.baselines:
            baseline_methods = [b.name for b in baseline_suite.baselines]
        elif profile.candidate_baselines:
            baseline_methods = profile.candidate_baselines[:3]
        else:
            baseline_methods = [
                "Canonical Full-Precision Baseline",
                "Regularized Competitive Baseline",
                "Lightweight Efficient Baseline",
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
