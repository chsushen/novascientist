"""NovaScientist Research Planner Agent.

Formulates, structures, and validates typed research plans from user research questions,
domains, constraints, and target publication formats.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.core.universal_engine import ComputationalDomain, UniversalDomainDispatcher


@dataclass
class ResearchPlan:
    """Typed schema representing a structured research plan."""
    plan_id: str
    research_question: str
    topic_title: str
    domain: ComputationalDomain
    domain_display_name: str
    model_acronym: str
    model_full_name: str
    objectives: List[str] = field(default_factory=list)
    sub_questions: List[str] = field(default_factory=list)
    literature_requirements: List[str] = field(default_factory=list)
    methodology_plan: List[str] = field(default_factory=list)
    experiment_plan: List[str] = field(default_factory=list)
    evaluation_plan: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    target_format: str = "8_12_pages_journal"
    constraints: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = False
    validation_notes: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary format."""
        d = asdict(self)
        d["domain"] = self.domain.value if isinstance(self.domain, ComputationalDomain) else str(self.domain)
        return d


class ResearchPlannerAgent:
    """Agent responsible for formulating, refining, and validating research plans."""

    def __init__(self) -> None:
        pass

    def create_plan(
        self,
        research_question: str,
        domain: Optional[ComputationalDomain] = None,
        constraints: Optional[Dict[str, Any]] = None,
        target_format: str = "8_12_pages_journal",
        num_seeds: int = 5,
    ) -> ResearchPlan:
        """Synthesize a complete structured research plan for a given research question."""
        cleaned_q = research_question.strip()
        if not cleaned_q:
            cleaned_q = "Dynamic Low-Compute Machine Learning under Memory Bounds"

        classification = UniversalDomainDispatcher.classify_topic(cleaned_q)
        active_domain = domain or classification.domain
        
        q_hash = hashlib.sha256(cleaned_q.encode("utf-8")).hexdigest()[:8]
        plan_id = f"plan_{q_hash}"

        constraints_dict = constraints or {
            "max_memory_mb": 128.0,
            "min_seeds": num_seeds,
            "device_target": "commodity_workstation",
            "double_blind": True,
        }

        objectives = [
            f"Formulate and prove mathematical bounds for {classification.model_full_name} ({classification.model_acronym}).",
            f"Implement dynamic block-floating discretization aligned with 64-byte L1/L2 cache lines for SIMD acceleration.",
            f"Empirically validate {classification.primary_metric_name} across k={num_seeds} deterministic seeds against dense and quantized baselines.",
            f"Perform DerSimonian-Laird random-effects meta-analysis to quantify pooled effect size and inter-seed heterogeneity.",
        ]

        sub_questions = [
            f"Can dynamic block quantization eliminate memory bus stalls without degrading {classification.primary_metric_name}?",
            f"What is the theoretical error bound between continuous operators and {classification.model_acronym} discretizations?",
            f"Does stochastic cache-line alignment yield reproducible latency acceleration on commodity CPU/MPS hardware?",
        ]

        literature_requirements = [
            f"Foundational papers on {classification.domain_display_name} architectures and neural operators.",
            "Theoretical literature on post-training quantization, straight-through estimators, and low-rank approximation.",
            "Empirical benchmarks and canonical baseline implementations for comparative validation.",
        ]

        methodology_plan = [
            f"Define mathematical formulation for {classification.model_acronym} with Straight-Through Estimators (STE).",
            "Derive variance bounds for dynamic block-floating tensor scaling under stochastic noise.",
            "Formulate straight-through backward propagation equations with variance stabilization.",
        ]

        experiment_plan = [
            f"Evaluate Dense FP32 baseline across k={num_seeds} deterministic random seeds.",
            f"Evaluate Static INT8 post-training quantization baseline across k={num_seeds} seeds.",
            f"Evaluate Sparsified baseline across k={num_seeds} seeds.",
            f"Evaluate Proposed {classification.model_acronym} across k={num_seeds} seeds under identical data partitions.",
            "Execute component ablation suite (w/o block scaling, w/o tile caching, w/o variance stabilization).",
            "Compute 2D hyperparameter sensitivity grid across quantization depths and cache tile dimensions.",
        ]

        evaluation_plan = [
            f"Primary Task Metric: {classification.primary_metric_name}",
            "Peak Working Memory Footprint (MB)",
            "Per-Sample Inference Latency (ms) & Arithmetic Throughput (samples/sec)",
            "DerSimonian-Laird Meta-Analysis: Pooled Effect Size (%), 95% Confidence Interval, Z-statistic, and I² Heterogeneity",
            "AST Dataflow Verification for strict train/test partition isolation.",
        ]

        risk_factors = [
            "Quantization noise divergence along high-frequency gradient boundaries.",
            "Cache miss penalties if tensor widths do not align with 64-byte SIMD boundaries.",
            "Data leakage between evaluation folds if random seeds are not strictly isolated.",
        ]

        expected_outputs = [
            "Structured ExperimentRecord telemetry for all evaluation folds.",
            "5-figure publication vector suite (Architecture, Convergence, Pareto, Ablation, Sensitivity).",
            "Complete 10-section IEEE Transactions LaTeX manuscript with verified BibTeX citations.",
            "Self-contained Overleaf ZIP package and downloadable publication PDF.",
        ]

        plan = ResearchPlan(
            plan_id=plan_id,
            research_question=cleaned_q,
            topic_title=cleaned_q,
            domain=active_domain,
            domain_display_name=classification.domain_display_name,
            model_acronym=classification.model_acronym,
            model_full_name=classification.model_full_name,
            objectives=objectives,
            sub_questions=sub_questions,
            literature_requirements=literature_requirements,
            methodology_plan=methodology_plan,
            experiment_plan=experiment_plan,
            evaluation_plan=evaluation_plan,
            risk_factors=risk_factors,
            expected_outputs=expected_outputs,
            target_format=target_format,
            constraints=constraints_dict,
        )

        is_valid, validation_notes = self.validate_plan(plan)
        plan.is_valid = is_valid
        plan.validation_notes = validation_notes
        return plan

    def validate_plan(self, plan: ResearchPlan) -> Tuple[bool, List[str]]:
        """Validate research plan against structural and scientific consistency criteria."""
        notes: List[str] = []

        if not plan.research_question or len(plan.research_question) < 5:
            notes.append("Research question is missing or too brief.")

        if len(plan.objectives) < 2:
            notes.append("Plan must define at least 2 clear scientific objectives.")

        if len(plan.sub_questions) < 1:
            notes.append("Plan must specify at least 1 sub-question.")

        if len(plan.experiment_plan) < 3:
            notes.append("Experiment plan must include baseline, ablation, and proposed evaluations.")

        if not plan.model_acronym:
            notes.append("Proposed model acronym must be specified.")

        is_valid = len(notes) == 0
        return is_valid, notes
