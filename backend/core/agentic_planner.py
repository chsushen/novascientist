"""NovaScientist Research Planner Agent.

Formulates, structures, and validates typed research plans from user research questions,
domains, constraints, and target publication formats.
Topic-adaptive: dynamically constructs objectives, literature requirements,
methodology plans, and experiment suites from TopicResearchProfile.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from backend.core.topic_profile import TopicProfileExtractor, TopicResearchProfile
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
    objectives: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    literature_requirements: list[str] = field(default_factory=list)
    methodology_plan: list[str] = field(default_factory=list)
    experiment_plan: list[str] = field(default_factory=list)
    evaluation_plan: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    target_format: str = "8_12_pages_journal"
    constraints: dict[str, Any] = field(default_factory=dict)
    is_valid: bool = False
    validation_notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert plan to dictionary format."""
        d = asdict(self)
        d["domain"] = (
            self.domain.value
            if isinstance(self.domain, ComputationalDomain)
            else str(self.domain)
        )
        return d


class ResearchPlannerAgent:
    """Agent responsible for formulating, refining, and validating research plans."""

    def __init__(self) -> None:
        pass

    def create_plan(
        self,
        research_question: str,
        domain: ComputationalDomain | None = None,
        constraints: dict[str, Any] | None = None,
        target_format: str = "8_12_pages_journal",
        num_seeds: int = 5,
        topic_profile: TopicResearchProfile | None = None,
    ) -> ResearchPlan:
        """Synthesize a complete structured research plan for a given research question."""
        cleaned_q = research_question.strip()
        if not cleaned_q:
            cleaned_q = "Dynamic Low-Compute Machine Learning under Memory Bounds"

        classification = UniversalDomainDispatcher.classify_topic(cleaned_q)
        active_domain = domain or classification.domain

        # Topic profile extraction for dynamic adaptive intelligence
        profile = topic_profile or TopicProfileExtractor.extract(
            cleaned_q, domain=active_domain.value
        )

        q_hash = hashlib.sha256(cleaned_q.encode("utf-8")).hexdigest()[:8]
        plan_id = f"plan_{q_hash}"

        constraints_dict = constraints or {
            "max_memory_mb": 128.0,
            "min_seeds": num_seeds,
            "device_target": "commodity_workstation",
            "double_blind": True,
        }

        # Dynamic topic-driven objectives
        metric_str = (
            ", ".join(profile.candidate_metrics[:2])
            if profile.candidate_metrics
            else classification.primary_metric_name
        )
        baselines_str = (
            ", ".join(profile.candidate_baselines[:3])
            if profile.candidate_baselines
            else "standard baselines"
        )

        objectives = [
            f"Formulate and evaluate the {classification.model_full_name} ({classification.model_acronym}) architecture for {profile.subdomain}.",
            f"Develop domain-appropriate inductive biases and algorithmic operators optimized for {profile.task_type.value}.",
            f"Empirically validate performance across {metric_str} using k={num_seeds} deterministic seeds against {baselines_str}.",
            "Perform DerSimonian-Laird random-effects meta-analysis to quantify pooled effect size and inter-seed heterogeneity.",
        ]

        sub_questions = [
            f"How does {classification.model_acronym} impact {metric_str} compared to canonical {profile.subdomain} baselines?",
            f"What theoretical guarantees or convergence properties govern {classification.model_acronym} on {profile.data_modality.value} data?",
            f"Are the empirical gains consistent across multi-seed replications on {', '.join(profile.candidate_datasets[:2])}?",
        ]

        literature_requirements = [
            f"Foundational literature on {profile.domain} with focus on {profile.subdomain}.",
            f"Canonical benchmark studies evaluating {', '.join(profile.candidate_baselines[:3])}.",
            f"Empirical validation protocols and metrics for {profile.task_type.value}.",
        ]

        methodology_plan = [
            f"Define mathematical formulation for {classification.model_acronym} leveraging {', '.join(profile.mathematical_objects[:3])}.",
            f"Synthesize architectural operators tailored to {profile.research_paradigm.value}.",
            f"Formulate loss objectives and computational complexity bounds for {profile.task_type.value}.",
        ]

        experiment_plan = [
            f"Evaluate canonical baselines ({', '.join(profile.candidate_baselines[:3])}) across k={num_seeds} seeds.",
            f"Evaluate proposed {classification.model_acronym} across k={num_seeds} seeds under strictly identical data partitions.",
            "Execute component ablation suite isolating core architectural mechanisms.",
            "Compute 2D hyperparameter sensitivity grid across key operational regimes.",
        ]

        evaluation_plan = [
            f"Primary Task Metrics: {', '.join(profile.candidate_metrics[:3])}",
            "Peak Working Memory Footprint (MB)",
            "Per-Sample Inference Latency (ms) & Arithmetic Throughput",
            "DerSimonian-Laird Meta-Analysis: Pooled Effect Size (%), 95% Confidence Interval, Z-statistic, and I² Heterogeneity",
            "AST Dataflow Verification for strict train/test partition isolation.",
        ]

        risk_factors = [
            f"Stochastic divergence during optimization across complex {profile.data_modality.value} distributions.",
            "Potential domain shift or partition sensitivity between benchmark datasets.",
            "Data leakage between evaluation folds if random seeds are not strictly isolated.",
        ]

        expected_outputs = [
            "Structured ExperimentRecord telemetry for all evaluation folds.",
            "Topic-adaptive publication vector figure suite.",
            "Complete IEEE / ACM formatted LaTeX manuscript with verified BibTeX citations.",
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

    def validate_plan(self, plan: ResearchPlan) -> tuple[bool, list[str]]:
        """Validate research plan against structural and scientific consistency criteria."""
        notes: list[str] = []

        if not plan.research_question or len(plan.research_question) < 5:
            notes.append("Research question is missing or too brief.")

        if len(plan.objectives) < 2:
            notes.append("Plan must define at least 2 clear scientific objectives.")

        if len(plan.sub_questions) < 1:
            notes.append("Plan must specify at least 1 sub-question.")

        if len(plan.experiment_plan) < 3:
            notes.append(
                "Experiment plan must include baseline, ablation, and proposed evaluations."
            )

        if not plan.model_acronym:
            notes.append("Proposed model acronym must be specified.")

        is_valid = len(notes) == 0
        return is_valid, notes
