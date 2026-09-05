"""NovaScientist Manuscript Planning Agent.

Dynamically structures research papers, section plans, figure placements,
and content budgets based on TopicResearchProfile, literature synthesis,
mathematical decisions, and target venue conventions.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from backend.core.figure_planner import FigurePlanItem
from backend.core.literature_advisor import LiteratureSynthesisReport
from backend.core.math_agent import FormalTheorem, TheoremDecisionType
from backend.core.methodology_agent import MethodologySpec
from backend.core.topic_profile import TopicResearchProfile


class VenueFormat(str, Enum):
    """Publication venue format and page constraints."""

    SHORT_CONFERENCE = "short_conference"  # 4-6 pages
    FULL_CONFERENCE = "full_conference"  # 6-8 pages
    EXTENDED_JOURNAL = "extended_journal"  # 8-12+ pages


@dataclass
class ManuscriptSpecification:
    """Rigorous contract-driven specification governing entire manuscript assembly."""

    title: str
    abstract_requirements: list[str]
    sections: list[str]
    subsections: dict[str, list[str]]
    equations: list[dict[str, str]]
    tables: list[dict[str, Any]]
    figures: list[FigurePlanItem]
    claims: list[str]
    references: list[dict[str, str]]
    limitations: list[str]
    statistical_reporting: dict[str, Any]
    reproducibility_requirements: list[str]
    page_target: str  # e.g., '6-8' or '8-12'
    contract_id: str
    provenance_hash: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["figures"] = [
            f.to_dict() if hasattr(f, "to_dict") else f for f in self.figures
        ]
        return d


@dataclass
class SectionPlanItem:
    """Structured plan for an individual manuscript section."""

    section_id: str
    number: int
    title: str
    target_words: int
    estimated_pages: float
    key_points: list[str] = field(default_factory=list)
    figure_ids: list[str] = field(default_factory=list)
    has_equations: bool = False
    has_theorems: bool = False
    subsections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ManuscriptPlan:
    """Comprehensive manuscript organization and page-budget plan."""

    plan_id: str
    topic_title: str
    venue_format: VenueFormat
    target_page_min: int
    target_page_max: int
    total_target_words: int
    sections: list[SectionPlanItem] = field(default_factory=list)
    figure_allocations: dict[str, str] = field(
        default_factory=dict
    )  # fig_id -> section_id
    special_blocks: list[str] = field(
        default_factory=list
    )  # e.g., ["formal_theorem", "paired_hypothesis_testing"]
    specification: ManuscriptSpecification | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["venue_format"] = (
            self.venue_format.value
            if isinstance(self.venue_format, VenueFormat)
            else str(self.venue_format)
        )
        d["sections"] = [s.to_dict() for s in self.sections]
        if self.specification:
            d["specification"] = self.specification.to_dict()
        return d


class ManuscriptPlanningAgent:
    """Plans topic-adaptive scientific paper structures and page budgets."""

    def __init__(self) -> None:
        pass

    def plan_manuscript(
        self,
        topic_profile: TopicResearchProfile,
        literature_report: LiteratureSynthesisReport | None,
        methodology: MethodologySpec,
        theorem: FormalTheorem | None,
        figures: list[FigurePlanItem],
        venue_format: VenueFormat = VenueFormat.EXTENDED_JOURNAL,
        contract: Any | None = None,
    ) -> ManuscriptPlan:
        """Construct a dynamic, topic-tailored manuscript structure."""
        plan_hash = hashlib.sha256(
            f"{topic_profile.profile_id}_{venue_format.value}".encode()
        ).hexdigest()[:8]
        plan_id = f"mplan_{plan_hash}"

        if venue_format == VenueFormat.SHORT_CONFERENCE:
            target_page_min, target_page_max = 4, 6
            base_word_budget = 3200
        elif venue_format == VenueFormat.FULL_CONFERENCE:
            target_page_min, target_page_max = 6, 8
            base_word_budget = 4800
        else:
            target_page_min, target_page_max = 8, 12
            base_word_budget = 7500

        has_theorem = (
            theorem is not None
            and theorem.decision
            in [
                TheoremDecisionType.THEOREM_REQUIRED,
                TheoremDecisionType.PROPOSITION_LEMMA,
            ]
            and getattr(theorem, "is_verified", False)
        )

        sections: list[SectionPlanItem] = []
        fig_alloc: dict[str, str] = {}
        fig_map = {f.figure_id: f for f in figures}

        # 1. Introduction
        intro_figs = [
            f.figure_id
            for f in figures
            if "architecture" in f.figure_type.value
            or "overview" in f.figure_type.value
        ][:1]
        for f_id in intro_figs:
            fig_alloc[f_id] = "sec_intro"
        sections.append(
            SectionPlanItem(
                section_id="sec_intro",
                number=1,
                title="Introduction",
                target_words=int(base_word_budget * 0.14),
                estimated_pages=1.2,
                key_points=[
                    f"Contextualize the significance of {topic_profile.domain} with emphasis on {topic_profile.subdomain}.",
                    f"Identify open limitations in existing approaches for {topic_profile.task_type.value}.",
                    "State core research gap synthesized from canonical literature.",
                    f"Present the core contributions of the proposed {methodology.model_acronym} framework.",
                    "Outline the structural organization of the manuscript.",
                ],
                figure_ids=intro_figs,
                subsections=[
                    "Motivation and Problem Setting",
                    "Summary of Contributions",
                    "Paper Organization",
                ],
            )
        )

        # 2. Related Work
        sections.append(
            SectionPlanItem(
                section_id="sec_related",
                number=2,
                title="Related Work",
                target_words=int(base_word_budget * 0.12),
                estimated_pages=1.0,
                key_points=[
                    f"Survey foundational literature in {topic_profile.subdomain}.",
                    "Discuss canonical baseline paradigms and their structural constraints.",
                    "Contextualize empirical and theoretical distinctions of the proposed work.",
                ],
                subsections=[
                    f"Foundations of {topic_profile.subdomain}",
                    "Contemporary Baseline Methods",
                    "Comparative Positioning",
                ],
            )
        )

        # 3. Problem Formulation & Analytical Framework
        theory_title = (
            "Problem Formulation & Theoretical Foundations"
            if has_theorem
            else "Problem Formulation & Analytical Framework"
        )
        theory_subsections = [
            "Notational Framework and Mathematical Setup",
            "Formal Problem Definition",
        ]
        if has_theorem:
            theory_subsections.append(
                f"Theoretical Properties & {theorem.theorem_type.capitalize()} Guarantees"
            )
        sections.append(
            SectionPlanItem(
                section_id="sec_theory",
                number=3,
                title=theory_title,
                target_words=int(base_word_budget * 0.15),
                estimated_pages=1.3 if has_theorem else 1.0,
                key_points=[
                    f"Define formal mathematical space using objects: {', '.join(topic_profile.mathematical_objects[:4])}.",
                    "Formalize task objectives and loss functions.",
                ]
                + (
                    [
                        f"Present formal {theorem.theorem_type} on {theorem.title} with verified proof."
                    ]
                    if has_theorem
                    else []
                ),
                has_equations=True,
                has_theorems=has_theorem,
                subsections=theory_subsections,
            )
        )

        # 4. Proposed Methodology / Architecture
        method_figs = [
            f.figure_id
            for f in figures
            if f.figure_id not in fig_alloc
            and (
                "architecture" in f.figure_type.value
                or "workflow" in f.figure_type.value
            )
        ][:1]
        for f_id in method_figs:
            fig_alloc[f_id] = "sec_method"
        sections.append(
            SectionPlanItem(
                section_id="sec_method",
                number=4,
                title=f"The {methodology.model_acronym} Methodology",
                target_words=int(base_word_budget * 0.18),
                estimated_pages=1.5,
                key_points=[
                    f"Detailed architectural walkthrough of {methodology.model_full_name}.",
                    "Algorithmic execution flow and computational complexity guarantees.",
                    "Engineering implementation rationales and inductive biases.",
                ],
                figure_ids=method_figs,
                has_equations=True,
                subsections=[
                    "Architectural Architecture",
                    "Algorithmic Formulation",
                    "Computational Complexity & Implementation",
                ],
            )
        )

        # 5. Experimental Setup & Benchmarking
        sections.append(
            SectionPlanItem(
                section_id="sec_setup",
                number=5,
                title="Experimental Setup & Benchmarking Protocol",
                target_words=int(base_word_budget * 0.10),
                estimated_pages=1.0,
                key_points=[
                    f"Benchmark datasets evaluated ({', '.join(topic_profile.candidate_datasets[:3])}) and acquisition status.",
                    f"Baseline comparative suite ({', '.join(methodology.baseline_methods[:4])}).",
                    f"Target evaluation metrics ({', '.join(topic_profile.candidate_metrics[:4])}).",
                    "Hardware constraints, seed configurations, and replication protocol.",
                ],
                subsections=[
                    "Datasets & Modality",
                    "Comparative Baseline Models",
                    "Evaluation Metrics & Protocol",
                    "Hardware & Compute Environment",
                ],
            )
        )

        # 6. Empirical Evaluation & Comparative Results
        stat_req_val = getattr(contract, "statistical_requirement", None)
        if hasattr(stat_req_val, "value"):
            stat_req_str = stat_req_val.value
        else:
            stat_req_str = str(stat_req_val or "paired_t_test")

        if stat_req_str == "random_effects_meta_analysis":
            stat_sub = "DerSimonian-Laird Random-Effects Meta-Analysis"
            stat_point = "Statistical meta-analysis summary (DerSimonian-Laird random effects, pooled effect size, I²)."
            stat_block_name = "random_effects_meta_analysis"
        elif stat_req_str == "paired_t_test":
            stat_sub = "Multi-Seed Paired Hypothesis Testing & Cohen's d"
            stat_point = "Paired Student's t-test hypothesis testing with Cohen's d effect size evaluation across deterministic seeds."
            stat_block_name = "paired_hypothesis_testing"
        elif stat_req_str == "bootstrap_confidence_interval":
            stat_sub = "Empirical Bootstrap Resampling & Confidence Bounds"
            stat_point = "Non-parametric bootstrap resampling confidence interval estimation (B=10,000 resamples)."
            stat_block_name = "bootstrap_confidence_intervals"
        elif stat_req_str == "wilcoxon_signed_rank":
            stat_sub = "Non-Parametric Wilcoxon Signed-Rank Test"
            stat_point = "Wilcoxon signed-rank test evaluating median paired differences across evaluation seeds."
            stat_block_name = "wilcoxon_signed_rank"
        else:
            stat_sub = "Descriptive Statistical Variance & Seed Dispersion"
            stat_point = "Descriptive statistics reporting sample mean and standard error across deterministic evaluation runs."
            stat_block_name = "descriptive_statistics"

        result_figs = [
            f.figure_id
            for f in figures
            if f.figure_id not in fig_alloc
            and (
                "convergence" in f.figure_type.value
                or "pareto" in f.figure_type.value
                or "comparison" in f.figure_type.value
                or "forecast" in f.figure_type.value
                or "depth" in f.figure_type.value
            )
        ][:2]
        for f_id in result_figs:
            fig_alloc[f_id] = "sec_results"
        sections.append(
            SectionPlanItem(
                section_id="sec_results",
                number=6,
                title="Empirical Evaluation & Results",
                target_words=int(base_word_budget * 0.16),
                estimated_pages=1.5,
                key_points=[
                    "Multi-seed comparative benchmark across all evaluated baselines.",
                    stat_point,
                    "Rigorous hypothesis evaluation against empirical criteria.",
                ],
                figure_ids=result_figs,
                has_equations=True,
                subsections=[
                    "Primary Performance Benchmark",
                    stat_sub,
                    "Hypothesis Evaluation Results",
                ],
            )
        )

        # 7. Ablation Studies & Sensitivity Analysis
        ablation_figs = [f.figure_id for f in figures if f.figure_id not in fig_alloc]
        for f_id in ablation_figs:
            fig_alloc[f_id] = "sec_ablation"
        sections.append(
            SectionPlanItem(
                section_id="sec_ablation",
                number=7,
                title="Ablation Studies & Sensitivity Analysis",
                target_words=int(base_word_budget * 0.12),
                estimated_pages=1.2,
                key_points=[
                    "Component-wise ablation isolating isolated contributions of each architectural module.",
                    "Hyperparameter sensitivity analysis across key operational regimes.",
                    "Empirical robustness and parameter scaling properties.",
                ],
                figure_ids=ablation_figs,
                subsections=[
                    "Component Ablation Study",
                    "Hyperparameter Sensitivity",
                    "Scaling Behavior",
                ],
            )
        )

        # 8. Discussion, Limitations & Epistemic Boundaries
        sections.append(
            SectionPlanItem(
                section_id="sec_discussion",
                number=8,
                title="Discussion & Epistemic Boundaries",
                target_words=int(base_word_budget * 0.08),
                estimated_pages=0.8,
                key_points=[
                    "Honest examination of operational trade-offs and observed failure modes.",
                    "Negative empirical results and computational boundary conditions.",
                    "Reproducibility guarantees and artifact availability.",
                ],
                subsections=[
                    "Operational Trade-offs",
                    "Identified Limitations & Boundary Conditions",
                    "Ethical Considerations",
                ],
            )
        )

        # 9. Conclusion & Future Directions
        sections.append(
            SectionPlanItem(
                section_id="sec_conclusion",
                number=9,
                title="Conclusion & Future Directions",
                target_words=int(base_word_budget * 0.05),
                estimated_pages=0.5,
                key_points=[
                    f"Summary of primary findings for {methodology.model_acronym}.",
                    "Broader impact on the scientific community.",
                    "Promising directions for future extensions.",
                ],
                subsections=["Summary of Findings", "Future Horizons"],
            )
        )

        total_words = sum(s.target_words for s in sections)
        special_blocks = [stat_block_name, "provenance_graph"]
        if has_theorem:
            special_blocks.append("formal_theorem")

        # Build ManuscriptSpecification
        spec_hash = hashlib.sha256(f"{plan_id}_{total_words}".encode()).hexdigest()[:12]
        manuscript_spec = ManuscriptSpecification(
            title=topic_profile.topic,
            abstract_requirements=[
                f"Problem contextualization in {topic_profile.domain}",
                "Research gap formalization",
                f"Proposed {methodology.model_acronym} framework summary",
                f"Empirical benchmark findings on {topic_profile.candidate_datasets[0] if topic_profile.candidate_datasets else 'canonical dataset'}",
                stat_point,
            ],
            sections=[s.title for s in sections],
            subsections={s.title: s.subsections for s in sections},
            equations=[
                {
                    "section": "Theoretical Formulation",
                    "type": getattr(
                        contract, "mathematical_requirement", "empirical_only"
                    ),
                }
            ],
            tables=[
                {"name": "Literature Taxonomy", "type": "comparative_taxonomy"},
                {"name": "Main Benchmark Results", "type": "multi_seed_comparison"},
                {"name": "Sub-Task Breakdown", "type": "stratum_analysis"},
                {"name": "Hyperparameters", "type": "experimental_configuration"},
            ],
            figures=figures,
            claims=[
                f"Proposed {methodology.model_acronym} outperforms canonical baselines.",
                "Performance retention is verified across deterministic random seeds.",
            ],
            references=[
                {"bibkey": getattr(b, "bibkey", getattr(b, "doi", "ref"))}
                for b in getattr(literature_report, "recommended_baselines", [])
            ],
            limitations=[
                f"Evaluations constrained to {topic_profile.candidate_datasets[0] if topic_profile.candidate_datasets else 'benchmark dataset'}.",
                "Resource envelopes bounded by standard compute budgets.",
            ],
            statistical_reporting={
                "method": stat_req_str,
                "confidence_level": 0.95,
                "effect_metric": getattr(
                    getattr(contract, "statistical_plan", None),
                    "effect_size",
                    "Cohen's d",
                ),
            },
            reproducibility_requirements=[
                "Deterministic fixed random seeds",
                "Strict pre-split partition isolation (zero data leakage)",
                "Full code and configuration artifact archival",
            ],
            page_target=f"{target_page_min}-{target_page_max}",
            contract_id=getattr(contract, "contract_id", "contract_default"),
            provenance_hash=spec_hash,
        )

        return ManuscriptPlan(
            plan_id=plan_id,
            topic_title=topic_profile.topic,
            venue_format=venue_format,
            target_page_min=target_page_min,
            target_page_max=target_page_max,
            total_target_words=total_words,
            sections=sections,
            figure_allocations=fig_alloc,
            special_blocks=special_blocks,
            specification=manuscript_spec,
            metadata={
                "task_type": topic_profile.task_type.value,
                "research_paradigm": topic_profile.research_paradigm.value,
                "data_modality": topic_profile.data_modality.value,
                "has_formal_theorem": has_theorem,
                "num_figures_planned": len(figures),
            },
        )
