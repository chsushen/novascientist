"""Test suite for Topic-Adaptive Scientific Intelligence (v2.1).

Validates that NovaScientist dynamically adapts across domains (NLP, Vision, Time Series, Graph ML, Federated Learning):
- Extracts accurate TopicResearchProfiles (task type, paradigm, data modality, candidate metrics).
- Dynamically selects domain-appropriate baselines.
- Generates task-grounded datasets with acquisition status.
- Makes justified mathematical decisions (theorems/lemmas vs empirical formulations).
- Plans diverse, topic-specific figures.
- Dynamically budgets manuscript sections and physical page targets.
"""

import pytest
from backend.core.topic_profile import TopicProfileExtractor, TaskType, ResearchParadigm, DataModality
from backend.core.literature_advisor import LiteratureAdvisor
from backend.core.baseline_selector import DynamicBaselineSelector
from backend.core.dataset_finder import DatasetFinder
from backend.core.math_agent import MathematicalFormulationAgent, TheoremDecisionType
from backend.core.figure_planner import FigurePlanningAgent, FigureType
from backend.core.manuscript_planner import ManuscriptPlanningAgent, VenueFormat
from backend.core.page_controller import PhysicalPageController, PageBudgetStatus
from backend.core.evidence_agent import EvidenceBundle, ClaimRecord, SourceRecord
from backend.core.methodology_agent import MethodologyAgent
from backend.core.agentic_planner import ResearchPlannerAgent


def test_topic_profile_nlp_extraction():
    topic = "Parameter-Efficient Fine-Tuning and Low-Rank Adaptation in Causal Language Models"
    profile = TopicProfileExtractor.extract(topic, domain="nlp")

    assert profile.domain == "nlp"
    assert profile.task_type in [TaskType.GENERATION, TaskType.LANGUAGE_MODELING, TaskType.SEQUENCE_GENERATION, TaskType.CLASSIFICATION]
    assert profile.data_modality == DataModality.NATURAL_LANGUAGE_TEXT
    assert any("perplexity" in m.lower() or "bleu" in m.lower() or "rouge" in m.lower() or "exact match" in m.lower() or "f1" in m.lower() for m in profile.candidate_metrics)
    assert any("lora" in b.lower() or "prompt" in b.lower() or "fine-tuning" in b.lower() for b in profile.candidate_baselines)


def test_topic_profile_vision_extraction():
    topic = "High-Resolution Spatial Attention and Diffusion Priors for Medical Image Segmentation"
    profile = TopicProfileExtractor.extract(topic, domain="vision")

    assert profile.domain == "vision"
    assert profile.data_modality == DataModality.IMAGE_VOLUMETRIC
    assert any("dice" in m.lower() or "iou" in m.lower() or "accuracy" in m.lower() for m in profile.candidate_metrics)
    assert any("unet" in b.lower() or "vit" in b.lower() or "resnet" in b.lower() for b in profile.candidate_baselines)


def test_topic_profile_time_series_extraction():
    topic = "Multivariate Spatiotemporal Forecasting for Epidemic Trajectories and Clinical Risk"
    profile = TopicProfileExtractor.extract(topic, domain="time_series")

    assert profile.domain == "time_series"
    assert profile.task_type == TaskType.TIMESERIES_FORECASTING
    assert profile.data_modality == DataModality.MULTIVARIATE_TIME_SERIES
    assert any("rmse" in m.lower() or "mae" in m.lower() or "crps" in m.lower() for m in profile.candidate_metrics)
    assert any("patchtst" in b.lower() or "dlinear" in b.lower() or "arima" in b.lower() or "informer" in b.lower() for b in profile.candidate_baselines)


def test_topic_profile_federated_learning():
    topic = "Decentralized Federated Optimization under Severe Client Drift and Non-IID Partitions"
    profile = TopicProfileExtractor.extract(topic, domain="federated")

    assert profile.domain == "federated"
    assert profile.research_paradigm in [ResearchParadigm.OPTIMIZATION, ResearchParadigm.SYSTEMS_OPTIMIZATION, ResearchParadigm.THEORETICAL_ALGORITHMIC]
    assert any("drift" in m.lower() or "accuracy" in m.lower() or "rounds" in m.lower() for m in profile.candidate_metrics)
    assert any("fedavg" in b.lower() or "fedprox" in b.lower() or "scaffold" in b.lower() for b in profile.candidate_baselines)


def test_dynamic_baseline_selector_cross_domain():
    selector = DynamicBaselineSelector()

    nlp_profile = TopicProfileExtractor.extract("LoRA adaptation for LLMs", domain="nlp")
    nlp_baselines = selector.select_baselines(nlp_profile)
    assert len(nlp_baselines.baselines) >= 3
    assert any("lora" in b.name.lower() or "dense" in b.name.lower() or "int8" in b.name.lower() for b in nlp_baselines.baselines)

    ts_profile = TopicProfileExtractor.extract("Long horizon solar energy forecasting", domain="time_series")
    ts_baselines = selector.select_baselines(ts_profile)
    assert len(ts_baselines.baselines) >= 3
    assert any("patchtst" in b.name.lower() or "dlinear" in b.name.lower() or "linear" in b.name.lower() for b in ts_baselines.baselines)


def test_dataset_discovery_and_acquisition_status():
    dataset = DatasetFinder.discover("Traffic flow prediction on California highways", domain="graph")
    assert dataset.name is not None
    assert dataset.acquisition_status in ["discovered", "literature_verified", "locally_available", "loaded"]
    assert dataset.task_compatibility_score >= 0.0


def test_mathematical_formulation_agent_theorem_decision():
    math_agent = MathematicalFormulationAgent()

    nlp_profile = TopicProfileExtractor.extract("Empirical benchmark of prompt engineering", domain="nlp")
    planner = ResearchPlannerAgent()
    plan = planner.create_plan(nlp_profile.topic, topic_profile=nlp_profile)
    evidence = EvidenceBundle(topic=nlp_profile.topic, domain="nlp")
    method_agent = MethodologyAgent()
    methodology = method_agent.synthesize_methodology(plan, evidence, topic_profile=nlp_profile)

    theorem = math_agent.formulate(nlp_profile, methodology, has_theoretical_claims=True)
    assert theorem.decision in [
        TheoremDecisionType.THEOREM_REQUIRED,
        TheoremDecisionType.PROPOSITION_LEMMA,
        TheoremDecisionType.ANALYTICAL_DERIVATION,
        TheoremDecisionType.EMPIRICAL_STUDY,
    ]
    assert len(theorem.assumptions) > 0 or len(theorem.proof_sketch) > 0
    latex_block = theorem.to_latex()
    assert len(latex_block) > 0


def test_figure_planning_agent_adaptive_types():
    fig_agent = FigurePlanningAgent()
    ts_profile = TopicProfileExtractor.extract("Spatiotemporal epidemic trajectory prediction", domain="time_series")
    mock_metrics = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 0.91, "mean_memory_mb": 65.0, "mean_latency_ms": 7.5},
            "dense_baseline": {"mean_accuracy": 0.84, "mean_memory_mb": 340.0, "mean_latency_ms": 32.0},
        }
    }
    planned = fig_agent.plan_figures(ts_profile, mock_metrics, output_dir="./dist/test_figs")
    assert len(planned) == 5
    types = [f.figure_type for f in planned]
    assert FigureType.ARCHITECTURE in types
    assert FigureType.CONVERGENCE_BAND in types


def test_manuscript_planner_and_page_controller():
    manuscript_planner = ManuscriptPlanningAgent()
    page_controller = PhysicalPageController()

    profile = TopicProfileExtractor.extract("Hamiltonian Neural Operator Physics Modeling", domain="physics_surrogate")
    planner = ResearchPlannerAgent()
    plan = planner.create_plan(profile.topic, topic_profile=profile)
    evidence = EvidenceBundle(topic=profile.topic, domain="physics_surrogate")
    method_agent = MethodologyAgent()
    methodology = method_agent.synthesize_methodology(plan, evidence, topic_profile=profile)

    mplan = manuscript_planner.plan_manuscript(
        topic_profile=profile,
        literature_report=None,
        methodology=methodology,
        theorem=None,
        figures=[],
        venue_format=VenueFormat.EXTENDED_JOURNAL,
    )

    assert mplan.target_page_min == 8
    assert mplan.target_page_max == 12
    assert len(mplan.sections) >= 8

    # Page controller test
    sample_latex = r"\section{Introduction} " + ("word " * 5000)
    eval_res = page_controller.evaluate_page_budget(
        target_min=8,
        target_max=12,
        tex_content=sample_latex,
        num_figures=5,
    )
    assert eval_res.measured_pages > 0
    assert eval_res.status in [PageBudgetStatus.IN_RANGE, PageBudgetStatus.UNDER_BUDGET, PageBudgetStatus.OVER_BUDGET, PageBudgetStatus.ESTIMATED]
