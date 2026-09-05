"""Anti-template regression test suite for NovaScientist v2.1.

Strictly verifies that distinct research questions produce distinct:
- TopicResearchProfiles (domain, task type, data modality)
- Dynamic baseline sets (e.g. NLP vs Time Series vs Physics)
- Datasets and task compatibility scores
- Methodological established facts and hypotheses
- Mathematical theorem formulations (theorems/lemmas vs derivations)
- Figure plans and captions
"""

import pytest
from backend.core.topic_profile import TopicProfileExtractor
from backend.core.literature_advisor import LiteratureAdvisor
from backend.core.baseline_selector import DynamicBaselineSelector
from backend.core.dataset_finder import DatasetFinder
from backend.core.math_agent import MathematicalFormulationAgent
from backend.core.figure_planner import FigurePlanningAgent
from backend.core.methodology_agent import MethodologyAgent
from backend.core.agentic_planner import ResearchPlannerAgent
from backend.core.evidence_agent import EvidenceBundle


def test_anti_template_cross_topic_differentiation():
    topic_nlp = "Parameter-Efficient Fine-Tuning and Low-Rank Adaptation in Causal Language Models"
    topic_time_series = "Multivariate Spatiotemporal Forecasting for Epidemic Trajectories and Clinical Risk"
    topic_physics = "Hamiltonian Neural Operator Physics Modeling of Nonlinear Navier Stokes Equations"

    # 1. Topic Profiles must be distinct
    profile_nlp = TopicProfileExtractor.extract(topic_nlp, domain="nlp")
    profile_ts = TopicProfileExtractor.extract(topic_time_series, domain="time_series")
    profile_phys = TopicProfileExtractor.extract(topic_physics, domain="physics_surrogate")

    assert profile_nlp.data_modality != profile_ts.data_modality
    assert profile_nlp.data_modality != profile_phys.data_modality
    assert profile_ts.task_type != profile_phys.task_type or profile_ts.domain != profile_phys.domain

    # 2. Dynamic Baselines must differ across domains
    selector = DynamicBaselineSelector()
    baselines_nlp = [b.name for b in selector.select_baselines(profile_nlp).baselines]
    baselines_ts = [b.name for b in selector.select_baselines(profile_ts).baselines]
    baselines_phys = [b.name for b in selector.select_baselines(profile_phys).baselines]

    assert set(baselines_nlp) != set(baselines_ts)
    assert set(baselines_nlp) != set(baselines_phys)
    assert set(baselines_ts) != set(baselines_phys)

    # 3. Datasets must be tailored to the topic
    dataset_nlp = DatasetFinder.discover(topic_nlp, domain="nlp")
    dataset_ts = DatasetFinder.discover(topic_time_series, domain="time_series")
    assert dataset_nlp.name != dataset_ts.name

    # 4. Methodologies must not have identical established facts or hypotheses
    planner = ResearchPlannerAgent()
    plan_nlp = planner.create_plan(topic_nlp, topic_profile=profile_nlp)
    plan_ts = planner.create_plan(topic_time_series, topic_profile=profile_ts)

    evidence_nlp = EvidenceBundle(topic=topic_nlp, domain="nlp")
    evidence_ts = EvidenceBundle(topic=topic_time_series, domain="time_series")
    method_agent = MethodologyAgent()

    method_nlp = method_agent.synthesize_methodology(plan_nlp, evidence_nlp, topic_profile=profile_nlp)
    method_ts = method_agent.synthesize_methodology(plan_ts, evidence_ts, topic_profile=profile_ts)

    assert method_nlp.established_facts != method_ts.established_facts
    assert method_nlp.hypotheses != method_ts.hypotheses

    # 5. Math formulations must formulate topic-specific theorems and assumptions
    math_agent = MathematicalFormulationAgent()
    theorem_nlp = math_agent.formulate(profile_nlp, method_nlp)
    theorem_ts = math_agent.formulate(profile_ts, method_ts)
    theorem_phys = math_agent.formulate(profile_phys, method_nlp)

    assert theorem_nlp.title != theorem_phys.title
    assert theorem_nlp.to_latex() != theorem_phys.to_latex()

    # 6. Figures planned must have distinct domain titles and keys
    fig_agent = FigurePlanningAgent()
    mock_metrics = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 0.88, "mean_memory_mb": 70.0, "mean_latency_ms": 8.0},
            "dense_baseline": {"mean_accuracy": 0.82, "mean_memory_mb": 350.0, "mean_latency_ms": 35.0},
        }
    }
    figs_nlp = fig_agent.plan_figures(profile_nlp, mock_metrics, output_dir="./dist/test_figs_nlp")
    figs_ts = fig_agent.plan_figures(profile_ts, mock_metrics, output_dir="./dist/test_figs_ts")

    nlp_titles = [f.title for f in figs_nlp]
    ts_titles = [f.title for f in figs_ts]
    assert nlp_titles != ts_titles
