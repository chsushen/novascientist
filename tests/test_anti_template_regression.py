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


def test_proposed_method_spec_task_grounded():
    """Verify that proposed method specs are grounded in task characteristics."""
    topic_nlp = "Parameter-Efficient Fine-Tuning with Low-Rank Tensor Adapters"
    topic_ts = "Multivariate Spatiotemporal Forecasting for Epidemic Trajectories"

    profile_nlp = TopicProfileExtractor.extract(topic_nlp, domain="nlp")
    profile_ts = TopicProfileExtractor.extract(topic_ts, domain="time_series")

    selector = DynamicBaselineSelector()
    suite_nlp = selector.select_baselines(profile_nlp)
    suite_ts = selector.select_baselines(profile_ts)

    # Proposed method specs must exist and reflect task-specific components
    assert suite_nlp.proposed_method_spec is not None
    assert suite_ts.proposed_method_spec is not None
    assert suite_nlp.proposed_method_spec.architecture_definition != suite_ts.proposed_method_spec.architecture_definition
    assert suite_nlp.proposed_method_spec.loss_objective != suite_ts.proposed_method_spec.loss_objective
    assert suite_nlp.proposed_method_spec.training_procedure != suite_ts.proposed_method_spec.training_procedure


def test_literature_advisor_zero_doi_cycling_and_gap_extraction():
    """Verify zero DOI cycling and evidence-grounded research gap synthesis."""
    from backend.core.evidence_agent import EvidenceBundle, SourceRecord, ClaimRecord, EvidenceScope, VerificationStatus
    from backend.core.topic_profile import TopicProfileExtractor

    topic = "Multivariate Spatiotemporal Forecasting for Epidemic Trajectories"
    profile = TopicProfileExtractor.extract(topic, domain="time_series")

    # Create synthetic evidence with specific limitation claim
    claim1 = ClaimRecord(
        claim_id="clm_001",
        source_id="src_001",
        claim_text="Existing models suffer from compounding error accumulation over extended forecast horizons exceeding 12 steps.",
        supporting_text="Existing models suffer from compounding error accumulation over extended forecast horizons exceeding 12 steps.",
        supporting_location="abstract",
        evidence_scope=EvidenceScope.ABSTRACT,
        category="limitation",
        verification_status=VerificationStatus.GROUNDED,
    )
    source1 = SourceRecord(
        source_id="src_001",
        title="Spatiotemporal Graph Neural Networks for Traffic Forecasting",
        authors=["Yu et al."],
        year=2018,
        venue="IJCAI",
        doi="10.5555/3304415.3304532",
        url="https://doi.org/10.5555/3304415.3304532",
        evidence_scope=EvidenceScope.ABSTRACT,
        claims=[claim1],
    )
    evidence = EvidenceBundle(topic=topic, domain="time_series", sources=[source1], claims=[claim1])

    advisor = LiteratureAdvisor()
    report = advisor.synthesize(evidence, profile)

    # 1. Research gaps must be extracted from the limitation claim with exact source_id and passage
    assert len(report.candidate_gaps) > 0
    gap = report.candidate_gaps[0]
    assert "src_001" in gap.supporting_source_ids
    assert "error accumulation" in gap.description.lower() or "horizon" in gap.description.lower()

    # 2. Baseline selector with this report must NOT cycle src_001's DOI to unrelated baselines
    selector = DynamicBaselineSelector()
    suite = selector.select_baselines(profile, report)
    for b in suite.baselines:
        if "stgcn" in b.baseline_id.lower() or "graph" in b.name.lower():
            # May be grounded
            pass
        else:
            # Unrelated baselines must NOT have been assigned src_001's DOI via modulo cycling
            assert getattr(b, "doi", None) != "10.5555/3304415.3304532" or b.is_corpus_grounded is True


def test_mathematical_verification_engine_gate():
    """Verify the independent mathematical verification engine and failure downgrade."""
    from backend.core.math_agent import (
        MathematicalVerificationEngine,
        FormalTheorem,
        MathematicalDecision,
    )

    engine = MathematicalVerificationEngine()

    # Case A: Well-formed theorem
    valid_theorem = FormalTheorem(
        theorem_id="thm_001",
        title="Error Bound Guarantee",
        decision_type=MathematicalDecision.THEOREM_REQUIRED,
        formal_objects=["\\mathcal{H}", "\\theta"],
        assumptions=["Lipschitz continuity: \\|\\nabla f(x) - \\nabla f(y)\\| \\le L \\|x - y\\|."],
        statement="Under the Lipschitz assumption, the error satisfies \\|e_t\\| \\le \\mathcal{O}(1/\\sqrt{T}).",
        latex_statement=r"\begin{theorem}[\textbf{Error Bound}]" + "\n" + r"Under Lipschitz continuity, the error satisfies $\|e_t\| \le \mathcal{O}(1/\sqrt{T})$." + "\n" + r"\end{theorem}",
        proof_steps=["By standard telescoping sums over $t=1,\\dots,T$, we obtain the bound."],
        latex_proof=r"\begin{proof}" + "\n" + r"By standard telescoping sums, the result follows." + "\n" + r"\end{proof}",
    )
    is_valid, notes = engine.verify(valid_theorem)
    assert is_valid is True
    assert valid_theorem.is_verified is True
    assert r"\begin{theorem}" in valid_theorem.to_latex()

    # Case B: Malformed theorem with unbalanced braces
    malformed_theorem = FormalTheorem(
        theorem_id="thm_002",
        title="Malformed Proof Statement",
        decision_type=MathematicalDecision.THEOREM_REQUIRED,
        assumptions=[],
        statement=r"Convergence holds under \mathcal{O(1/T.",  # Unbalanced brace
        latex_statement=r"\begin{theorem}{Unbalanced brace \textbf{test}\end{theorem}",
    )
    is_valid_malformed, notes_malformed = engine.verify(malformed_theorem)
    assert is_valid_malformed is False
    assert malformed_theorem.is_verified is False
    # Unverified theorems must be downgraded to observation in LaTeX rendering
    assert r"\noindent\textbf{Observation:}" in malformed_theorem.to_latex() or "% Unverified" in malformed_theorem.to_latex()


def test_figure_provenance_and_cryptographic_data_hashing():
    """Verify cryptographic SHA-256 data hashing and provenance across distinct tasks."""
    import os
    import shutil
    from backend.core.figure_planner import FigurePlanningAgent
    from backend.core.topic_profile import TopicProfileExtractor

    topic_nlp = "Parameter-Efficient Low-Rank Adaptation for Large Models"
    topic_ts = "Multivariate Spatiotemporal Forecasting for Epidemic Trajectories"

    profile_nlp = TopicProfileExtractor.extract(topic_nlp, domain="nlp")
    profile_ts = TopicProfileExtractor.extract(topic_ts, domain="time_series")

    mock_metrics_nlp = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 88.5, "mean_memory_mb": 72.0, "mean_latency_ms": 8.5},
            "dense_baseline": {"mean_accuracy": 82.0, "mean_memory_mb": 340.0, "mean_latency_ms": 32.0},
        }
    }
    mock_metrics_ts = {
        "methods": {
            "proposed_mb_qgt": {"mean_accuracy": 91.2, "mean_memory_mb": 45.0, "mean_latency_ms": 5.2},
            "dense_baseline": {"mean_accuracy": 79.5, "mean_memory_mb": 210.0, "mean_latency_ms": 22.0},
        }
    }

    fig_agent = FigurePlanningAgent()
    out_dir_nlp = "./dist/test_figs_nlp_hash"
    out_dir_ts = "./dist/test_figs_ts_hash"

    figs_nlp = fig_agent.plan_figures(profile_nlp, mock_metrics_nlp, output_dir=out_dir_nlp)
    figs_ts = fig_agent.plan_figures(profile_ts, mock_metrics_ts, output_dir=out_dir_ts)

    res_nlp = fig_agent.generate_figures(figs_nlp, mock_metrics_nlp, profile_nlp, output_dir=out_dir_nlp)
    res_ts = fig_agent.generate_figures(figs_ts, mock_metrics_ts, profile_ts, output_dir=out_dir_ts)

    # Every planned figure must have a non-empty SHA-256 data hash
    for f in figs_nlp:
        assert len(f.data_hash) == 64
        assert f.research_question_addressed is not None
        assert f.output_filename in res_nlp

    for f in figs_ts:
        assert len(f.data_hash) == 64
        assert f.research_question_addressed is not None
        assert f.output_filename in res_ts

    # Convergence / Pareto figures across distinct tasks must produce distinct data hashes
    nlp_hashes = {f.figure_type.value: f.data_hash for f in figs_nlp}
    ts_hashes = {f.figure_type.value: f.data_hash for f in figs_ts}

    for ftype in nlp_hashes:
        if ftype in ts_hashes:
            assert nlp_hashes[ftype] != ts_hashes[ftype]

    # Clean up test directories
    shutil.rmtree(out_dir_nlp, ignore_errors=True)
    shutil.rmtree(out_dir_ts, ignore_errors=True)


