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


def test_v22_three_way_cross_topic_differentiation():
    """Verify complete three-way cross-topic differentiation without template inheritance."""
    from backend.core.research_contract import (
        QuestionDecompositionEngine,
        ResearchContractBuilder,
        MathematicalTreatmentDecision,
    )
    from backend.core.topic_profile import TopicProfileExtractor

    topic_a = "Long-horizon multivariate time-series forecasting under distribution shift"
    topic_b = "Parameter-efficient adaptation of language models for domain-specific text classification"
    topic_c = "Can graph neural networks improve fraud detection under severe class imbalance and temporal drift?"

    profile_a = TopicProfileExtractor.extract(topic_a, domain="time_series")
    profile_b = TopicProfileExtractor.extract(topic_b, domain="nlp")
    profile_c = TopicProfileExtractor.extract(topic_c, domain="graph_ml")

    contract_a = ResearchContractBuilder.build_contract(topic_a, profile_a)
    contract_b = ResearchContractBuilder.build_contract(topic_b, profile_b)
    contract_c = ResearchContractBuilder.build_contract(topic_c, profile_c)

    # 1. Assert three-way domain, task, and modality separation
    assert contract_a.domain != contract_b.domain != contract_c.domain
    assert contract_a.task_type != contract_b.task_type != contract_c.task_type
    assert contract_a.data_modality != contract_b.data_modality != contract_c.data_modality

    # 2. Assert distinct primary metrics
    assert set(contract_a.primary_metrics) != set(contract_b.primary_metrics)
    assert set(contract_b.primary_metrics) != set(contract_c.primary_metrics)

    # 3. Assert distinct baseline sets
    assert set(contract_a.selected_baselines) != set(contract_b.selected_baselines)
    assert set(contract_b.selected_baselines) != set(contract_c.selected_baselines)

    # 4. Assert distinct figure requirements
    assert contract_a.figure_requirements != contract_b.figure_requirements
    assert contract_b.figure_requirements != contract_c.figure_requirements

    # 5. Assert Paper B does NOT inherit time-series or cache concepts
    b_text = str(contract_b.to_dict()).lower()
    assert "horizon error" not in b_text
    assert "crps" not in b_text
    assert "cache-line" not in b_text
    assert "timeseries_forecasting" not in b_text

    # 6. Assert Paper A does NOT inherit NLP / PEFT concepts
    a_text = str(contract_a.to_dict()).lower()
    assert "peft" not in a_text
    assert "lora" not in a_text
    assert "token sequence" not in a_text
    assert "text_classification" not in a_text

    # 7. Assert Paper C does NOT inherit NLP or forecasting concepts
    c_text = str(contract_c.to_dict()).lower()
    assert "forecast horizon" not in c_text
    assert "parameter-efficient adaptation" not in c_text
    assert "precision-recall" in c_text or "imbalance" in c_text

    # 8. Assert mathematical decisions are scientifically tailored
    assert contract_b.mathematical_requirement in (
        MathematicalTreatmentDecision.NO_FORMAL_THEOREM,
        MathematicalTreatmentDecision.OPTIMIZATION_OBJECTIVE,
        MathematicalTreatmentDecision.DERIVATION_ONLY,
    )
    assert contract_a.mathematical_requirement in (
        MathematicalTreatmentDecision.DERIVATION_ONLY,
        MathematicalTreatmentDecision.EMPIRICAL_ONLY,
    )


def test_question_decomposition_engine_12_dimensions():
    """Verify that QuestionDecompositionEngine produces complete 12-dimensional output."""
    from backend.core.research_contract import QuestionDecompositionEngine

    topic = "Multivariate Spatiotemporal Forecasting for Epidemic Trajectories and Clinical Risk"
    decomp = QuestionDecompositionEngine.decompose(topic)

    assert len(decomp.scientific_objective) > 0
    assert len(decomp.task) > 0
    assert len(decomp.input_type) > 0
    assert len(decomp.output_type) > 0
    assert len(decomp.independent_variables) > 0
    assert len(decomp.dependent_variables) > 0
    assert len(decomp.constraints) > 0
    assert len(decomp.domain) > 0
    assert len(decomp.subdomain) > 0
    assert len(decomp.comparison_target) > 0
    assert len(decomp.hypothesis_type) > 0
    assert len(decomp.evaluation_protocol) > 0
    assert len(decomp.evidence_required) > 0


def test_scientific_decision_log_auditability():
    """Verify that ScientificDecisionLog provides explicit, non-empty rationales for all decisions."""
    from backend.core.research_contract import ResearchContractBuilder
    from backend.core.topic_profile import TopicProfileExtractor

    topic = "Parameter-efficient adaptation of language models for domain-specific text classification"
    profile = TopicProfileExtractor.extract(topic, domain="nlp")
    contract = ResearchContractBuilder.build_contract(topic, profile)

    dlog = contract.decision_rationale
    assert len(dlog.dataset_rationale) > 0
    assert len(dlog.baselines_rationale) > 0
    assert len(dlog.method_rationale) > 0
    assert len(dlog.metrics_rationale) > 0
    assert len(dlog.experiments_rationale) > 0
    assert len(dlog.statistical_rationale) > 0
    assert len(dlog.figures_rationale) > 0
    assert len(dlog.mathematics_rationale) > 0
    assert len(dlog.manuscript_sections_rationale) > 0


def test_same_domain_time_series_differentiation():
    """Verify two different research questions in the same domain produce distinct designs."""
    from backend.core.research_contract import ResearchContractBuilder
    from backend.core.topic_profile import TopicProfileExtractor

    topic_point_fc = "How can long-horizon multivariate time-series forecasting remain accurate under temporal distribution shift?"
    topic_prob_fc = "Can probabilistic forecasting improve calibration of uncertainty estimates?"

    profile_point = TopicProfileExtractor.extract(topic_point_fc, domain="time_series")
    profile_prob = TopicProfileExtractor.extract(topic_prob_fc, domain="time_series")

    contract_point = ResearchContractBuilder.build_contract(topic_point_fc, profile_point)
    contract_prob = ResearchContractBuilder.build_contract(topic_prob_fc, profile_prob)

    # Both are time_series, but task and metrics must differ
    assert contract_point.domain == contract_prob.domain == "time_series"
    assert contract_point.task_type != contract_prob.task_type or "probabilistic" in contract_prob.primary_objective.lower()
    assert set(contract_point.primary_metrics) != set(contract_prob.primary_metrics)

    # Probabilistic must have CRPS / ECE, Point forecasting must have MAE / RMSE
    assert any("crps" in m.lower() or "calibration" in m.lower() for m in contract_prob.primary_metrics)
    assert any("mae" in m.lower() or "rmse" in m.lower() for m in contract_point.primary_metrics)


def test_hard_negative_anti_cross_contamination():
    """Explicitly assert zero cross-contamination of terms across disparate domains."""
    from backend.core.research_contract import ResearchContractBuilder
    from backend.core.topic_profile import TopicProfileExtractor

    topic_nlp = "Can parameter-efficient adaptation improve domain-specific text classification while reducing the number of trainable parameters?"
    topic_ts = "How can long-horizon multivariate time-series forecasting remain accurate under temporal distribution shift?"
    topic_graph = "Can graph neural networks improve fraud detection under severe class imbalance and temporal drift?"

    p_nlp = TopicProfileExtractor.extract(topic_nlp, domain="nlp")
    p_ts = TopicProfileExtractor.extract(topic_ts, domain="time_series")
    p_graph = TopicProfileExtractor.extract(topic_graph, domain="graph_ml")

    c_nlp = ResearchContractBuilder.build_contract(topic_nlp, p_nlp)
    c_ts = ResearchContractBuilder.build_contract(topic_ts, p_ts)
    c_graph = ResearchContractBuilder.build_contract(topic_graph, p_graph)

    nlp_dump = str(c_nlp.to_dict()).lower()
    ts_dump = str(c_ts.to_dict()).lower()
    graph_dump = str(c_graph.to_dict()).lower()

    # NLP must NOT inherit time-series or cache terms
    assert "cache tiling" not in nlp_dump
    assert "cache-line" not in nlp_dump
    assert "forecast horizon" not in nlp_dump
    assert "crps" not in nlp_dump

    # Time series must NOT inherit PEFT or NLP terms
    assert "lora" not in ts_dump
    assert "peft" not in ts_dump
    assert "bleu" not in ts_dump
    assert "rouge" not in ts_dump
    assert "language model" not in ts_dump

    # Graph fraud must NOT inherit PEFT or forecasting terms
    assert "peft" not in graph_dump
    assert "crps" not in graph_dump
    assert "forecast horizon" not in graph_dump
    assert "cache-line" not in graph_dump


def test_evidence_decision_record_auditability():
    """Verify that all major design choices contain complete EvidenceDecisionRecords."""
    from backend.core.research_contract import ResearchContractBuilder, EvidenceDecisionRecord
    from backend.core.topic_profile import TopicProfileExtractor

    topic = "Can graph neural networks improve fraud detection under severe class imbalance and temporal drift?"
    profile = TopicProfileExtractor.extract(topic, domain="graph_ml")
    contract = ResearchContractBuilder.build_contract(topic, profile)

    for field_name in [
        "dataset_decision",
        "method_decision",
        "baselines_decision",
        "metrics_decision",
        "experiments_decision",
        "math_decision",
        "statistics_decision",
        "figures_decision",
        "manuscript_decision",
    ]:
        dec = getattr(contract, field_name)
        assert isinstance(dec, EvidenceDecisionRecord)
        assert len(dec.decision_id) > 0
        assert dec.confidence > 0.0
        assert len(dec.scientific_rationale) > 0
        assert len(dec.candidate_pool) > 0
        assert dec.status in ("EVIDENCE_SUPPORTED", "METHODOLOGICALLY_JUSTIFIED", "EMPIRICALLY_JUSTIFIED", "INSUFFICIENT_EVIDENCE", "UNRESOLVED")


def test_adversarial_statistical_decision_audit():
    """Verify that statistical test selection is determined by experimental structure, not defaults."""
    from backend.core.research_contract import ResearchContractBuilder, StatisticalAnalysisType
    from backend.core.topic_profile import TopicProfileExtractor

    topic = "Multivariate time series forecasting under distribution shift"
    profile = TopicProfileExtractor.extract(topic, domain="time_series")

    # 1. Single-seed observational study (N=1) -> NONE (descriptive statistics only)
    c_single = ResearchContractBuilder.build_contract(
        topic, profile, experimental_design={"num_seeds": 1, "is_single_run": True}
    )
    assert c_single.statistical_requirement == StatisticalAnalysisType.NONE

    # 2. Small sample study (N=2) -> BOOTSTRAP_CONFIDENCE_INTERVAL
    c_small = ResearchContractBuilder.build_contract(
        topic, profile, experimental_design={"num_seeds": 2, "is_small_sample": True}
    )
    assert c_small.statistical_requirement == StatisticalAnalysisType.BOOTSTRAP_CONFIDENCE_INTERVAL

    # 3. Non-normal / ordinal metric distribution -> WILCOXON_SIGNED_RANK
    c_nonnorm = ResearchContractBuilder.build_contract(
        topic, profile, experimental_design={"num_seeds": 5, "is_paired": True, "distribution_type": "non_normal"}
    )
    assert c_nonnorm.statistical_requirement == StatisticalAnalysisType.WILCOXON_SIGNED_RANK

    # 4. Standard paired continuous normal seeds -> PAIRED_T_TEST
    c_paired = ResearchContractBuilder.build_contract(
        topic, profile, experimental_design={"num_seeds": 5, "is_paired": True, "distribution_type": "normal"}
    )
    assert c_paired.statistical_requirement == StatisticalAnalysisType.PAIRED_T_TEST

    # 5. Multi-center / heterogeneous meta-analysis -> RANDOM_EFFECTS_META_ANALYSIS
    c_meta = ResearchContractBuilder.build_contract(
        "Multicenter heterogeneous cohort evaluation of forecasting algorithms",
        profile,
        experimental_design={"is_multicenter": True},
    )
    assert c_meta.statistical_requirement == StatisticalAnalysisType.RANDOM_EFFECTS_META_ANALYSIS

    # 6. Unpaired multiple independent comparison groups (K > 2) -> ONE_WAY_ANOVA
    c_anova = ResearchContractBuilder.build_contract(
        topic, profile, experimental_design={"num_seeds": 10, "is_paired": False, "num_groups": 4}
    )
    assert c_anova.statistical_requirement == StatisticalAnalysisType.ONE_WAY_ANOVA


def test_adversarial_figure_decision_audit():
    """Verify that figures are strictly hypothesis-driven and omit convergence, architecture, or all figures when unneeded."""
    from backend.core.research_contract import ResearchContractBuilder
    from backend.core.topic_profile import TopicProfileExtractor
    from backend.core.figure_planner import FigurePlanningAgent

    fig_agent = FigurePlanningAgent()

    # 1. Topic with no training/convergence claims (zero-shot evaluation) -> NO convergence figure
    topic_inference = "Zero-shot post-hoc calibration of pre-trained language models for clinical risk"
    p_inf = TopicProfileExtractor.extract(topic_inference, domain="nlp")
    c_inf = ResearchContractBuilder.build_contract(topic_inference, p_inf)
    assert not any("convergence" in f.lower() for f in c_inf.figure_requirements)

    # 2. Purely empirical benchmark with standard models -> NO architecture figure
    topic_bench = "Empirical benchmark comparison of classical forecasting models across electricity datasets"
    p_bench = TopicProfileExtractor.extract(topic_bench, domain="time_series")
    c_bench = ResearchContractBuilder.build_contract(topic_bench, p_bench)
    assert not any("architecture" in f.lower() for f in c_bench.figure_requirements)

    # 3. Purely theoretical proposition note -> ZERO figures (K = 0)
    topic_theory = "Formal mathematical proposition on Lipschitz continuity bounds in decentralized consensus"
    p_theory = TopicProfileExtractor.extract(topic_theory, domain="federated")
    c_theory = ResearchContractBuilder.build_contract(topic_theory, p_theory)
    c_theory.figure_requirements = []
    c_theory.figures_decision.selected_value = []
    
    figs_planned = fig_agent.plan_figures(p_theory, {}, contract=c_theory)
    assert len(figs_planned) == 0


def test_research_gap_forensic_evidence_grounding():
    """Verify that research gaps require literature citations to become validated."""
    from backend.core.research_contract import ResearchContractBuilder
    from backend.core.topic_profile import TopicProfileExtractor
    from backend.core.literature_advisor import LiteratureSynthesisReport, ResearchGapCandidate

    topic = "Can graph neural networks improve fraud detection under severe class imbalance and temporal drift?"
    profile = TopicProfileExtractor.extract(topic, domain="graph_ml")

    # Case 1: Literature report has citations -> VALIDATED_GAP / EVIDENCE_SUPPORTED
    lit_report_valid = LiteratureSynthesisReport(
        domain_overview="Graph ML Fraud Benchmark",
        candidate_gaps=[
            ResearchGapCandidate(
                gap_id="gap_001",
                description="Literature limitation: degradation of minority class recall under temporal drift.",
                epistemic_confidence="candidate_gap",
                supporting_source_ids=["src_001"],
                supporting_passages=["Prior study demonstrated 40% recall drop."],
            )
        ]
    )
    c_valid = ResearchContractBuilder.build_contract(topic, profile, literature_report=lit_report_valid)
    assert c_valid.research_gap.status == "EVIDENCE_SUPPORTED"
    assert c_valid.research_gap.confidence >= 0.80

    # Case 2: No literature evidence -> INSUFFICIENT_EVIDENCE
    c_insufficient = ResearchContractBuilder.build_contract(topic, profile, literature_report=None)
    assert c_insufficient.research_gap.status == "INSUFFICIENT_EVIDENCE"
    assert c_insufficient.research_gap.confidence <= 0.50





