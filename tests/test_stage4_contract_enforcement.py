"""
Unit and integration tests for Downstream Scientific Contract Enforcement.
Ensures ScientificResearchContract acts as the authoritative source of truth
across execution, assemblers, provenance, and UI reporting.
"""

import os
import json
import pytest
import tempfile
from pathlib import Path

from backend.core.topic_profile import TopicProfileExtractor
from backend.core.research_contract import (
    ScientificResearchContract,
    ResearchContractBuilder,
    ScientificContractViolation,
    ScientificContractViolationError,
    StatisticalRequirement,
    MathematicalTreatmentDecision,
    generate_contract_consistency_report,
)
from backend.core.latex_assembler import CompliantLaTeXAssembler, AuthorProfile
from backend.core.deep_journal_assembler import DeepJournalAssembler


def make_contract(topic: str) -> ScientificResearchContract:
    profile = TopicProfileExtractor.extract(topic)
    return ResearchContractBuilder.build_contract(topic=topic, profile=profile)


def test_contract_freezing_immutability():
    """Test that freeze() locks all fields and raises ScientificContractViolation on modification."""
    contract = make_contract(
        "Can Retrieval-Augmented Generation Improve Factual Consistency in Domain-Specific Question Answering?"
    )
    assert not contract.is_frozen
    
    # Pre-freeze modification is allowed
    contract.primary_objective = "Updated Objective"
    assert contract.primary_objective == "Updated Objective"
    
    # Freeze
    contract.freeze()
    assert contract.is_frozen
    
    # Post-freeze modification must raise violation
    with pytest.raises((ScientificContractViolation, ScientificContractViolationError)):
        contract.primary_objective = "Illegal Mutation"
        
    with pytest.raises((ScientificContractViolation, ScientificContractViolationError)):
        contract.selected_dataset = "IllegalDataset"


def test_rag_contract_and_latex_assembler_consistency():
    """Test that RAG contract produces 100% consistent LaTeX with zero uncontracted hardware figures."""
    topic = "Can Retrieval-Augmented Generation Improve Factual Consistency in Domain-Specific Question Answering?"
    contract = make_contract(topic)
    contract.freeze()
    
    author = AuthorProfile(name="Dr. Jane Doe", affiliation="AI Institute", email="jane@ai.org")
    assembler = CompliantLaTeXAssembler(
        metrics_data={"topic": topic, "methods": {}},
        papers=[],
        author=author,
        contract=contract,
    )
    latex_code = assembler.generate_latex()
    
    # Check consistency
    report = generate_contract_consistency_report(
        contract=contract,
        latex_content=latex_code,
        figures={},
        metrics_dict={},
    )
    
    assert report["status"] == "PASS", f"Report failed with unauthorized elements: {report}"
    assert len(report["unauthorized_figures"]) == 0
    assert len(report["unauthorized_methodology"]) == 0
    assert len(report["unauthorized_dataset"]) == 0


def test_peft_contract_and_deep_journal_assembler():
    """Test PEFT topic contract consistency with DeepJournalAssembler."""
    topic = "Can parameter-efficient adaptation improve domain-specific text classification while reducing trainable parameters?"
    contract = make_contract(topic)
    contract.freeze()
    
    author = AuthorProfile(name="Dr. Alex Smith", affiliation="Stanford", email="alex@stanford.edu")
    assembler = DeepJournalAssembler(
        metrics_dict={"topic": topic, "methods": {}},
        papers=[],
        author=author,
        contract=contract,
    )
    latex_code = assembler.generate_journal_latex()
    
    report = generate_contract_consistency_report(
        contract=contract,
        latex_content=latex_code,
        figures={},
        metrics_dict={},
    )
    
    assert report["status"] == "PASS", f"Report failed with unauthorized elements: {report}"
    assert len(report["unauthorized_methodology"]) == 0
    assert len(report["unauthorized_mathematics"]) == 0


def test_forecasting_contract_derivation_math():
    """Test forecasting contract enforces DERIVATION_ONLY mathematical treatment."""
    topic = "How can long-horizon multivariate forecasting remain accurate under temporal distribution shift?"
    contract = make_contract(topic)
    assert contract.mathematical_requirement == MathematicalTreatmentDecision.DERIVATION_ONLY
    contract.freeze()
    
    author = AuthorProfile(name="Dr. Forecast", affiliation="Time Lab", email="forecast@lab.org")
    assembler = CompliantLaTeXAssembler(
        metrics_data={"topic": topic, "methods": {}},
        papers=[],
        author=author,
        contract=contract,
    )
    latex_code = assembler.generate_latex()
    
    report = generate_contract_consistency_report(
        contract=contract,
        latex_content=latex_code,
        figures={},
        metrics_dict={},
    )
    assert report["status"] == "PASS", f"Report failed: {report}"


def test_statistical_selector_adversarial_design_derivation():
    """Adversarial test proving Statistical Selector derives method from experimental structure."""
    topic = "Comparative Evaluation of Deep Learning Architectures"
    profile = TopicProfileExtractor.extract(topic)
    
    # 1. Paired, Normal, k >= 3 -> PAIRED_T_TEST
    c_paired = ResearchContractBuilder.build_contract(
        topic, profile, experimental_design={"is_paired": True, "num_seeds": 5, "distribution_type": "normal"}
    )
    assert c_paired.statistical_requirement == StatisticalRequirement.PAIRED_T_TEST
    
    # 2. Non-Normal / Ordinal / Heavy-Tailed -> WILCOXON_SIGNED_RANK
    c_wilcoxon = ResearchContractBuilder.build_contract(
        topic, profile, experimental_design={"is_paired": True, "num_seeds": 5, "distribution_type": "non_normal"}
    )
    assert c_wilcoxon.statistical_requirement == StatisticalRequirement.WILCOXON_SIGNED_RANK
    
    # 3. Small Sample (k = 2) -> BOOTSTRAP_CONFIDENCE_INTERVAL
    c_boot = ResearchContractBuilder.build_contract(
        topic, profile, experimental_design={"is_paired": True, "num_seeds": 2, "is_small_sample": True}
    )
    assert c_boot.statistical_requirement == StatisticalRequirement.BOOTSTRAP_CONFIDENCE_INTERVAL
    
    # 4. Single-Run / N = 1 -> NONE (Descriptive Statistics only)
    c_single = ResearchContractBuilder.build_contract(
        topic, profile, experimental_design={"is_single_run": True, "num_seeds": 1}
    )
    assert c_single.statistical_requirement == StatisticalRequirement.NONE
    
    # 5. Independent Groups (>2 groups, unpaired) -> ONE_WAY_ANOVA
    c_anova = ResearchContractBuilder.build_contract(
        topic, profile, experimental_design={"is_paired": False, "num_groups": 4, "num_seeds": 5}
    )
    assert c_anova.statistical_requirement == StatisticalRequirement.ONE_WAY_ANOVA
    
    # 6. Multicenter / Multi-Site Heterogeneous Variance -> RANDOM_EFFECTS_META_ANALYSIS
    c_meta = ResearchContractBuilder.build_contract(
        topic, profile, experimental_design={"is_multicenter": True, "num_seeds": 5}
    )
    assert c_meta.statistical_requirement == StatisticalRequirement.RANDOM_EFFECTS_META_ANALYSIS


def test_hypothesis_mismatch_adversarial_detection():
    """Adversarial test: evaluating variance with accuracy or meta-analysis with speedup is flagged/blocked."""
    topic = "Can Retrieval-Augmented Generation Improve Factual Consistency in Domain-Specific Question Answering?"
    contract = make_contract(topic)
    contract.freeze()

    from backend.core.research_contract import HypothesisEvaluation, HypothesisStatus
    # Corrupt evaluation: evaluating variance hypothesis with accuracy delta metric
    corrupt_evals = [
        HypothesisEvaluation(
            hypothesis_id="H2",
            statement="Performance across multi-seed random splits remains stable with variance within 1.0%.",
            metric_name="accuracy_delta",  # MISMATCH: should be seed_variance_std
            metric_direction="maximize",
            threshold=-1.50,
            comparison_target=">= -1.50%",
            experiment_ids=["exp_01"],
            statistical_test="paired_t_test",
            raw_observations=[5.2],
            observed_value=5.2,
            effect_size=0.5,
            confidence_interval=None,
            p_value=0.01,
            decision=HypothesisStatus.SUPPORTED,
            rationale="Recycled accuracy delta to evaluate variance.",
        )
    ]

    val = contract.validate_downstream_state(hypothesis_evaluations=corrupt_evals)
    assert not val["is_valid"]
    assert val["status"] == "BLOCKED"
    assert any("Hypothesis" in v and "H2" in v and "variance" in v for v in val["violations"])


def test_experiment_contamination_rejection():
    """Adversarial test: uncontracted or forbidden legacy methods trigger BLOCKED status."""
    topic = "Can Retrieval-Augmented Generation Improve Factual Consistency in Domain-Specific Question Answering?"
    contract = make_contract(topic)
    contract.freeze()

    from dataclasses import dataclass
    @dataclass
    class MockExpRecord:
        experiment_id: str
        method_name: str
        method_id: str
        seed: int

    # Inject forbidden quantization method into RAG run
    contaminated_records = [
        MockExpRecord(experiment_id="exp_01", method_name="Static INT8 Quantization", method_id="post_int8", seed=42),
    ]

    val = contract.validate_downstream_state(experiment_records=contaminated_records)
    assert not val["is_valid"]
    assert val["status"] == "BLOCKED"
    assert any("quantization" in v.lower() or "int8" in v.lower() for v in val["violations"])


def test_physical_pdf_page_budget_fail_closed():
    """Test that physical PDF outside 6–8 pages is strictly BLOCKED."""
    topic = "Can Retrieval-Augmented Generation Improve Factual Consistency in Domain-Specific Question Answering?"
    contract = make_contract(topic)
    contract.freeze()

    # Create dummy 3-page PDF
    import pypdf
    writer = pypdf.PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=612, height=792)
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name
        writer.write(f)

    try:
        val = contract.validate_downstream_state(pdf_path=pdf_path)
        assert not val["is_valid"]
        assert val["status"] == "BLOCKED"
        assert val["physical_pdf_pages"] == 3
        assert any("violates Standard Conference target" in v for v in val["violations"])
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


def test_cross_topic_differentiation():
    """Verify RAG, PEFT, and Forecasting contracts produce distinct, uncontaminated specifications."""
    c_rag = make_contract("Can Retrieval-Augmented Generation Improve Factual Consistency in Domain-Specific Question Answering?")
    c_peft = make_contract("Can parameter-efficient adaptation improve domain-specific text classification while reducing trainable parameters?")
    c_fore = make_contract("How can long-horizon multivariate forecasting remain accurate under temporal distribution shift?")

    # 1. Models & Methods
    assert c_rag.selected_method != c_peft.selected_method
    assert c_rag.selected_method != c_fore.selected_method
    assert "rag" in c_rag.selected_method.lower() or "ada-rag" in c_rag.selected_method.lower() or "retrieval" in c_rag.selected_method.lower()
    assert "peft" in c_peft.selected_method.lower() or "adapter" in c_peft.selected_method.lower() or "ada-peft" in c_peft.selected_method.lower()
    assert "forecast" in c_fore.selected_method.lower() or "tempshift" in c_fore.selected_method.lower() or "temporal" in c_fore.selected_method.lower()

    # 2. Datasets
    assert c_rag.selected_dataset != c_fore.selected_dataset
    assert any(k in c_rag.selected_dataset.lower() for k in ["qa", "pubmed", "trivia", "squad", "natural"])
    assert any(k in c_fore.selected_dataset.lower() for k in ["ett", "weather", "traffic", "electricity", "forecast", "timeseries"])

    # 3. Primary Metrics
    assert c_rag.primary_metrics != c_fore.primary_metrics
    assert any(any(k in m.lower() for k in ["factual", "exact match", "f1", "qa", "attribution"]) for m in c_rag.primary_metrics)
    assert any(any(k in m.lower() for k in ["mae", "rmse", "wape", "crps", "forecast", "error"]) for m in c_fore.primary_metrics)


def test_same_domain_nlp_differentiation():
    """Verify RAG QA vs Text Classification in NLP domain produce differentiated contracts."""
    c_rag = make_contract("Can Retrieval-Augmented Generation Improve Factual Consistency in Domain-Specific Question Answering?")
    c_peft = make_contract("Can parameter-efficient adaptation improve domain-specific text classification while reducing trainable parameters?")

    assert c_rag.domain == c_peft.domain  # Both are NLP / Language Modeling
    assert c_rag.subdomain != c_peft.subdomain or c_rag.selected_dataset != c_peft.selected_dataset
    assert c_rag.selected_method != c_peft.selected_method
    assert c_rag.selected_baselines != c_peft.selected_baselines
