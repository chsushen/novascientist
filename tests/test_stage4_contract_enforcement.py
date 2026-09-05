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
