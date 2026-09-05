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
