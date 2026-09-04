"""Unit and integration test suite for NovaScientist v2 features."""

import pytest
from backend.core.universal_engine import (
    ComputationalDomain,
    UniversalDomainDispatcher,
    UniversalBenchmarkEngine,
)
from backend.core.venue_matcher import VenueMatcher
from backend.core.reviewer_swarm import (
    StatisticalAuditor,
    RhetoricLinter,
    AdversarialReviewerSwarm,
)
from backend.core.latex_assembler import (
    AuthorProfile,
    CompliantLaTeXAssembler,
    ComplianceViolationError,
)
from backend.core.literature import PaperMetadata


def test_domain_dispatcher_classification():
    # Physics surrogate
    res_pde = UniversalDomainDispatcher.classify_topic("Physics-Informed Dynamic Neural Surrogates under Bounded Memory")
    assert res_pde.domain == ComputationalDomain.PHYSICS_SURROGATE
    assert res_pde.confidence >= 0.70

    # Graph
    res_graph = UniversalDomainDispatcher.classify_topic("Low-Compute Dynamic Graph Representation under Quantized Memory")
    assert res_graph.domain == ComputationalDomain.GRAPH

    # Vision
    res_vision = UniversalDomainDispatcher.classify_topic("Efficient Convolutional Segmentation Transformers on Edge Devices")
    assert res_vision.domain == ComputationalDomain.VISION

    # NLP
    res_nlp = UniversalDomainDispatcher.classify_topic("Sub-Linear Memory LLM Attention Mechanisms for Sequence Modeling")
    assert res_nlp.domain == ComputationalDomain.NLP

    # Bioinformatics
    res_bio = UniversalDomainDispatcher.classify_topic("High-Throughput Metagenomic Binning and Contig Taxonomic Classification")
    assert res_bio.domain == ComputationalDomain.BIOINFORMATICS
    assert res_bio.model_acronym == "MetaGraph-Trans"

    # Quantum
    res_quantum = UniversalDomainDispatcher.classify_topic("Variational Quantum Tensor Networks for Molecular Ground-State Eigensolvers")
    assert res_quantum.domain == ComputationalDomain.QUANTUM
    assert res_quantum.model_acronym == "VQ-TensorNet"


def test_universal_zero_shot_domain_synthesis():
    res_cloaking = UniversalDomainDispatcher.classify_topic("Metamaterial Terahertz Resonator Cloaking under Anisotropic Dielectrics")
    assert res_cloaking.model_acronym != ""
    assert res_cloaking.confidence >= 0.70
    assert "MTR" in res_cloaking.model_acronym or "Clo" in res_cloaking.model_acronym
    assert "Fidelity" in res_cloaking.primary_metric_name

    res_acoustic = UniversalDomainDispatcher.classify_topic("Acoustic Levitation Particle Trapping Dynamics")
    assert res_acoustic.model_acronym != ""
    assert res_acoustic.confidence >= 0.70


def test_venue_matcher_ranking():
    recs = VenueMatcher.match_venues(
        "Physics-Informed Neural Surrogates under Bounded Memory",
        ComputationalDomain.PHYSICS_SURROGATE,
        top_k=3,
    )
    assert len(recs) == 3
    assert recs[0].rank == 1
    assert any("TNNLS" in r.venue.short_name or "JCP" in r.venue.short_name for r in recs)
    for r in recs:
        assert r.relevance_score > 0.5
        assert r.venue.template_class != ""


def test_author_profile_compliance_gate():
    # Default double-blind anonymous author profile
    anon_author = AuthorProfile()
    anon_author.validate()  # Must pass without error
    assert anon_author.name == "Anonymous Author(s)"

    # Valid human author
    valid_author = AuthorProfile(
        name="Dr. Jane Doe",
        affiliation="Laboratory for Information and Decision Systems, MIT",
        email="janedoe@mit.edu",
    )
    valid_author.validate()  # Should not raise

    # Disallowed AI tokens
    ai_authors = [
        "AI Agent",
        "Autonomous Research Engine",
        "NovaScientist Bot",
        "ChatGPT",
        "Claude LLM",
    ]
    for bad_name in ai_authors:
        bad_profile = AuthorProfile(
            name=bad_name,
            affiliation="Research Lab",
            email="test@domain.org",
        )
        with pytest.raises(ComplianceViolationError):
            bad_profile.validate()

    # Empty name or missing email
    with pytest.raises(ComplianceViolationError):
        AuthorProfile(name="", affiliation="University", email="test@domain.org").validate()

    with pytest.raises(ComplianceViolationError):
        AuthorProfile(name="Researcher", affiliation="University", email="invalid_email").validate()


def test_statistical_auditor():
    # Underpowered (k=3 seeds)
    bad_metrics = {
        "seeds": [42, 137, 2024],
        "methods": {
            "m1": {"name": "M1", "std_accuracy": 0.01},
            "m2": {"name": "M2", "std_accuracy": 0.01},
        },
        "meta_analysis": {"i_squared_percent": 10.0},
    }
    passed, issues = StatisticalAuditor.audit_experiment_package(bad_metrics)
    assert not passed
    assert any("Statistical Power Violation" in i for i in issues)

    # Valid (k=5 seeds)
    good_metrics = {
        "seeds": [42, 179, 316, 453, 590],
        "methods": {
            "m1": {"name": "M1", "std_accuracy": 0.012},
            "m2": {"name": "M2", "std_accuracy": 0.009},
        },
        "meta_analysis": {"i_squared_percent": 0.0},
    }
    passed_good, issues_good = StatisticalAuditor.audit_experiment_package(good_metrics)
    assert passed_good
    assert len(issues_good) == 0


def test_rhetoric_linter():
    raw_text = "Our method completely solves all memory issues and achieves unbeatable SOTA accuracy with a universal solution."
    cleaned, mods = RhetoricLinter.lint_and_refactor(raw_text)

    assert "completely solves" not in cleaned
    assert "effectively mitigates" in cleaned
    assert "state-of-the-art" in cleaned
    assert "broadly applicable framework" in cleaned
    assert len(mods) >= 3


def test_compliant_latex_assembler_domain_equations_and_resizebox():
    engine = UniversalBenchmarkEngine(
        topic="Physics-Informed Dynamic Neural Surrogates under Bounded Memory",
        num_seeds=5,
    )
    pkg = engine.run_experiments()
    from dataclasses import asdict
    metrics_dict = asdict(pkg)

    papers = [
        PaperMetadata(
            doi="10.1109/TPAMI.2021.3099999",
            title="Adaptive Quantization and Memory-Bounded Graph Neural Networks",
            authors=["Kipf, Thomas", "Welling, Max"],
            year=2022,
            venue="IEEE TPAMI",
            bibkey="kipf2022_adaptive",
            source_origin="test_fixture",
            text_origin="test_fixture",
        )
    ]

    # Default anonymous author
    assembler = CompliantLaTeXAssembler(metrics_dict, papers)
    latex_doc = assembler.generate_latex()

    # Verify anonymous double-blind author is in LaTeX
    assert "Anonymous Author(s)" in latex_doc
    assert "Affiliation Withheld for Double-Blind Review" in latex_doc

    # Verify domain-specific PDE residual equation (not GNN)
    assert r"\mathcal{L}_{\text{pde}}" in latex_doc
    assert r"\partial_t u_\theta" in latex_doc
    assert "gnn_aggregation" not in latex_doc

    # Verify table is wrapped with resizebox to prevent clipping
    assert r"\resizebox{\textwidth}{!}" in latex_doc

    # Verify AI disclosure section
    assert r"\section{Ethical Statement and AI-Assistance Acknowledgment}" in latex_doc
    assert "In compliance with IEEE and ACM 2024+ authorship policies" in latex_doc

    # Verify figure inclusions are unique (each declared exactly once)
    assert latex_doc.count("figures/convergence_frontier.pdf") == 1
    assert latex_doc.count("figures/pareto_tradeoff.pdf") == 1
    assert latex_doc.count("figures/meta_forest_plot.pdf") == 1

    # Verify numerical invariants
    errors = assembler.validate_numerical_invariants(latex_doc, metrics_dict)
    assert len(errors) == 0
