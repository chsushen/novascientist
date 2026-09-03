"""Tests for DeepJournalAssembler 8-12 page journal synthesis."""

import pytest
from backend.core.deep_journal_assembler import DeepJournalAssembler
from backend.core.real_trainer import RealPyTorchTrainer
from backend.core.dataset_finder import DatasetFinder
from backend.core.literature import PaperMetadata
from dataclasses import asdict


def test_deep_journal_assembler_structure(tmp_path):
    trainer = RealPyTorchTrainer(
        topic="Low-Rank Dynamic Graph Attention for Smart Disaster Resilience",
        num_seeds=2,
        num_epochs=3,
        experiments_dir=str(tmp_path / "experiments"),
    )
    pkg = trainer.run_full_benchmark()
    metrics_dict = asdict(pkg)

    papers = [
        PaperMetadata(doi="10.1145/3209978.3210006", title="Spatial-Temporal Graph Networks", authors=["Li, Y."], year=2018, venue="KDD"),
        PaperMetadata(doi="10.1109/TPAMI.2020.1001", title="Dynamic Quantization", authors=["Chen, X."], year=2020, venue="TPAMI"),
    ]
    dataset = DatasetFinder.discover("Low-Rank Dynamic Graph Attention", "graph")

    assembler = DeepJournalAssembler(metrics_dict, papers, dataset=dataset)
    latex_text = assembler.generate_journal_latex()

    # Verify 10 complete sections & IEEE requirements
    assert r"\section{Introduction}" in latex_text
    assert r"\section{Related Work and Taxonomic Survey}" in latex_text
    assert r"\section{Theoretical Formulation and Mathematical Foundations}" in latex_text
    assert r"\section{System Architecture and Algorithm}" in latex_text
    assert r"\section{Experimental Setup and Hardware Profiling}" in latex_text
    assert r"\section{Empirical Results and Meta-Analytic Synthesis}" in latex_text
    assert r"\section{Component Ablation and Sensitivity Analysis}" in latex_text
    assert r"\section{In-Depth Technical Discussion and Complexity Analysis}" in latex_text
    assert r"\section{Ethical Statement and AI-Assistance Acknowledgment}" in latex_text
    assert r"\section{Conclusion and Future Trajectories}" in latex_text
    assert "Theorem" in latex_text
    assert "Lemma" in latex_text
    assert "Proposition" in latex_text
    assert "fig1_system_architecture" in latex_text
    assert "fig5_sensitivity_heatmap" in latex_text


def test_latex_formatting_and_string_sanitation(tmp_path):
    import re
    trainer = RealPyTorchTrainer(
        topic="Physics-Informed Surrogates for Compressible Aerodynamics",
        num_seeds=2,
        num_epochs=2,
        experiments_dir=str(tmp_path / "experiments"),
    )
    pkg = trainer.run_full_benchmark()
    metrics_dict = asdict(pkg)

    papers = [
        PaperMetadata(doi="10.1145/3209978.3210006", title="PDE Surrogates", authors=["Smith, A."], year=2021, venue="JCP"),
    ]
    dataset = DatasetFinder.discover("Physics-Informed Surrogates", "physics_surrogate")

    assembler = DeepJournalAssembler(metrics_dict, papers, dataset=dataset)
    latex_text = assembler.generate_journal_latex()

    # Strict formatting assertions
    assert not re.search(r"\(\s*\)", latex_text), "Found empty parentheses () in LaTeX"
    assert not re.search(r"\[95\% CI:\s*,\s*\]", latex_text), "Malformed confidence interval"
    assert not re.search(r"\\textbf\{\s*\%", latex_text), "Missing number before %"
    assert "Ham-QNO" in latex_text
    assert "Fidelity Index" in latex_text
