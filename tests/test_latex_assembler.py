"""Unit tests for LaTeXAssembler and Numerical Provenance Validation."""

import json
import pytest
from backend.core.latex_assembler import LaTeXAssembler
from backend.core.literature import PaperMetadata
from backend.core.surrogate_engine import SurrogateBenchmarkEngine


def test_latex_generation_and_provenance():
    engine = SurrogateBenchmarkEngine(topic="Low-Compute Quantization", num_seeds=5)
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

    assembler = LaTeXAssembler(metrics_dict, papers)
    latex_code = assembler.generate_latex()

    # Structural checks
    assert r"\documentclass[journal,10pt,twocolumn]{IEEEtran}" in latex_code
    assert r"\begin{document}" in latex_code
    assert r"\end{document}" in latex_code
    assert r"\bibliography{references}" in latex_code
    assert r"\cite{kipf2022_adaptive}" in latex_code

    # Numerical Invariant validation
    inv_errors = assembler.validate_numerical_invariants(latex_code, metrics_dict)
    assert len(inv_errors) == 0, f"Invariant errors: {inv_errors}"
