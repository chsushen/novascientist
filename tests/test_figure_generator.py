"""Tests for ScientificFigureSuite 5-figure generation."""

import json
import pytest
from backend.core.figure_generator import ScientificFigureSuite
from backend.core.real_trainer import RealPyTorchTrainer
from dataclasses import asdict


def test_scientific_figure_suite_generation(tmp_path):
    trainer = RealPyTorchTrainer(
        topic="Physics Informed Surrogates",
        num_seeds=2,
        num_epochs=3,
        experiments_dir=str(tmp_path / "experiments"),
    )
    pkg = trainer.run_full_benchmark()
    metrics_dict = asdict(pkg)

    fig_dir = tmp_path / "figures"
    fig_suite = ScientificFigureSuite(metrics_dict, output_dir=str(fig_dir))
    figs = fig_suite.generate_all_figures()

    assert len(figs) == 5
    for key in ["fig1_system_architecture", "fig2_convergence_curves", "fig3_pareto_frontier", "fig4_ablation_study", "fig5_sensitivity_heatmap"]:
        assert key in figs
        assert (fig_dir / f"{key}.pdf").exists()
        assert (fig_dir / f"{key}.png").exists()
