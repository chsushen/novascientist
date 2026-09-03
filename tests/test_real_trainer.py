"""Tests for RealPyTorchTrainer and hardware execution."""

import os
import shutil
import pytest
from backend.core.real_trainer import RealPyTorchTrainer, get_torch_device


def test_hardware_device_detection():
    dev_type, dev_name = get_torch_device()
    assert dev_type in ["cuda", "mps", "cpu"]
    assert len(dev_name) > 0


def test_real_pytorch_trainer_execution(tmp_path):
    exp_dir = tmp_path / "experiments"
    trainer = RealPyTorchTrainer(
        topic="Low-Rank Dynamic Graph Attention for Evacuation Forecasting",
        num_seeds=3,
        num_epochs=5,
        experiments_dir=str(exp_dir),
    )
    pkg = trainer.run_full_benchmark()

    assert pkg.topic == "Low-Rank Dynamic Graph Attention for Evacuation Forecasting"
    assert len(pkg.seeds) == 3
    assert "proposed_mb_qgt" in pkg.methods
    assert "dense_baseline" in pkg.methods
    assert pkg.methods["proposed_mb_qgt"].mean_accuracy > 0.70
    assert pkg.meta_analysis.pooled_effect_size > 0.0

    # Verify log output and checkpoints
    assert (exp_dir / "logs" / "training_log.json").exists()
