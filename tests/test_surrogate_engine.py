"""Unit tests for Surrogate Benchmark Engine & DerSimonian-Laird Meta-Analysis."""

import json
import tempfile
import numpy as np
import pytest
from backend.core.surrogate_engine import DerSimonianLairdEstimator, SurrogateBenchmarkEngine


def test_dersimonian_laird_math():
    # Test with synthetic 5-study data
    effect_sizes = [0.065, 0.058, 0.071, 0.062, 0.064]
    std_errors = [0.006, 0.007, 0.005, 0.006, 0.008]

    res = DerSimonianLairdEstimator.compute(effect_sizes, std_errors)

    assert res.degrees_of_freedom == 4
    assert res.cochran_q >= 0
    assert 0.0 <= res.i_squared_percent <= 100.0
    assert res.ci_95_lower < res.pooled_effect_size < res.ci_95_upper
    assert res.z_statistic > 0
    assert res.p_value_z < 0.05
    assert len(res.study_weights) == 5
    assert round(sum(res.study_weights), 1) == 100.0


def test_surrogate_engine_reproducibility():
    engine = SurrogateBenchmarkEngine(topic="Test Topic", num_seeds=5)
    pkg = engine.run_experiments()

    assert len(pkg.seeds) == 5
    assert "proposed_mb_qgt" in pkg.methods
    assert "dense_baseline" in pkg.methods

    prop = pkg.methods["proposed_mb_qgt"]
    dense = pkg.methods["dense_baseline"]

    # Proposed model should outperform baseline in accuracy and memory
    assert prop.mean_accuracy > dense.mean_accuracy
    assert prop.mean_memory_mb < dense.mean_memory_mb
    assert prop.mean_latency_ms < dense.mean_latency_ms

    # Meta analysis output
    assert pkg.meta_analysis.pooled_effect_size > 0


def test_export_metrics_json():
    engine = SurrogateBenchmarkEngine(topic="Test Topic", num_seeds=3)
    pkg = engine.run_experiments()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name

    engine.export_metrics_json(pkg, path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["topic"] == "Test Topic"
    assert "methods" in data
    assert "meta_analysis" in data
