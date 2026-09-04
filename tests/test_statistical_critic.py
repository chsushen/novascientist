"""Tests for NovaScientist Statistical Critic Agent (Phase 3).

Verifies real data-driven statistical evaluation of multi-seed experiment telemetry,
hypothesis test selection (paired t-test / Wilcoxon), Cohen's d effect sizes,
hard failure vs warning separation, and single-seed non-fabrication guarantees.
"""

import math
from dataclasses import asdict
import numpy as np
import pytest
import scipy.stats as stats

from backend.core.statistical_critic import (
    MetricSummary,
    MethodStatisticalSummary,
    PairwiseComparison,
    StatisticalCriticAgent,
    StatisticalCritique,
)
from backend.core.surrogate_engine import SurrogateBenchmarkEngine


def test_single_seed_hard_failure_and_non_fabrication():
    """Requirement 4: 1 seed must explicitly trigger hard failure and not fabricate data."""
    critic = StatisticalCriticAgent()
    metrics_single_seed = {
        "seeds": [42],
        "methods": {
            "proposed_mb_qgt": {
                "name": "Proposed MB-QGT",
                "seed_results": [{"final_accuracy": 0.88, "peak_memory_mb": 75.0, "inference_latency_ms": 8.5}],
            },
            "dense_baseline": {
                "name": "Dense Baseline",
                "seed_results": [{"final_accuracy": 0.82, "peak_memory_mb": 390.0, "inference_latency_ms": 34.0}],
            },
        },
    }

    critique = critic.evaluate_statistics(metrics_single_seed)
    assert critique.passed is False
    assert critique.sample_size_sufficient is False
    assert critique.num_seeds == 1

    # Invariant: Must contain explicit insufficient repeated-seed warning
    assert any("Insufficient repeated-seed evidence" in issue for issue in critique.critical_issues)
    assert any("only 1 seed evaluated" in issue for issue in critique.critical_issues)

    # Invariant: Summary statistics for n=1 must have std=0.0, se=0.0, CI=None (no fabricated intervals)
    prop_acc = critique.method_summaries["proposed_mb_qgt"].metrics["accuracy"]
    assert prop_acc.sample_size == 1
    assert prop_acc.mean == 0.88
    assert prop_acc.std == 0.0
    assert prop_acc.se == 0.0
    assert prop_acc.ci_95_lower is None
    assert prop_acc.ci_95_upper is None

    # Invariant: Pairwise comparison must declare insufficient_samples, no fake p-values
    assert len(critique.pairwise_comparisons) == 1
    cmp_res = critique.pairwise_comparisons[0]
    assert cmp_res.test_used == "insufficient_samples"
    assert cmp_res.p_value is None
    assert cmp_res.statistic is None


def test_two_seeds_underpowered_hard_failure():
    """Sample size k=2 is underpowered for statistical significance (minimum k=3)."""
    critic = StatisticalCriticAgent()
    metrics_2_seeds = {
        "seeds": [42, 179],
        "methods": {
            "proposed_mb_qgt": {
                "name": "Proposed MB-QGT",
                "seed_results": [
                    {"final_accuracy": 0.88, "peak_memory_mb": 75.0, "inference_latency_ms": 8.5},
                    {"final_accuracy": 0.89, "peak_memory_mb": 74.0, "inference_latency_ms": 8.3},
                ],
            },
            "dense_baseline": {
                "name": "Dense Baseline",
                "seed_results": [
                    {"final_accuracy": 0.82, "peak_memory_mb": 390.0, "inference_latency_ms": 34.0},
                    {"final_accuracy": 0.83, "peak_memory_mb": 388.0, "inference_latency_ms": 33.5},
                ],
            },
        },
    }

    critique = critic.evaluate_statistics(metrics_2_seeds)
    assert critique.passed is False
    assert critique.sample_size_sufficient is False
    assert any("Sample Size Violation" in issue for issue in critique.critical_issues)


def test_three_and_four_seeds_pass_threshold_with_recommendation():
    """Sample size k=3 passes hard threshold but generates recommendation for k=5."""
    critic = StatisticalCriticAgent()
    metrics_3_seeds = {
        "seeds": [42, 179, 316],
        "methods": {
            "proposed_mb_qgt": {
                "name": "Proposed MB-QGT",
                "seed_results": [
                    {"final_accuracy": 0.882, "peak_memory_mb": 75.0, "inference_latency_ms": 8.5},
                    {"final_accuracy": 0.891, "peak_memory_mb": 74.0, "inference_latency_ms": 8.3},
                    {"final_accuracy": 0.886, "peak_memory_mb": 74.5, "inference_latency_ms": 8.4},
                ],
            },
            "dense_baseline": {
                "name": "Dense Baseline",
                "seed_results": [
                    {"final_accuracy": 0.821, "peak_memory_mb": 390.0, "inference_latency_ms": 34.0},
                    {"final_accuracy": 0.830, "peak_memory_mb": 388.0, "inference_latency_ms": 33.5},
                    {"final_accuracy": 0.825, "peak_memory_mb": 389.0, "inference_latency_ms": 33.8},
                ],
            },
        },
    }

    critique = critic.evaluate_statistics(metrics_3_seeds)
    assert critique.passed is True
    assert critique.sample_size_sufficient is True
    assert any("Low Seed Count" in w for w in critique.warnings)


def test_empty_or_malformed_metrics_dictionary():
    """Empty or None metrics dictionary triggers critical hard failure."""
    critic = StatisticalCriticAgent()

    critique_none = critic.evaluate_statistics(None)
    assert critique_none.passed is False
    assert any("Missing Telemetry" in issue for issue in critique_none.critical_issues)

    critique_empty = critic.evaluate_statistics({})
    assert critique_empty.passed is False
    assert any("Missing Telemetry" in issue for issue in critique_empty.critical_issues)

    critique_no_methods = critic.evaluate_statistics({"seeds": [1, 2, 3], "methods": {}})
    assert critique_no_methods.passed is False
    assert any("Missing Methods" in issue for issue in critique_no_methods.critical_issues)


def test_missing_baselines_hard_failure():
    """Evaluation with only 1 method triggers missing baseline hard failure."""
    critic = StatisticalCriticAgent()
    metrics_single_method = {
        "seeds": [42, 179, 316, 453, 590],
        "methods": {
            "proposed_mb_qgt": {
                "name": "Proposed MB-QGT",
                "seed_results": [{"accuracy": 0.88 + i * 0.001} for i in range(5)],
            }
        },
    }

    critique = critic.evaluate_statistics(metrics_single_method)
    assert critique.passed is False
    assert any("Missing Baselines" in issue for issue in critique.critical_issues)


def test_malformed_nan_inf_telemetry_hard_failure():
    """NaN or Inf values in metric telemetry must be caught as hard failures."""
    critic = StatisticalCriticAgent()
    metrics_nan = {
        "seeds": [42, 179, 316, 453, 590],
        "methods": {
            "proposed_mb_qgt": {
                "name": "Proposed MB-QGT",
                "seed_results": [
                    {"accuracy": 0.88},
                    {"accuracy": float("nan")},
                    {"accuracy": 0.89},
                    {"accuracy": 0.885},
                    {"accuracy": 0.887},
                ],
            },
            "dense_baseline": {
                "name": "Dense Baseline",
                "seed_results": [{"accuracy": 0.82} for _ in range(5)],
            },
        },
    }

    critique = critic.evaluate_statistics(metrics_nan)
    assert critique.passed is False
    assert any("Malformed Telemetry" in issue for issue in critique.critical_issues)


def test_metric_summary_mathematical_accuracy():
    """Verify mean, std (ddof=1), SE, 95% Student's t CI, min, max, and median."""
    raw_accuracies = [0.850, 0.860, 0.870, 0.880, 0.890]
    n = 5
    expected_mean = float(np.mean(raw_accuracies))  # 0.870
    expected_std = float(np.std(raw_accuracies, ddof=1))  # 0.0158113883
    expected_se = float(expected_std / math.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=4))  # ~2.7764451
    expected_ci_lower = expected_mean - t_crit * expected_se
    expected_ci_upper = expected_mean + t_crit * expected_se

    summary = StatisticalCriticAgent._compute_metric_summary("accuracy", raw_accuracies)

    assert summary.sample_size == 5
    assert pytest.approx(summary.mean, abs=1e-5) == expected_mean
    assert pytest.approx(summary.std, abs=1e-5) == expected_std
    assert pytest.approx(summary.se, abs=1e-5) == expected_se
    assert pytest.approx(summary.ci_95_lower, abs=1e-5) == expected_ci_lower
    assert pytest.approx(summary.ci_95_upper, abs=1e-5) == expected_ci_upper
    assert summary.min == 0.850
    assert summary.max == 0.890
    assert summary.median == 0.870


def test_deterministic_hypothesis_testing_paired_t_test():
    """Paired t-test is deterministically selected for normally distributed paired differences."""
    prop_accs = [0.885, 0.892, 0.888, 0.890, 0.889]
    base_accs = [0.820, 0.825, 0.822, 0.824, 0.823]

    cmp_res = StatisticalCriticAgent._conduct_pairwise_test(
        proposed_id="proposed_mb_qgt",
        proposed_name="Proposed MB-QGT",
        baseline_id="dense_baseline",
        baseline_name="Dense Baseline",
        metric_name="accuracy",
        proposed_vals=prop_accs,
        baseline_vals=base_accs,
    )

    assert cmp_res.test_used == "paired_t_test"
    assert cmp_res.degrees_of_freedom == 4
    assert cmp_res.statistic is not None and cmp_res.statistic > 0
    assert cmp_res.p_value is not None and cmp_res.p_value < 0.001
    assert cmp_res.is_significant is True
    assert cmp_res.effect_size_magnitude == "large"
    assert cmp_res.effect_size_cohens_d is not None and cmp_res.effect_size_cohens_d > 2.0


def test_cohens_d_magnitude_classification():
    """Cohen's d magnitudes are classified according to standard conventions."""
    # Large effect
    g1_large = [10.0, 10.2, 10.1, 10.3, 10.0]
    g2_large = [2.0, 2.1, 2.0, 2.2, 2.1]
    d_l, mag_l = StatisticalCriticAgent._compute_cohens_d(g1_large, g2_large)
    assert mag_l == "large"
    assert d_l is not None and d_l > 0.8

    # Negligible effect
    g1_neg = [0.850, 0.852, 0.851, 0.849, 0.850]
    g2_neg = [0.850, 0.851, 0.850, 0.852, 0.849]
    d_n, mag_n = StatisticalCriticAgent._compute_cohens_d(g1_neg, g2_neg)
    assert mag_n == "negligible"
    assert d_n is not None and abs(d_n) < 0.2


def test_zero_variance_detection_and_warning():
    """Zero variance across multiple seeds generates zero-variance warning."""
    critic = StatisticalCriticAgent()
    metrics_zero_var = {
        "seeds": [42, 179, 316, 453, 590],
        "methods": {
            "proposed_mb_qgt": {
                "name": "Proposed MB-QGT",
                "seed_results": [{"accuracy": 0.88} for _ in range(5)],  # All identical
            },
            "dense_baseline": {
                "name": "Dense Baseline",
                "seed_results": [{"accuracy": 0.82 + i * 0.002} for i in range(5)],
            },
        },
    }

    critique = critic.evaluate_statistics(metrics_zero_var)
    assert critique.variance_bounded is False
    assert any("Zero-Variance Warning" in w for w in critique.warnings)


def test_excessive_heterogeneity_hard_failure():
    """Meta-analysis I^2 > 75% triggers hard failure for excessive heterogeneity."""
    critic = StatisticalCriticAgent()
    metrics_hetero = {
        "seeds": [42, 179, 316, 453, 590],
        "methods": {
            "proposed_mb_qgt": {
                "name": "Proposed MB-QGT",
                "seed_results": [{"accuracy": 0.88 + i * 0.005} for i in range(5)],
            },
            "dense_baseline": {
                "name": "Dense Baseline",
                "seed_results": [{"accuracy": 0.82 + i * 0.002} for i in range(5)],
            },
        },
        "meta_analysis": {
            "z_statistic": 4.5,
            "i_squared_percent": 84.5,  # Exceeds 75.0% threshold
        },
    }

    critique = critic.evaluate_statistics(metrics_hetero)
    assert critique.passed is False
    assert critique.heterogeneity_acceptable is False
    assert any("Excessive Heterogeneity" in issue for issue in critique.critical_issues)


def test_end_to_end_surrogate_engine_telemetry_integration():
    """Full integration test verifying StatisticalCritic against real SurrogateBenchmarkEngine package."""
    engine = SurrogateBenchmarkEngine("Quantum Tensor Networks under Memory Bounds")
    pkg = engine.run_experiments()
    metrics_dict = asdict(pkg)

    critic = StatisticalCriticAgent()
    critique = critic.evaluate_statistics(metrics_dict)

    assert critique.passed is True
    assert critique.num_seeds == 5
    assert critique.sample_size_sufficient is True
    assert critique.variance_bounded is True
    assert critique.meta_analysis_significant is True
    assert critique.heterogeneity_acceptable is True

    # Verify method summaries populated for all 4 methods
    assert len(critique.method_summaries) == 4
    assert "proposed_mb_qgt" in critique.method_summaries
    assert "dense_baseline" in critique.method_summaries

    prop_summary = critique.method_summaries["proposed_mb_qgt"]
    assert "accuracy" in prop_summary.metrics
    assert "memory_mb" in prop_summary.metrics
    assert "latency_ms" in prop_summary.metrics
    assert prop_summary.metrics["accuracy"].sample_size == 5
    assert prop_summary.metrics["accuracy"].std > 0.0
    assert prop_summary.metrics["accuracy"].ci_95_lower is not None

    # Verify pairwise comparisons populated
    assert len(critique.pairwise_comparisons) == 3
    for cmp_res in critique.pairwise_comparisons:
        assert cmp_res.proposed_id == "proposed_mb_qgt"
        assert cmp_res.test_used in ["paired_t_test", "wilcoxon_signed_rank", "constant_difference"]
        assert cmp_res.p_value is not None
        assert cmp_res.effect_size_cohens_d is not None
