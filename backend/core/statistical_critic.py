"""NovaScientist Statistical Critic Agent.

Performs adversarial statistical evaluation of empirical experiment packages,
multi-seed variance distributions, DerSimonian-Laird random-effects estimators,
hypothesis testing (paired t-test / Wilcoxon signed-rank), Cohen's d effect sizes,
cherry-picking risks, and confidence interval estimation on actual telemetry.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class MetricSummary:
    """Summary statistics for a specific metric across seeds."""

    metric_name: str
    sample_size: int
    mean: float
    std: float
    se: float
    ci_95_lower: float | None
    ci_95_upper: float | None
    min: float
    max: float
    median: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MethodStatisticalSummary:
    """Statistical profile for an evaluation method across all metrics."""

    method_id: str
    method_name: str
    sample_size: int
    metrics: dict[str, MetricSummary] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "method_name": self.method_name,
            "sample_size": self.sample_size,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }


@dataclass
class PairwiseComparison:
    """Statistical hypothesis test comparing proposed method against a baseline."""

    baseline_id: str
    baseline_name: str
    proposed_id: str
    proposed_name: str
    metric_name: str
    test_used: str  # 'paired_t_test', 'wilcoxon_signed_rank', 'insufficient_samples', 'constant_difference'
    statistic: float | None
    p_value: float | None
    effect_size_cohens_d: float | None
    effect_size_magnitude: str  # 'negligible', 'small', 'medium', 'large', 'undefined'
    is_significant: bool  # p < 0.05
    mean_difference: float
    degrees_of_freedom: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StatisticalCritique:
    """Structured findings from the statistical critic agent."""

    critique_id: str
    passed: bool
    num_seeds: int
    sample_size_sufficient: bool
    variance_bounded: bool
    meta_analysis_significant: bool
    heterogeneity_acceptable: bool
    cherry_picking_risk: str  # 'low', 'medium', 'high'
    critical_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    method_summaries: dict[str, MethodStatisticalSummary] = field(default_factory=dict)
    pairwise_comparisons: list[PairwiseComparison] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["method_summaries"] = {
            k: v.to_dict() if hasattr(v, "to_dict") else v
            for k, v in self.method_summaries.items()
        }
        res["pairwise_comparisons"] = [
            c.to_dict() if hasattr(c, "to_dict") else c
            for c in self.pairwise_comparisons
        ]
        return res


class StatisticalCriticAgent:
    """Adversarial statistical agent ensuring data-driven significance without data fabrication."""

    MIN_SEEDS_HARD_THRESHOLD = 3
    RECOMMENDED_SEEDS = 5
    MAX_ALLOWED_HETEROGENEITY_I2 = 75.0
    SIGNIFICANCE_ALPHA = 0.05

    METRIC_ALIASES: dict[str, list[str]] = {
        "accuracy": [
            "final_accuracy",
            "accuracy",
            "acc",
            "val_acc",
            "test_acc",
            "mean_accuracy",
        ],
        "memory_mb": [
            "peak_memory_mb",
            "memory_mb",
            "memory",
            "mem_mb",
            "mem",
            "mean_memory_mb",
        ],
        "latency_ms": [
            "inference_latency_ms",
            "latency_ms",
            "latency",
            "lat_ms",
            "lat",
            "mean_latency_ms",
        ],
        "throughput": [
            "throughput_samples_sec",
            "throughput",
            "throughput_fps",
            "mean_throughput",
        ],
        "compression_ratio": [
            "compression_ratio",
            "compression",
            "comp_ratio",
            "mean_compression_ratio",
        ],
    }

    def __init__(self) -> None:
        pass

    @staticmethod
    def _extract_metric_series(method_data: Any, metric_canonical: str) -> list[float]:
        """Extract a series of numeric observations for a canonical metric."""
        aliases = StatisticalCriticAgent.METRIC_ALIASES.get(
            metric_canonical, [metric_canonical]
        )

        if method_data is None:
            return []

        # Check if method_data is a dict or object
        if isinstance(method_data, dict):
            runs = (
                method_data.get("seed_runs")
                or method_data.get("seed_results")
                or method_data.get("runs")
            )
            if runs and isinstance(runs, list):
                extracted: list[float] = []
                for r in runs:
                    val = None
                    if isinstance(r, dict):
                        for a in aliases:
                            if a in r and r[a] is not None:
                                val = r[a]
                                break
                    elif hasattr(r, "__dict__"):
                        for a in aliases:
                            if hasattr(r, a) and getattr(r, a) is not None:
                                val = getattr(r, a)
                                break
                    if val is not None:
                        try:
                            extracted.append(float(val))
                        except (TypeError, ValueError):
                            extracted.append(float("nan"))
                if extracted:
                    return extracted

            # Check direct series in dict
            for a in aliases:
                if a in method_data and isinstance(method_data[a], list):
                    return [float(x) for x in method_data[a] if x is not None]

            # Check single summary value in dict
            for a in aliases:
                if a in method_data and isinstance(method_data[a], (int, float)):
                    return [float(method_data[a])]

        elif hasattr(method_data, "__dict__"):
            runs = getattr(method_data, "seed_runs", None) or getattr(
                method_data, "seed_results", None
            )
            if runs and isinstance(runs, list):
                extracted = []
                for r in runs:
                    val = None
                    for a in aliases:
                        if hasattr(r, a) and getattr(r, a) is not None:
                            val = getattr(r, a)
                            break
                        elif isinstance(r, dict) and a in r:
                            val = r[a]
                            break
                    if val is not None:
                        try:
                            extracted.append(float(val))
                        except (TypeError, ValueError):
                            extracted.append(float("nan"))
                if extracted:
                    return extracted

            for a in aliases:
                if hasattr(method_data, a):
                    val = getattr(method_data, a)
                    if isinstance(val, (int, float)):
                        return [float(val)]

        return []

    @staticmethod
    def _compute_metric_summary(metric_name: str, values: list[float]) -> MetricSummary:
        """Calculate mean, std, se, 95% CI, min, max, median for actual values."""
        n = len(values)
        if n == 0:
            return MetricSummary(
                metric_name=metric_name,
                sample_size=0,
                mean=0.0,
                std=0.0,
                se=0.0,
                ci_95_lower=None,
                ci_95_upper=None,
                min=0.0,
                max=0.0,
                median=0.0,
            )

        # Check for NaN / Inf
        clean_values = [
            float(v) for v in values if not math.isnan(v) and not math.isinf(v)
        ]
        if len(clean_values) != n:
            # Propagate NaN if present
            return MetricSummary(
                metric_name=metric_name,
                sample_size=n,
                mean=float("nan"),
                std=float("nan"),
                se=float("nan"),
                ci_95_lower=None,
                ci_95_upper=None,
                min=float("nan"),
                max=float("nan"),
                median=float("nan"),
            )

        mean_val = float(np.mean(clean_values))
        min_val = float(np.min(clean_values))
        max_val = float(np.max(clean_values))
        median_val = float(np.median(clean_values))

        if n >= 2:
            std_val = float(np.std(clean_values, ddof=1))
            se_val = float(std_val / math.sqrt(n))
            # 95% Confidence Interval via Student's t distribution
            t_crit = float(
                stats.t.ppf(
                    1.0 - StatisticalCriticAgent.SIGNIFICANCE_ALPHA / 2.0, df=n - 1
                )
            )
            ci_lower = float(mean_val - t_crit * se_val)
            ci_upper = float(mean_val + t_crit * se_val)
        else:
            std_val = 0.0
            se_val = 0.0
            ci_lower = None
            ci_upper = None

        return MetricSummary(
            metric_name=metric_name,
            sample_size=n,
            mean=round(mean_val, 6),
            std=round(std_val, 6),
            se=round(se_val, 6),
            ci_95_lower=round(ci_lower, 6) if ci_lower is not None else None,
            ci_95_upper=round(ci_upper, 6) if ci_upper is not None else None,
            min=round(min_val, 6),
            max=round(max_val, 6),
            median=round(median_val, 6),
        )

    @staticmethod
    def _compute_cohens_d(
        group1: list[float], group2: list[float]
    ) -> tuple[float | None, str]:
        """Compute Cohen's d effect size and classify its magnitude."""
        n1 = len(group1)
        n2 = len(group2)
        if n1 < 2 or n2 < 2:
            return None, "undefined"

        var1 = float(np.var(group1, ddof=1))
        var2 = float(np.var(group2, ddof=1))
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

        if pooled_var <= 1e-12:
            diff = float(np.mean(group1) - np.mean(group2))
            if abs(diff) < 1e-9:
                return 0.0, "negligible"
            return (10.0 if diff > 0 else -10.0), "large"

        d = float((np.mean(group1) - np.mean(group2)) / math.sqrt(pooled_var))
        abs_d = abs(d)

        if abs_d < 0.2:
            magnitude = "negligible"
        elif abs_d < 0.5:
            magnitude = "small"
        elif abs_d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        return round(d, 4), magnitude

    @staticmethod
    def _conduct_pairwise_test(
        proposed_id: str,
        proposed_name: str,
        baseline_id: str,
        baseline_name: str,
        metric_name: str,
        proposed_vals: list[float],
        baseline_vals: list[float],
    ) -> PairwiseComparison:
        """Conduct deterministic hypothesis testing between proposed and baseline methods."""
        n = min(len(proposed_vals), len(baseline_vals))
        mean_diff = (
            float(np.mean(proposed_vals) - np.mean(baseline_vals)) if n > 0 else 0.0
        )

        if n < StatisticalCriticAgent.MIN_SEEDS_HARD_THRESHOLD:
            return PairwiseComparison(
                baseline_id=baseline_id,
                baseline_name=baseline_name,
                proposed_id=proposed_id,
                proposed_name=proposed_name,
                metric_name=metric_name,
                test_used="insufficient_samples",
                statistic=None,
                p_value=None,
                effect_size_cohens_d=None,
                effect_size_magnitude="undefined",
                is_significant=False,
                mean_difference=round(mean_diff, 6),
                degrees_of_freedom=None,
            )

        p_subset = proposed_vals[:n]
        b_subset = baseline_vals[:n]
        diffs = [p - b for p, b in zip(p_subset, b_subset)]

        # Check for constant differences
        diff_var = float(np.var(diffs, ddof=1)) if n > 1 else 0.0
        if diff_var <= 1e-12:
            d_val, d_mag = StatisticalCriticAgent._compute_cohens_d(p_subset, b_subset)
            return PairwiseComparison(
                baseline_id=baseline_id,
                baseline_name=baseline_name,
                proposed_id=proposed_id,
                proposed_name=proposed_name,
                metric_name=metric_name,
                test_used="constant_difference",
                statistic=None,
                p_value=0.0 if abs(mean_diff) > 1e-6 else 1.0,
                effect_size_cohens_d=d_val,
                effect_size_magnitude=d_mag,
                is_significant=(abs(mean_diff) > 1e-6),
                mean_difference=round(mean_diff, 6),
                degrees_of_freedom=n - 1,
            )

        # Normality check of paired differences via Shapiro-Wilk
        shapiro_p = 1.0
        if n >= 6:
            try:
                shapiro_res = stats.shapiro(diffs)
                shapiro_p = float(shapiro_res.pvalue)
            except Exception:
                shapiro_p = 1.0

        d_val, d_mag = StatisticalCriticAgent._compute_cohens_d(p_subset, b_subset)

        # For n <= 5, two-sided Wilcoxon minimum p-value is 0.0625, so paired t-test is mathematically necessary
        if n <= 5 or shapiro_p >= 0.01:
            # Parametric: Paired Student's t-test
            t_res = stats.ttest_rel(p_subset, b_subset)
            stat = float(t_res.statistic)
            pval = float(t_res.pvalue)
            return PairwiseComparison(
                baseline_id=baseline_id,
                baseline_name=baseline_name,
                proposed_id=proposed_id,
                proposed_name=proposed_name,
                metric_name=metric_name,
                test_used="paired_t_test",
                statistic=round(stat, 4),
                p_value=round(pval, 6),
                effect_size_cohens_d=d_val,
                effect_size_magnitude=d_mag,
                is_significant=(pval < StatisticalCriticAgent.SIGNIFICANCE_ALPHA),
                mean_difference=round(mean_diff, 6),
                degrees_of_freedom=n - 1,
            )
        else:
            # Non-parametric: Wilcoxon signed-rank test
            try:
                w_res = stats.wilcoxon(p_subset, b_subset)
                stat = float(w_res.statistic)
                pval = float(w_res.pvalue)
                return PairwiseComparison(
                    baseline_id=baseline_id,
                    baseline_name=baseline_name,
                    proposed_id=proposed_id,
                    proposed_name=proposed_name,
                    metric_name=metric_name,
                    test_used="wilcoxon_signed_rank",
                    statistic=round(stat, 4),
                    p_value=round(pval, 6),
                    effect_size_cohens_d=d_val,
                    effect_size_magnitude=d_mag,
                    is_significant=(pval < StatisticalCriticAgent.SIGNIFICANCE_ALPHA),
                    mean_difference=round(mean_diff, 6),
                    degrees_of_freedom=None,
                )
            except Exception:
                t_res = stats.ttest_rel(p_subset, b_subset)
                return PairwiseComparison(
                    baseline_id=baseline_id,
                    baseline_name=baseline_name,
                    proposed_id=proposed_id,
                    proposed_name=proposed_name,
                    metric_name=metric_name,
                    test_used="paired_t_test_fallback",
                    statistic=round(float(t_res.statistic), 4),
                    p_value=round(float(t_res.pvalue), 6),
                    effect_size_cohens_d=d_val,
                    effect_size_magnitude=d_mag,
                    is_significant=(
                        float(t_res.pvalue) < StatisticalCriticAgent.SIGNIFICANCE_ALPHA
                    ),
                    mean_difference=round(mean_diff, 6),
                    degrees_of_freedom=n - 1,
                )
            except Exception:
                # Fallback to t-test if Wilcoxon cannot rank zero ties
                t_res = stats.ttest_rel(p_subset, b_subset)
                return PairwiseComparison(
                    baseline_id=baseline_id,
                    baseline_name=baseline_name,
                    proposed_id=proposed_id,
                    proposed_name=proposed_name,
                    metric_name=metric_name,
                    test_used="paired_t_test_fallback",
                    statistic=round(float(t_res.statistic), 4),
                    p_value=round(float(t_res.pvalue), 6),
                    effect_size_cohens_d=d_val,
                    effect_size_magnitude=d_mag,
                    is_significant=(
                        float(t_res.pvalue) < StatisticalCriticAgent.SIGNIFICANCE_ALPHA
                    ),
                    mean_difference=round(mean_diff, 6),
                    degrees_of_freedom=n - 1,
                )

    def evaluate_statistics(
        self, metrics_dict: dict[str, Any] | None
    ) -> StatisticalCritique:
        """Conduct rigorous data-driven statistical audit across all evaluation folds."""
        critical_issues: list[str] = []
        warnings: list[str] = []
        recommendations: list[str] = []

        if not metrics_dict or not isinstance(metrics_dict, dict):
            critical_issues.append(
                "Missing Telemetry: Experiment metrics dictionary is empty or invalid."
            )
            return StatisticalCritique(
                critique_id="stat_critique_err",
                passed=False,
                num_seeds=0,
                sample_size_sufficient=False,
                variance_bounded=False,
                meta_analysis_significant=False,
                heterogeneity_acceptable=False,
                cherry_picking_risk="high",
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=[
                    "Re-run experiment pipeline to produce valid metrics.json artifact."
                ],
            )

        methods = metrics_dict.get("methods", {})
        if not isinstance(methods, dict) or len(methods) == 0:
            critical_issues.append(
                "Missing Methods: No evaluation methods found in experiment telemetry."
            )
            return StatisticalCritique(
                critique_id="stat_critique_err",
                passed=False,
                num_seeds=0,
                sample_size_sufficient=False,
                variance_bounded=False,
                meta_analysis_significant=False,
                heterogeneity_acceptable=False,
                cherry_picking_risk="high",
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=[
                    "Provide at least 2 distinct evaluation methods (proposed + baseline)."
                ],
            )

        seeds = metrics_dict.get("seeds", [])
        num_seeds = len(seeds) if isinstance(seeds, list) else 0

        # Build method statistical profiles from actual telemetry
        method_summaries: dict[str, MethodStatisticalSummary] = {}
        all_sample_sizes: list[int] = []

        for m_id, m_data in methods.items():
            m_name = (
                m_data.get("name", m_id)
                if isinstance(m_data, dict)
                else getattr(m_data, "name", m_id)
            )
            metric_summaries: dict[str, MetricSummary] = {}
            m_sizes = []

            for canonical_metric in [
                "accuracy",
                "memory_mb",
                "latency_ms",
                "throughput",
                "compression_ratio",
            ]:
                series = self._extract_metric_series(m_data, canonical_metric)
                if series:
                    # Check for NaN / Inf
                    if any(math.isnan(v) or math.isinf(v) for v in series):
                        critical_issues.append(
                            f"Malformed Telemetry: NaN or Inf detected in method '{m_id}' metric '{canonical_metric}'."
                        )
                    summary = self._compute_metric_summary(canonical_metric, series)
                    metric_summaries[canonical_metric] = summary
                    m_sizes.append(summary.sample_size)

            sample_size = max(m_sizes) if m_sizes else 0
            all_sample_sizes.append(sample_size)
            method_summaries[m_id] = MethodStatisticalSummary(
                method_id=m_id,
                method_name=m_name,
                sample_size=sample_size,
                metrics=metric_summaries,
            )

        effective_seeds = (
            num_seeds
            if num_seeds > 0
            else (max(all_sample_sizes) if all_sample_sizes else 0)
        )

        # 1. Sample Size Rigorous Evaluation
        if effective_seeds == 0:
            sample_size_ok = False
            critical_issues.append(
                "Missing Telemetry: No experimental seeds or sample runs evaluated."
            )
            recommendations.append(
                f"Execute multi-seed evaluation with at least k={self.RECOMMENDED_SEEDS} random seeds."
            )
        elif effective_seeds == 1:
            sample_size_ok = False
            critical_issues.append(
                "Insufficient repeated-seed evidence: only 1 seed evaluated (minimum 3 required for statistical significance)."
            )
            recommendations.append(
                f"Execute multi-seed evaluation with at least k={self.RECOMMENDED_SEEDS} random seeds."
            )
        elif effective_seeds < self.MIN_SEEDS_HARD_THRESHOLD:
            sample_size_ok = False
            critical_issues.append(
                f"Sample Size Violation: Evaluated {effective_seeds} seeds. Minimum required for statistical significance is k={self.MIN_SEEDS_HARD_THRESHOLD}."
            )
            recommendations.append(
                f"Increase random seed count to at least k={self.RECOMMENDED_SEEDS}."
            )
        elif effective_seeds < self.RECOMMENDED_SEEDS:
            sample_size_ok = True
            warnings.append(
                f"Low Seed Count: Evaluated {effective_seeds} seeds. While k={effective_seeds} allows hypothesis testing, k={self.RECOMMENDED_SEEDS} is recommended for publication."
            )
            recommendations.append(
                f"Consider scaling seed evaluation to k={self.RECOMMENDED_SEEDS} for tighter confidence bounds."
            )
        else:
            sample_size_ok = True

        # 2. Baseline Availability Check
        if len(methods) < 2:
            critical_issues.append(
                f"Missing Baselines: Comparative evaluation requires at least 2 distinct methods. Found {len(methods)}."
            )
            recommendations.append(
                "Include standard baseline architectures for comparative benchmark."
            )

        # 3. Variance & Dispersion Check
        variance_ok = True
        for m_id, summary in method_summaries.items():
            acc_summary = summary.metrics.get("accuracy")
            if acc_summary and acc_summary.sample_size >= 2:
                if acc_summary.std == 0.0:
                    variance_ok = False
                    warnings.append(
                        f"Zero-Variance Warning: Method '{m_id}' has std=0.0 across {acc_summary.sample_size} seeds (potential seed re-use or static constants)."
                    )
                elif acc_summary.mean > 0:
                    cv = acc_summary.std / acc_summary.mean
                    if cv > 0.25:
                        warnings.append(
                            f"High Relative Variance: Method '{m_id}' exhibits coefficient of variation CV={cv:.2f}."
                        )

        # 4. Pairwise Hypothesis Testing & Effect Sizes
        proposed_key = None
        for k in methods.keys():
            if "proposed" in k.lower() or "mb_qgt" in k.lower():
                proposed_key = k
                break
        if not proposed_key and methods:
            proposed_key = list(methods.keys())[0]

        pairwise_comparisons: list[PairwiseComparison] = []
        if proposed_key and proposed_key in method_summaries:
            prop_summary = method_summaries[proposed_key]
            prop_acc_vals = self._extract_metric_series(
                methods[proposed_key], "accuracy"
            )

            for base_key, base_summary in method_summaries.items():
                if base_key == proposed_key:
                    continue
                base_acc_vals = self._extract_metric_series(
                    methods[base_key], "accuracy"
                )
                if prop_acc_vals and base_acc_vals:
                    cmp_res = self._conduct_pairwise_test(
                        proposed_id=proposed_key,
                        proposed_name=prop_summary.method_name,
                        baseline_id=base_key,
                        baseline_name=base_summary.method_name,
                        metric_name="accuracy",
                        proposed_vals=prop_acc_vals,
                        baseline_vals=base_acc_vals,
                    )
                    pairwise_comparisons.append(cmp_res)
                    if cmp_res.p_value is not None and not cmp_res.is_significant:
                        warnings.append(
                            f"Insignificant Advantage: Accuracy gain of '{prop_summary.method_name}' over '{base_summary.method_name}' is not statistically significant (p={cmp_res.p_value:.4f}, d={cmp_res.effect_size_cohens_d})."
                        )

        # 5. Meta-Analysis Evaluation (DerSimonian-Laird)
        meta = metrics_dict.get("meta_analysis", {})
        if isinstance(meta, dict) and meta:
            z_stat = meta.get("z_statistic", 0.0)
            i_sq = meta.get("i_squared_percent", 0.0)
            p_val_z = meta.get("p_value_z", 0.0 if abs(z_stat) >= 1.96 else 0.10)

            meta_sig = abs(z_stat) >= 1.96 or p_val_z < self.SIGNIFICANCE_ALPHA
            if not meta_sig:
                warnings.append(
                    f"Marginal Effect Size: DerSimonian-Laird Z={z_stat:.2f} does not reach p<{self.SIGNIFICANCE_ALPHA} significance."
                )

            hetero_ok = i_sq <= self.MAX_ALLOWED_HETEROGENEITY_I2
            if not hetero_ok:
                critical_issues.append(
                    f"Excessive Heterogeneity: I² = {i_sq:.1f}% exceeds maximum publication threshold ({self.MAX_ALLOWED_HETEROGENEITY_I2:.1f}%)."
                )
            elif i_sq > 50.0:
                warnings.append(
                    f"Moderate Heterogeneity: I² = {i_sq:.1f}% observed across evaluation folds."
                )
        else:
            meta_sig = True
            hetero_ok = True

        # 6. Cherry-Picking Risk Assessment
        cherry_risk = "low"
        if proposed_key and proposed_key in method_summaries:
            prop_acc = method_summaries[proposed_key].metrics.get("accuracy")
            if prop_acc and prop_acc.sample_size >= 2:
                spread = prop_acc.max - prop_acc.min
                if spread > 0.20:
                    cherry_risk = "high"
                    critical_issues.append(
                        f"Excessive Seed Dispersion: Accuracy spread of {spread * 100:.1f}% in proposed method indicates unstable optimization."
                    )
                elif spread > 0.12:
                    cherry_risk = "medium"
                    warnings.append(
                        f"Moderate Seed Dispersion: Accuracy spread of {spread * 100:.1f}% observed in proposed method across seeds."
                    )

        passed = len(critical_issues) == 0

        return StatisticalCritique(
            critique_id="stat_critique_001",
            passed=passed,
            num_seeds=effective_seeds,
            sample_size_sufficient=sample_size_ok,
            variance_bounded=variance_ok,
            meta_analysis_significant=meta_sig,
            heterogeneity_acceptable=hetero_ok,
            cherry_picking_risk=cherry_risk,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations,
            method_summaries=method_summaries,
            pairwise_comparisons=pairwise_comparisons,
        )
