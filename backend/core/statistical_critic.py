"""NovaScientist Statistical Critic Agent.

Performs adversarial statistical evaluation of empirical experiment packages,
multi-seed variance distributions, DerSimonian-Laird random-effects estimators,
cherry-picking risks, and effect size statistical significance.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StatisticalCriticAgent:
    """Adversarial statistical agent ensuring zero fake significance or cherry-picking."""

    MIN_SEEDS = 5
    MAX_ALLOWED_HETEROGENEITY_I2 = 75.0

    def __init__(self) -> None:
        pass

    def evaluate_statistics(self, metrics_dict: Dict[str, Any]) -> StatisticalCritique:
        """Conduct rigorous statistical audit across all evaluation folds."""
        critical_issues: List[str] = []
        warnings: List[str] = []
        recommendations: List[str] = []

        seeds = metrics_dict.get("seeds", [])
        num_seeds = len(seeds)
        sample_size_ok = (num_seeds >= self.MIN_SEEDS)

        if not sample_size_ok:
            critical_issues.append(
                f"Sample Size Violation: Evaluated {num_seeds} seeds. Minimum required for publication is k={self.MIN_SEEDS}."
            )
            recommendations.append(f"Increase random seed count to at least k={self.MIN_SEEDS}.")

        methods = metrics_dict.get("methods", {})
        if len(methods) < 2:
            critical_issues.append("Missing Baselines: Comparative evaluation requires at least 2 distinct methods.")

        variance_ok = True
        for m_id, m_data in methods.items():
            std_acc = m_data.get("std_accuracy", 0.0)
            if std_acc == 0.0:
                variance_ok = False
                warnings.append(f"Zero-Variance Warning: Method '{m_id}' has std=0.0 across seeds (potential seed re-use).")

        meta = metrics_dict.get("meta_analysis", {})
        pooled_es = meta.get("pooled_effect_size", 0.0)
        z_stat = meta.get("z_statistic", 0.0)
        i_sq = meta.get("i_squared_percent", 0.0)

        meta_sig = (abs(z_stat) >= 1.96)  # p < 0.05
        if not meta_sig:
            warnings.append(f"Marginal Effect Size: DerSimonian-Laird Z={z_stat:.2f} does not reach p<0.05 significance.")

        hetero_ok = (i_sq <= self.MAX_ALLOWED_HETEROGENEITY_I2)
        if not hetero_ok:
            critical_issues.append(f"Excessive Heterogeneity: I² = {i_sq:.1f}% exceeds maximum publication threshold (75.0%).")

        # Cherry-picking risk analysis
        cherry_risk = "low"
        prop = methods.get("proposed_mb_qgt", {})
        prop_runs = prop.get("seed_runs", prop.get("seed_results", []))
        prop_seed_accs = [sr.get("final_accuracy", sr.get("accuracy", 0.0)) for sr in prop_runs]
        if prop_seed_accs:
            max_acc = max(prop_seed_accs)
            min_acc = min(prop_seed_accs)
            if (max_acc - min_acc) > 0.15:  # >15% spread
                cherry_risk = "medium"
                warnings.append(f"High Seed Dispersion: Spread of {(max_acc-min_acc)*100:.1f}% observed in proposed method.")

        passed = (len(critical_issues) == 0)

        return StatisticalCritique(
            critique_id="stat_critique_001",
            passed=passed,
            num_seeds=num_seeds,
            sample_size_sufficient=sample_size_ok,
            variance_bounded=variance_ok,
            meta_analysis_significant=meta_sig,
            heterogeneity_acceptable=hetero_ok,
            cherry_picking_risk=cherry_risk,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations,
        )
