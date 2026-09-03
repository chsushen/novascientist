"""Adversarial Reviewer Swarm: Statistical Auditor & Rhetoric Linter.

Audits machine learning experiment packages and manuscript prose prior to submission:
1. StatisticalAuditor: Enforces k>=5 paired seeds, variance bounds, and heterogeneity sanity.
2. RhetoricLinter: Replaces and flags unhedged overclaiming tokens with scoped scientific qualifiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


class StatisticalAuditError(Exception):
    """Raised when experimental evaluation violates statistical rigor standards."""
    pass


@dataclass
class ReviewerAuditReport:
    """Comprehensive adversarial reviewer audit findings."""
    passed: bool
    statistical_issues: List[str] = field(default_factory=list)
    rhetoric_modifications: List[Tuple[str, str]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class StatisticalAuditor:
    """Audits raw metrics.json data for statistical power, paired seeds, and variance bounds."""

    MIN_SEEDS_REQUIRED = 5

    @classmethod
    def audit_experiment_package(cls, metrics_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify that all baseline comparisons meet rigorous statistical requirements."""
        issues: List[str] = []
        seeds = metrics_data.get("seeds", [])
        num_seeds = len(seeds)

        if num_seeds < cls.MIN_SEEDS_REQUIRED:
            issues.append(
                f"Statistical Power Violation: Only {num_seeds} evaluation seeds provided. "
                f"A minimum of k={cls.MIN_SEEDS_REQUIRED} deterministic seeds is required for publication."
            )

        methods = metrics_data.get("methods", {})
        if len(methods) < 2:
            issues.append("Comparative Baseline Violation: At least 2 methods (baseline and proposed) must be evaluated.")

        for m_id, m_data in methods.items():
            name = m_data.get("name", m_id)
            std_acc = m_data.get("std_accuracy", 0.0)
            if std_acc == 0.0:
                issues.append(f"Zero-Variance Warning: Method '{name}' exhibits 0.0 variance across seeds, indicating potential non-stochastic duplication.")

        meta = metrics_data.get("meta_analysis", {})
        i_squared = meta.get("i_squared_percent", 0.0)
        if i_squared > 80.0:
            issues.append(f"Excessive Heterogeneity Warning: DerSimonian-Laird I² = {i_squared:.1f}% indicates severe inconsistency across evaluation folds.")

        passed = len(issues) == 0
        return passed, issues


class RhetoricLinter:
    """Scans and transforms unhedged hyperbole and marketing claims into rigorous scientific prose."""

    OVERCLAIM_PATTERNS = [
        (r"\b(completely solves|solves all)\b", "effectively mitigates"),
        (r"\b(universal solution|universal paradigm)\b", "broadly applicable framework"),
        (r"\b(unbeatable SOTA|the definitive SOTA|SOTA)\b", "state-of-the-art empirical performance"),
        (r"\b(flawless performance|flawless accuracy)\b", "high empirical fidelity"),
        (r"\b(revolutionary breakthrough)\b", "substantive architectural refinement"),
        (r"\b(guarantees optimal)\b", "theoretically bounds"),
        (r"\b(vastly superior to all)\b", "empirically outperforming evaluated baselines"),
    ]

    @classmethod
    def lint_and_refactor(cls, manuscript_text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Scan manuscript text, replace overclaims with scoped qualifiers, and log changes."""
        modified_text = manuscript_text
        modifications: List[Tuple[str, str]] = []

        for pattern, replacement in cls.OVERCLAIM_PATTERNS:
            matches = re.findall(pattern, modified_text, flags=re.IGNORECASE)
            if matches:
                for match in set(matches):
                    # Keep original casing if possible
                    modifications.append((match, replacement))
                modified_text = re.sub(pattern, replacement, modified_text, flags=re.IGNORECASE)

        return modified_text, modifications


class AdversarialReviewerSwarm:
    """Coordinates statistical auditing and rhetoric linting."""

    @classmethod
    def review_manuscript(cls, metrics_data: Dict[str, Any], raw_latex: str) -> Tuple[str, ReviewerAuditReport]:
        """Perform full adversarial review on metrics and LaTeX text."""
        stat_passed, stat_issues = StatisticalAuditor.audit_experiment_package(metrics_data)
        cleaned_latex, rhetoric_mods = RhetoricLinter.lint_and_refactor(raw_latex)

        recommendations = []
        if not stat_passed:
            recommendations.append("Increase seed budget to k >= 5 and re-run baseline simulations.")
        if rhetoric_mods:
            recommendations.append(f"Refactored {len(rhetoric_mods)} unhedged assertions into qualified scientific language.")

        report = ReviewerAuditReport(
            passed=stat_passed,
            statistical_issues=stat_issues,
            rhetoric_modifications=rhetoric_mods,
            recommendations=recommendations,
        )
        return cleaned_latex, report


class ReviewerSwarm:
    """Instantiable coordinator for adversarial reviewer audits."""
    def __init__(self, latex_content: str, metrics_dict: Dict[str, Any]) -> None:
        self.latex_content = latex_content
        self.metrics_dict = metrics_dict

    def conduct_audit(self) -> ReviewerAuditReport:
        cleaned, report = AdversarialReviewerSwarm.review_manuscript(self.metrics_dict, self.latex_content)
        return report

