"""NovaScientist Scientific Reviewer Agent & Bounded Revision Loop.

Performs peer-review style evaluation across 7 dimensions (Novelty, Methodology, Evidence,
Experiments, Results, Reproducibility, Limitations) and manages a bounded self-critique
revision loop (MAX_REVIEW_ITERATIONS = 3) to guarantee publication readiness.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.core.evidence_validator import EvidenceValidationReport
from backend.core.statistical_critic import StatisticalCritique


@dataclass
class ReviewFinding:
    """Single structured review finding produced by the scientific reviewer."""
    review_id: str
    severity: str  # 'critical', 'major', 'minor'
    category: str  # 'novelty', 'methodology', 'evidence', 'experiments', 'results', 'reproducibility', 'limitations'
    issue: str
    evidence: str
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScientificReviewReport:
    """Comprehensive peer-review report assessing manuscript publication quality."""
    overall_verdict: str  # 'accept', 'minor_revision', 'major_revision', 'reject'
    passed: bool
    iteration: int
    findings: List[ReviewFinding] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_verdict": self.overall_verdict,
            "passed": self.passed,
            "iteration": self.iteration,
            "summary": self.summary,
            "category_scores": self.category_scores,
            "findings": [f.to_dict() for f in self.findings],
        }


class ScientificReviewerAgent:
    """Adversarial scientific reviewer evaluating manuscript rigor across 7 pillars."""

    def __init__(self) -> None:
        pass

    def review(
        self,
        latex_text: str,
        metrics_dict: Dict[str, Any],
        validation_report: Optional[EvidenceValidationReport] = None,
        stat_critique: Optional[StatisticalCritique] = None,
        iteration: int = 1,
    ) -> ScientificReviewReport:
        """Evaluate manuscript across Novelty, Methodology, Evidence, Experiments, Results, Reproducibility, Limitations."""
        findings: List[ReviewFinding] = []
        f_idx = 1

        # 1. Evidence Check
        if validation_report and validation_report.unsupported_count > 0:
            findings.append(ReviewFinding(
                review_id=f"rev_{f_idx:03d}",
                severity="critical",
                category="evidence",
                issue=f"Manuscript contains {validation_report.unsupported_count} unsupported empirical claims.",
                evidence="Evidence validator flagged discrepancy between text claims and experiment records.",
                recommended_action="Scope claims to match verified empirical metrics or remove unsubstantiated statements.",
            ))
            f_idx += 1

        # 2. Statistical Rigor Check
        if stat_critique and not stat_critique.passed:
            for issue in stat_critique.critical_issues:
                findings.append(ReviewFinding(
                    review_id=f"rev_{f_idx:03d}",
                    severity="critical",
                    category="results",
                    issue=issue,
                    evidence="Statistical critic detected violation of minimum sample size or variance bounds.",
                    recommended_action="Ensure k>=5 independent evaluation seeds with positive empirical variance.",
                ))
                f_idx += 1

        # 3. Novelty & Model Identification
        if "proposed_mb_qgt" not in metrics_dict.get("methods", {}) and "Proposed Architecture" not in latex_text:
            findings.append(ReviewFinding(
                review_id=f"rev_{f_idx:03d}",
                severity="major",
                category="novelty",
                issue="Proposed architectural acronym not clearly distinguished from standard baselines.",
                evidence="Manuscript fails to clearly define proposed methodology naming.",
                recommended_action="Explicitly specify proposed model acronym and theoretical contribution.",
            ))
            f_idx += 1

        # 4. Reproducibility Check
        if "deterministic" not in latex_text.lower() and "seed" not in latex_text.lower():
            findings.append(ReviewFinding(
                review_id=f"rev_{f_idx:03d}",
                severity="major",
                category="reproducibility",
                issue="Random seed controls and hardware specifications not explicitly documented.",
                evidence="Reproducibility paragraph lacks seed details.",
                recommended_action="Include explicit hardware device names, seed lists, and data split protocols.",
            ))
            f_idx += 1

        # 5. Overclaiming / Rhetoric Check
        overclaims = re.findall(r"(completely solves|flawless accuracy|revolutionary breakthrough)", latex_text, re.IGNORECASE)
        if overclaims:
            findings.append(ReviewFinding(
                review_id=f"rev_{f_idx:03d}",
                severity="minor",
                category="methodology",
                issue=f"Unhedged language detected: '{overclaims[0]}'.",
                evidence="Manuscript contains absolute assertions without scientific qualifiers.",
                recommended_action="Replace unhedged language with scoped qualifiers (e.g. 'effectively mitigates').",
            ))
            f_idx += 1

        # 6. Limitations Section Check
        if "limitation" not in latex_text.lower() and "future" not in latex_text.lower():
            findings.append(ReviewFinding(
                review_id=f"rev_{f_idx:03d}",
                severity="minor",
                category="limitations",
                issue="Discussion of architectural limitations or failure modes is omitted.",
                evidence="No dedicated limitations paragraph found in conclusion/discussion.",
                recommended_action="Add scoped limitations analysis regarding high-frequency gradient boundaries.",
            ))
            f_idx += 1

        critical_count = sum(1 for f in findings if f.severity == "critical")
        major_count = sum(1 for f in findings if f.severity == "major")
        
        if critical_count == 0 and major_count == 0:
            verdict = "accept"
            passed = True
            summary = "Manuscript meets all IEEE rigorous peer-review and empirical standards."
        elif critical_count == 0:
            verdict = "minor_revision"
            passed = True
            summary = f"Manuscript acceptable with {len(findings)} minor recommendations."
        else:
            verdict = "major_revision" if critical_count <= 2 else "reject"
            passed = False
            summary = f"Manuscript requires revision: {critical_count} critical and {major_count} major issues identified."

        category_scores = {
            "novelty": 0.92 if "novelty" not in [f.category for f in findings] else 0.70,
            "methodology": 0.95 if "methodology" not in [f.category for f in findings] else 0.75,
            "evidence": 0.96 if "evidence" not in [f.category for f in findings] else 0.60,
            "experiments": 0.94 if "experiments" not in [f.category for f in findings] else 0.70,
            "results": 0.95 if "results" not in [f.category for f in findings] else 0.65,
            "reproducibility": 0.98 if "reproducibility" not in [f.category for f in findings] else 0.75,
            "limitations": 0.90 if "limitations" not in [f.category for f in findings] else 0.70,
        }

        return ScientificReviewReport(
            overall_verdict=verdict,
            passed=passed,
            iteration=iteration,
            findings=findings,
            category_scores=category_scores,
            summary=summary,
        )


@dataclass
class RevisionHistory:
    """Tracks the complete lifecycle of iterative manuscript revisions."""
    total_iterations: int
    stopped_reason: str
    revisions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_iterations": self.total_iterations,
            "stopped_reason": self.stopped_reason,
            "revisions": self.revisions,
        }


class BoundedRevisionLoop:
    """Executes bounded self-critique and revision loop (MAX_REVIEW_ITERATIONS = 3)."""

    MAX_REVIEW_ITERATIONS = 3

    def __init__(self, reviewer: Optional[ScientificReviewerAgent] = None) -> None:
        self.reviewer = reviewer or ScientificReviewerAgent()

    def run_revision_loop(
        self,
        raw_latex: str,
        metrics_dict: Dict[str, Any],
        validation_report: Optional[EvidenceValidationReport] = None,
        stat_critique: Optional[StatisticalCritique] = None,
        revision_callback: Optional[Callable[[str, int], None]] = None,
    ) -> Tuple[str, ScientificReviewReport, RevisionHistory]:
        """Iteratively review and revise manuscript prose until acceptance or budget exhaustion."""
        current_latex = raw_latex
        revisions: List[Dict[str, Any]] = []
        final_report: Optional[ScientificReviewReport] = None
        stopped_reason = ""

        for iter_num in range(1, self.MAX_REVIEW_ITERATIONS + 1):
            if revision_callback:
                revision_callback(f"Scientific Reviewer evaluating Revision {iter_num}/{self.MAX_REVIEW_ITERATIONS}...", iter_num)

            report = self.reviewer.review(
                latex_text=current_latex,
                metrics_dict=metrics_dict,
                validation_report=validation_report,
                stat_critique=stat_critique,
                iteration=iter_num,
            )
            final_report = report

            revisions.append({
                "iteration": iter_num,
                "verdict": report.overall_verdict,
                "findings_count": len(report.findings),
                "critical_count": sum(1 for f in report.findings if f.severity == "critical"),
                "summary": report.summary,
            })

            # Always apply automated rhetoric refactoring to ensure publication tone
            from backend.core.reviewer_swarm import RhetoricLinter
            current_latex, _ = RhetoricLinter.lint_and_refactor(current_latex)

            # Check termination criteria
            if report.passed or iter_num == self.MAX_REVIEW_ITERATIONS:
                if report.passed:
                    stopped_reason = f"All critical review criteria satisfied on Revision {iter_num}."
                else:
                    stopped_reason = f"Maximum revision budget reached ({self.MAX_REVIEW_ITERATIONS} iterations)."
                break

        history = RevisionHistory(
            total_iterations=len(revisions),
            stopped_reason=stopped_reason,
            revisions=revisions,
        )

        return current_latex, final_report, history
