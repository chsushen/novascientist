"""Tests for NovaScientist Scientific Reviewer Agent & Bounded Revision Loop (Phase 4).

Verifies 7-pillar peer-review evaluation, structured ReviewFinding generation,
targeted iterative LaTeX refactoring, strict iteration limits (k<=3), and
failure-safe handling of unresolvable critical issues.
"""

import pytest
from dataclasses import asdict

from backend.core.evidence_agent import ClaimRecord, EvidenceBundle, SourceRecord, VerificationStatus
from backend.core.evidence_validator import EvidenceValidationReport, EvidenceValidator, ValidatedClaim
from backend.core.scientific_reviewer import (
    BoundedRevisionLoop,
    ManuscriptRefactorer,
    ReviewFinding,
    RevisionHistory,
    ScientificReviewReport,
    ScientificReviewerAgent,
)
from backend.core.statistical_critic import StatisticalCriticAgent, StatisticalCritique


@pytest.fixture
def mock_clean_metrics():
    return {
        "topic": "Graph Quantization",
        "seeds": [42, 179, 316, 453, 590],
        "hardware_info": {"device": "CPU Multi-Core", "memory_mb": 512},
        "methods": {
            "proposed_mb_qgt": {
                "name": "Proposed MB-QGT",
                "seed_results": [{"accuracy": 0.88 + i * 0.002, "memory_mb": 75.0, "latency_ms": 8.5} for i in range(5)],
            },
            "dense_baseline": {
                "name": "Dense Baseline",
                "seed_results": [{"accuracy": 0.82 + i * 0.002, "memory_mb": 390.0, "latency_ms": 34.0} for i in range(5)],
            },
        },
        "meta_analysis": {
            "z_statistic": 12.5,
            "i_squared_percent": 0.0,
        },
    }


@pytest.fixture
def mock_compliant_latex():
    return (
        r"\documentclass{IEEEtran}" "\n"
        r"\title{Adaptive Quantized Graph Transformers}" "\n"
        r"\begin{document}" "\n"
        r"\section{Introduction}" "\n"
        r"We propose MB-QGT architecture for memory-bounded neural processing." "\n"
        r"\section{Methodology}" "\n"
        r"Comparative baseline evaluation against Dense Baseline under bounded memory." "\n"
        r"\section{Experiments}" "\n"
        r"\begin{table}[h]" "\n"
        r"Accuracy and memory comparison across seeds." "\n"
        r"\end{table}" "\n"
        r"\paragraph{Reproducibility and Determinism}" "\n"
        r"All runs evaluated across k=5 random seeds on CPU hardware." "\n"
        r"\section{Discussion and Limitations}" "\n"
        r"Discussion of boundary conditions and limitation analysis." "\n"
        r"\section{Conclusion}" "\n"
        r"Conclusion of empirical findings." "\n"
        r"\end{document}"
    )


def test_clean_manuscript_immediate_acceptance(mock_compliant_latex, mock_clean_metrics):
    """Compliant manuscript with valid telemetry passes review on iteration 1 with 'accept'."""
    reviewer = ScientificReviewerAgent()
    stat_critic = StatisticalCriticAgent()
    critique = stat_critic.evaluate_statistics(mock_clean_metrics)

    report = reviewer.review(
        latex_text=mock_compliant_latex,
        metrics_dict=mock_clean_metrics,
        stat_critique=critique,
        iteration=1,
    )

    assert report.passed is True
    assert report.overall_verdict == "accept"
    assert report.critical_count == 0
    assert report.major_count == 0
    assert report.iteration == 1
    for cat, score in report.category_scores.items():
        assert score >= 0.88


def test_review_finding_generation_for_unsupported_evidence(mock_compliant_latex, mock_clean_metrics):
    """Unsupported empirical claims generate critical 'evidence' ReviewFinding."""
    reviewer = ScientificReviewerAgent()
    val_report = EvidenceValidationReport(
        total_claims=3,
        supported_count=1,
        weak_count=0,
        unsupported_count=2,
        unsupported_rate=0.667,
        is_publishable=False,
        claims=[],
        flags=["Claim exceeds empirical accuracy delta"],
    )

    report = reviewer.review(
        latex_text=mock_compliant_latex,
        metrics_dict=mock_clean_metrics,
        validation_report=val_report,
    )

    assert report.passed is False
    assert report.critical_count >= 1
    evidence_findings = [f for f in report.findings if f.category == "evidence"]
    assert len(evidence_findings) >= 1
    assert evidence_findings[0].severity == "critical"
    assert "2 unsupported empirical claims" in evidence_findings[0].issue
    assert report.category_scores["evidence"] <= 0.50


def test_review_finding_generation_for_statistical_critical_issues(mock_compliant_latex, mock_clean_metrics):
    """Statistical critic hard failures propagate as critical 'results' ReviewFindings."""
    reviewer = ScientificReviewerAgent()
    critique_failed = StatisticalCritique(
        critique_id="stat_err",
        passed=False,
        num_seeds=1,
        sample_size_sufficient=False,
        variance_bounded=False,
        meta_analysis_significant=False,
        heterogeneity_acceptable=False,
        cherry_picking_risk="high",
        critical_issues=["Insufficient repeated-seed evidence: only 1 seed evaluated."],
    )

    report = reviewer.review(
        latex_text=mock_compliant_latex,
        metrics_dict=mock_clean_metrics,
        stat_critique=critique_failed,
    )

    assert report.passed is False
    assert report.critical_count >= 1
    stat_findings = [f for f in report.findings if f.category == "results"]
    assert any("Insufficient repeated-seed evidence" in f.issue for f in stat_findings)
    assert report.category_scores["results"] <= 0.50


def test_review_finding_generation_for_missing_reproducibility_and_limitations(mock_clean_metrics):
    """Missing reproducibility and limitations sections generate major findings."""
    reviewer = ScientificReviewerAgent()
    sparse_latex = (
        r"\documentclass{IEEEtran}" "\n"
        r"\title{Sparse Test}" "\n"
        r"\begin{document}" "\n"
        r"\section{Introduction} We propose MB-QGT architecture." "\n"
        r"\section{Methodology} Dense baseline comparison." "\n"
        r"\section{Experiments} \begin{table} Metrics \end{table}" "\n"
        r"\section{Conclusion} The end." "\n"
        r"\end{document}"
    )

    report = reviewer.review(
        latex_text=sparse_latex,
        metrics_dict=mock_clean_metrics,
    )

    assert report.passed is False
    assert report.overall_verdict in ["minor_revision", "major_revision"]
    categories = [f.category for f in report.findings]
    assert "reproducibility" in categories
    assert "limitations" in categories


def test_manuscript_refactorer_unhedged_rhetoric_and_section_injection(mock_clean_metrics):
    """ManuscriptRefactorer replaces hyperbolic words and injects missing sections."""
    unhedged_latex = (
        r"\section{Introduction} Our method completely solves memory bottlenecks with flawless accuracy." "\n"
        r"\section{Conclusion} We conclude." "\n"
        r"\end{document}"
    )
    findings = [
        ReviewFinding(
            review_id="rev_001",
            severity="minor",
            category="methodology",
            issue="Unhedged language detected.",
            evidence="contains completely solves",
            recommended_action="Replace unhedged language.",
        ),
        ReviewFinding(
            review_id="rev_002",
            severity="major",
            category="limitations",
            issue="Limitations omitted.",
            evidence="No limitations section",
            recommended_action="Add limitations.",
        ),
        ReviewFinding(
            review_id="rev_003",
            severity="major",
            category="reproducibility",
            issue="Reproducibility omitted.",
            evidence="No seeds documented",
            recommended_action="Add reproducibility.",
        ),
    ]

    revised_latex, actions = ManuscriptRefactorer.apply_targeted_revisions(
        latex_text=unhedged_latex,
        findings=findings,
        metrics_dict=mock_clean_metrics,
    )

    assert "effectively mitigates" in revised_latex
    assert "completely solves" not in revised_latex
    assert "high empirical precision" in revised_latex
    assert "flawless accuracy" not in revised_latex
    assert "Limitations and Boundary Conditions" in revised_latex
    assert "Reproducibility and Determinism" in revised_latex
    assert len(actions) >= 3


def test_bounded_revision_loop_resolves_issues_in_two_iterations(mock_clean_metrics):
    """BoundedRevisionLoop detects issues on iter 1, repairs them, and passes on iter 2."""
    imperfect_latex = (
        r"\documentclass{IEEEtran}" "\n"
        r"\title{Test Title}" "\n"
        r"\begin{document}" "\n"
        r"\section{Introduction} We propose MB-QGT architecture which completely solves dense complexity." "\n"
        r"\section{Methodology} Evaluation against Dense Baseline." "\n"
        r"\section{Experiments} \begin{table} Accuracy \end{table}" "\n"
        r"\section{Conclusion} Concluding remarks." "\n"
        r"\end{document}"
    )
    stat_critic = StatisticalCriticAgent()
    critique = stat_critic.evaluate_statistics(mock_clean_metrics)

    rev_loop = BoundedRevisionLoop()
    revised_latex, final_report, history = rev_loop.run_revision_loop(
        raw_latex=imperfect_latex,
        metrics_dict=mock_clean_metrics,
        stat_critique=critique,
    )

    assert history.total_iterations == 2
    assert final_report.passed is True
    assert final_report.overall_verdict == "accept"
    assert "Limitations and Boundary Conditions" in revised_latex
    assert "Reproducibility and Determinism" in revised_latex
    assert "completely solves" not in revised_latex
    assert "satisfied on Revision 2" in history.stopped_reason


def test_bounded_revision_loop_enforces_max_iterations_on_unresolvable_issues(mock_clean_metrics):
    """BoundedRevisionLoop strictly caps iterations at MAX_REVIEW_ITERATIONS = 3 when critical issues persist."""
    unfixable_critique = StatisticalCritique(
        critique_id="unfixable",
        passed=False,
        num_seeds=1,
        sample_size_sufficient=False,
        variance_bounded=False,
        meta_analysis_significant=False,
        heterogeneity_acceptable=False,
        cherry_picking_risk="high",
        critical_issues=["Insufficient repeated-seed evidence: only 1 seed evaluated."],
    )

    latex_source = r"\title{Test} \section{Introduction} Hello"

    rev_loop = BoundedRevisionLoop()
    revised_latex, final_report, history = rev_loop.run_revision_loop(
        raw_latex=latex_source,
        metrics_dict=mock_clean_metrics,
        stat_critique=unfixable_critique,
    )

    assert history.total_iterations == 3
    assert final_report.passed is False
    assert final_report.overall_verdict == "major_revision" or final_report.overall_verdict == "reject"
    assert "Maximum revision budget reached (3 iterations)" in history.stopped_reason
    assert len(history.revisions) == 3


def test_serialization_of_review_data_structures():
    """Verify to_dict serialization on ReviewFinding, ScientificReviewReport, and RevisionHistory."""
    finding = ReviewFinding(
        review_id="rev_001",
        severity="critical",
        category="evidence",
        issue="Test issue",
        evidence="Test evidence",
        recommended_action="Test recommendation",
    )
    f_dict = finding.to_dict()
    assert f_dict["review_id"] == "rev_001"
    assert f_dict["severity"] == "critical"

    report = ScientificReviewReport(
        overall_verdict="minor_revision",
        passed=False,
        iteration=1,
        findings=[finding],
        category_scores={"evidence": 0.40},
        summary="Test summary",
    )
    r_dict = report.to_dict()
    assert r_dict["overall_verdict"] == "minor_revision"
    assert r_dict["critical_count"] == 1
    assert len(r_dict["findings"]) == 1

    history = RevisionHistory(
        total_iterations=2,
        stopped_reason="Passed",
        revisions=[{"iteration": 1}, {"iteration": 2}],
    )
    h_dict = history.to_dict()
    assert h_dict["total_iterations"] == 2
    assert len(h_dict["revisions"]) == 2
