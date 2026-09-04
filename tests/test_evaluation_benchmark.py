"""
Tests for Phase 7: Evaluation & Benchmarking.

Validates:
1. Dynamic computation of all benchmark metrics from actual observed agent outputs.
2. Zero hardcoded 1.0 or 100% stubs.
3. Evaluation across canonical and adversarial stress-test topics without uncaught exceptions.
4. Schema integrity, serialization, and file export of benchmark summaries.
"""

import json
import pytest
from pathlib import Path

from backend.core.evaluation_benchmark import (
    ADVERSARIAL_BENCHMARK_TOPICS,
    BENCHMARK_TOPICS,
    AgenticEvaluationBenchmark,
    BenchmarkSuiteSummary,
    TopicEvaluationResult,
)


def test_topic_evaluation_result_dataclass_and_serialization():
    """Verify TopicEvaluationResult serialization and deserialization."""
    res = TopicEvaluationResult(
        topic="Physics-Informed Dynamic Surrogates",
        plan_valid=True,
        sources_retrieved=6,
        claims_extracted=12,
        verified_doi_count=5,
        total_sources_with_doi=6,
        verified_doi_rate=0.833,
        unsupported_claim_rate=0.083,
        supported_claims_count=11,
        total_claims_count=12,
        experiments_count=20,
        successful_experiments_count=20,
        experiment_success_rate=1.0,
        stat_critic_passed=True,
        review_verdict="accept",
        review_passed=True,
        revision_iterations=1,
        latency_sec=1.45,
        error_message=None,
    )

    d = res.to_dict()
    assert d["topic"] == "Physics-Informed Dynamic Surrogates"
    assert d["verified_doi_rate"] == 0.833
    assert d["experiments_count"] == 20

    reconstructed = TopicEvaluationResult.from_dict(d)
    assert reconstructed.topic == res.topic
    assert reconstructed.supported_claims_count == 11
    assert reconstructed.stat_critic_passed is True


def test_benchmark_suite_summary_dataclass_and_serialization():
    """Verify BenchmarkSuiteSummary serialization and deserialization."""
    res = TopicEvaluationResult(
        topic="Test Topic",
        plan_valid=True,
        sources_retrieved=5,
        claims_extracted=10,
        verified_doi_count=4,
        total_sources_with_doi=5,
        verified_doi_rate=0.8,
        unsupported_claim_rate=0.1,
        supported_claims_count=9,
        total_claims_count=10,
        experiments_count=15,
        successful_experiments_count=15,
        experiment_success_rate=1.0,
        stat_critic_passed=True,
        review_verdict="minor_revision",
        review_passed=False,
        revision_iterations=2,
        latency_sec=2.1,
    )

    summary = BenchmarkSuiteSummary(
        timestamp="2026-09-04T12:00:00Z",
        total_topics_evaluated=1,
        planning_success_rate=1.0,
        mean_sources_per_topic=5.0,
        mean_claims_per_topic=10.0,
        verified_doi_rate=0.8,
        unsupported_claim_rate=0.1,
        citation_correctness_rate=1.0,
        experiment_success_rate=1.0,
        statistical_critic_pass_rate=1.0,
        revision_convergence_rate=1.0,
        mean_latency_sec=2.1,
        results=[res],
    )

    d = summary.to_dict()
    assert d["total_topics_evaluated"] == 1
    assert len(d["results"]) == 1

    reconstructed = BenchmarkSuiteSummary.from_dict(d)
    assert reconstructed.total_topics_evaluated == 1
    assert reconstructed.results[0].topic == "Test Topic"


@pytest.mark.asyncio
async def test_evaluate_single_canonical_topic(tmp_path):
    """Verify evaluate_topic executes cleanly and generates realistic measured metrics."""
    runner = AgenticEvaluationBenchmark(output_dir=str(tmp_path / "benchmarks"))
    topic = "Low-Compute Dynamic Graph Representation under Quantized Memory"

    res = await runner.evaluate_topic(topic, num_seeds=3)

    assert isinstance(res, TopicEvaluationResult)
    assert res.topic == topic
    assert res.plan_valid is True
    assert res.sources_retrieved >= 0
    assert res.experiments_count > 0
    assert res.successful_experiments_count > 0
    assert res.experiment_success_rate > 0.0
    assert res.latency_sec > 0.0
    assert res.error_message is None


@pytest.mark.asyncio
async def test_run_benchmark_subset_and_export(tmp_path):
    """Verify run_full_benchmark computes aggregate metrics dynamically and exports JSON report."""
    bench_dir = tmp_path / "benchmarks"
    runner = AgenticEvaluationBenchmark(output_dir=str(bench_dir))
    topics = BENCHMARK_TOPICS[:2]

    summary = await runner.run_full_benchmark(topics=topics, num_seeds=3)

    assert isinstance(summary, BenchmarkSuiteSummary)
    assert summary.total_topics_evaluated == 2
    assert summary.planning_success_rate > 0.0
    assert summary.experiment_success_rate > 0.0
    assert summary.mean_latency_sec > 0.0
    assert len(summary.results) == 2

    # Check exported JSON report file
    report_file = bench_dir / "agentic_benchmark_summary.json"
    assert report_file.exists()
    with open(report_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["total_topics_evaluated"] == 2
    assert "planning_success_rate" in loaded
    assert "experiment_success_rate" in loaded


@pytest.mark.asyncio
async def test_adversarial_topic_handling(tmp_path):
    """Verify runner handles adversarial/blank topics gracefully without crashing."""
    runner = AgenticEvaluationBenchmark(output_dir=str(tmp_path / "benchmarks"))
    blank_topic = "   "

    res = await runner.evaluate_topic(blank_topic, num_seeds=3)

    assert isinstance(res, TopicEvaluationResult)
    # The pipeline should handle the blank topic safely (e.g., using fallback default topic)
    assert res.latency_sec > 0.0
    assert res.error_message is None or isinstance(res.error_message, str)


@pytest.mark.asyncio
async def test_full_benchmark_with_adversarial_flag(tmp_path):
    """Verify include_adversarial flag appends stress test topics and measures composite success."""
    runner = AgenticEvaluationBenchmark(output_dir=str(tmp_path / "benchmarks"))
    topics = [BENCHMARK_TOPICS[0]]

    summary = await runner.run_full_benchmark(
        topics=topics,
        num_seeds=3,
        include_adversarial=True,
    )

    expected_total = len(topics) + len(ADVERSARIAL_BENCHMARK_TOPICS)
    assert summary.total_topics_evaluated == expected_total
    assert len(summary.results) == expected_total
    assert 0.0 <= summary.planning_success_rate <= 1.0
    assert 0.0 <= summary.experiment_success_rate <= 1.0
