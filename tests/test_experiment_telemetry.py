"""
Tests for Phase 5: Real Experiment Telemetry & Hardware Sandbox.

Validates that:
1. Seed execution records contain genuine monotonic wall-clock runtimes and ISO timestamps.
2. Failure states (exceptions, OOM, divergence) capture exact error messages and 'failed' status.
3. Checkpoint paths are assigned strictly to completed runs of proposed architectures.
4. ExperimentRecord and ExperimentSpec schemas and serialization function correctly.
5. Hardware device metadata is faithfully captured.
"""

import pytest
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

from backend.core.experiment_agent import ExperimentAgent, ExperimentRecord, ExperimentSpec
from backend.core.methodology_agent import MethodologySpec
from backend.core.real_trainer import RealPyTorchTrainer, DenseFP32Baseline, ProposedMBQGT
from backend.core.surrogate_engine import SeedResult, SurrogateBenchmarkEngine


def test_seed_result_dataclass_fields():
    """Verify SeedResult includes all required telemetry fields with sensible defaults."""
    res = SeedResult(
        seed=42,
        train_loss_history=[1.2, 0.8, 0.4],
        val_accuracy_history=[0.5, 0.7, 0.85],
        final_accuracy=0.85,
        peak_memory_mb=128.5,
        inference_latency_ms=12.4,
        throughput_samples_sec=5160.0,
        compression_ratio=4.0,
        gradient_variance=0.012,
        runtime_sec=0.1425,
        start_time="2026-09-04T12:00:00+00:00",
        end_time="2026-09-04T12:00:00.142500+00:00",
        status="completed",
        error=None,
    )

    assert res.seed == 42
    assert res.final_accuracy == 0.85
    assert res.runtime_sec == 0.1425
    assert res.status == "completed"
    assert res.error is None
    assert res.start_time is not None
    assert res.end_time is not None


def test_experiment_record_dataclass_and_serialization():
    """Verify ExperimentRecord fields and dictionary serialization."""
    rec = ExperimentRecord(
        experiment_id="exp_001",
        method_id="proposed_mb_qgt",
        method_name="Proposed MB-QGT",
        seed=42,
        dataset="METR-LA Sensor Benchmark",
        accuracy=88.5,
        memory_mb=72.5,
        latency_ms=8.35,
        throughput=7664.7,
        compression_ratio=5.9,
        runtime_sec=1.2345,
        checkpoint_path="/path/to/checkpoint.pt",
        status="completed",
        error=None,
        start_time="2026-09-04T12:00:00+00:00",
        end_time="2026-09-04T12:00:01.234500+00:00",
        hardware_device="Apple Silicon MPS",
    )

    d = rec.to_dict()
    assert d["experiment_id"] == "exp_001"
    assert d["method_id"] == "proposed_mb_qgt"
    assert d["accuracy"] == 88.5
    assert d["runtime_sec"] == 1.2345
    assert d["checkpoint_path"] == "/path/to/checkpoint.pt"
    assert d["status"] == "completed"
    assert d["hardware_device"] == "Apple Silicon MPS"
    assert "timestamp" in d


def test_surrogate_engine_telemetry_timestamps_and_runtime():
    """Verify surrogate benchmark engine captures true monotonic runtime and ISO timestamps."""
    engine = SurrogateBenchmarkEngine(
        topic="Adaptive Quantization for Graph Neural Networks",
        num_seeds=3,
    )
    package = engine.run_experiments()

    assert package is not None
    assert len(package.seeds) == 3

    for m_id, m_metrics in package.methods.items():
        assert len(m_metrics.seed_runs) == 3
        for sr in m_metrics.seed_runs:
            assert isinstance(sr, SeedResult)
            assert sr.runtime_sec >= 0.0
            assert sr.status == "completed"
            assert sr.error is None
            assert sr.start_time is not None
            assert sr.end_time is not None
            # Validate ISO timestamp parsing
            dt_start = datetime.fromisoformat(sr.start_time)
            dt_end = datetime.fromisoformat(sr.end_time)
            assert dt_end >= dt_start


def test_real_trainer_telemetry_and_checkpoint_handling(tmp_path):
    """Verify RealPyTorchTrainer captures telemetry, ISO timestamps, and saves checkpoints safely."""
    trainer = RealPyTorchTrainer(
        topic="Sparse Graph Transformers",
        num_seeds=2,
        num_epochs=2,
        batch_size=16,
        experiments_dir=str(tmp_path / "experiments"),
    )

    # Train a single seed for baseline
    base_res = trainer.train_seed(DenseFP32Baseline, seed=42, is_proposed=False)
    assert base_res.status == "completed"
    assert base_res.runtime_sec >= 0.0
    assert base_res.error is None
    assert base_res.start_time is not None
    assert base_res.end_time is not None

    # Train a single seed for proposed model
    prop_res = trainer.train_seed(ProposedMBQGT, seed=42, is_proposed=True)
    assert prop_res.status == "completed"
    assert prop_res.runtime_sec >= 0.0
    assert prop_res.error is None

    # Check that checkpoint directory was created and checkpoint saved for proposed seed 0
    ckpt_file = tmp_path / "experiments" / "checkpoints" / "proposed_mb_qgt_weights.pt"
    assert ckpt_file.exists()


def test_real_trainer_failure_capture():
    """Verify RealPyTorchTrainer catches exceptions, marks status='failed', and records error."""
    trainer = RealPyTorchTrainer(
        topic="Fault Injection Test",
        num_seeds=1,
        num_epochs=1,
        experiments_dir="./dist/test_experiments",
    )

    class FailingModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("CUDA Out of Memory: simulated memory allocation failure")

    res = trainer.train_seed(FailingModel, seed=999, is_proposed=True)
    assert res.status == "failed"
    assert res.error is not None
    assert "CUDA Out of Memory" in res.error or "RuntimeError" in res.error
    assert res.final_accuracy == 0.0
    assert res.peak_memory_mb == 0.0
    assert res.runtime_sec >= 0.0


def test_experiment_agent_extract_records_proposed_checkpoint_safety():
    """Verify that extract_experiment_records attaches checkpoint_path ONLY to completed proposed runs."""
    agent = ExperimentAgent()

    metrics_dict = {
        "seeds": [42, 179],
        "device": "NVIDIA A100-SXM4-80GB",
        "methods": {
            "dense_baseline": {
                "name": "Dense Baseline",
                "mean_compression_ratio": 1.0,
                "seed_runs": [
                    {
                        "seed": 42,
                        "accuracy": 0.81,
                        "memory_mb": 350.0,
                        "latency_ms": 32.0,
                        "throughput": 2000.0,
                        "runtime_sec": 0.5,
                        "status": "completed",
                        "start_time": "2026-09-04T12:00:00+00:00",
                        "end_time": "2026-09-04T12:00:00.500000+00:00",
                    }
                ],
            },
            "proposed_mb_qgt": {
                "name": "Proposed MB-QGT",
                "mean_compression_ratio": 5.9,
                "seed_runs": [
                    {
                        "seed": 42,
                        "accuracy": 0.88,
                        "memory_mb": 72.0,
                        "latency_ms": 8.0,
                        "throughput": 8000.0,
                        "runtime_sec": 0.45,
                        "status": "completed",
                        "start_time": "2026-09-04T12:00:01+00:00",
                        "end_time": "2026-09-04T12:00:01.450000+00:00",
                    },
                    {
                        "seed": 179,
                        "accuracy": 0.0,
                        "memory_mb": 0.0,
                        "latency_ms": 0.0,
                        "throughput": 0.0,
                        "runtime_sec": 0.1,
                        "status": "failed",
                        "error": "DeviceOOMException: Memory threshold exceeded",
                        "start_time": "2026-09-04T12:00:02+00:00",
                        "end_time": "2026-09-04T12:00:02.100000+00:00",
                    },
                ],
            },
        },
    }

    records = agent.extract_experiment_records(
        metrics_dict,
        dataset_name="Test Dataset",
        checkpoint_path="/checkpoints/proposed_model.pt",
    )

    assert len(records) == 3

    # Baseline run -> checkpoint_path must be None
    dense_rec = [r for r in records if r.method_id == "dense_baseline"][0]
    assert dense_rec.checkpoint_path is None
    assert dense_rec.status == "completed"
    assert dense_rec.hardware_device == "NVIDIA A100-SXM4-80GB"

    # Completed proposed run -> checkpoint_path populated
    prop_completed = [r for r in records if r.method_id == "proposed_mb_qgt" and r.status == "completed"][0]
    assert prop_completed.checkpoint_path == "/checkpoints/proposed_model.pt"
    assert prop_completed.accuracy == 88.0
    assert prop_completed.runtime_sec == 0.45

    # Failed proposed run -> checkpoint_path must be None and error preserved
    prop_failed = [r for r in records if r.method_id == "proposed_mb_qgt" and r.status == "failed"][0]
    assert prop_failed.checkpoint_path is None
    assert prop_failed.status == "failed"
    assert "DeviceOOMException" in prop_failed.error


def test_experiment_spec_creation():
    """Verify ExperimentAgent creates complete ExperimentSpec from MethodologySpec."""
    agent = ExperimentAgent()
    methodology = MethodologySpec(
        methodology_id="meth_test_001",
        topic_title="Adaptive Quantization for Graph Neural Networks",
        domain="machine_learning",
        model_acronym="MB-QGT",
        model_full_name="Memory-Bounded Quantized Transformer",
        hardware_constraints={"max_memory_mb": 256},
    )

    spec = agent.create_spec(
        methodology=methodology,
        dataset_name="METR-LA",
        sample_count=2000,
        seeds=[42, 179, 316],
        num_epochs=20,
        batch_size=32,
    )

    assert isinstance(spec, ExperimentSpec)
    assert spec.methodology_id == "meth_test_001"
    assert spec.dataset_name == "METR-LA"
    assert spec.seeds == [42, 179, 316]
    assert len(spec.methods_to_evaluate) == 4
    assert len(spec.ablation_configurations) == 4

    spec_dict = spec.to_dict()
    assert spec_dict["num_epochs"] == 20
    assert spec_dict["batch_size"] == 32
