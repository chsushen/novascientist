"""Unit and integration tests for Asynchronous Job Queue and JobManager."""

import asyncio
import pytest
from pathlib import Path

from backend.jobs.job_manager import JobManager, JobState
from backend.storage.artifact_store import ArtifactStore
from backend.storage.workspace_manager import WorkspaceManager, RunStatus


@pytest.mark.asyncio
async def test_job_submission_and_completion(tmp_path):
    """Verify asynchronous job submission, execution to completion, and artifact registration."""
    store = ArtifactStore(base_dir=tmp_path / "artifacts")
    ws = WorkspaceManager(base_dir=tmp_path / "workspaces", artifact_store=store)
    jm = JobManager(workspace_manager=ws, artifact_store=store, max_concurrent_jobs=2)

    proj = ws.create_project(name="Async Test Project")
    
    job = jm.submit_job(
        project_id=proj.project_id,
        topic="Adaptive Sequence Learning for Text Classification",
        target_length="4_page_conference",
        execution_mode="fast_microbenchmark",
        num_seeds=2,
        num_epochs=2,
    )

    assert job.state in (JobState.QUEUED, JobState.RUNNING)
    assert job.job_id.startswith("job_")

    # Wait for completion (fast microbenchmark mode)
    max_wait = 30.0
    start_t = asyncio.get_event_loop().time()
    while job.state not in (JobState.COMPLETED, JobState.FAILED):
        await asyncio.sleep(0.5)
        if asyncio.get_event_loop().time() - start_t > max_wait:
            break

    assert job.state == JobState.COMPLETED
    assert job.progress_percent == 100.0

    run = ws.get_run(job.run_id)
    assert run.status == RunStatus.COMPLETED
    assert len(run.artifacts) > 0


@pytest.mark.asyncio
async def test_job_cancellation(tmp_path):
    """Verify job cancellation flips state to CANCELLED cleanly."""
    store = ArtifactStore(base_dir=tmp_path / "artifacts")
    ws = WorkspaceManager(base_dir=tmp_path / "workspaces", artifact_store=store)
    jm = JobManager(workspace_manager=ws, artifact_store=store)

    proj = ws.create_project(name="Cancel Test Project")
    job = jm.submit_job(
        project_id=proj.project_id,
        topic="Long-Running Test Topic for Cancellation",
    )

    # Cancel immediately
    cancelled = jm.cancel_job(job.job_id)
    assert cancelled is True
    assert job.state == JobState.CANCELLED
