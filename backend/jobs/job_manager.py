"""NovaScientist Asynchronous Research Job Queue & Executor.

Executes autonomous research jobs in the background decoupled from client browser sessions,
with progress tracking, checkpointing, cancellation, retries, and structured failure reporting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from backend.config import config
from backend.core.orchestrator import (
    AuthorProfile,
    ExecutionMode,
    NovaScientistOrchestrator,
    TargetPaperLength,
)
from backend.storage.artifact_store import ArtifactStore, ArtifactType
from backend.storage.workspace_manager import RunStatus, WorkspaceManager

logger = logging.getLogger("novascientist.jobs")


class JobState(str, Enum):
    """Execution state machine for research jobs."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CHECKPOINTING = "CHECKPOINTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class StructuredFailure:
    """Production-grade actionable failure report."""

    run_id: str
    stage: str
    reason: str
    recoverability: str  # "RETRYABLE", "FATAL_CONFIG", "USER_INPUT_REQUIRED"
    suggested_action: str
    raw_error: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobInfo:
    """Live state descriptor for an asynchronous research job."""

    job_id: str
    run_id: str
    project_id: str
    topic: str
    state: JobState = JobState.QUEUED
    progress_percent: float = 0.0
    current_stage: str = "Queued in runner"
    stage_message: str = "Awaiting execution slot"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error: StructuredFailure | None = None
    retries_attempted: int = 0
    max_retries: int = 2
    is_cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = (
            self.state.value if isinstance(self.state, JobState) else str(self.state)
        )
        if self.error:
            d["error"] = self.error.to_dict()
        return d


class JobManager:
    """Manages concurrent job execution, cancellation, and progress."""

    def __init__(
        self,
        workspace_manager: WorkspaceManager | None = None,
        artifact_store: ArtifactStore | None = None,
        max_concurrent_jobs: int | None = None,
    ) -> None:
        self.workspace_mgr = workspace_manager or WorkspaceManager()
        self.artifact_store = artifact_store or ArtifactStore()
        self.max_concurrent_jobs = max_concurrent_jobs or config.max_concurrent_jobs
        self._jobs: dict[str, JobInfo] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._semaphore = asyncio.Semaphore(self.max_concurrent_jobs)

    def submit_job(
        self,
        project_id: str,
        topic: str,
        author: AuthorProfile | None = None,
        target_length: str = "8_12_pages_journal",
        execution_mode: str = "fast_microbenchmark",
        num_seeds: int = 5,
        num_epochs: int = 5,
        run_id: str | None = None,
    ) -> JobInfo:
        """Submit a new research run to the background queue."""
        if run_id:
            run = self.workspace_mgr.get_run(run_id)
        else:
            run = self.workspace_mgr.create_run(project_id, topic)

        job_id = f"job_{run.run_id}"
        job_info = JobInfo(
            job_id=job_id,
            run_id=run.run_id,
            project_id=project_id,
            topic=topic,
            state=JobState.QUEUED,
        )
        self._jobs[job_id] = job_info

        # Launch non-blocking background async task
        task = asyncio.create_task(
            self._execute_job_wrapper(
                job_info=job_info,
                author=author or AuthorProfile(),
                target_length=target_length,
                execution_mode=execution_mode,
                num_seeds=num_seeds,
                num_epochs=num_epochs,
            )
        )
        self._tasks[job_id] = task
        return job_info

    def get_job(self, job_id: str) -> JobInfo:
        """Retrieve live job status."""
        if job_id not in self._jobs:
            raise KeyError(f"Job ID '{job_id}' not found.")
        return self._jobs[job_id]

    def get_job_by_run_id(self, run_id: str) -> JobInfo | None:
        """Look up job by research run ID."""
        for j in self._jobs.values():
            if j.run_id == run_id:
                return j
        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or queued job safely."""
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        job.is_cancelled = True
        job.state = JobState.CANCELLED
        job.stage_message = "Execution cancelled by user."
        job.completed_at = time.time()

        if job_id in self._tasks and not self._tasks[job_id].done():
            self._tasks[job_id].cancel()

        self.workspace_mgr.update_run(
            job.run_id,
            status=RunStatus.CANCELLED,
            metadata={"cancelled_at": time.time()},
        )
        return True

    def list_jobs(self) -> list[JobInfo]:
        """Return all managed jobs."""
        return list(self._jobs.values())

    async def _execute_job_wrapper(
        self,
        job_info: JobInfo,
        author: AuthorProfile,
        target_length: str,
        execution_mode: str,
        num_seeds: int,
        num_epochs: int,
    ) -> None:
        """Internal worker executing orchestrator inside concurrency semaphore."""
        async with self._semaphore:
            if job_info.is_cancelled:
                return

            job_info.state = JobState.RUNNING
            job_info.started_at = time.time()
            self.workspace_mgr.update_run(job_info.run_id, status=RunStatus.RUNNING)

            run_dist = (
                self.workspace_mgr.base_dir
                / job_info.project_id
                / job_info.run_id
                / "dist"
            )
            run_dist.mkdir(parents=True, exist_ok=True)

            orchestrator = NovaScientistOrchestrator(output_dir=str(run_dist))

            def progress_hook(msg: str, pct: float) -> None:
                if job_info.is_cancelled:
                    raise asyncio.CancelledError("Job was cancelled during execution.")
                job_info.stage_message = msg
                job_info.progress_percent = round(pct * 100.0, 1)
                job_info.current_stage = msg.split(":")[0] if ":" in msg else msg[:40]

            try:
                # Checkpointing start of synthesis
                self.workspace_mgr.save_checkpoint(
                    job_info.run_id,
                    stage="initializing",
                    state_data={
                        "topic": job_info.topic,
                        "started_at": job_info.started_at,
                    },
                )

                result = await orchestrator.execute(
                    topic=job_info.topic,
                    author=author,
                    target_length=TargetPaperLength(target_length)
                    if target_length in [e.value for e in TargetPaperLength]
                    else target_length,
                    execution_mode=ExecutionMode(execution_mode)
                    if execution_mode in [e.value for e in ExecutionMode]
                    else execution_mode,
                    num_seeds=num_seeds,
                    num_epochs=num_epochs,
                    progress_callback=progress_hook,
                )

                if job_info.is_cancelled:
                    return

                # Ingest generated outputs into immutable ArtifactStore
                artifact_ids: list[str] = []

                # 1. LaTeX manuscript
                if result.latex_content:
                    art_tex = self.artifact_store.store_text(
                        project_id=job_info.project_id,
                        run_id=job_info.run_id,
                        artifact_type=ArtifactType.MANUSCRIPT,
                        filename="manuscript.tex",
                        content=result.latex_content,
                    )
                    artifact_ids.append(art_tex.artifact_id)

                # 2. Metrics & Telemetry
                if result.metrics:
                    art_met = self.artifact_store.store_text(
                        project_id=job_info.project_id,
                        run_id=job_info.run_id,
                        artifact_type=ArtifactType.RESULT,
                        filename="metrics.json",
                        content=json.dumps(result.metrics, indent=2),
                    )
                    artifact_ids.append(art_met.artifact_id)

                # 3. Scientific Research Contract
                if result.research_contract:
                    art_contract = self.artifact_store.store_text(
                        project_id=job_info.project_id,
                        run_id=job_info.run_id,
                        artifact_type=ArtifactType.SCIENTIFIC_CONTRACT,
                        filename="scientific_contract.json",
                        content=json.dumps(result.research_contract, indent=2),
                    )
                    artifact_ids.append(art_contract.artifact_id)

                # 4. Provenance Graph
                if result.provenance_graph:
                    art_prov = self.artifact_store.store_text(
                        project_id=job_info.project_id,
                        run_id=job_info.run_id,
                        artifact_type=ArtifactType.PROVENANCE_GRAPH,
                        filename="provenance_graph.json",
                        content=json.dumps(result.provenance_graph, indent=2),
                    )
                    artifact_ids.append(art_prov.artifact_id)

                # 5. Compiled PDF
                pdf_art_id = None
                if result.pdf_path and Path(result.pdf_path).exists():
                    art_pdf = self.artifact_store.store_file(
                        project_id=job_info.project_id,
                        run_id=job_info.run_id,
                        artifact_type=ArtifactType.PDF,
                        source_file=result.pdf_path,
                        target_filename="publication.pdf",
                    )
                    pdf_art_id = art_pdf.artifact_id
                    artifact_ids.append(pdf_art_id)

                # 6. Overleaf ZIP package
                zip_art_id = None
                if result.zip_path and Path(result.zip_path).exists():
                    art_zip = self.artifact_store.store_file(
                        project_id=job_info.project_id,
                        run_id=job_info.run_id,
                        artifact_type=ArtifactType.OVERLEAF_PACKAGE,
                        source_file=result.zip_path,
                        target_filename="overleaf_package.zip",
                    )
                    zip_art_id = art_zip.artifact_id
                    artifact_ids.append(zip_art_id)

                # Update workspace run to COMPLETED
                self.workspace_mgr.update_run(
                    job_info.run_id,
                    status=RunStatus.COMPLETED,
                    contract_id=result.research_contract.get("contract_id")
                    if result.research_contract
                    else None,
                    contract_data=result.research_contract,
                    evidence_data=result.evidence,
                    experiment_data={"num_seeds": num_seeds, "num_epochs": num_epochs},
                    results_data=result.metrics,
                    statistics_data=result.stat_critique,
                    figures_data=result.figures,
                    claims_data=result.evidence.get("claims", [])
                    if result.evidence
                    else [],
                    provenance_data=result.provenance_graph,
                    manuscript_latex=result.latex_content,
                    pdf_artifact_id=pdf_art_id,
                    zip_artifact_id=zip_art_id,
                    page_count=result.page_count,
                    artifacts=artifact_ids,
                )

                job_info.state = JobState.COMPLETED
                job_info.progress_percent = 100.0
                job_info.stage_message = (
                    "Research pipeline successfully finished and verified."
                )
                job_info.completed_at = time.time()

            except asyncio.CancelledError:
                job_info.state = JobState.CANCELLED
                job_info.stage_message = "Research run was cancelled."
                job_info.completed_at = time.time()
                self.workspace_mgr.update_run(
                    job_info.run_id, status=RunStatus.CANCELLED
                )

            except Exception as exc:
                logger.exception(f"Job {job_info.job_id} failed: {exc}")
                failure = StructuredFailure(
                    run_id=job_info.run_id,
                    stage=job_info.current_stage,
                    reason=str(exc),
                    recoverability="RETRYABLE"
                    if "timeout" in str(exc).lower() or "network" in str(exc).lower()
                    else "FATAL_CONFIG",
                    suggested_action="Check dataset/topic constraints or re-run with fast_microbenchmark mode.",
                    raw_error=str(exc),
                )
                job_info.state = JobState.FAILED
                job_info.error = failure
                job_info.stage_message = (
                    f"Run failed at stage '{job_info.current_stage}': {exc}"
                )
                job_info.completed_at = time.time()

                self.workspace_mgr.update_run(
                    job_info.run_id,
                    status=RunStatus.FAILED,
                    error=failure.to_dict(),
                )
