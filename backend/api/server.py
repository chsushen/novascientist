"""NovaScientist Production REST API Server.

Provides a clean, decoupled service layer for creating projects, submitting asynchronous runs,
streaming execution status, retrieving verified evidence, and downloading immutable artifacts.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.config import config
from backend.core.orchestrator import AuthorProfile
from backend.jobs.job_manager import JobManager, JobState
from backend.reproducibility.manifest_generator import ReproducibilityGenerator
from backend.storage.artifact_store import ArtifactStore
from backend.storage.workspace_manager import WorkspaceManager, RunStatus

logger = logging.getLogger("novascientist.api")


# Pydantic Schemas
class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    user_id: str = Field(default="default_user")


class CreateRunRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=500)
    author_name: str = Field(default="Anonymous Author(s)")
    author_affiliation: str = Field(default="Affiliation Withheld for Double-Blind Review")
    author_email: str = Field(default="anonymous@conference-review.org")
    target_length: str = Field(default="8_12_pages_journal")
    execution_mode: str = Field(default="fast_microbenchmark")
    num_seeds: int = Field(default=5, ge=1, le=10)
    num_epochs: int = Field(default=5, ge=1, le=50)


def create_app(
    workspace_manager: Optional[WorkspaceManager] = None,
    artifact_store: Optional[ArtifactStore] = None,
    job_manager: Optional[JobManager] = None,
) -> FastAPI:
    """Application factory for NovaScientist API Server."""
    
    app = FastAPI(
        title="NovaScientist Production Research API",
        description="Evidence-First Autonomous Research Infrastructure for Scientific Investigation",
        version=config.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS Configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Subsystem Singletons
    store = artifact_store or ArtifactStore()
    ws_mgr = workspace_manager or WorkspaceManager(artifact_store=store)
    job_mgr = job_manager or JobManager(workspace_manager=ws_mgr, artifact_store=store)

    @app.get("/health", tags=["Diagnostics"])
    async def health_check() -> Dict[str, Any]:
        """System liveness and readiness probe."""
        return {
            "status": "healthy",
            "app_version": config.app_version,
            "environment": config.environment,
            "timestamp": time.time(),
        }

    @app.get("/diagnostics", tags=["Diagnostics"])
    async def diagnostics() -> Dict[str, Any]:
        """Comprehensive operational telemetry and environment report."""
        git_sha = ReproducibilityGenerator.get_git_sha()
        jobs = job_mgr.list_jobs()
        return {
            "application": "NovaScientist v2.3",
            "git_sha": git_sha,
            "version": config.app_version,
            "active_jobs_count": len([j for j in jobs if j.state == JobState.RUNNING]),
            "queued_jobs_count": len([j for j in jobs if j.state == JobState.QUEUED]),
            "total_projects": len(ws_mgr.list_projects()),
            "total_runs": len(ws_mgr.list_runs()),
            "data_directory": str(config.data_dir),
            "demo_mode": config.demo_mode,
        }

    # ==================== PROJECTS ====================

    @app.post("/api/v1/projects", status_code=status.HTTP_201_CREATED, tags=["Projects"])
    async def create_project(req: CreateProjectRequest) -> Dict[str, Any]:
        """Create a new research workspace project."""
        proj = ws_mgr.create_project(
            name=req.name,
            description=req.description,
            user_id=req.user_id,
        )
        return proj.to_dict()

    @app.get("/api/v1/projects", tags=["Projects"])
    async def list_projects(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all projects."""
        projects = ws_mgr.list_projects(user_id=user_id)
        return [p.to_dict() for p in projects]

    @app.get("/api/v1/projects/{project_id}", tags=["Projects"])
    async def get_project(project_id: str) -> Dict[str, Any]:
        """Retrieve project metadata and constituent research runs."""
        try:
            proj = ws_mgr.get_project(project_id)
            runs = ws_mgr.list_runs(project_id=project_id)
            d = proj.to_dict()
            d["run_records"] = [r.to_dict() for r in runs]
            return d
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    # ==================== RUNS ====================

    @app.post("/api/v1/projects/{project_id}/runs", status_code=status.HTTP_202_ACCEPTED, tags=["Research Runs"])
    async def submit_run(project_id: str, req: CreateRunRequest) -> Dict[str, Any]:
        """Submit and queue a new asynchronous research run."""
        try:
            ws_mgr.get_project(project_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

        author = AuthorProfile(
            name=req.author_name,
            affiliation=req.author_affiliation,
            email=req.author_email,
        )

        job_info = job_mgr.submit_job(
            project_id=project_id,
            topic=req.topic,
            author=author,
            target_length=req.target_length,
            execution_mode=req.execution_mode,
            num_seeds=req.num_seeds,
            num_epochs=req.num_epochs,
        )
        return {
            "status": "QUEUED",
            "job_id": job_info.job_id,
            "run_id": job_info.run_id,
            "project_id": project_id,
            "topic": req.topic,
        }

    @app.get("/api/v1/runs/{run_id}", tags=["Research Runs"])
    async def get_run(run_id: str) -> Dict[str, Any]:
        """Get full details and status of a research run."""
        try:
            run = ws_mgr.get_run(run_id)
            return run.to_dict()
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    @app.get("/api/v1/runs/{run_id}/status", tags=["Research Runs"])
    async def get_run_status(run_id: str) -> Dict[str, Any]:
        """Poll live execution progress for a run."""
        job = job_mgr.get_job_by_run_id(run_id)
        if job:
            return job.to_dict()
        
        try:
            run = ws_mgr.get_run(run_id)
            return {
                "run_id": run.run_id,
                "project_id": run.project_id,
                "topic": run.topic,
                "state": run.status.value,
                "progress_percent": 100.0 if run.status == RunStatus.COMPLETED else 0.0,
                "stage_message": "Run finished." if run.status == RunStatus.COMPLETED else "Status persisted.",
            }
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    @app.post("/api/v1/runs/{run_id}/cancel", tags=["Research Runs"])
    async def cancel_run(run_id: str) -> Dict[str, Any]:
        """Cancel a running or queued job."""
        job = job_mgr.get_job_by_run_id(run_id)
        if job:
            success = job_mgr.cancel_job(job.job_id)
            return {"cancelled": success, "run_id": run_id}
        raise HTTPException(status_code=404, detail=f"Active job for run '{run_id}' not found.")

    # ==================== RUN SUB-RESOURCES ====================

    @app.get("/api/v1/runs/{run_id}/contract", tags=["Artifacts & Evidence"])
    async def get_run_contract(run_id: str) -> Dict[str, Any]:
        """Retrieve frozen scientific research contract."""
        run = ws_mgr.get_run(run_id)
        if not run.contract_data:
            raise HTTPException(status_code=404, detail="Scientific contract not yet generated for this run.")
        return run.contract_data

    @app.get("/api/v1/runs/{run_id}/evidence", tags=["Artifacts & Evidence"])
    async def get_run_evidence(run_id: str) -> Dict[str, Any]:
        """Retrieve literature evidence and citations."""
        run = ws_mgr.get_run(run_id)
        if not run.evidence_data:
            raise HTTPException(status_code=404, detail="Literature evidence not yet gathered for this run.")
        return run.evidence_data

    @app.get("/api/v1/runs/{run_id}/results", tags=["Artifacts & Evidence"])
    async def get_run_results(run_id: str) -> Dict[str, Any]:
        """Retrieve empirical multi-seed telemetry and metrics."""
        run = ws_mgr.get_run(run_id)
        if not run.results_data:
            raise HTTPException(status_code=404, detail="Results not yet compiled for this run.")
        return run.results_data

    @app.get("/api/v1/runs/{run_id}/statistics", tags=["Artifacts & Evidence"])
    async def get_run_statistics(run_id: str) -> Dict[str, Any]:
        """Retrieve statistical critique and validation report."""
        run = ws_mgr.get_run(run_id)
        if not run.statistics_data:
            raise HTTPException(status_code=404, detail="Statistical analysis not yet completed for this run.")
        return run.statistics_data

    @app.get("/api/v1/runs/{run_id}/provenance", tags=["Artifacts & Evidence"])
    async def get_run_provenance(run_id: str) -> Dict[str, Any]:
        """Retrieve complete provenance DAG."""
        run = ws_mgr.get_run(run_id)
        if not run.provenance_data:
            raise HTTPException(status_code=404, detail="Provenance graph not yet finalized for this run.")
        return run.provenance_data

    @app.get("/api/v1/runs/{run_id}/paper", tags=["Artifacts & Evidence"])
    async def get_run_paper(run_id: str) -> Dict[str, Any]:
        """Retrieve LaTeX manuscript source and physical page metrics."""
        run = ws_mgr.get_run(run_id)
        if not run.manuscript_latex:
            raise HTTPException(status_code=404, detail="Paper not yet compiled for this run.")
        return {
            "run_id": run.run_id,
            "latex_content": run.manuscript_latex,
            "page_count": run.page_count,
            "pdf_artifact_id": run.pdf_artifact_id,
            "zip_artifact_id": run.zip_artifact_id,
        }

    @app.get("/api/v1/runs/{run_id}/artifacts/{artifact_id}/download", tags=["Artifacts & Evidence"])
    async def download_artifact(run_id: str, artifact_id: str) -> FileResponse:
        """Download an immutable verified artifact file."""
        try:
            art = store.get_artifact(artifact_id, verify_integrity=True)
            if art.run_id != run_id:
                raise HTTPException(status_code=403, detail="Artifact does not belong to specified run.")
            return FileResponse(
                path=art.location,
                filename=Path(art.location).name,
                media_type="application/octet-stream",
            )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Artifact '{artifact_id}' not found.")

    return app


# Default ASGI application instance
app = create_app()
