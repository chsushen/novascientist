"""NovaScientist Persistent Research Workspaces.

Implements the User -> Project -> ResearchRun hierarchy with full persistence,
run inspection, checkpointing, and cross-run comparison.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.config import config
from backend.storage.artifact_store import ArtifactStore, StoredArtifact


class RunStatus(str, Enum):
    """Lifecycle states for research runs."""
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CHECKPOINTING = "CHECKPOINTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class ResearchRun:
    """A single end-to-end scientific research run."""
    run_id: str
    project_id: str
    topic: str
    status: RunStatus = RunStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    contract_id: Optional[str] = None
    contract_data: Optional[Dict[str, Any]] = None
    evidence_data: Optional[Dict[str, Any]] = None
    experiment_data: Optional[Dict[str, Any]] = None
    results_data: Optional[Dict[str, Any]] = None
    statistics_data: Optional[Dict[str, Any]] = None
    figures_data: Optional[Dict[str, Any]] = None
    claims_data: Optional[List[Dict[str, Any]]] = None
    provenance_data: Optional[Dict[str, Any]] = None
    manuscript_latex: Optional[str] = None
    pdf_artifact_id: Optional[str] = None
    zip_artifact_id: Optional[str] = None
    page_count: int = 0
    checkpoint: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    artifacts: List[str] = field(default_factory=list)  # List of artifact_ids
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize run record to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, RunStatus) else str(self.status)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResearchRun:
        """Construct run record from dictionary."""
        d = dict(data)
        if "status" in d:
            d["status"] = RunStatus(d["status"])
        return cls(**d)


@dataclass
class Project:
    """Top-level research project containing multiple runs."""
    project_id: str
    user_id: str
    name: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    runs: List[str] = field(default_factory=list)  # List of run_ids
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize project to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Project:
        """Construct project from dictionary."""
        return cls(**data)


class WorkspaceManager:
    """Manages persistent project and research run directories and metadata."""

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else config.data_dir / "workspaces"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_store = artifact_store or ArtifactStore()
        self.projects_file = self.base_dir / "projects_index.json"
        self.runs_file = self.base_dir / "runs_index.json"
        self._projects: Dict[str, Project] = {}
        self._runs: Dict[str, ResearchRun] = {}
        self._load_indexes()

    def _load_indexes(self) -> None:
        """Load index files from disk."""
        if self.projects_file.exists():
            try:
                with open(self.projects_file, "r", encoding="utf-8") as f:
                    raw_p = json.load(f)
                    self._projects = {k: Project.from_dict(v) for k, v in raw_p.items()}
            except Exception:
                self._projects = {}

        if self.runs_file.exists():
            try:
                with open(self.runs_file, "r", encoding="utf-8") as f:
                    raw_r = json.load(f)
                    self._runs = {k: ResearchRun.from_dict(v) for k, v in raw_r.items()}
            except Exception:
                self._runs = {}

    def _save_indexes(self) -> None:
        """Save index files atomically."""
        tmp_p = self.projects_file.with_suffix(".tmp")
        with open(tmp_p, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in self._projects.items()}, f, indent=2)
        shutil.move(str(tmp_p), str(self.projects_file))

        tmp_r = self.runs_file.with_suffix(".tmp")
        with open(tmp_r, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in self._runs.items()}, f, indent=2)
        shutil.move(str(tmp_r), str(self.runs_file))

    def create_project(
        self,
        name: str,
        description: str = "",
        user_id: str = "default_user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Project:
        """Create and persist a new research project."""
        proj_hash = hashlib.sha256(f"{name}_{user_id}_{time.time()}".encode("utf-8")).hexdigest()[:10]
        project_id = f"proj_{proj_hash}"

        proj = Project(
            project_id=project_id,
            user_id=user_id,
            name=name,
            description=description,
            metadata=metadata or {},
        )
        self._projects[project_id] = proj
        (self.base_dir / project_id).mkdir(parents=True, exist_ok=True)
        self._save_indexes()
        return proj

    def get_project(self, project_id: str) -> Project:
        """Retrieve project by ID."""
        if project_id not in self._projects:
            raise KeyError(f"Project '{project_id}' not found.")
        return self._projects[project_id]

    def list_projects(self, user_id: Optional[str] = None) -> List[Project]:
        """List all projects or filter by user."""
        if user_id:
            return [p for p in self._projects.values() if p.user_id == user_id]
        return list(self._projects.values())

    def create_run(
        self,
        project_id: str,
        topic: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ResearchRun:
        """Create and queue a new research run under a project."""
        project = self.get_project(project_id)
        run_hash = hashlib.sha256(f"{project_id}_{topic}_{time.time()}".encode("utf-8")).hexdigest()[:10]
        run_id = f"run_{run_hash}"

        run = ResearchRun(
            run_id=run_id,
            project_id=project_id,
            topic=topic,
            status=RunStatus.QUEUED,
            metadata=metadata or {},
        )
        self._runs[run_id] = run
        project.runs.append(run_id)
        project.updated_at = time.time()

        run_dir = self.base_dir / project_id / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._save_indexes()
        return run

    def get_run(self, run_id: str) -> ResearchRun:
        """Retrieve research run by ID."""
        if run_id not in self._runs:
            raise KeyError(f"Research Run '{run_id}' not found.")
        return self._runs[run_id]

    def list_runs(self, project_id: Optional[str] = None) -> List[ResearchRun]:
        """List runs across all projects or for a specific project."""
        if project_id:
            return [r for r in self._runs.values() if r.project_id == project_id]
        return list(self._runs.values())

    def update_run(self, run_id: str, **kwargs: Any) -> ResearchRun:
        """Update fields on a research run and persist."""
        run = self.get_run(run_id)
        for k, v in kwargs.items():
            if hasattr(run, k):
                setattr(run, k, v)
        run.updated_at = time.time()
        self._save_indexes()
        return run

    def save_checkpoint(self, run_id: str, stage: str, state_data: Dict[str, Any]) -> ResearchRun:
        """Save execution checkpoint enabling recovery on failure."""
        checkpoint_entry = {
            "stage": stage,
            "timestamp": time.time(),
            "state": state_data,
        }
        return self.update_run(
            run_id,
            status=RunStatus.CHECKPOINTING,
            checkpoint=checkpoint_entry,
        )

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Generate a structured comparison between multiple research runs."""
        comparison: Dict[str, Any] = {
            "comparison_id": f"comp_{int(time.time())}",
            "runs_evaluated": len(run_ids),
            "runs": {},
        }
        for rid in run_ids:
            run = self.get_run(rid)
            comparison["runs"][rid] = {
                "topic": run.topic,
                "status": run.status.value,
                "contract_id": run.contract_id,
                "primary_metric": run.contract_data.get("primary_metrics", ["Accuracy (%)"])[0] if run.contract_data else None,
                "selected_dataset": run.contract_data.get("selected_dataset") if run.contract_data else None,
                "selected_method": run.contract_data.get("selected_method") if run.contract_data else None,
                "page_count": run.page_count,
                "results_summary": run.results_data.get("meta_analysis", {}) if run.results_data else {},
            }
        return comparison
