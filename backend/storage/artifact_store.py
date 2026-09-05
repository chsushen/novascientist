"""NovaScientist Immutable Artifact Store.

Manages persistent, immutable research artifacts with cryptographic SHA-256 verification
and bidirectional provenance tracking.
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


class ArtifactType(str, Enum):
    """Categorization of immutable research artifacts."""
    DATASET = "dataset"
    EXPERIMENT_SPEC = "experiment_spec"
    RESULT = "result"
    FIGURE = "figure"
    STATISTICS = "statistics"
    MANUSCRIPT = "manuscript"
    PDF = "pdf"
    OVERLEAF_PACKAGE = "overleaf_package"
    MODEL_WEIGHTS = "model_weights"
    REPRODUCIBILITY_MANIFEST = "reproducibility_manifest"
    SCIENTIFIC_CONTRACT = "scientific_contract"
    PROVENANCE_GRAPH = "provenance_graph"


class ArtifactIntegrityError(Exception):
    """Raised when an artifact fails SHA-256 integrity verification."""
    pass


@dataclass
class StoredArtifact:
    """Metadata record for an immutable research artifact."""
    artifact_id: str
    project_id: str
    run_id: str
    artifact_type: ArtifactType
    location: str
    sha256: str
    created_at: float = field(default_factory=time.time)
    source: str = "novascientist_pipeline"
    provenance_nodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary."""
        d = asdict(self)
        d["artifact_type"] = self.artifact_type.value if isinstance(self.artifact_type, ArtifactType) else str(self.artifact_type)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StoredArtifact:
        """Construct record from dictionary."""
        d = dict(data)
        if "artifact_type" in d:
            d["artifact_type"] = ArtifactType(d["artifact_type"])
        return cls(**d)


class ArtifactStore:
    """Manages persistent immutable storage for research runs."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else config.data_dir / "artifacts"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.base_dir / "artifact_index.json"
        self._index: Dict[str, StoredArtifact] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load artifact catalog from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    self._index = {k: StoredArtifact.from_dict(v) for k, v in raw.items()}
            except Exception:
                self._index = {}

    def _save_index(self) -> None:
        """Persist catalog to disk atomically."""
        tmp_path = self.index_file.with_suffix(".tmp")
        data = {k: v.to_dict() for k, v in self._index.items()}
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        shutil.move(str(tmp_path), str(self.index_file))

    @staticmethod
    def compute_sha256(content: Union[str, bytes, Path]) -> str:
        """Compute SHA-256 hash of string, bytes, or file on disk."""
        h = hashlib.sha256()
        if isinstance(content, str):
            h.update(content.encode("utf-8"))
        elif isinstance(content, bytes):
            h.update(content)
        elif isinstance(content, Path):
            with open(content, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
        return h.hexdigest()

    def store_bytes(
        self,
        project_id: str,
        run_id: str,
        artifact_type: ArtifactType,
        filename: str,
        content: bytes,
        provenance_nodes: Optional[List[str]] = None,
        source: str = "novascientist_pipeline",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredArtifact:
        """Store binary payload immutably."""
        sha256_hash = self.compute_sha256(content)
        artifact_id = f"art_{sha256_hash[:12]}"

        target_dir = self.base_dir / project_id / run_id
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / filename

        with open(file_path, "wb") as f:
            f.write(content)

        record = StoredArtifact(
            artifact_id=artifact_id,
            project_id=project_id,
            run_id=run_id,
            artifact_type=artifact_type,
            location=str(file_path),
            sha256=sha256_hash,
            provenance_nodes=provenance_nodes or [],
            source=source,
            metadata=metadata or {},
        )
        self._index[artifact_id] = record
        self._save_index()
        return record

    def store_text(
        self,
        project_id: str,
        run_id: str,
        artifact_type: ArtifactType,
        filename: str,
        content: str,
        provenance_nodes: Optional[List[str]] = None,
        source: str = "novascientist_pipeline",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredArtifact:
        """Store text string (e.g. LaTeX, JSON, Markdown)."""
        return self.store_bytes(
            project_id=project_id,
            run_id=run_id,
            artifact_type=artifact_type,
            filename=filename,
            content=content.encode("utf-8"),
            provenance_nodes=provenance_nodes,
            source=source,
            metadata=metadata,
        )

    def store_file(
        self,
        project_id: str,
        run_id: str,
        artifact_type: ArtifactType,
        source_file: Union[str, Path],
        target_filename: Optional[str] = None,
        provenance_nodes: Optional[List[str]] = None,
        source: str = "novascientist_pipeline",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredArtifact:
        """Copy and register an existing file from disk."""
        src_path = Path(source_file)
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")

        sha256_hash = self.compute_sha256(src_path)
        artifact_id = f"art_{sha256_hash[:12]}"
        dest_filename = target_filename or src_path.name

        target_dir = self.base_dir / project_id / run_id
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_path = target_dir / dest_filename

        shutil.copy2(str(src_path), str(dest_path))

        record = StoredArtifact(
            artifact_id=artifact_id,
            project_id=project_id,
            run_id=run_id,
            artifact_type=artifact_type,
            location=str(dest_path),
            sha256=sha256_hash,
            provenance_nodes=provenance_nodes or [],
            source=source,
            metadata=metadata or {},
        )
        self._index[artifact_id] = record
        self._save_index()
        return record

    def get_artifact(self, artifact_id: str, verify_integrity: bool = True) -> StoredArtifact:
        """Retrieve artifact metadata with optional SHA-256 verification."""
        if artifact_id not in self._index:
            raise KeyError(f"Artifact ID '{artifact_id}' not found in store.")
        
        record = self._index[artifact_id]
        if verify_integrity:
            loc = Path(record.location)
            if not loc.exists():
                raise FileNotFoundError(f"Artifact file missing at: {record.location}")
            current_hash = self.compute_sha256(loc)
            if current_hash != record.sha256:
                raise ArtifactIntegrityError(
                    f"Integrity violation on artifact '{artifact_id}': "
                    f"expected {record.sha256}, got {current_hash}"
                )
        return record

    def read_bytes(self, artifact_id: str) -> bytes:
        """Read artifact binary content with fail-closed integrity check."""
        record = self.get_artifact(artifact_id, verify_integrity=True)
        with open(record.location, "rb") as f:
            return f.read()

    def read_text(self, artifact_id: str) -> str:
        """Read artifact text content with fail-closed integrity check."""
        return self.read_bytes(artifact_id).decode("utf-8")

    def list_run_artifacts(self, run_id: str) -> List[StoredArtifact]:
        """List all artifacts generated for a given research run."""
        return [art for art in self._index.values() if art.run_id == run_id]

    def list_project_artifacts(self, project_id: str) -> List[StoredArtifact]:
        """List all artifacts for an entire project workspace."""
        return [art for art in self._index.values() if art.project_id == project_id]
