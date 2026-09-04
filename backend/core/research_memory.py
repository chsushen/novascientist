"""NovaScientist Persistent Research Memory Subsystem.

Stores and retrieves structured research artifacts, verified claims, experiment configurations,
empirical findings, and reviewer decisions across sessions without complex database overhead.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ResearchMemoryItem:
    """Single structured entry stored in research memory."""
    task_id: str
    topic: str
    domain: str
    plan_id: str
    sources_count: int
    claims_count: int
    top_claims: List[str] = field(default_factory=list)
    methods_evaluated: List[str] = field(default_factory=list)
    proposed_acc: float = 0.0
    baseline_acc: float = 0.0
    mem_reduction_pct: float = 0.0
    speedup_ratio: float = 0.0
    review_status: str = "passed"
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResearchMemory:
    """Persistent, file-backed research memory store."""

    DEFAULT_STORE_PATH = Path("./artifacts/research_memory.json")

    def __init__(self, store_path: Optional[Path] = None) -> None:
        self.store_path = store_path or self.DEFAULT_STORE_PATH
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Load existing research memory from disk."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _persist(self) -> None:
        """Save in-memory state to disk safely."""
        try:
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(self._memory, f, indent=2)
        except Exception:
            pass

    def store_task(
        self,
        task_id: str,
        topic: str,
        domain: str,
        plan_id: str,
        sources: List[Any],
        claims: List[Any],
        metrics: Dict[str, Any],
        review_passed: bool = True,
    ) -> ResearchMemoryItem:
        """Record a completed research cycle into memory."""
        methods = metrics.get("methods", {})
        prop = methods.get("proposed_mb_qgt", {})
        dense = methods.get("dense_baseline", {})

        p_acc = prop.get("mean_accuracy", 0.0) * 100.0
        d_acc = dense.get("mean_accuracy", 0.0) * 100.0
        p_mem = prop.get("mean_memory_mb", 0.0)
        d_mem = dense.get("mean_memory_mb", 0.0)
        p_lat = prop.get("mean_latency_ms", 0.0)
        d_lat = dense.get("mean_latency_ms", 0.0)

        mem_red = ((d_mem - p_mem) / d_mem * 100.0) if d_mem > 0 else 0.0
        speedup = (d_lat / p_lat) if p_lat > 0 else 1.0

        top_claims_list = []
        for c in claims[:3]:
            if hasattr(c, "claim_text"):
                top_claims_list.append(c.claim_text)
            elif isinstance(c, dict):
                top_claims_list.append(c.get("claim_text", ""))

        item = ResearchMemoryItem(
            task_id=task_id,
            topic=topic,
            domain=domain,
            plan_id=plan_id,
            sources_count=len(sources),
            claims_count=len(claims),
            top_claims=top_claims_list,
            methods_evaluated=list(methods.keys()),
            proposed_acc=round(p_acc, 2),
            baseline_acc=round(d_acc, 2),
            mem_reduction_pct=round(mem_red, 1),
            speedup_ratio=round(speedup, 2),
            review_status="passed" if review_passed else "flagged",
            timestamp=metrics.get("timestamp", ""),
        )

        self._memory[task_id] = item.to_dict()
        self._persist()
        return item

    def find_relevant_knowledge(self, topic: str, domain: str) -> List[Dict[str, Any]]:
        """Retrieve relevant prior research findings matching topic/domain keywords."""
        topic_lower = topic.lower()
        results: List[Dict[str, Any]] = []
        for task_id, data in self._memory.items():
            if data.get("domain", "").lower() == domain.lower():
                results.append(data)
            elif any(w in data.get("topic", "").lower() for w in topic_lower.split() if len(w) > 4):
                results.append(data)
        return results

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Return all recorded memory entries."""
        return list(self._memory.values())

    def clear(self) -> None:
        """Reset memory store."""
        self._memory = {}
        self._persist()
