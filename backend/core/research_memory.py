"""NovaScientist Persistent Research Memory Subsystem.

Stores and retrieves structured research artifacts, verified claims, experiment configurations,
empirical findings, and reviewer decisions across sessions without complex database overhead.
Implements atomic file-backed JSON persistence, corrupted store recovery, and semantic keyword/domain retrieval.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Common English stopwords for query tokenization
STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "of",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "that",
    "which",
    "this",
    "these",
    "those",
    "using",
    "under",
    "via",
    "over",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
}


def tokenize_text(text: str) -> set[str]:
    """Extract clean alphanumeric tokens excluding stopwords."""
    words = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


@dataclass
class ResearchMemoryItem:
    """Single structured entry stored in research memory."""

    task_id: str
    topic: str
    domain: str
    plan_id: str
    sources_count: int
    claims_count: int
    top_claims: list[str] = field(default_factory=list)
    methods_evaluated: list[str] = field(default_factory=list)
    proposed_acc: float = 0.0
    baseline_acc: float = 0.0
    mem_reduction_pct: float = 0.0
    speedup_ratio: float = 0.0
    review_status: str = "passed"
    timestamp: str = ""
    model_acronym: str = ""
    dataset_name: str = ""
    meta_effect_size: float = 0.0
    meta_i_squared: float = 0.0
    provenance_summary: dict[str, int] = field(default_factory=dict)
    relevance_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchMemoryItem:
        """Construct a ResearchMemoryItem safely from a dictionary."""
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class ResearchMemory:
    """Persistent, file-backed research memory store with atomic I/O and recovery."""

    DEFAULT_STORE_PATH = Path("./artifacts/research_memory.json")

    def __init__(self, store_path: str | Path | None = None) -> None:
        self.store_path = Path(store_path) if store_path else self.DEFAULT_STORE_PATH
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        """Load existing research memory from disk with corrupt-file recovery."""
        if not self.store_path.exists():
            return {}

        try:
            with open(self.store_path, encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                data = json.loads(content)
                if isinstance(data, dict):
                    return data
                # If content is a valid JSON list of items, convert to dict keyed by task_id
                elif isinstance(data, list):
                    res: dict[str, dict[str, Any]] = {}
                    for idx, item in enumerate(data):
                        if isinstance(item, dict):
                            t_id = item.get("task_id", f"task_{idx}")
                            res[t_id] = item
                    return res
                else:
                    raise ValueError(
                        f"Expected dict or list root in memory JSON, got {type(data).__name__}"
                    )
        except Exception:
            # Backup corrupted file safely and reset
            backup_path = self.store_path.with_suffix(
                f".corrupted.{int(time.time())}.bak"
            )
            try:
                shutil.copy2(self.store_path, backup_path)
            except Exception:
                pass
            return {}

    def _persist(self) -> None:
        """Save in-memory state to disk atomically using temporary file rename."""
        try:
            temp_path = self.store_path.with_suffix(
                f".tmp.{os.getpid()}_{int(time.time() * 1000)}"
            )
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._memory, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.store_path)
        except Exception:
            # Fallback direct write if atomic replacement fails on restricted systems
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
        sources: list[Any],
        claims: list[Any],
        metrics: dict[str, Any],
        review_passed: bool = True,
        model_acronym: str = "",
        dataset_name: str = "",
        provenance_graph: dict[str, Any] | None = None,
    ) -> ResearchMemoryItem:
        """Record a completed research cycle into memory."""
        methods = metrics.get("methods", {})
        prop = methods.get("proposed_mb_qgt", {})
        dense = methods.get("dense_baseline", {})

        # Extract accuracy
        p_acc_val = (
            prop.get("mean_accuracy", 0.0)
            if isinstance(prop, dict)
            else getattr(prop, "mean_accuracy", 0.0)
        )
        d_acc_val = (
            dense.get("mean_accuracy", 0.0)
            if isinstance(dense, dict)
            else getattr(dense, "mean_accuracy", 0.0)
        )
        p_acc = p_acc_val * 100.0 if p_acc_val <= 1.0 else p_acc_val
        d_acc = d_acc_val * 100.0 if d_acc_val <= 1.0 else d_acc_val

        # Extract memory
        p_mem = (
            prop.get("mean_memory_mb", 0.0)
            if isinstance(prop, dict)
            else getattr(prop, "mean_memory_mb", 0.0)
        )
        d_mem = (
            dense.get("mean_memory_mb", 0.0)
            if isinstance(dense, dict)
            else getattr(dense, "mean_memory_mb", 0.0)
        )

        # Extract latency
        p_lat = (
            prop.get("mean_latency_ms", 0.0)
            if isinstance(prop, dict)
            else getattr(prop, "mean_latency_ms", 0.0)
        )
        d_lat = (
            dense.get("mean_latency_ms", 0.0)
            if isinstance(dense, dict)
            else getattr(dense, "mean_latency_ms", 0.0)
        )

        mem_red = ((d_mem - p_mem) / d_mem * 100.0) if d_mem > 0 else 0.0
        speedup = (d_lat / p_lat) if p_lat > 0 else 1.0

        top_claims_list: list[str] = []
        for c in claims[:5]:
            if hasattr(c, "claim_text"):
                top_claims_list.append(c.claim_text)
            elif isinstance(c, dict):
                top_claims_list.append(c.get("claim_text", ""))

        meta_res = metrics.get("meta_analysis", {})
        if isinstance(meta_res, dict):
            eff_size = meta_res.get("pooled_effect_size", 0.0)
            i_sq = meta_res.get("i_squared_percent", 0.0)
        else:
            eff_size = getattr(meta_res, "pooled_effect_size", 0.0)
            i_sq = getattr(meta_res, "i_squared_percent", 0.0)

        # Summarize provenance if available
        prov_summary: dict[str, int] = {}
        if provenance_graph and isinstance(provenance_graph, dict):
            nodes = provenance_graph.get("nodes", [])
            for n in nodes:
                nt = (
                    n.get("node_type", "unknown")
                    if isinstance(n, dict)
                    else getattr(n, "node_type", "unknown")
                )
                prov_summary[nt] = prov_summary.get(nt, 0) + 1

        timestamp = metrics.get("timestamp") or datetime.now(UTC).isoformat()

        item = ResearchMemoryItem(
            task_id=task_id,
            topic=topic,
            domain=domain,
            plan_id=plan_id,
            sources_count=len(sources),
            claims_count=len(claims),
            top_claims=top_claims_list,
            methods_evaluated=list(methods.keys()) if isinstance(methods, dict) else [],
            proposed_acc=round(float(p_acc), 2),
            baseline_acc=round(float(d_acc), 2),
            mem_reduction_pct=round(float(mem_red), 1),
            speedup_ratio=round(float(speedup), 2),
            review_status="passed" if review_passed else "flagged",
            timestamp=str(timestamp),
            model_acronym=model_acronym,
            dataset_name=dataset_name,
            meta_effect_size=round(float(eff_size), 4),
            meta_i_squared=round(float(i_sq), 2),
            provenance_summary=prov_summary,
        )

        self._memory[task_id] = item.to_dict()
        self._persist()
        return item

    def find_relevant_knowledge(
        self,
        topic: str,
        domain: str,
        top_k: int = 5,
        min_score: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant prior research findings matching topic/domain keywords with ranked scoring."""
        query_tokens = tokenize_text(topic)
        domain_tokens = tokenize_text(domain)
        all_query_tokens = query_tokens.union(domain_tokens)

        scored_results: list[tuple[float, dict[str, Any]]] = []

        for task_id, data in self._memory.items():
            score = 0.0
            rec_domain = data.get("domain", "")
            rec_topic = data.get("topic", "")
            rec_acronym = data.get("model_acronym", "")
            rec_claims = data.get("top_claims", [])

            # Domain exact or partial match
            if rec_domain.lower() == domain.lower():
                score += 3.0
            elif any(
                d in rec_domain.lower() for d in domain.lower().split() if len(d) > 3
            ):
                score += 1.5

            # Topic token overlap
            rec_topic_tokens = tokenize_text(rec_topic)
            if query_tokens and rec_topic_tokens:
                overlap = query_tokens.intersection(rec_topic_tokens)
                if overlap:
                    jaccard = len(overlap) / len(query_tokens.union(rec_topic_tokens))
                    score += 2.5 * jaccard + 0.5 * len(overlap)

            # Model acronym matching
            if rec_acronym and rec_acronym.lower() in topic.lower():
                score += 2.0

            # Claim token overlap
            claims_text = " ".join(rec_claims)
            rec_claim_tokens = tokenize_text(claims_text)
            if all_query_tokens and rec_claim_tokens:
                claim_overlap = all_query_tokens.intersection(rec_claim_tokens)
                if claim_overlap:
                    score += 0.3 * len(claim_overlap)

            if score >= min_score:
                entry_copy = dict(data)
                entry_copy["relevance_score"] = round(score, 3)
                scored_results.append((score, entry_copy))

        # Sort descending by score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_results[:top_k]]

    def query_prior_knowledge(
        self,
        topic: str,
        domain: str,
        top_k: int = 5,
    ) -> list[ResearchMemoryItem]:
        """Retrieve strongly-typed ResearchMemoryItem objects matching topic/domain."""
        raw_results = self.find_relevant_knowledge(topic, domain, top_k=top_k)
        return [ResearchMemoryItem.from_dict(d) for d in raw_results]

    def get_summary(self) -> dict[str, Any]:
        """Return aggregate statistics across all recorded tasks."""
        if not self._memory:
            return {
                "total_tasks": 0,
                "domains_represented": [],
                "avg_proposed_accuracy": 0.0,
                "avg_baseline_accuracy": 0.0,
                "avg_memory_reduction_pct": 0.0,
                "avg_speedup_ratio": 0.0,
                "review_pass_rate": 0.0,
            }

        entries = list(self._memory.values())
        total = len(entries)
        domains = sorted(
            list({e.get("domain", "") for e in entries if e.get("domain")})
        )
        p_accs = [e.get("proposed_acc", 0.0) for e in entries]
        d_accs = [e.get("baseline_acc", 0.0) for e in entries]
        m_reds = [e.get("mem_reduction_pct", 0.0) for e in entries]
        speedups = [e.get("speedup_ratio", 1.0) for e in entries]
        passes = [1 for e in entries if e.get("review_status") == "passed"]

        return {
            "total_tasks": total,
            "domains_represented": domains,
            "avg_proposed_accuracy": round(sum(p_accs) / total, 2),
            "avg_baseline_accuracy": round(sum(d_accs) / total, 2),
            "avg_memory_reduction_pct": round(sum(m_reds) / total, 1),
            "avg_speedup_ratio": round(sum(speedups) / total, 2),
            "review_pass_rate": round(len(passes) / total, 3),
        }

    def export_knowledge_graph(self) -> dict[str, Any]:
        """Export research memory as a structured knowledge graph with nodes and relationships."""
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        seen_domains: set[str] = set()
        seen_methods: set[str] = set()

        for task_id, item in self._memory.items():
            # Task Node
            nodes.append(
                {
                    "id": task_id,
                    "type": "Task",
                    "label": item.get("topic", task_id),
                    "properties": {
                        "proposed_acc": item.get("proposed_acc"),
                        "mem_reduction_pct": item.get("mem_reduction_pct"),
                        "review_status": item.get("review_status"),
                        "timestamp": item.get("timestamp"),
                    },
                }
            )

            # Domain Node & Edge
            dom = item.get("domain", "")
            if dom:
                dom_id = f"dom_{dom.lower().replace(' ', '_')}"
                if dom_id not in seen_domains:
                    seen_domains.add(dom_id)
                    nodes.append(
                        {"id": dom_id, "type": "Domain", "label": dom, "properties": {}}
                    )
                edges.append(
                    {
                        "source": task_id,
                        "target": dom_id,
                        "relation": "BELONGS_TO_DOMAIN",
                    }
                )

            # Method Nodes & Edges
            methods = item.get("methods_evaluated", [])
            for m in methods:
                m_id = f"method_{m}"
                if m_id not in seen_methods:
                    seen_methods.add(m_id)
                    nodes.append(
                        {"id": m_id, "type": "Method", "label": m, "properties": {}}
                    )
                edges.append(
                    {
                        "source": task_id,
                        "target": m_id,
                        "relation": "EVALUATED_METHOD",
                    }
                )

        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }

    def get_entry(self, task_id: str) -> dict[str, Any] | None:
        """Retrieve a specific research task entry by task_id."""
        return self._memory.get(task_id)

    def delete_entry(self, task_id: str) -> bool:
        """Remove a specific research task entry from memory."""
        if task_id in self._memory:
            del self._memory[task_id]
            self._persist()
            return True
        return False

    def get_all_entries(self) -> list[dict[str, Any]]:
        """Return all recorded memory entries."""
        return list(self._memory.values())

    def clear(self) -> None:
        """Reset memory store."""
        self._memory = {}
        self._persist()
