"""NovaScientist Literature & Evidence Agent.

Identifies relevant literature, retrieves verified paper metadata via CrossRef/OpenAlex,
extracts real evidence-grounded scientific claims directly from accessible scholarly text (abstracts/full text),
enforces strict supporting passage requirements, and guarantees zero claims are generated from metadata alone.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from backend.core.doi_verifier import (
    DOIVerificationResult,
    DOIVerificationStatus,
    DOIVerifier,
    calculate_verified_doi_rate,
)
from backend.core.literature import LiteratureService, PaperMetadata
from backend.core.universal_engine import UniversalDomainDispatcher


class EvidenceScope(str, Enum):
    """Scope of accessible scholarly text for an evidence source."""

    FULL_TEXT = "full_text"
    ABSTRACT = "abstract"
    METADATA_ONLY = "metadata_only"


class VerificationStatus(str, Enum):
    """Evidence-grounding status of an extracted scientific claim.

    NOTE ON SCIENTIFIC TERMINOLOGY:
    'grounded' (or 'verified') strictly denotes that the claim is directly supported by
    and excerpted from an accessible text passage in retrieved scholarly literature.
    It does NOT certify or claim independent absolute external scientific truth.
    """

    GROUNDED = "grounded"
    VERIFIED = "grounded"  # Compatibility alias for exact passage-grounded assertions
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    UNVERIFIABLE = "unverifiable"


@dataclass
class ClaimRecord:
    """Evidence-grounded scientific claim directly linked to a source text passage."""

    claim_id: str
    claim_text: str
    source_id: str
    supporting_text: str = ""
    supporting_location: str = "abstract"
    evidence_scope: EvidenceScope = EvidenceScope.ABSTRACT
    category: str = (
        "empirical"  # 'theoretical', 'empirical', 'methodology', 'limitation'
    )
    polarity: str = "supports"  # 'supports', 'contradicts', 'neutral'
    confidence: float | None = None
    extraction_method: str = "passage_extraction"
    verification_status: VerificationStatus = VerificationStatus.GROUNDED
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.evidence_scope, str):
            self.evidence_scope = EvidenceScope(self.evidence_scope)
        if isinstance(self.verification_status, str):
            self.verification_status = VerificationStatus(self.verification_status)

        # Enforce strict supporting passage requirement for grounded claims
        if self.verification_status in (
            VerificationStatus.GROUNDED,
            VerificationStatus.VERIFIED,
        ):
            if not self.supporting_text or not self.supporting_text.strip():
                raise ValueError(
                    f"Invalid claim {self.claim_id}: Claims marked 'grounded' must have non-empty supporting_text."
                )
            if not self.source_id or not self.source_id.strip():
                raise ValueError(
                    f"Invalid claim {self.claim_id}: Claims marked 'grounded' must have an associated source_id."
                )
            if not self.supporting_location or not self.supporting_location.strip():
                raise ValueError(
                    f"Invalid claim {self.claim_id}: Claims marked 'grounded' must have a supporting_location."
                )
            if self.evidence_scope == EvidenceScope.METADATA_ONLY:
                raise ValueError(
                    f"Invalid claim {self.claim_id}: Claims cannot be marked 'grounded' when evidence_scope is 'metadata_only'."
                )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence_scope"] = (
            self.evidence_scope.value
            if isinstance(self.evidence_scope, EvidenceScope)
            else str(self.evidence_scope)
        )
        d["verification_status"] = (
            self.verification_status.value
            if isinstance(self.verification_status, VerificationStatus)
            else str(self.verification_status)
        )
        return d


@dataclass
class SourceRecord:
    """Structured scholarly source record with accessible text tracking and associated claims."""

    source_id: str
    title: str
    authors: list[str]
    year: int
    doi: str
    url: str
    venue: str
    citation_count: int = 0
    relevance_score: float = 1.0
    evidence_scope: EvidenceScope = EvidenceScope.METADATA_ONLY
    source_origin: str = (
        "openalex"  # 'crossref', 'openalex', 'open_access_fulltext', 'test_fixture'
    )
    text_origin: str = "none"  # 'crossref_abstract', 'openalex_abstract_inverted_index', 'open_access_fulltext', 'test_fixture', 'none'
    doi_normalized: str | None = None
    doi_syntax_valid: bool = False
    doi_resolved: bool = False
    doi_metadata_match: bool = False
    doi_verification_status: str = "syntax_valid_only"  # 'verified', 'syntax_valid_only', 'metadata_mismatch', 'unresolvable', 'missing'
    doi_final_url: str | None = None
    doi_http_status: int | None = None
    retrieved_text_available: bool = False
    retrieved_text: str | None = None
    claims: list[ClaimRecord] = field(default_factory=list)
    bibkey: str = ""
    retraction_status: str = "active"  # 'active', 'retracted', 'unknown'

    def __post_init__(self) -> None:
        if re.search(
            r"\b(retracted|retraction|withdrawn)\b", self.title, re.IGNORECASE
        ):
            self.retraction_status = "retracted"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence_scope"] = (
            self.evidence_scope.value
            if isinstance(self.evidence_scope, EvidenceScope)
            else str(self.evidence_scope)
        )
        d["claims"] = [c.to_dict() for c in self.claims]
        return d


@dataclass
class EvidenceBundle:
    """Collection of verified sources, extracted claims, and detected conflicts."""

    topic: str
    domain: str
    sources: list[SourceRecord] = field(default_factory=list)
    claims: list[ClaimRecord] = field(default_factory=list)
    conflicting_claims: list[tuple[str, str, str]] = field(default_factory=list)
    total_sources_retrieved: int = 0
    verified_doi_rate: float | None = None
    doi_verification_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "domain": self.domain,
            "total_sources": len(self.sources),
            "total_claims": len(self.claims),
            "conflicts_count": len(self.conflicting_claims),
            "verified_doi_rate": self.verified_doi_rate,
            "doi_verification_summary": self.doi_verification_summary,
            "sources": [s.to_dict() for s in self.sources],
            "claims": [c.to_dict() for c in self.claims],
            "conflicting_claims": self.conflicting_claims,
        }


class LiteratureAgent:
    """Agent responsible for scholarly evidence discovery and grounded claim extraction."""

    def __init__(
        self,
        lit_service: LiteratureService | None = None,
        doi_verifier: DOIVerifier | None = None,
    ) -> None:
        self.lit_service = lit_service or LiteratureService()
        self.doi_verifier = doi_verifier or DOIVerifier()

    async def gather_evidence(self, topic: str, limit: int = 8) -> EvidenceBundle:
        """Retrieve verified literature, perform real DOI resolution, and extract evidence-grounded claims."""
        classification = UniversalDomainDispatcher.classify_topic(topic)
        papers: list[PaperMetadata] = await self.lit_service.search_literature(
            topic, limit=limit
        )

        if not papers:
            # Safe Fallback: When no external scholarly sources are accessible, return empty evidence bundle
            return EvidenceBundle(
                topic=topic,
                domain=classification.domain_display_name,
                sources=[],
                claims=[],
                conflicting_claims=[],
                total_sources_retrieved=0,
                verified_doi_rate=None,
                doi_verification_summary={
                    "total_doi_bearing_sources": 0,
                    "verified_count": 0,
                },
            )

        sources: list[SourceRecord] = []
        all_claims: list[ClaimRecord] = []
        claim_counter = 1

        for idx, paper in enumerate(papers, 1):
            source_id = f"src_{idx:03d}"
            rel_score = round(max(0.70, 1.0 - (idx - 1) * 0.04), 2)

            accessible_text = paper.accessible_text
            has_text = bool(
                accessible_text
                and len(accessible_text.strip()) > 20
                and paper.text_origin != "none"
            )

            scope = EvidenceScope.ABSTRACT if has_text else EvidenceScope.METADATA_ONLY
            if paper.full_text and len(paper.full_text.strip()) > 100:
                scope = EvidenceScope.FULL_TEXT

            p_claims: list[ClaimRecord] = []
            if has_text and accessible_text is not None:
                p_claims = self._extract_claims_from_text(
                    text=accessible_text,
                    source_id=source_id,
                    scope=scope,
                    start_idx=claim_counter,
                )
                claim_counter += len(p_claims)
                all_claims.extend(p_claims)
            else:
                # ACCESSIBLE TEXT RULE: If no accessible abstract/full text exists, claims = []
                scope = EvidenceScope.METADATA_ONLY
                p_claims = []

            # Perform active DOI verification if not already verified test fixture
            doi_norm = paper.doi_normalized
            doi_syntax = paper.doi_syntax_valid
            doi_res = paper.doi_resolved
            doi_match = paper.doi_metadata_match
            doi_status = paper.doi_verification_status
            doi_furl = paper.doi_final_url
            doi_hstatus = paper.doi_http_status

            if paper.source_origin != "test_fixture" and paper.doi and doi_syntax:
                ver_res: DOIVerificationResult = await self.doi_verifier.verify_doi(
                    doi=paper.doi,
                    expected_title=paper.title,
                    expected_year=paper.year,
                )
                doi_norm = ver_res.doi_normalized
                doi_syntax = ver_res.doi_syntax_valid
                doi_res = ver_res.doi_resolved
                doi_match = ver_res.doi_metadata_match
                doi_status = ver_res.doi_verification_status.value
                doi_furl = ver_res.final_url
                doi_hstatus = ver_res.http_status

            source = SourceRecord(
                source_id=source_id,
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
                doi=paper.doi,
                url=paper.url or f"https://doi.org/{paper.doi}",
                venue=paper.venue,
                citation_count=paper.citation_count,
                relevance_score=rel_score,
                evidence_scope=scope,
                source_origin=paper.source_origin,
                text_origin=paper.text_origin if has_text else "none",
                doi_normalized=doi_norm,
                doi_syntax_valid=doi_syntax,
                doi_resolved=doi_res,
                doi_metadata_match=doi_match,
                doi_verification_status=doi_status,
                doi_final_url=doi_furl,
                doi_http_status=doi_hstatus,
                retrieved_text_available=has_text,
                retrieved_text=accessible_text if has_text else None,
                claims=p_claims,
                bibkey=paper.bibkey,
                retraction_status=paper.retraction_status,
            )
            sources.append(source)

        conflicts = self._detect_conflicts(all_claims)
        v_rate = calculate_verified_doi_rate(sources)

        doi_summary = {
            "total_sources": len(sources),
            "doi_bearing_sources": sum(1 for s in sources if s.doi and s.doi.strip()),
            "verified_dois": sum(
                1
                for s in sources
                if s.doi_verification_status
                in ("verified", DOIVerificationStatus.VERIFIED.value)
            ),
            "syntax_valid_only": sum(
                1
                for s in sources
                if s.doi_verification_status
                == DOIVerificationStatus.SYNTAX_VALID_ONLY.value
            ),
            "resolved_metadata_unavailable": sum(
                1
                for s in sources
                if s.doi_verification_status
                == DOIVerificationStatus.RESOLVED_METADATA_UNAVAILABLE.value
            ),
            "metadata_mismatch": sum(
                1
                for s in sources
                if s.doi_verification_status
                == DOIVerificationStatus.METADATA_MISMATCH.value
            ),
            "unresolvable": sum(
                1
                for s in sources
                if s.doi_verification_status == DOIVerificationStatus.UNRESOLVABLE.value
            ),
            "missing": sum(
                1
                for s in sources
                if s.doi_verification_status == DOIVerificationStatus.MISSING.value
            ),
            "verified_doi_rate": v_rate,
        }

        return EvidenceBundle(
            topic=topic,
            domain=classification.domain_display_name,
            sources=sources,
            claims=all_claims,
            conflicting_claims=conflicts,
            total_sources_retrieved=len(sources),
            verified_doi_rate=v_rate,
            doi_verification_summary=doi_summary,
        )

    def _extract_claims_from_text(
        self,
        text: str,
        source_id: str,
        scope: EvidenceScope,
        start_idx: int,
    ) -> list[ClaimRecord]:
        """Extract concrete scientific claims directly grounded in retrieved text passages."""
        claims: list[ClaimRecord] = []

        # Split text into candidate sentences
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 25]

        idx = start_idx
        for s in sentences:
            # Detect substantive assertion keywords (findings, metrics, methods, constraints)
            is_empirical = bool(
                re.search(
                    r"\b(achieve|achieves|achieved|improves|reduces|reduction|outperform|accuracy|speedup|latency|memory|bound|bounds|error|convergence|proves|demonstrate|variance)\b",
                    s,
                    re.IGNORECASE,
                )
            )
            is_method = bool(
                re.search(
                    r"\b(propose|formulate|investigate|introduce|develop|algorithm|architecture|quantization|transformer|neural|operator|framework)\b",
                    s,
                    re.IGNORECASE,
                )
            )
            is_limitation = bool(
                re.search(
                    r"\b(bottleneck|constrained|limitation|degradation|trade-off|overhead|saturation|expensive)\b",
                    s,
                    re.IGNORECASE,
                )
            )

            if is_empirical or is_method or is_limitation:
                cat = (
                    "limitation"
                    if is_limitation
                    else ("empirical" if is_empirical else "methodology")
                )

                # Claim text preserves exact statement without synthetic exaggeration
                claim_text = s

                claim = ClaimRecord(
                    claim_id=f"claim_{idx:03d}",
                    claim_text=claim_text,
                    source_id=source_id,
                    supporting_text=s,
                    supporting_location="abstract"
                    if scope == EvidenceScope.ABSTRACT
                    else "full_text",
                    evidence_scope=scope,
                    category=cat,
                    polarity="supports",
                    confidence=None,  # No hardcoded fake confidence
                    extraction_method="passage_extraction",
                    verification_status=VerificationStatus.GROUNDED,
                    tags=re.findall(r"\b[a-zA-Z]{4,}\b", s)[:4],
                )
                claims.append(claim)
                idx += 1

                # Cap at 2 salient claims per source to maintain high relevance density
                if len(claims) >= 2:
                    break

        return claims

    def _detect_conflicts(
        self, claims: list[ClaimRecord]
    ) -> list[tuple[str, str, str]]:
        """Detect contrasting scientific paradigms across literature claims with explicit evidence."""
        conflicts: list[tuple[str, str, str]] = []
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                c1, c2 = claims[i], claims[j]
                if c1.polarity == "supports" and c2.polarity == "contradicts":
                    # Shared topic tags
                    shared_tags = set(t.lower() for t in c1.tags) & set(
                        t.lower() for t in c2.tags
                    )
                    if shared_tags:
                        conflicts.append(
                            (
                                c1.claim_id,
                                c2.claim_id,
                                f"Contrasting findings regarding {list(shared_tags)[0]} between {c1.source_id} and {c2.source_id}.",
                            )
                        )
        return conflicts
