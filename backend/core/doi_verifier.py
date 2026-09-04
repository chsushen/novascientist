"""NovaScientist Real DOI Verifier & Metadata Validation Module.

Provides deterministic DOI normalization, strict standard syntax validation (ISO 26324),
asynchronous HTTP resolution to canonical publisher endpoints with redirect tracking,
deterministic normalized title and publication year cross-checking, in-memory caching,
and strictly measured verified DOI rates.

NOTE ON SCIENTIFIC TERMINOLOGY:
'DOI Verified' establishes strictly that the digital object identifier resolves
to a valid publisher endpoint and matches the expected bibliographic metadata.
It does NOT certify or imply that scientific claims within the paper are true.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import httpx


class DOIVerificationStatus(str, Enum):
    """Structured DOI verification states."""
    VERIFIED = "verified"                                  # Valid syntax, final 2xx HTTP resolution, required metadata obtained, title & year matched
    SYNTAX_VALID_ONLY = "syntax_valid_only"                # Valid standard syntax, but not yet actively resolved
    RESOLVED_METADATA_UNAVAILABLE = "resolved_metadata_unavailable" # HTTP 2xx resolved, but required title/year metadata was unavailable from endpoint
    METADATA_MISMATCH = "metadata_mismatch"                # HTTP 2xx resolved, but publisher title or year contradicts source metadata
    UNRESOLVABLE = "unresolvable"                          # HTTP 404, 410, 429, 5xx, DNS failure, timeout, connection error, or invalid syntax
    MISSING = "missing"                                    # No DOI provided or empty string


# Standard DOI prefix regex: 10. followed by 4-9 digits, a slash, and suffix
DOI_SYNTAX_PATTERN = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$", re.IGNORECASE)


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Deterministically clean and normalize DOI strings.
    
    Handles:
    - https://doi.org/10.xxxx/abc
    - http://doi.org/10.xxxx/abc
    - https://dx.doi.org/10.xxxx/abc
    - doi:10.xxxx/abc
    - 10.xxxx/abc
    - Leading/trailing whitespace and accidental trailing punctuation.
    """
    if not doi or not isinstance(doi, str):
        return None
    
    clean = doi.strip()
    if not clean:
        return None

    # Strip URL prefixes
    url_prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi.org/",
        "dx.doi.org/",
    ]
    for prefix in url_prefixes:
        if clean.lower().startswith(prefix):
            clean = clean[len(prefix):].strip()
            break

    # Strip doi: prefix
    if clean.lower().startswith("doi:"):
        clean = clean[4:].strip()

    # Strip accidental trailing punctuation: trailing dots, commas, semicolons
    clean = re.sub(r"[.,;]+$", "", clean).strip()

    if not clean.startswith("10."):
        return None

    return clean


def validate_doi_syntax(doi: Optional[str]) -> bool:
    """Validate whether normalized DOI matches canonical ISO 26324 / CrossRef syntax."""
    if not doi or not isinstance(doi, str):
        return False
    norm = normalize_doi(doi)
    if not norm:
        return False
    return bool(DOI_SYNTAX_PATTERN.match(norm))


def normalize_title_for_comparison(title: Optional[str]) -> str:
    """Normalize title string for robust cross-checking against publisher metadata."""
    if not title or not isinstance(title, str):
        return ""
    # Remove LaTeX commands, punctuation, extra spaces, lowercase
    clean = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})?", "", title)
    clean = re.sub(r"[^\w\s]", " ", clean).lower()
    return re.sub(r"\s+", " ", clean).strip()


def is_title_match(expected_title: Optional[str], resolved_title: Optional[str], threshold: float = 0.50) -> bool:
    """Evaluate whether resolved publisher title matches expected source title via deterministic lexical matching."""
    t1 = normalize_title_for_comparison(expected_title)
    t2 = normalize_title_for_comparison(resolved_title)
    
    if not t1 or not t2:
        return False
    
    if t1 == t2 or t1 in t2 or t2 in t1:
        return True
    
    tokens1: Set[str] = set(w for w in t1.split() if len(w) > 2)
    tokens2: Set[str] = set(w for w in t2.split() if len(w) > 2)
    
    if not tokens1 or not tokens2:
        return False
        
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    jaccard = len(intersection) / len(union) if union else 0.0
    
    return jaccard >= threshold


def is_year_match(expected_year: Optional[int], resolved_year: Optional[int], tolerance: int = 1) -> bool:
    """Evaluate whether resolved year matches expected year.
    
    Documented tolerance rule:
    - If expected_year is None: True (no year constraint specified).
    - If expected_year is provided but resolved_year is None: False.
    - If both provided: |expected_year - resolved_year| <= tolerance.
      (Tolerance of 1 year accommodates online-first vs print publication year differences).
    """
    if expected_year is None:
        return True
    if resolved_year is None:
        return False
    try:
        return abs(int(expected_year) - int(resolved_year)) <= tolerance
    except (ValueError, TypeError):
        return False


def extract_metadata_from_response(resp: httpx.Response) -> Tuple[Optional[str], Optional[int]]:
    """Extract resolved title and publication year from HTTP response.
    
    Precedence for year extraction:
    1. published-print (date-parts)
    2. published-online (date-parts)
    3. issued (date-parts)
    4. created (date-parts)
    5. direct numeric publication_year or year
    """
    resolved_title = None
    resolved_year = None

    content_type = resp.headers.get("content-type", "")
    if "json" in content_type:
        try:
            meta = resp.json()
            if isinstance(meta, dict):
                # Title extraction
                title_val = meta.get("title")
                if isinstance(title_val, list) and title_val:
                    resolved_title = str(title_val[0]).strip()
                elif title_val:
                    resolved_title = str(title_val).strip()

                # Year extraction in precedence order
                for date_key in ("published-print", "published-online", "issued", "created"):
                    date_obj = meta.get(date_key)
                    if isinstance(date_obj, dict) and "date-parts" in date_obj:
                        dp = date_obj["date-parts"]
                        if isinstance(dp, list) and dp and isinstance(dp[0], list) and dp[0]:
                            try:
                                resolved_year = int(dp[0][0])
                                break
                            except (ValueError, TypeError):
                                pass

                if resolved_year is None:
                    for y_key in ("publication_year", "year"):
                        if y_key in meta and meta[y_key] is not None:
                            try:
                                resolved_year = int(meta[y_key])
                                break
                            except (ValueError, TypeError):
                                pass
        except Exception:
            pass

    return resolved_title, resolved_year


@dataclass
class DOIVerificationResult:
    """Outcome of active DOI verification."""
    doi: str
    doi_normalized: Optional[str]
    doi_syntax_valid: bool
    doi_resolved: bool
    doi_metadata_match: bool
    doi_verification_status: DOIVerificationStatus
    http_status: Optional[int] = None
    final_url: Optional[str] = None
    resolved_title: Optional[str] = None
    resolved_year: Optional[int] = None
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["doi_verification_status"] = self.doi_verification_status.value
        return d


class DOIVerifier:
    """Asynchronous, cached, security-hardened DOI resolver and metadata cross-checker."""

    def __init__(self, timeout: float = 5.0, email: str = "novascientist@research.org") -> None:
        self.timeout = timeout
        self.email = email
        self.headers = {
            "User-Agent": f"NovaScientist-DOIVerifier/2.0 (mailto:{self.email})",
            "Accept": "application/vnd.citationstyles.csl+json, application/json;q=0.9, text/html;q=0.5",
        }
        self._cache: Dict[str, DOIVerificationResult] = {}

    async def verify_doi(
        self,
        doi: Optional[str],
        expected_title: Optional[str] = None,
        expected_year: Optional[int] = None,
    ) -> DOIVerificationResult:
        """Verify a single DOI with syntax validation, HTTP resolution, and metadata cross-check."""
        if not doi or not str(doi).strip():
            return DOIVerificationResult(
                doi=doi or "",
                doi_normalized=None,
                doi_syntax_valid=False,
                doi_resolved=False,
                doi_metadata_match=False,
                doi_verification_status=DOIVerificationStatus.MISSING,
                error_type="missing_doi",
            )

        norm = normalize_doi(doi)
        if not norm:
            return DOIVerificationResult(
                doi=doi,
                doi_normalized=None,
                doi_syntax_valid=False,
                doi_resolved=False,
                doi_metadata_match=False,
                doi_verification_status=DOIVerificationStatus.UNRESOLVABLE,
                error_type="malformed_prefix",
            )

        syntax_ok = validate_doi_syntax(norm)
        if not syntax_ok:
            return DOIVerificationResult(
                doi=doi,
                doi_normalized=norm,
                doi_syntax_valid=False,
                doi_resolved=False,
                doi_metadata_match=False,
                doi_verification_status=DOIVerificationStatus.UNRESOLVABLE,
                error_type="invalid_syntax",
            )

        # Cache key must include DOI, expected title, and expected year
        cache_key = f"{norm}::{expected_title or ""}::{expected_year if expected_year is not None else ""}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Security: construct URL only from validated normalized DOI
        encoded_doi = urllib.parse.quote(norm, safe="/:.-_()")
        resolver_url = f"https://doi.org/{encoded_doi}"

        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.get(resolver_url)
                status_code = resp.status_code
                final_url = str(resp.url)

                # Tighten HTTP success semantics: only final 2xx is considered resolved
                if 200 <= status_code < 300:
                    doi_resolved = True
                    resolved_title, resolved_year = extract_metadata_from_response(resp)

                    title_ok = False
                    title_mismatch = False
                    if expected_title:
                        if resolved_title:
                            if is_title_match(expected_title, resolved_title):
                                title_ok = True
                            else:
                                title_mismatch = True
                        else:
                            title_ok = False
                    else:
                        title_ok = True if resolved_title else False

                    year_ok = False
                    year_mismatch = False
                    if expected_year is not None:
                        if resolved_year is not None:
                            if is_year_match(expected_year, resolved_year, tolerance=1):
                                year_ok = True
                            else:
                                year_mismatch = True
                        else:
                            year_ok = False
                    else:
                        year_ok = True

                    if title_mismatch or year_mismatch:
                        doi_metadata_match = False
                        ver_status = DOIVerificationStatus.METADATA_MISMATCH
                    elif (expected_title and not resolved_title) or (expected_year is not None and resolved_year is None):
                        # Case C: Resolved via HTTP 200, but required metadata was unavailable
                        doi_metadata_match = False
                        ver_status = DOIVerificationStatus.RESOLVED_METADATA_UNAVAILABLE
                    elif title_ok and year_ok:
                        # Case A: Title & year matched
                        doi_metadata_match = True
                        ver_status = DOIVerificationStatus.VERIFIED
                    else:
                        doi_metadata_match = False
                        ver_status = DOIVerificationStatus.RESOLVED_METADATA_UNAVAILABLE

                    result = DOIVerificationResult(
                        doi=doi,
                        doi_normalized=norm,
                        doi_syntax_valid=True,
                        doi_resolved=doi_resolved,
                        doi_metadata_match=doi_metadata_match,
                        doi_verification_status=ver_status,
                        http_status=status_code,
                        final_url=final_url,
                        resolved_title=resolved_title,
                        resolved_year=resolved_year,
                    )
                elif status_code in (404, 410):
                    result = DOIVerificationResult(
                        doi=doi,
                        doi_normalized=norm,
                        doi_syntax_valid=True,
                        doi_resolved=False,
                        doi_metadata_match=False,
                        doi_verification_status=DOIVerificationStatus.UNRESOLVABLE,
                        http_status=status_code,
                        final_url=final_url,
                        error_type=f"http_{status_code}_not_found" if status_code == 404 else "http_410_gone",
                    )
                elif status_code == 429:
                    result = DOIVerificationResult(
                        doi=doi,
                        doi_normalized=norm,
                        doi_syntax_valid=True,
                        doi_resolved=False,
                        doi_metadata_match=False,
                        doi_verification_status=DOIVerificationStatus.UNRESOLVABLE,
                        http_status=status_code,
                        final_url=final_url,
                        error_type="http_429_rate_limited",
                    )
                elif status_code >= 500:
                    result = DOIVerificationResult(
                        doi=doi,
                        doi_normalized=norm,
                        doi_syntax_valid=True,
                        doi_resolved=False,
                        doi_metadata_match=False,
                        doi_verification_status=DOIVerificationStatus.UNRESOLVABLE,
                        http_status=status_code,
                        final_url=final_url,
                        error_type=f"http_{status_code}_server_error",
                    )
                else:
                    result = DOIVerificationResult(
                        doi=doi,
                        doi_normalized=norm,
                        doi_syntax_valid=True,
                        doi_resolved=False,
                        doi_metadata_match=False,
                        doi_verification_status=DOIVerificationStatus.UNRESOLVABLE,
                        http_status=status_code,
                        final_url=final_url,
                        error_type=f"http_{status_code}_unresolved",
                    )

        except httpx.TimeoutException:
            result = DOIVerificationResult(
                doi=doi,
                doi_normalized=norm,
                doi_syntax_valid=True,
                doi_resolved=False,
                doi_metadata_match=False,
                doi_verification_status=DOIVerificationStatus.UNRESOLVABLE,
                error_type="timeout",
            )
        except httpx.RequestError as exc:
            result = DOIVerificationResult(
                doi=doi,
                doi_normalized=norm,
                doi_syntax_valid=True,
                doi_resolved=False,
                doi_metadata_match=False,
                doi_verification_status=DOIVerificationStatus.UNRESOLVABLE,
                error_type=f"connection_error: {type(exc).__name__}",
            )
        except Exception as exc:
            result = DOIVerificationResult(
                doi=doi,
                doi_normalized=norm,
                doi_syntax_valid=True,
                doi_resolved=False,
                doi_metadata_match=False,
                doi_verification_status=DOIVerificationStatus.UNRESOLVABLE,
                error_type=f"unexpected_error: {str(exc)}",
            )

        self._cache[cache_key] = result
        return result


def calculate_verified_doi_rate(sources: List[Any]) -> Optional[float]:
    """Calculate verified DOI rate from actual source records.
    
    Formula:
    verified_doi_rate = verified_doi_count / total_doi_bearing_sources
    
    Returns:
    - None if total_doi_bearing_sources == 0 (explicitly unavailable, NEVER 1.0)
    - float between 0.0 and 1.0 otherwise.
    """
    if not sources:
        return None

    doi_bearing_sources = [
        s for s in sources
        if getattr(s, "doi", None) and str(getattr(s, "doi", "")).strip()
    ]
    
    if not doi_bearing_sources:
        return None

    verified_count = sum(
        1 for s in doi_bearing_sources
        if getattr(s, "doi_verification_status", "") in (
            "verified",
            DOIVerificationStatus.VERIFIED,
            DOIVerificationStatus.VERIFIED.value,
        )
    )

    return round(verified_count / len(doi_bearing_sources), 4)
