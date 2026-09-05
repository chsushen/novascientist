"""NovaScientist Physical Page Controller.

Feedback controller that measures compiled PDF physical page counts against
target venue specifications and generates deterministic adjustments
to satisfy strict publication page boundaries.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class PageBudgetStatus(str, Enum):
    """Evaluation status of physical page count against budget bounds."""

    IN_RANGE = "IN_RANGE"
    UNDER_BUDGET = "UNDER_BUDGET"
    OVER_BUDGET = "OVER_BUDGET"
    ESTIMATED = "ESTIMATED"


@dataclass
class PageBudgetEvaluation:
    """Detailed evaluation of physical page budget compliance."""

    measured_pages: int
    target_min: int
    target_max: int
    status: PageBudgetStatus
    delta_from_target: int
    word_count: int
    suggested_adjustments: list[str] = field(default_factory=list)
    expansion_factor: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = (
            self.status.value
            if isinstance(self.status, PageBudgetStatus)
            else str(self.status)
        )
        return d


class PhysicalPageController:
    """Measures physical compiled PDF page count and advises on layout tuning."""

    def __init__(self) -> None:
        pass

    def measure_pdf_pages(self, pdf_path: str | Path) -> int | None:
        """Attempt to measure the exact physical page count of a compiled PDF."""
        p = Path(pdf_path)
        if not p.exists() or p.stat().st_size == 0:
            return None

        # 1. Try pypdf / pymupdf if available
        try:
            import pypdf

            reader = pypdf.PdfReader(str(p))
            return len(reader.pages)
        except Exception:
            pass

        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(p))
            pages = len(doc)
            doc.close()
            return pages
        except Exception:
            pass

        # 2. Try pdfinfo command line utility
        try:
            res = subprocess.run(
                ["pdfinfo", str(p)], capture_output=True, text=True, timeout=5
            )
            if res.returncode == 0:
                match = re.search(r"Pages:\s+(\d+)", res.stdout)
                if match:
                    return int(match.group(1))
        except Exception:
            pass

        # 3. Fast binary scan for /Type /Page objects
        try:
            with open(str(p), "rb") as f:
                content = f.read()
                matches = re.findall(rb"/Type\s*/Page\b", content)
                pages_count = len(matches)
                # Ensure we don't count /Pages
                pages_obj = re.findall(rb"/Type\s*/Pages\b", content)
                net_pages = pages_count - len(pages_obj)
                if net_pages > 0:
                    return net_pages
                if pages_count > 0:
                    return pages_count
        except Exception:
            pass

        return None

    def estimate_pages_from_content(
        self,
        tex_content: str,
        num_figures: int = 5,
        num_tables: int = 2,
    ) -> int:
        """Heuristic estimate of page count based on word count, equations, and floats."""
        # Strip comments
        clean_tex = re.sub(r"(?m)^%.*$", "", tex_content)
        words = len(re.findall(r"\b[A-Za-z0-9_-]+\b", clean_tex))
        equations = len(re.findall(r"\\begin\{equation\}|\\\[", clean_tex))

        # Standard two-column IEEE / ACM layout has ~900-1000 words per text page.
        # Each figure/table occupies ~0.25 to 0.5 pages.
        text_pages = words / 900.0
        float_pages = (num_figures * 0.35) + (num_tables * 0.25)
        eq_pages = equations * 0.05

        est_pages = max(1, int(round(text_pages + float_pages + eq_pages)))
        return est_pages

    def evaluate_page_budget(
        self,
        target_min: int,
        target_max: int,
        pdf_path: str | Path | None = None,
        tex_content: str | None = None,
        num_figures: int = 5,
    ) -> PageBudgetEvaluation:
        """Evaluate page budget and suggest concrete adjustments."""
        measured = None
        if pdf_path:
            measured = self.measure_pdf_pages(pdf_path)

        words = 0
        if tex_content:
            clean_tex = re.sub(r"(?m)^%.*$", "", tex_content)
            words = len(re.findall(r"\b[A-Za-z0-9_-]+\b", clean_tex))

        if measured is None:
            if tex_content:
                measured = self.estimate_pages_from_content(
                    tex_content, num_figures=num_figures
                )
                is_estimate = True
            else:
                measured = target_min
                is_estimate = True
        else:
            is_estimate = False

        adjustments: list[str] = []
        expansion_factor = 1.0

        if measured < target_min:
            status = PageBudgetStatus.UNDER_BUDGET
            delta = target_min - measured
            expansion_factor = round(target_min / max(1, measured), 2)
            adjustments.append(
                f"Expand literature discussion and detailed proof derivations (deficit: {delta} pages)."
            )
            adjustments.append(
                "Add extensive hyperparameter sensitivity sub-sections and ablation tables."
            )
            adjustments.append(
                "Include deeper epistemic limitations and broader impact analysis."
            )
        elif measured > target_max:
            status = PageBudgetStatus.OVER_BUDGET
            delta = measured - target_max
            expansion_factor = round(target_max / measured, 2)
            adjustments.append(
                f"Compress methodology details into algorithmic floats (overflow: {delta} pages)."
            )
            adjustments.append(
                "Move secondary proof lemmas and extended baseline tables to Appendix."
            )
            adjustments.append(
                "Use compact multi-panel figures and tighten paragraph spacing."
            )
        else:
            status = (
                PageBudgetStatus.IN_RANGE
                if not is_estimate
                else PageBudgetStatus.ESTIMATED
            )
            delta = 0
            adjustments.append(
                "Physical page length satisfies publication venue constraints."
            )

        return PageBudgetEvaluation(
            measured_pages=measured,
            target_min=target_min,
            target_max=target_max,
            status=status,
            delta_from_target=delta,
            word_count=words,
            suggested_adjustments=adjustments,
            expansion_factor=expansion_factor,
        )
