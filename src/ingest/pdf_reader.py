"""PDF -> intermediate chunks.

Pages are classified one by one: a page with a usable text layer is read with PyMuPDF
(doc_type "pdf_text"); a page without one (scan / photo) goes to the OCR pipeline
(doc_type "pdf_scan"). Tables found on digital pages are emitted one chunk per row.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from statistics import median

import pymupdf

from .blocks import TABLE, TEXT, TITLE, Block, blocks_to_chunks, join_lines, merge_across_pages
from .schema import Chunk, default_doc_id, source_name

log = logging.getLogger(__name__)
if hasattr(pymupdf, "no_recommend_layout"):  # silence the "use pymupdf_layout" notice
    pymupdf.no_recommend_layout()

MIN_TEXT_CHARS = 20          # fewer selectable characters than this -> treat the page as a scan
MARGIN_RATIO = 0.07          # top/bottom band where running headers / footers / page numbers live
_PAGE_NO_RE = re.compile(r"^\s*(trang|page)?\s*\d+\s*(/\s*\d+)?\s*$", re.I)


def page_has_text_layer(page: pymupdf.Page, min_chars: int = MIN_TEXT_CHARS) -> bool:
    text = page.get_text("text")
    return len(re.sub(r"\s+", "", text)) >= min_chars


def classify_pdf(path: str | Path, min_chars: int = MIN_TEXT_CHARS) -> list[str]:
    """Per-page type: "pdf_text" (text layer) or "pdf_scan"."""
    with pymupdf.open(path) as doc:
        return ["pdf_text" if page_has_text_layer(p, min_chars) else "pdf_scan" for p in doc]


def _clean_cell(text) -> str | None:
    if text is None:
        return None
    return join_lines(str(text).splitlines())


def _fill_merged(rows: list[list[str | None]]) -> list[list[str]]:
    """PyMuPDF returns None for cells covered by a merged cell: header row -> fill from the
    left (group header), other rows -> fill from above (vertical merge)."""
    out: list[list[str]] = []
    for r, row in enumerate(rows):
        new = []
        for c, val in enumerate(row):
            if val is None:
                if r == 0:
                    val = new[c - 1] if c > 0 else ""
                else:
                    val = out[r - 1][c] if c < len(out[r - 1]) else ""
            new.append(val)
        out.append(new)
    return out


def _inside(bbox, rect, tol=2.0) -> bool:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return rect[0] - tol <= cx <= rect[2] + tol and rect[1] - tol <= cy <= rect[3] + tol


def _margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def extract_digital_page(page: pymupdf.Page, skip_margin_keys: set[str] = frozenset()) -> list[Block]:
    """Text blocks + tables of one digital page, in reading order (top -> bottom)."""
    page_no = page.number + 1
    height = page.rect.height
    items: list[tuple[float, float, Block]] = []

    table_rects = []
    try:
        tables = page.find_tables().tables
    except Exception as e:  # table detection is best effort
        log.warning("find_tables failed on page %d: %s", page_no, e)
        tables = []
    for t in tables:
        raw = t.extract()
        if t.row_count < 2 or t.col_count < 2:
            continue
        rows = _fill_merged([[_clean_cell(c) for c in r] for r in raw])
        rows = [r for r in rows if any(r)]
        if len(rows) < 2:
            continue
        table_rects.append(t.bbox)
        items.append((t.bbox[1], t.bbox[0], Block(TABLE, page_no, "pdf_text", rows=rows, bbox=tuple(t.bbox))))

    data = page.get_text("dict", sort=True)
    sizes = Counter()
    for b in data["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                sizes[round(s["size"], 1)] += len(s["text"].strip())
    body = median(sizes.elements()) if sizes else 10.0

    for b in data["blocks"]:
        if b.get("type") != 0:
            continue
        bbox = b["bbox"]
        if any(_inside(bbox, r) for r in table_rects):
            continue
        lines = ["".join(s["text"] for s in l["spans"]) for l in b["lines"]]
        text = join_lines(lines)
        if not text:
            continue
        in_margin = bbox[3] < height * MARGIN_RATIO or bbox[1] > height * (1 - MARGIN_RATIO)
        if in_margin and (_PAGE_NO_RE.match(text) or _margin_key(text) in skip_margin_keys):
            continue
        spans = [s for l in b["lines"] for s in l["spans"] if s["text"].strip()]
        size = max((s["size"] for s in spans), default=body)
        bold = spans and all("bold" in s["font"].lower() or s["flags"] & 16 for s in spans)
        is_title = len(text) <= 150 and (size >= body * 1.15 or (bold and len(b["lines"]) <= 2))
        items.append((bbox[1], bbox[0], Block(TITLE if is_title else TEXT, page_no, "pdf_text", text=text, bbox=tuple(bbox))))

    items.sort(key=lambda it: (round(it[0]), it[1]))
    return [it[2] for it in items]


def _repeated_margin_texts(doc: pymupdf.Document, pages: list[int]) -> set[str]:
    """Running headers/footers: margin text repeated on at least half of the digital pages."""
    if len(pages) < 2:
        return set()
    counts = Counter()
    for i in pages:
        page = doc[i]
        h = page.rect.height
        seen = set()
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
            if y1 < h * MARGIN_RATIO or y0 > h * (1 - MARGIN_RATIO):
                seen.add(_margin_key(text))
        counts.update(seen)
    return {k for k, n in counts.items() if k and n >= max(2, len(pages) / 2)}


def read_pdf_blocks(path: str | Path, ocr_backend: str = "auto", min_chars: int = MIN_TEXT_CHARS,
                    dpi: int = 300) -> list[Block]:
    path = Path(path)
    with pymupdf.open(path) as doc:
        types = ["pdf_text" if page_has_text_layer(p, min_chars) else "pdf_scan" for p in doc]
        digital = [i for i, t in enumerate(types) if t == "pdf_text"]
        scans = [i for i, t in enumerate(types) if t == "pdf_scan"]
        log.info("%s: %d digital page(s), %d scanned page(s)", path.name, len(digital), len(scans))

        skip = _repeated_margin_texts(doc, digital)
        per_page: dict[int, list[Block]] = {i: extract_digital_page(doc[i], skip) for i in digital}
        if scans:
            from .ocr_pipeline import ocr_pages  # lazy: OCR deps only needed for scans

            per_page.update(ocr_pages(doc, scans, backend=ocr_backend, dpi=dpi))
    blocks = [b for i in sorted(per_page) for b in per_page[i]]
    return merge_across_pages(blocks)


def read_pdf(path: str | Path, doc_id: str | None = None, ocr_backend: str = "auto",
             min_chars: int = MIN_TEXT_CHARS, max_chars: int = 800, dpi: int = 300,
             start_index: int = 1) -> list[Chunk]:
    path = Path(path)
    blocks = read_pdf_blocks(path, ocr_backend=ocr_backend, min_chars=min_chars, dpi=dpi)
    return blocks_to_chunks(blocks, doc_id or default_doc_id(path), source_name(path),
                            max_chars=max_chars, start_index=start_index)
