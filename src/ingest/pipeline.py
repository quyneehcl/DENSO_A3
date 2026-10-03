"""Single entry point: ingest any supported file into intermediate chunks."""
from __future__ import annotations

from pathlib import Path

from .excel_reader import read_excel
from .pdf_reader import classify_pdf, read_pdf
from .schema import Chunk

EXCEL_EXT = {".xlsx", ".xlsm", ".xls", ".csv"}
PDF_EXT = {".pdf"}
SUPPORTED_EXT = EXCEL_EXT | PDF_EXT


def file_category(path: str | Path) -> str:
    """"excel", "pdf_text" (all pages digital), "pdf_scan" (all scanned) or "pdf_mixed"."""
    ext = Path(path).suffix.lower()
    if ext in EXCEL_EXT:
        return "excel"
    if ext in PDF_EXT:
        kinds = set(classify_pdf(path))
        return kinds.pop() if len(kinds) == 1 else "pdf_mixed"
    raise ValueError(f"unsupported file type: {path}")


def ingest_file(path: str | Path, doc_id: str | None = None, ocr_backend: str = "auto",
                max_chars: int = 800, dpi: int = 300) -> list[Chunk]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in EXCEL_EXT:
        return read_excel(path, doc_id=doc_id)
    if ext in PDF_EXT:
        return read_pdf(path, doc_id=doc_id, ocr_backend=ocr_backend, max_chars=max_chars, dpi=dpi)
    raise ValueError(f"unsupported file type: {path} (supported: {sorted(SUPPORTED_EXT)})")
