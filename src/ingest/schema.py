"""Intermediate chunk format shared by every ingest module.

Every reader MUST emit exactly these fields — no more, no less:

    {
      "chunk_id": "doc001_chunk003",
      "text": "...",
      "source_file": "ten_file_goc.xlsx",
      "location": "Sheet1, hàng 5",
      "doc_type": "excel_table",
      "language": "vi"
    }
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

CHUNK_FIELDS = ("chunk_id", "text", "source_file", "location", "doc_type", "language")
VALID_DOC_TYPES = {"pdf_text", "pdf_scan", "excel_table"}
CHUNK_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+_chunk\d{3,}$")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    location: str
    doc_type: str
    language: str

    def to_dict(self) -> dict:
        return asdict(self)


def make_chunk_id(doc_id: str, index: int) -> str:
    """index is 1-based -> doc001_chunk001."""
    return f"{doc_id}_chunk{index:03d}"


def default_doc_id(path: str | Path) -> str:
    """Stable doc id derived from the file name (ASCII slug)."""
    stem = Path(path).stem
    stem = unicodedata.normalize("NFKD", stem.replace("đ", "d").replace("Đ", "D"))
    stem = stem.encode("ascii", "ignore").decode()
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return stem or "doc"


# Vietnamese-specific letters (base letters + tone-marked vowels) that do not occur in English.
_VI_CHARS = set(
    "ăâđêôơưĂÂĐÊÔƠƯ"
    "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
    "ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ"
)


def detect_language(text: str, default: str = "vi") -> str:
    """Lightweight, dependency-free language guess: 'vi', 'ja' or 'en'."""
    text = unicodedata.normalize("NFC", text)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return default
    ja = sum(1 for c in letters if "぀" <= c <= "ヿ" or "一" <= c <= "鿿")
    if ja / len(letters) > 0.2:
        return "ja"
    vi = sum(1 for c in letters if c in _VI_CHARS)
    if vi / len(letters) > 0.02:
        return "vi"
    if all(c.isascii() for c in letters):
        return "en"
    return default


def normalize_text(text: str) -> str:
    """NFC-normalize, collapse whitespace inside lines, drop empty lines."""
    text = unicodedata.normalize("NFC", text).replace(" ", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def validate_chunk(chunk: dict) -> list[str]:
    """Return a list of problems (empty list == valid)."""
    errors = []
    if not isinstance(chunk, dict):
        return ["chunk is not an object"]
    keys = set(chunk)
    if keys != set(CHUNK_FIELDS):
        missing = set(CHUNK_FIELDS) - keys
        extra = keys - set(CHUNK_FIELDS)
        if missing:
            errors.append(f"missing fields: {sorted(missing)}")
        if extra:
            errors.append(f"unexpected fields: {sorted(extra)}")
    for f in CHUNK_FIELDS:
        if f in chunk and not isinstance(chunk[f], str):
            errors.append(f"field {f!r} must be a string")
    if isinstance(chunk.get("chunk_id"), str) and not CHUNK_ID_RE.match(chunk["chunk_id"]):
        errors.append(f"bad chunk_id: {chunk['chunk_id']!r}")
    if isinstance(chunk.get("text"), str) and not chunk["text"].strip():
        errors.append("empty text")
    if chunk.get("doc_type") not in VALID_DOC_TYPES:
        errors.append(f"invalid doc_type: {chunk.get('doc_type')!r}")
    for f in ("source_file", "location", "language"):
        if isinstance(chunk.get(f), str) and not chunk[f].strip():
            errors.append(f"empty {f}")
    return errors


def validate_chunks(chunks: Iterable[dict]) -> list[str]:
    errors, seen = [], set()
    for i, c in enumerate(chunks):
        errors += [f"chunk[{i}]: {e}" for e in validate_chunk(c)]
        cid = c.get("chunk_id") if isinstance(c, dict) else None
        if cid in seen:
            errors.append(f"chunk[{i}]: duplicate chunk_id {cid!r}")
        seen.add(cid)
    return errors


def write_chunks(chunks: Iterable[Chunk | dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.to_dict() if isinstance(c, Chunk) else c for c in chunks]
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
