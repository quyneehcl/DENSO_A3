"""Helpers shared by Excel and PDF table handling: cell formatting, header detection,
row -> structured sentence, HTML table parsing (PaddleOCR-VL table output)."""
from __future__ import annotations

import datetime as dt
import math
import re
from html.parser import HTMLParser


def format_cell(value) -> str:
    """Render a cell value as clean text ('' for empty)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Có" if value else "Không"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    if hasattr(value, "to_pydatetime"):  # pandas Timestamp
        value = value.to_pydatetime()
    if isinstance(value, dt.datetime):
        if value.time() == dt.time(0, 0):
            return value.date().isoformat()
        return value.isoformat(sep=" ", timespec="minutes")
    if isinstance(value, (dt.date, dt.time)):
        return value.isoformat()
    text = str(value).replace(" ", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def _lower_first(s: str) -> str:
    # Keep acronyms / codes (e.g. "PLC", "IP") as they are.
    if len(s) > 1 and s[:2].isupper():
        return s
    return s[:1].lower() + s[1:]


def row_to_sentence(headers: list[str], values: list, language: str = "vi") -> str:
    """Turn one table row into a self-contained sentence keeping column/value pairs.

    ["Tên linh kiện", "Mã lỗi", "Cách xử lý"] + ["Cảm biến nhiệt", "E01", "Thay cảm biến"]
    -> "Tên linh kiện là Cảm biến nhiệt, mã lỗi là E01, cách xử lý là Thay cảm biến."
    """
    link = {"vi": "là", "en": "is"}.get(language, ":")
    parts = []
    for i, val in enumerate(values):
        val = format_cell(val)
        if not val:
            continue
        head = format_cell(headers[i]) if i < len(headers) else ""
        if not head:
            parts.append(val)
            continue
        if parts:
            head = _lower_first(head)
        parts.append(f"{head}: {val}" if link == ":" else f"{head} {link} {val}")
    if not parts:
        return ""
    sentence = "; ".join(parts) if any("," in p for p in parts) else ", ".join(parts)
    sentence = sentence[0].upper() + sentence[1:]
    return sentence if sentence[-1] in ".!?" else sentence + "."


_NUMERIC_RE = re.compile(r"[-+]?[\d.,]+\s*%?")


def is_header_like(cells: list[str]) -> bool:
    """A header row: >= 2 non-empty cells, none of them a pure number."""
    filled = [c for c in cells if c]
    if len(filled) < 2:
        return False
    return all(not _NUMERIC_RE.fullmatch(c) for c in filled)


def combine_header_rows(rows: list[list[str]]) -> list[str]:
    """Merge multi-level headers column-wise: ["Thông số", "Điện áp"] -> "Thông số - Điện áp"."""
    width = max(len(r) for r in rows)
    out = []
    for col in range(width):
        parts: list[str] = []
        for r in rows:
            v = r[col] if col < len(r) else ""
            if v and v not in parts:
                parts.append(v)
        out.append(" - ".join(parts))
    return out


def dedupe_headers(headers: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, h in enumerate(headers):
        h = h or f"Cột {i + 1}"
        if h in seen:
            seen[h] += 1
            h = f"{h} ({seen[h]})"
        else:
            seen[h] = 1
        out.append(h)
    return out


def split_header(rows: list[list[str]], max_header_rows: int = 2) -> tuple[list[str], int]:
    """Detect the header of a raw table (list of rows) -> (headers, number of header rows).

    The first header-like row is the header. A second header-like row directly below is
    treated as a sub-header when the first row repeats a value across columns (merged group).
    """
    if not rows:
        return [], 0
    header_rows = [rows[0]]
    first = [c for c in rows[0] if c]
    has_group = len(first) != len(set(first)) or any(not c for c in rows[0])
    if max_header_rows > 1 and len(rows) > 1 and has_group and is_header_like(rows[1]):
        header_rows.append(rows[1])
    headers = dedupe_headers(combine_header_rows(header_rows))
    return headers, len(header_rows)


class _HTMLTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._colspan = 1
        self._rowspan = 1
        self._pending: dict[int, tuple[int, str]] = {}  # col -> (rows left, text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.tables.append([])
            self._pending = {}
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
            self._colspan = int(attrs.get("colspan") or 1)
            self._rowspan = int(attrs.get("rowspan") or 1)
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def _fill_pending(self):
        while self._row is not None and len(self._row) in self._pending:
            col = len(self._row)
            left, text = self._pending[col]
            self._row.append(text)
            if left <= 1:
                del self._pending[col]
            else:
                self._pending[col] = (left - 1, text)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._fill_pending()
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            for _ in range(self._colspan):
                if self._rowspan > 1:
                    self._pending[len(self._row)] = (self._rowspan - 1, text)
                self._row.append(text)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self._fill_pending()
            if self.tables and any(self._row):
                self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_tables_to_rows(html: str) -> list[list[list[str]]]:
    """Parse every <table> in an HTML string into rows (colspan/rowspan expanded)."""
    p = _HTMLTableParser()
    p.feed(html)
    return [t for t in p.tables if t]
