"""Page blocks shared by the digital PDF reader and the OCR pipeline, plus the logic that
stitches blocks across pages and turns them into intermediate chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import Chunk, detect_language, make_chunk_id, normalize_text
from .table_utils import row_to_sentence, split_header

TEXT, TITLE, TABLE = "text", "title", "table"
_SENTENCE_END = tuple(".!?:;…。)")
_BULLET_RE = re.compile(r"^\s*([-•●▪*–]|\d+[.)]|[a-zA-Z][.)])\s+")


def join_lines(lines: list[str]) -> str:
    """Re-flow wrapped lines into one paragraph; keep list items on their own line."""
    out = ""
    for ln in (l.strip() for l in lines):
        if not ln:
            continue
        if not out:
            out = ln
        elif _BULLET_RE.match(ln):
            out += "\n" + ln
        elif out.endswith("-") and not out.endswith(" -") and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out += " " + ln
    return out


@dataclass
class Block:
    kind: str                      # "text" | "title" | "table"
    page: int                      # 1-based page where the block starts
    doc_type: str                  # "pdf_text" | "pdf_scan"
    text: str = ""
    rows: list[list[str]] = field(default_factory=list)   # tables only, header rows included
    row_pages: list[int] = field(default_factory=list)    # page of each row (tables only)
    end_page: int | None = None    # text merged across pages
    bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if self.kind == TABLE and not self.row_pages:
            self.row_pages = [self.page] * len(self.rows)


def _is_continuation(prev: str, nxt: str) -> bool:
    prev, nxt = prev.rstrip(), nxt.lstrip()
    if not prev or not nxt:
        return False
    if prev.endswith("-") and not prev.endswith(" -"):
        return True
    return not prev.endswith(_SENTENCE_END) and (nxt[0].islower() or nxt[0] in ",;)")


def _join_text(prev: str, nxt: str) -> str:
    prev = prev.rstrip()
    if prev.endswith("-") and nxt[:1].islower():
        return prev[:-1] + nxt.lstrip()
    return prev + " " + nxt.lstrip()


def _last_page(b: Block) -> int:
    if b.kind == TABLE and b.row_pages:
        return b.row_pages[-1]
    return b.end_page or b.page


def merge_across_pages(blocks: list[Block]) -> list[Block]:
    """Join a paragraph / table that is cut by a page break.

    * text: last text block of page N + first text block of page N+1 when the first one does
      not end a sentence and the second one continues it (lowercase start, hyphenation);
    * table: last block of page N and first block of page N+1 are tables with the same
      number of columns -> one table (a repeated header row on the new page is dropped).
    """
    out: list[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
        if prev is not None and b.page == _last_page(prev) + 1 and prev.doc_type == b.doc_type:
            if prev.kind == TEXT and b.kind == TEXT and _is_continuation(prev.text, b.text):
                prev.text = _join_text(prev.text, b.text)
                prev.end_page = b.end_page or b.page
                continue
            if prev.kind == TABLE and b.kind == TABLE and prev.rows and b.rows \
                    and len(prev.rows[0]) == len(b.rows[0]):
                _, n_head = split_header(prev.rows)
                rows, pages = b.rows, b.row_pages
                k = 0
                while k < min(n_head, len(rows)) and rows[k] == prev.rows[k]:
                    k += 1
                prev.rows += rows[k:]
                prev.row_pages += pages[k:]
                continue
        out.append(b)
    return out


def _page_label(start: int, end: int | None) -> str:
    return f"Trang {start}" if not end or end == start else f"Trang {start}-{end}"


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    parts, cur = [], ""
    for s in sentences:
        while len(s) > max_chars:  # no sentence boundary: hard cut on whitespace
            cut = s.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(s[:cut].strip())
            s = s[cut:].strip()
        if cur and len(cur) + 1 + len(s) > max_chars:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def blocks_to_chunks(blocks: list[Block], doc_id: str, source_file: str,
                     max_chars: int = 800, start_index: int = 1) -> list[Chunk]:
    """Text: consecutive paragraphs of one page grouped up to `max_chars`; a title opens a
    new chunk and stays attached to the text below it. Tables: one chunk per data row."""
    chunks: list[Chunk] = []

    def emit(text: str, location: str, doc_type: str):
        text = normalize_text(text)
        if text:
            chunks.append(Chunk(make_chunk_id(doc_id, start_index + len(chunks)), text,
                                source_file, location, doc_type, detect_language(text)))

    buf: list[str] = []
    buf_start = buf_end = 0
    buf_type = ""
    only_title = False

    def flush():
        for part in _split_long("\n".join(buf), max_chars) if buf else []:
            emit(part, _page_label(buf_start, buf_end), buf_type)
        buf.clear()

    table_no: dict[int, int] = {}
    for b in blocks:
        if b.kind == TABLE:
            flush()
            if len(b.rows) < 2:  # a single row cannot have header + data: keep it as text
                for r in b.rows:
                    emit(" | ".join(c for c in r if c), _page_label(b.page, None), b.doc_type)
                continue
            table_no[b.page] = table_no.get(b.page, 0) + 1
            headers, n_head = split_header(b.rows)
            data = [(r, p) for r, p in zip(b.rows[n_head:], b.row_pages[n_head:]) if r not in b.rows[:n_head]]
            for i, (row, page) in enumerate(data, start=1):
                lang = detect_language(" ".join(headers + row))
                emit(row_to_sentence(headers, row, lang),
                     f"Trang {page}, bảng {table_no[b.page]}, hàng {i}", b.doc_type)
            continue

        text = b.text.strip()
        if not text:
            continue
        same_place = buf and b.page == buf_end and b.doc_type == buf_type
        fits = len("\n".join(buf)) + len(text) + 1 <= max_chars
        # a title left alone at the bottom of a page belongs to the text on the next page
        title_carry = only_title and b.page == buf_end + 1 and b.doc_type == buf_type
        if b.kind == TEXT and ((same_place and (fits or only_title)) or title_carry):
            buf.append(text)
            only_title = False
        elif b.kind == TITLE and only_title and same_place:  # title wrapped on several lines
            buf[-1] += " " + text
        else:
            flush()
            buf.append(text)
            buf_start, buf_type = b.page, b.doc_type
            only_title = b.kind == TITLE
        buf_end = b.end_page or b.page
    flush()
    return chunks
