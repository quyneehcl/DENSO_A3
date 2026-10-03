"""OCR pipeline for scanned PDF pages (doc_type "pdf_scan").

    render page -> (orientation fix, deskew) -> layout (text / title / table regions)
                -> recognise each region -> reading order -> Blocks

Backends
  * "paddle_vl"  PaddleOCR-VL: PP-DocLayoutV2 layout analysis + PaddleOCR-VL-0.9B recognition.
                 Needs `pip install -r requirements-paddle.txt` (GPU strongly recommended).
  * "tesseract"  Fallback that runs on CPU: Tesseract (vie+eng) for text, ruled-grid detection
                 with numpy for tables (each cell OCR'd separately, merged cells supported).
  * "auto"       paddle_vl when PaddleOCR is installed and a GPU is visible, else tesseract.
  * "none"       skip scanned pages (only a warning is logged).

Cross-page stitching of paragraphs / tables is done afterwards by blocks.merge_across_pages.
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

from .blocks import TABLE, TEXT, TITLE, Block, join_lines
from .table_utils import html_tables_to_rows, markdown_table_to_rows

log = logging.getLogger(__name__)
DOC_TYPE = "pdf_scan"
_PAGE_NO_RE = re.compile(r"^\s*(trang|page)?\s*\d+\s*(/\s*\d+)?\s*$", re.I)


def render_page(page: pymupdf.Page, dpi: int = 300) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


class OCRBackend:
    name = "base"

    def recognize(self, image: Image.Image, page_no: int) -> list[Block]:
        raise NotImplementedError


# =============================================================================== PaddleOCR-VL
class PaddleOCRVLBackend(OCRBackend):
    """PaddleOCR-VL document parsing (PaddleOCR >= 3.3)."""

    name = "paddle_vl"
    SKIP = {"header", "footer", "number", "header_image", "footer_image", "seal", "image",
            "figure", "aside_text", "footnote", "vision_footnote"}
    TITLES = {"doc_title", "paragraph_title", "title", "figure_title", "table_title", "chart_title"}

    def __init__(self, use_doc_orientation_classify: bool = True, use_doc_unwarping: bool = False,
                 device: str | None = None):
        from paddleocr import PaddleOCRVL  # noqa: deferred heavy import

        kwargs = {"use_doc_orientation_classify": use_doc_orientation_classify,
                  "use_doc_unwarping": use_doc_unwarping}
        if device:
            kwargs["device"] = device
        try:
            self.pipeline = PaddleOCRVL(**kwargs)
        except TypeError:  # older/newer signature: fall back to defaults
            self.pipeline = PaddleOCRVL()

    @staticmethod
    def _result_dict(res) -> dict:
        data = getattr(res, "json", None)
        if callable(data):
            data = data()
        if data is None and isinstance(res, dict):
            data = res
        data = data or {}
        return data.get("res", data)

    def recognize(self, image: Image.Image, page_no: int) -> list[Block]:
        with tempfile.TemporaryDirectory() as d:
            img_path = Path(d) / f"page_{page_no}.png"
            image.convert("RGB").save(img_path)
            results = list(self.pipeline.predict(str(img_path)))
        blocks: list[Block] = []
        for res in results:
            for item in self._result_dict(res).get("parsing_res_list", []):
                if not isinstance(item, dict):  # some versions return objects
                    item = {k: getattr(item, k, None) for k in ("block_label", "block_content", "block_bbox",
                                                                 "label", "content", "bbox")}
                label = (item.get("block_label") or item.get("label") or "text").lower()
                content = (item.get("block_content") or item.get("content") or "").strip()
                bbox = item.get("block_bbox") or item.get("bbox")
                bbox = tuple(float(v) for v in bbox[:4]) if bbox is not None and len(bbox) >= 4 else None
                if not content or label in self.SKIP:
                    continue
                if label == "table":
                    tables = html_tables_to_rows(content) or markdown_table_to_rows(content)
                    if tables:
                        blocks += [Block(TABLE, page_no, DOC_TYPE, rows=t, bbox=bbox) for t in tables]
                        continue
                kind = TITLE if label in self.TITLES else TEXT
                text = join_lines(re.sub(r"^#+\s*", "", content).splitlines())
                blocks.append(Block(kind, page_no, DOC_TYPE, text=text, bbox=bbox))
        return blocks


# =============================================================================== Tesseract
def _otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total, sum_all = gray.size, np.dot(np.arange(256), hist)
    w0 = sum0 = 0.0
    best, thresh = -1.0, 128
    for t in range(256):
        w0 += hist[t]
        if w0 == 0 or w0 == total:
            continue
        sum0 += t * hist[t]
        m0, m1 = sum0 / w0, (sum_all - sum0) / (total - w0)
        var = w0 * (total - w0) * (m0 - m1) ** 2
        if var > best:
            best, thresh = var, t
    return thresh


def estimate_skew(gray: np.ndarray, max_angle: float = 5.0) -> float:
    """Skew angle (degrees, PIL convention) maximising the row-profile sharpness."""
    img = Image.fromarray(gray)
    scale = 1000 / max(img.size)
    small = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    arr = np.asarray(small)
    ink = Image.fromarray(((arr < _otsu(arr)) * 255).astype(np.uint8))

    def score(a: float) -> float:
        prof = np.asarray(ink.rotate(a, resample=Image.NEAREST, fillcolor=0)).sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(prof) ** 2))

    best = max(np.arange(-max_angle, max_angle + 1e-6, 0.5), key=score)
    return float(max(np.arange(best - 0.5, best + 0.5 + 1e-6, 0.1), key=score))


def _runs(mask_1d: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    """[start, end] index ranges of True values, bridging gaps <= max_gap."""
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            out.append((int(start), int(prev)))
            start = i
        prev = i
    out.append((int(start), int(prev)))
    return out


class _Grid:
    """A ruled table: horizontal rule y's, vertical rule x's, merged-cell groups."""

    def __init__(self, ys, xs, groups):
        self.ys, self.xs, self.groups = ys, xs, groups  # groups: list of (cells, bbox)

    @property
    def bbox(self):
        return self.xs[0], self.ys[0], self.xs[-1], self.ys[-1]


def detect_ruled_tables(ink: np.ndarray, min_rule_ratio: float = 0.04) -> list[_Grid]:
    """Find ruled tables in a binary ink mask (True = dark)."""
    h, w = ink.shape
    L = max(20, int(w * min_rule_ratio))
    # -- horizontal rules: pixels belonging to a dark horizontal run of length >= L
    cs = np.cumsum(np.pad(ink, ((0, 0), (1, 0))), axis=1, dtype=np.int32)
    win = (cs[:, L:] - cs[:, :-L]) >= int(L * 0.9)          # window starting at x is (almost) all ink
    rows_with_line = win.any(axis=1)
    max_thick = max(6, int(h * 0.006))       # rules are thin; bold text lines are not
    min_len = int(w * 0.10)
    rules = []
    for y0, y1 in _runs(rows_with_line, max_gap=1):
        if y1 - y0 + 1 > max_thick:
            continue
        cols = win[y0:y1 + 1].any(axis=0)
        for x0, x1 in _runs(cols, max_gap=4):
            if x1 + L - x0 >= min_len:
                rules.append([(y0 + y1) / 2, x0, x1 + L, y0, y1])
    rules.sort()
    # -- group rules with overlapping x extent into table candidates
    groups, cur = [], []
    for r in rules:
        if cur:
            p = cur[-1]
            overlap = min(p[2], r[2]) - max(p[1], r[1])
            if overlap >= 0.7 * min(p[2] - p[1], r[2] - r[1]) and r[0] - p[0] <= 0.25 * h:
                cur.append(r)
                continue
            groups.append(cur)
        cur = [r]
    if cur:
        groups.append(cur)

    grids = []
    for g in groups:
        if len(g) < 2:
            continue
        x0, x1 = int(min(r[1] for r in g)), int(max(r[2] for r in g))
        ys = [r[0] for r in g]
        thick = [(r[3], r[4]) for r in g]
        bands = [(int(thick[i][1]) + 3, int(thick[i + 1][0]) - 3) for i in range(len(g) - 1)]
        if any(b <= a for a, b in bands):
            continue
        # vertical rule present in a band = (nearly) every pixel row of the band is dark there
        region = ink[:, max(0, x0 - 3):min(w, x1 + 4)]
        dil = region.copy()
        dil[:, 1:] |= region[:, :-1]
        dil[:, :-1] |= region[:, 1:]
        present = np.stack([dil[a:b + 1].mean(axis=0) >= 0.85 for a, b in bands])
        xs_runs = _runs(present.any(axis=0), max_gap=3)
        xs = [max(0, x0 - 3) + (a + b) / 2 for a, b in xs_runs]
        if len(xs) < 3:  # need left + right border + at least one inner rule
            continue
        col_present = [[bool(present[bi, a:b + 1].any()) for a, b in xs_runs] for bi in range(len(bands))]
        # a table band must be closed by the left and right border; split the grid elsewhere
        bi = 0
        while bi < len(bands):
            if not (col_present[bi][0] and col_present[bi][-1]):
                bi += 1
                continue
            bj = bi
            while bj + 1 < len(bands) and col_present[bj + 1][0] and col_present[bj + 1][-1]:
                bj += 1
            grids.append(_build_grid(ink, ys[bi:bj + 2], xs, thick[bi:bj + 2], col_present[bi:bj + 1]))
            bi = bj + 1
    return grids


def _build_grid(ink, ys, xs, thick, col_present) -> _Grid:
    nb, nc = len(ys) - 1, len(xs) - 1
    parent = list(range(nb * nc))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    for b in range(nb):
        for c in range(nc - 1):
            if not col_present[b][c + 1]:              # no vertical rule -> colspan
                union(b * nc + c, b * nc + c + 1)
    for b in range(nb - 1):
        ya, yb = int(thick[b + 1][0]) - 1, int(thick[b + 1][1]) + 1
        for c in range(nc):
            xa, xb = int(xs[c]) + 4, int(xs[c + 1]) - 4
            if xb <= xa:
                continue
            seg = ink[max(0, ya):yb + 1, xa:xb + 1].any(axis=0).mean()
            if seg < 0.8:                               # horizontal rule missing -> rowspan
                union(b * nc + c, (b + 1) * nc + c)
    members: dict[int, list[tuple[int, int]]] = {}
    for b in range(nb):
        for c in range(nc):
            members.setdefault(find(b * nc + c), []).append((b, c))
    groups = []
    for cells in members.values():
        bs, cs = [b for b, _ in cells], [c for _, c in cells]
        groups.append((cells, (xs[min(cs)], ys[min(bs)], xs[max(cs) + 1], ys[max(bs) + 1])))
    return _Grid(ys, xs, groups)


class TesseractBackend(OCRBackend):
    name = "tesseract"

    def __init__(self, lang: str = "vie+eng", fix_orientation: bool = True, deskew: bool = True,
                 min_conf: float = 30.0):
        import pytesseract

        self.tess = pytesseract
        try:
            available = set(pytesseract.get_languages(config=""))
        except Exception as e:
            raise RuntimeError(f"Tesseract not usable ({e}). Install tesseract-ocr + tesseract-ocr-vie.") from e
        langs = [l for l in lang.split("+") if l in available]
        if not langs:
            raise RuntimeError(f"No Tesseract language data for {lang!r} (installed: {sorted(available)})")
        if len(langs) < len(lang.split("+")):
            log.warning("Tesseract: missing language data, using %s", "+".join(langs))
        self.lang = "+".join(langs)
        self.fix_orientation, self.deskew, self.min_conf = fix_orientation, deskew, min_conf

    # ---------------------------------------------------------------- preprocessing
    def preprocess(self, image: Image.Image) -> Image.Image:
        gray = image.convert("L")
        if self.fix_orientation:
            try:
                osd = self.tess.image_to_osd(gray, config="--psm 0 -c min_characters_to_try=10")
                rot = int(re.search(r"Rotate:\s+(\d+)", osd).group(1))
                conf = float(re.search(r"Orientation confidence:\s+([\d.]+)", osd).group(1))
                if rot and conf >= 2.0:
                    gray = gray.rotate(-rot, expand=True, fillcolor=255)
            except Exception:  # OSD fails on pages with little text: keep as is
                pass
        if self.deskew:
            angle = estimate_skew(np.asarray(gray))
            if abs(angle) >= 0.1:
                gray = gray.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
        return gray

    # ---------------------------------------------------------------- recognition
    def _ocr_cell(self, gray: Image.Image, box) -> str:
        x0, y0, x1, y1 = (int(v) for v in box)
        pad = 4
        crop = gray.crop((x0 + pad, y0 + pad, x1 - pad, y1 - pad))
        if crop.width < 5 or crop.height < 5:
            return ""
        canvas = Image.new("L", (crop.width + 40, crop.height + 40), 255)
        canvas.paste(crop, (20, 20))
        text = self.tess.image_to_string(canvas, lang=self.lang, config="--psm 6")
        return join_lines(_clean_ocr_text(text).splitlines())

    def _lines(self, gray: Image.Image) -> list[dict]:
        """Tesseract words grouped into text lines (reading order), noise filtered out."""
        d = self.tess.image_to_data(gray, lang=self.lang, config="--psm 3", output_type=self.tess.Output.DICT)
        lines: dict[tuple, dict] = {}
        for i, word in enumerate(d["text"]):
            word = word.strip()
            conf = float(d["conf"][i])
            if not word or conf < 0:
                continue
            if conf < self.min_conf and not re.search(r"\w{2,}", word):
                continue  # isolated noise speckles
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            ln = lines.setdefault(key, {"block": d["block_num"][i], "words": [], "box": [1e9, 1e9, 0, 0], "hs": []})
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            ln["words"].append((x, word))
            ln["box"] = [min(ln["box"][0], x), min(ln["box"][1], y), max(ln["box"][2], x + w), max(ln["box"][3], y + h)]
            ln["hs"].append(h)
        out = []
        for ln in lines.values():
            text = _clean_ocr_text(" ".join(w for _, w in sorted(ln["words"])))
            # a line made only of 1-char tokens is accent marks / noise read as a separate line
            if not any(len(re.findall(r"\w", tok)) >= 2 for tok in text.split()):
                continue
            ln["text"], ln["h"] = text, float(np.median(ln["hs"]))
            out.append(ln)
        return out

    def _paragraphs(self, gray: Image.Image, page_no: int) -> list[Block]:
        """Rebuild paragraphs from line geometry (Tesseract's own paragraphing is unreliable)."""
        lines = self._lines(gray)
        if not lines:
            return []
        body_h = float(np.median([ln["h"] for ln in lines]))
        H = gray.height
        groups: list[list[dict]] = []
        for ln in lines:
            if groups:
                prev = groups[-1][-1]
                gap = ln["box"][1] - prev["box"][3]
                same_size = max(ln["h"], prev["h"]) <= 1.2 * min(ln["h"], prev["h"])
                if ln["block"] == prev["block"] and -body_h < gap <= 0.6 * max(ln["h"], prev["h"]) and same_size:
                    groups[-1].append(ln)
                    continue
            groups.append([ln])
        blocks = []
        for g in groups:
            text = join_lines([ln["text"] for ln in g])
            box = (min(l["box"][0] for l in g), min(l["box"][1] for l in g),
                   max(l["box"][2] for l in g), max(l["box"][3] for l in g))
            in_margin = box[3] < H * 0.07 or box[1] > H * 0.93
            if in_margin and _PAGE_NO_RE.match(text):
                continue
            big = np.median([l["h"] for l in g]) >= 1.2 * body_h
            short = len(g) <= 2 and len(text) <= 80 and not text.rstrip().endswith((".", ",", ";"))
            kind = TITLE if (big and len(g) <= 3) or (len(g) == 1 and short) else TEXT
            blocks.append(Block(kind, page_no, DOC_TYPE, text=text, bbox=box))
        return blocks

    def recognize(self, image: Image.Image, page_no: int) -> list[Block]:
        gray = self.preprocess(image)
        arr = np.asarray(gray)
        ink = arr < min(_otsu(arr), 160)
        items: list[Block] = []
        masked = arr.copy()
        for grid in detect_ruled_tables(ink):
            nb, nc = len(grid.ys) - 1, len(grid.xs) - 1
            rows = [["" for _ in range(nc)] for _ in range(nb)]
            for cells, box in grid.groups:
                text = self._ocr_cell(gray, box)
                for b, c in cells:
                    rows[b][c] = text
            rows = [r for r in rows if any(r)]
            if rows:
                items.append(Block(TABLE, page_no, DOC_TYPE, rows=rows, bbox=grid.bbox))
            x0, y0, x1, y1 = (int(v) for v in grid.bbox)
            masked[max(0, y0 - 6):y1 + 7, max(0, x0 - 6):x1 + 7] = 255
        items += self._paragraphs(Image.fromarray(masked), page_no)
        items.sort(key=lambda b: (b.bbox[1], b.bbox[0]) if b.bbox else (0, 0))
        return items


def _clean_ocr_text(text: str) -> str:
    text = text.replace("|", " ").replace("——", " ")
    return "\n".join(re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip())


# =============================================================================== backend selection
_BACKENDS: dict[str, OCRBackend] = {}
_FAILED: dict[str, Exception] = {}  # a backend that failed to load is not retried (re-import is unsafe)


def _paddle_gpu_available() -> bool:
    try:
        import importlib.util

        if importlib.util.find_spec("paddleocr") is None or importlib.util.find_spec("paddle") is None:
            return False
        import paddle

        return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        return False


def get_backend(name: str = "auto", **kwargs) -> OCRBackend | None:
    """Return a (cached) backend instance. "auto" prefers PaddleOCR-VL on GPU, else Tesseract."""
    name = (name or "auto").lower()
    if name == "none":
        return None
    if name in _BACKENDS:
        return _BACKENDS[name]
    if name in _FAILED:
        raise RuntimeError(f"OCR backend {name!r} failed to load earlier: {_FAILED[name]}") from _FAILED[name]
    if name == "auto":
        if _paddle_gpu_available():
            try:
                backend = _BACKENDS["auto"] = get_backend("paddle_vl", **kwargs)
                return backend
            except Exception as e:
                log.warning("PaddleOCR-VL unavailable (%s) -> falling back to Tesseract", e)
        else:
            log.info("PaddleOCR-VL/GPU not available -> using Tesseract")
        backend = _BACKENDS["auto"] = get_backend("tesseract", **kwargs)
        return backend
    factories = {"paddle_vl": PaddleOCRVLBackend, "tesseract": TesseractBackend}
    if name not in factories:
        raise ValueError(f"unknown OCR backend {name!r} (auto|paddle_vl|tesseract|none)")
    try:
        backend = factories[name](**kwargs)
    except Exception as e:
        _FAILED[name] = e
        raise
    _BACKENDS[name] = backend
    return backend


def ocr_pages(doc: pymupdf.Document, page_indexes: list[int], backend: str | OCRBackend = "auto",
              dpi: int = 300) -> dict[int, list[Block]]:
    """OCR the given 0-based pages of an open PDF -> {page_index: blocks in reading order}."""
    engine = backend if isinstance(backend, OCRBackend) else get_backend(backend)
    if engine is None:
        log.warning("%d scanned page(s) skipped (OCR backend 'none')", len(page_indexes))
        return {}
    out = {}
    for i in page_indexes:
        img = render_page(doc[i], dpi=dpi)
        out[i] = engine.recognize(img, i + 1)
        log.debug("page %d: %d block(s) via %s", i + 1, len(out[i]), engine.name)
    return out
