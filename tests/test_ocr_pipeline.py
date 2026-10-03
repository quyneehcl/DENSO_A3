import shutil
from pathlib import Path

import numpy as np
import pytest

from ingest import ocr_pipeline
from ingest.blocks import TABLE, TITLE, Block, blocks_to_chunks, merge_across_pages
from ingest.ocr_pipeline import PaddleOCRVLBackend, detect_ruled_tables
from ingest.table_utils import markdown_table_to_rows

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HAS_TESS = shutil.which("tesseract") is not None


def _draw_grid(h=600, w=800):
    """3 columns x 4 rows; cell (rows 1-2, col 0) merged vertically."""
    ink = np.zeros((h, w), dtype=bool)
    ys, xs = [100, 160, 220, 280, 340], [100, 300, 500, 700]
    for y in ys:
        ink[y:y + 3, 100:703] = True
    ink[220:223, 100:300] = False          # missing rule -> rowspan in column 0
    for x in xs:
        ink[100:343, x:x + 3] = True
    ink[400:420, 150:600] = True           # thick bar (bold text) must not be a rule
    return ink


def test_detect_ruled_table_with_rowspan():
    grids = detect_ruled_tables(_draw_grid())
    assert len(grids) == 1
    g = grids[0]
    assert len(g.ys) == 5 and len(g.xs) == 4
    sizes = sorted(len(cells) for cells, _ in g.groups)
    assert sizes == [1] * 10 + [2]          # 12 cells, two of them merged


def test_markdown_table():
    md = "| Mã | Mô tả |\n|---|---|\n| E1 | Quá nhiệt |"
    assert markdown_table_to_rows(md) == [[["Mã", "Mô tả"], ["E1", "Quá nhiệt"]]]


class _FakeVL:
    """Mimics PaddleOCRVL.predict() output (parsing_res_list)."""

    def predict(self, path):
        return [{"res": {"parsing_res_list": [
            {"block_label": "header", "block_content": "DENSO - nội bộ", "block_bbox": [0, 0, 10, 10]},
            {"block_label": "doc_title", "block_content": "# BIÊN BẢN SỰ CỐ", "block_bbox": [0, 20, 10, 30]},
            {"block_label": "text", "block_content": "Máy CNC số 3 dừng đột ngột\nlúc 9 giờ 40.", "block_bbox": [0, 40, 10, 50]},
            {"block_label": "table", "block_bbox": [0, 60, 10, 90], "block_content":
                "<table><tr><td>Thời gian</td><td>Hành động</td></tr><tr><td>09:45</td><td>Ngắt nguồn</td></tr></table>"},
            {"block_label": "number", "block_content": "1", "block_bbox": [0, 95, 10, 99]},
        ]}}]


class _FakeVLTwoPages:
    """Real PaddleOCR-VL 1.6 output: a lone paragraph on page 2 labelled 'aside_text'."""

    def predict(self, path):
        if path.endswith("page_1.png"):
            items = [{"block_label": "text", "block_content": "Áp suất tối đa 8 bar."},
                     {"block_label": "paragraph_title", "block_content": "Dừng máy"}]
        else:
            items = [{"block_label": "aside_text", "block_content": "Nhấn nút dừng, máy chạy không tải 30 giây."}]
        return [{"res": {"parsing_res_list": items}}]


def test_paddle_vl_result_parsing():
    backend = PaddleOCRVLBackend.__new__(PaddleOCRVLBackend)
    backend.pipeline = _FakeVL()
    from PIL import Image

    blocks = backend.recognize(Image.new("RGB", (20, 20), "white"), page_no=1)
    assert [b.kind for b in blocks] == [TITLE, "text", TABLE]
    chunks = blocks_to_chunks(merge_across_pages(blocks), "doc005", "scan.pdf")
    assert chunks[0].text == "BIÊN BẢN SỰ CỐ\nMáy CNC số 3 dừng đột ngột lúc 9 giờ 40."
    assert chunks[1].text == "Thời gian là 09:45, hành động là Ngắt nguồn."
    assert chunks[1].location == "Trang 1, bảng 1, hàng 1"
    assert all(c.doc_type == "pdf_scan" for c in chunks)


def test_cross_page_table_merge_drops_repeated_header():
    head = ["Mã", "Mô tả"]
    blocks = [Block(TABLE, 1, "pdf_scan", rows=[head, ["E1", "a"]]),
              Block(TABLE, 2, "pdf_scan", rows=[head, ["E2", "b"]])]
    merged = merge_across_pages(blocks)
    assert len(merged) == 1 and merged[0].rows == [head, ["E1", "a"], ["E2", "b"]]
    assert merged[0].row_pages == [1, 1, 2]


@pytest.mark.skipif(not HAS_TESS or not (RAW / "scan_huong_dan_van_hanh.pdf").exists(),
                    reason="needs tesseract + sample data")
def test_tesseract_scan_end_to_end():
    from ingest.pdf_reader import read_pdf

    chunks = read_pdf(RAW / "scan_huong_dan_van_hanh.pdf", doc_id="doc004", ocr_backend="tesseract")
    assert all(c.doc_type == "pdf_scan" for c in chunks)
    rows = [c for c in chunks if ", bảng " in c.location]
    assert len(rows) == 4
    assert rows[0].text == ("Thông số là Áp suất đầu ra, giá trị bình thường là 7 bar, "
                            "ngưỡng cảnh báo là Trên 8 bar.")
    # paragraph title at the bottom of page 1 + its text on page 2
    assert any(c.location == "Trang 1-2" and c.text.startswith("Dừng máy") for c in chunks)


def test_backend_none_skips_scans():
    assert ocr_pipeline.get_backend("none") is None


def test_failed_backend_is_not_reloaded(monkeypatch):
    calls = []

    def boom(**kw):
        calls.append(1)
        raise ImportError("libtorch_cuda.so: undefined symbol")

    monkeypatch.setattr(ocr_pipeline, "PaddleOCRVLBackend", boom)
    monkeypatch.setattr(ocr_pipeline, "_FAILED", {})
    monkeypatch.setattr(ocr_pipeline, "_BACKENDS", {})
    with pytest.raises(ImportError):
        ocr_pipeline.get_backend("paddle_vl")
    with pytest.raises(RuntimeError, match="undefined symbol"):
        ocr_pipeline.get_backend("paddle_vl")
    assert len(calls) == 1


def test_paddle_vl_keeps_aside_text_and_carries_title():
    from PIL import Image

    backend = PaddleOCRVLBackend.__new__(PaddleOCRVLBackend)
    backend.pipeline = _FakeVLTwoPages()
    img = Image.new("RGB", (20, 20), "white")
    blocks = backend.recognize(img, 1) + backend.recognize(img, 2)
    chunks = blocks_to_chunks(merge_across_pages(blocks), "doc005", "scan.pdf")
    assert chunks[-1].text == "Dừng máy\nNhấn nút dừng, máy chạy không tải 30 giây."
    assert chunks[-1].location == "Trang 1-2"
