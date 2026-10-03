from pathlib import Path

import pymupdf
import pytest

from ingest.pdf_reader import classify_pdf, read_pdf
from ingest.schema import validate_chunks

FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def _make_pdf(path, pages):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_font(fontname="vn", fontfile=str(FONT))
            page.insert_textbox(pymupdf.Rect(60, 60, 540, 780), text, fontname="vn", fontsize=11)
    doc.save(path)


@pytest.mark.skipif(not FONT.exists(), reason="DejaVu font not installed")
def test_classify_and_cross_page_paragraph(tmp_path):
    p = tmp_path / "a.pdf"
    _make_pdf(p, ["Kiểm tra mức dầu thủy lực trước khi khởi động máy và",
                  "ghi kết quả vào sổ theo dõi.", ""])
    assert classify_pdf(p) == ["pdf_text", "pdf_text", "pdf_scan"]
    chunks = [c.to_dict() for c in read_pdf(p, doc_id="doc009", ocr_backend="none")]
    assert validate_chunks(chunks) == []
    assert len(chunks) == 1
    assert chunks[0]["text"] == ("Kiểm tra mức dầu thủy lực trước khi khởi động máy và "
                                 "ghi kết quả vào sổ theo dõi.")
    assert chunks[0]["location"] == "Trang 1-2"
    assert chunks[0]["doc_type"] == "pdf_text"


@pytest.mark.skipif(not (RAW / "huong_dan_bao_tri.pdf").exists(), reason="run scripts/make_sample_data.py")
def test_table_split_across_pages():
    chunks = read_pdf(RAW / "huong_dan_bao_tri.pdf", doc_id="doc001")
    rows = [c for c in chunks if ", bảng " in c.location]
    assert len(rows) == 12  # repeated header on page 2 not counted as a row
    assert rows[0].text == ("Hạng mục là Kiểm tra mức dầu thủy lực, chu kỳ là Hàng ngày, "
                            "người thực hiện là Công nhân vận hành, ghi chú là Bổ sung dầu ISO VG46.")
    assert rows[3].location == "Trang 2, bảng 1, hàng 4"
    assert any(c.location == "Trang 2-3" and "xác nhận an toàn" in c.text for c in chunks)


@pytest.mark.skipif(not (RAW / "quy_trinh_kiem_tra.pdf").exists(), reason="run scripts/make_sample_data.py")
def test_merged_cells_in_pdf_table():
    rows = [c.text for c in read_pdf(RAW / "quy_trinh_kiem_tra.pdf") if ", bảng " in c.location]
    assert rows[2] == "Thiết bị là Máy hàn reflow, hạng mục kiểm tra là Tốc độ băng tải, tiêu chuẩn là 0.9 m/phút."
