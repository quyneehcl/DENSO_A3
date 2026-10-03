from ingest.schema import detect_language, make_chunk_id, validate_chunk, validate_chunks
from ingest.table_utils import html_tables_to_rows, row_to_sentence

GOOD = {
    "chunk_id": "doc001_chunk003",
    "text": "Nội dung đoạn đã xử lý",
    "source_file": "ten_file_goc.xlsx",
    "location": "Sheet1, hàng 5",
    "doc_type": "excel_table",
    "language": "vi",
}


def test_valid_chunk():
    assert validate_chunk(GOOD) == []
    assert make_chunk_id("doc001", 3) == "doc001_chunk003"


def test_invalid_chunks():
    assert validate_chunk({**GOOD, "page": 1})
    assert validate_chunk({k: v for k, v in GOOD.items() if k != "language"})
    assert validate_chunk({**GOOD, "doc_type": "word"})
    assert validate_chunk({**GOOD, "text": "  "})
    assert validate_chunks([GOOD, GOOD])  # duplicate id


def test_language():
    assert detect_language("Kiểm tra áp suất dầu") == "vi"
    assert detect_language("Check the oil pressure") == "en"
    assert detect_language("油圧を確認してください") == "ja"


def test_row_to_sentence():
    s = row_to_sentence(["Tên linh kiện", "Mã lỗi", "Cách xử lý"], ["Bơm", "E05", None])
    assert s == "Tên linh kiện là Bơm, mã lỗi là E05."
    assert row_to_sentence(["PLC", "IP"], ["S7", "10.0.0.1"]) == "PLC là S7, IP là 10.0.0.1."


def test_html_table_spans():
    html = ("<table><tr><td rowspan=2>Bơm</td><td>E1</td></tr><tr><td>E2</td></tr>"
            "<tr><td colspan=2>Ghi chú</td></tr></table>")
    assert html_tables_to_rows(html) == [[["Bơm", "E1"], ["Bơm", "E2"], ["Ghi chú", "Ghi chú"]]]
