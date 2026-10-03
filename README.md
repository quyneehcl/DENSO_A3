# A3 — Giai đoạn ingest: PDF + Excel → JSON trung gian

Đọc tài liệu PDF (có text layer hoặc scan) và Excel, chuyển thành các **chunk JSON theo format trung gian chuẩn**
để dùng cho các bước sau (embedding, retrieval, chatbot — chưa thuộc phạm vi bản build này).
Chạy hoàn toàn local/on-prem, không gọi API cloud.

## Format đầu ra (bắt buộc)

Mỗi chunk có đúng 6 field, không thêm/bớt:

```json
{
  "chunk_id": "doc001_chunk003",
  "text": "Tên linh kiện là Động cơ servo, mã lỗi là E10, nguyên nhân là Quá tải động cơ, cách xử lý là ...",
  "source_file": "bang_ma_loi.xlsx",
  "location": "Máy ép, hàng 6",
  "doc_type": "excel_table",
  "language": "vi"
}
```

| doc_type      | Nguồn                          | location ví dụ                               |
|---------------|--------------------------------|----------------------------------------------|
| `pdf_text`    | trang PDF có text layer        | `Trang 3`, `Trang 2-3`, `Trang 2, bảng 1, hàng 4` |
| `pdf_scan`    | trang PDF scan / ảnh (qua OCR) | như trên                                     |
| `excel_table` | Excel / CSV                    | `Sheet1, hàng 5` (số hàng thật trong Excel)  |

`language`: `vi` / `en` / `ja` (đoán theo ký tự, không cần thư viện ngoài).

## Cấu trúc

```
data/raw/                 tài liệu gốc (6 file mẫu tự sinh) + ground_truth/ để đánh giá
data/processed/           <tên file>.json, all_chunks.jsonl, _manifest.json, _report.json, _evaluation.json
src/ingest/
  schema.py               format chunk, kiểm tra hợp lệ, đoán ngôn ngữ
  table_utils.py          hàng bảng -> câu có cấu trúc, header 2 tầng, parse bảng HTML/Markdown
  excel_reader.py         pandas (multi-sheet) + openpyxl khi có ô gộp
  pdf_reader.py           phân loại từng trang digital/scan, PyMuPDF cho trang digital
  ocr_pipeline.py         PaddleOCR-VL (chính) / Tesseract (dự phòng) cho trang scan
  blocks.py               ghép đoạn/bảng bị ngắt giữa trang, chia chunk
  pipeline.py             ingest_file(): chọn reader theo loại file
scripts/
  make_sample_data.py     sinh bộ dữ liệu test + ground truth
  run_ingest.py           chạy toàn bộ data/raw -> data/processed, đo thời gian
  evaluate.py             so output với ground truth (độ chính xác text, bảng, thời gian)
tests/                    pytest + sample_outputs/ (JSON mẫu đúng format)
```

## Cài đặt (máy local / on-prem)

```bash
git clone https://github.com/quyneehcl/DENSO_A3.git && cd DENSO_A3
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

OCR dự phòng (Tesseract, chạy CPU):
- Ubuntu/Debian: `sudo apt install tesseract-ocr tesseract-ocr-vie`
- Windows: cài bản UB-Mannheim, tick thêm ngôn ngữ **Vietnamese**, thêm thư mục cài vào `PATH`.

OCR chính (PaddleOCR-VL, cần GPU ~16–24GB VRAM khuyến nghị):
```bash
# cài paddlepaddle-gpu đúng với bản CUDA của máy (xem https://www.paddlepaddle.org.cn/install), ví dụ CUDA 12.6:
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install -r requirements-paddle.txt
```
Lần chạy đầu PaddleOCR-VL tự tải model (~2GB). Trên máy không có mạng: tải model trước rồi copy vào cache của PaddleX.

## Chạy

```bash
python scripts/make_sample_data.py          # (tuỳ chọn) sinh lại 6 file mẫu vào data/raw/
python scripts/run_ingest.py                # mọi PDF/Excel trong data/raw/ -> data/processed/
python scripts/evaluate.py                  # độ chính xác + thời gian (cần ground truth)
pytest tests/
```

Tham số `run_ingest.py`:
- `--ocr-backend auto|paddle_vl|tesseract|none` — `auto` (mặc định): dùng PaddleOCR-VL nếu đã cài và có GPU, ngược lại dùng Tesseract.
- `--dpi 300` — độ phân giải render trang scan. `--max-chars 800` — độ dài tối đa một chunk văn bản.
- `--input`, `--output` — đổi thư mục. Tài liệu thật: bỏ vào `data/raw/` (hoặc thư mục riêng qua `--input`) rồi chạy lại.
  ⚠️ Không commit tài liệu thật / output của tài liệu thật lên git — nên dùng thư mục ngoài repo, ví dụ `--input /data/a3/raw --output /data/a3/processed`.
- File `.xls` (Excel cũ) cần thêm `pip install xlrd`.

`doc_id` (`doc001`, `doc002`, ...) được lưu trong `data/processed/_manifest.json` nên giữ nguyên giữa các lần chạy.

Dùng trong code:
```python
import sys; sys.path.insert(0, "src")
from ingest.pipeline import ingest_file
chunks = ingest_file("data/raw/bang_ma_loi.xlsx", doc_id="doc001")
print(chunks[0].to_dict())
```

## Chạy thử PaddleOCR-VL trên Google Colab

> ⚠️ Colab là cloud bên ngoài. **Chỉ dùng dữ liệu mẫu / không mật**, không đưa tài liệu thật của công ty lên Colab.

Runtime → Change runtime type → **T4 GPU**, rồi chạy:
```python
!git clone -b claude/wizardly-dirac-9lj4gz https://github.com/quyneehcl/DENSO_A3.git
%cd DENSO_A3
!apt -qq install -y tesseract-ocr tesseract-ocr-vie
!pip install -q -r requirements.txt
!pip install -q paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
!pip install -q -r requirements-paddle.txt
!python scripts/run_ingest.py --ocr-backend paddle_vl
!python scripts/evaluate.py
```
Repo private thì cần token GitHub để clone. Session Colab tắt là mất file — tải `data/processed/` về nếu cần.
Lệnh cài paddlepaddle-gpu có thể thay đổi theo phiên bản; nếu lỗi, xem hướng dẫn cài trên trang PaddlePaddle.

## Xử lý chi tiết

**Excel** — mỗi hàng dữ liệu = 1 chunk, dạng câu `"<cột 1> là <giá trị 1>, <cột 2> là <giá trị 2>, ..."`
(nếu giá trị có dấu phẩy thì phân cách bằng `;` để không lẫn cột). Hỗ trợ: nhiều sheet; dòng tiêu đề phía trên bảng;
ô gộp (đọc bằng openpyxl, giá trị ô gộp được điền cho mọi ô trong vùng); header 2 tầng
(`Thông số điện` + `Điện áp (V)` → `Thông số điện - Điện áp (V)`); số/ngày được format gọn (`22`, `2021-05-14`).
Ô có công thức lấy giá trị đã tính sẵn mà Excel lưu trong file. File tạo bằng code và chưa từng mở bằng Excel sẽ không có giá trị này, nên ô đó bị coi là rỗng.

**PDF** — phân loại **từng trang** (ít hơn 20 ký tự chọn được → trang scan), nên PDF lẫn trang scan vẫn xử lý được.
- Trang digital: PyMuPDF lấy block văn bản theo thứ tự đọc, phát hiện tiêu đề (cỡ chữ/đậm), bỏ số trang và
  header/footer lặp lại; bảng (`find_tables`) → mỗi hàng 1 chunk, ô gộp được điền lại.
- Trang scan: render ảnh → OCR (xem dưới) → cùng logic chia chunk.
- Đoạn văn bị ngắt giữa 2 trang được nối lại (`Trang 2-3`); bảng kéo dài sang trang sau được nối thành 1 bảng,
  header lặp lại ở trang sau bị bỏ, số hàng đánh tiếp.
- Văn bản: tiêu đề + các đoạn của cùng trang gom thành chunk ≤ 800 ký tự.

**OCR**
- `paddle_vl`: PaddleOCR-VL (PP-DocLayoutV2 phân tích layout + PaddleOCR-VL-0.9B đọc từng vùng), có xoay đúng hướng;
  bỏ header/footer/số trang/ảnh; bảng (HTML) → hàng.
- `tesseract` (dự phòng, CPU): xoay hướng (OSD), chỉnh nghiêng, phát hiện **bảng có kẻ ô** bằng phân tích đường kẻ
  (hỗ trợ ô gộp ngang/dọc) rồi OCR từng ô, phần văn bản còn lại ghép đoạn theo khoảng cách dòng.
  Hạn chế: bảng không kẻ ô sẽ ra dạng văn bản; tiêu đề in hoa đậm hay mất dấu.

## Kết quả trên bộ mẫu

| Loại | Đoạn văn tìm thấy | Hàng bảng đúng chính xác | Thời gian TB/file |
|------|-------------------|--------------------------|-------------------|
| PDF digital (2 file) | 20/20 | 18/18 | ~0.07 s |
| Excel (2 file)       | –     | 17/17 | ~0.05 s |
| PDF scan – Tesseract (CPU)          | 13/14 | 7/8 | ~10 s |
| PDF scan – PaddleOCR-VL 1.6 (Colab T4) | 11/14 | 4/8 | ~22 s |

Trên bộ scan mẫu (ảnh tự làm nhiễu/mờ/nén JPEG):
- PaddleOCR-VL tách bố cục (tiêu đề / đoạn / bảng) chuẩn, nhưng **đọc sai dấu tiếng Việt** nhiều hơn Tesseract
  (`CỐ`→`CÓ`, `dừng`→`dùng`, `Ngưỡng`→`Nguồng`...). Một chữ sai trong header bảng làm cả 4 hàng của bảng đó không khớp.
- Tesseract: tiêu đề in hoa đậm hay mất dấu (`BIEN BAN SỰ CO`), `45 A` → `45A`; không đọc được bảng không kẻ ô.
- Bộ scan mẫu là giả lập → cần so 2 backend trên vài trang scan thật (trên máy nội bộ) trước khi chọn.

Ghi chú khi chạy PaddleOCR-VL trên Colab:
- Cài `paddlepaddle-gpu` có thể làm hỏng PyTorch của Colab (`undefined symbol: ncclCommWindowDeregister`):
  chạy `pip install -U "nvidia-nccl-cu12>=2.27"` rồi **Runtime → Restart session**.
- Không nạp model trong notebook rồi lại chạy `!python scripts/run_ingest.py` — hai bản model không vừa GPU 16GB
  (`Out of memory`). Restart session trước khi chạy script.

Dữ liệu mẫu là nội dung kỹ thuật **tự tạo** (không phải tài liệu thật). Cần thay bằng tài liệu thật để đánh giá chính xác.
