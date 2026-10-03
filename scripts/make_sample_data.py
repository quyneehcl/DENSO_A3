"""Generate the sample test set in data/raw/ (+ ground truth in data/raw/ground_truth/):

  * 2 digital PDFs (text layer)     huong_dan_bao_tri.pdf, quy_trinh_kiem_tra.pdf
  * 2 scanned PDFs (image only)     scan_bien_ban_su_co.pdf, scan_huong_dan_van_hanh.pdf
  * 2 Excel files                   bang_ma_loi.xlsx, thong_so_may.xlsx

The content is synthetic (no real DENSO data). Ground truth = the exact paragraphs and
table rows written into each file, used by scripts/evaluate.py.

Usage: python scripts/make_sample_data.py [--out data/raw]
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import random
from pathlib import Path

import numpy as np
import pymupdf
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, A5
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "C:/Windows/Fonts", "/Library/Fonts"]


def _register_fonts():
    for d in FONT_DIRS:
        reg, bold = Path(d) / "DejaVuSans.ttf", Path(d) / "DejaVuSans-Bold.ttf"
        if reg.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("VN", str(reg)))
            pdfmetrics.registerFont(TTFont("VN-Bold", str(bold)))
            return
    raise SystemExit("Cần font DejaVuSans (hỗ trợ tiếng Việt): apt install fonts-dejavu-core")


# --------------------------------------------------------------------------- content
# Each document = list of ("h1"|"h2"|"p", text) or ("table", {"headers", "rows", "spans"}).
# "spans": list of (col, first_row, last_row) vertical merges, indices in data rows.

DOC_BAO_TRI = [
    ("h1", "HƯỚNG DẪN BẢO TRÌ MÁY ÉP NHỰA MX-200"),
    ("p", "Tài liệu này hướng dẫn quy trình bảo trì định kỳ cho máy ép nhựa MX-200 đang vận hành tại "
          "xưởng sản xuất số 2. Người thực hiện bảo trì phải được đào tạo và có chứng chỉ vận hành hợp lệ."),
    ("h2", "1. Yêu cầu an toàn"),
    ("p", "Trước khi bảo trì, phải ngắt nguồn điện chính, treo biển cảnh báo và khóa cầu dao theo quy trình "
          "LOTO. Chờ ít nhất 30 phút để nhiệt độ xi lanh giảm xuống dưới 50 độ C trước khi tháo vỏ bảo vệ."),
    ("p", "Luôn mang găng tay chịu nhiệt, kính bảo hộ và giày an toàn. Không được làm việc một mình khi "
          "kiểm tra hệ thống thủy lực có áp suất cao."),
    ("h2", "2. Lịch bảo dưỡng định kỳ"),
    ("p", "Bảng dưới đây liệt kê các hạng mục bảo dưỡng, chu kỳ thực hiện và bộ phận chịu trách nhiệm. "
          "Các hạng mục quá hạn phải được báo cáo cho trưởng ca ngay trong ngày."),
    ("table", {
        "headers": ["Hạng mục", "Chu kỳ", "Người thực hiện", "Ghi chú"],
        "rows": [
            ["Kiểm tra mức dầu thủy lực", "Hàng ngày", "Công nhân vận hành", "Bổ sung dầu ISO VG46"],
            ["Vệ sinh lưới lọc dầu", "Hàng tuần", "Kỹ thuật viên", "Thay nếu rách"],
            ["Bôi trơn thanh dẫn hướng", "Hàng tuần", "Kỹ thuật viên", "Dùng mỡ EP2"],
            ["Kiểm tra cảm biến nhiệt", "Hàng tháng", "Kỹ sư điện", "Sai số cho phép 2 độ C"],
            ["Kiểm tra áp suất kẹp khuôn", "Hàng tháng", "Kỹ sư cơ khí", "Áp suất 140 bar"],
            ["Thay dầu thủy lực", "6 tháng", "Kỹ thuật viên", "Khoảng 200 lít"],
            ["Hiệu chuẩn bộ điều khiển PLC", "12 tháng", "Nhà cung cấp", "Lưu biên bản hiệu chuẩn"],
            ["Kiểm tra vít xoắn và xi lanh", "12 tháng", "Kỹ sư cơ khí", "Đo độ mòn bằng thước cặp"],
            ["Kiểm tra van an toàn", "12 tháng", "Kỹ sư cơ khí", "Thử ở áp suất 160 bar"],
            ["Vệ sinh tủ điện", "3 tháng", "Kỹ sư điện", "Dùng khí nén khô"],
            ["Kiểm tra dây curoa quạt", "3 tháng", "Kỹ thuật viên", "Thay nếu nứt"],
            ["Kiểm tra bơm làm mát", "Hàng tháng", "Kỹ thuật viên", "Lưu lượng tối thiểu 20 lít/phút"],
        ],
    }),
    ("h2", "3. Xử lý dầu thải"),
    ("p", "Dầu thủy lực đã qua sử dụng phải được chứa trong thùng kín có dán nhãn chất thải nguy hại. "
          "Không được đổ dầu thải xuống cống hoặc trộn lẫn với chất thải sinh hoạt. Bộ phận môi trường sẽ "
          "thu gom dầu thải vào thứ sáu hàng tuần và ghi nhận khối lượng vào sổ theo dõi chất thải. Mọi sự cố "
          "rò rỉ dầu trong quá trình thay thế phải được thấm hút ngay bằng vật liệu chuyên dụng, sau đó báo cáo "
          "cho bộ phận an toàn để lập biên bản. Trường hợp dầu tràn ra sàn với diện tích lớn hơn một mét "
          "vuông, phải khoanh vùng và dừng hoạt động của các máy lân cận cho đến khi khu vực được làm sạch hoàn "
          "toàn và được trưởng ca xác nhận an toàn."),
    ("h2", "4. Ghi chép sau bảo trì"),
    ("p", "Sau mỗi lần bảo trì, kỹ thuật viên ghi kết quả vào phiếu bảo trì, bao gồm ngày thực hiện, hạng mục, "
          "linh kiện đã thay và chữ ký xác nhận. Phiếu bảo trì được lưu trữ tối thiểu ba năm."),
]

DOC_KIEM_TRA = [
    ("h1", "QUY TRÌNH KIỂM TRA CHẤT LƯỢNG ĐẦU CA"),
    ("p", "Quy trình áp dụng cho tất cả các dây chuyền lắp ráp linh kiện điện tử. Trưởng ca có trách nhiệm "
          "đảm bảo việc kiểm tra được hoàn thành trong 15 phút đầu tiên của ca làm việc."),
    ("h2", "Bước 1: Kiểm tra thiết bị"),
    ("p", "Kiểm tra theo bảng dưới đây. Nếu bất kỳ hạng mục nào không đạt tiêu chuẩn, dừng dây chuyền và "
          "báo cho bộ phận bảo trì."),
    ("table", {
        "headers": ["Thiết bị", "Hạng mục kiểm tra", "Tiêu chuẩn"],
        "rows": [
            ["Máy hàn reflow", "Nhiệt độ vùng 1", "150 ± 5 độ C"],
            ["Máy hàn reflow", "Nhiệt độ vùng peak", "245 ± 3 độ C"],
            ["Máy hàn reflow", "Tốc độ băng tải", "0.9 m/phút"],
            ["Máy gắp đặt linh kiện", "Áp suất khí nén", "0.5 MPa"],
            ["Máy gắp đặt linh kiện", "Độ chính xác đặt", "Sai lệch dưới 0.05 mm"],
            ["Máy kiểm tra AOI", "Mẫu chuẩn", "Phát hiện đủ 10/10 lỗi mẫu"],
        ],
        "spans": [(0, 0, 2), (0, 3, 4)],
    }),
    ("h2", "Bước 2: Kiểm tra vật tư"),
    ("p", "Đối chiếu mã linh kiện trên cuộn tape với danh sách BOM của sản phẩm đang chạy. Kiểm tra hạn sử "
          "dụng của kem hàn; kem hàn đã mở nắp quá 24 giờ phải được loại bỏ."),
    ("p", "Note: operators must sign the checklist in English and Vietnamese for export products."),
    ("h2", "Bước 3: Sản phẩm mẫu đầu ca"),
    ("p", "Chạy 5 sản phẩm mẫu, kiểm tra ngoại quan và đo điện. Chỉ khi cả 5 sản phẩm đạt yêu cầu mới được "
          "chạy sản xuất hàng loạt. Kết quả được ghi vào phiếu kiểm tra đầu ca."),
]

DOC_SU_CO = [
    ("h1", "BIÊN BẢN SỰ CỐ THIẾT BỊ"),
    ("p", "Ngày 12 tháng 3 năm 2025, lúc 9 giờ 40 phút, máy CNC số 3 tại xưởng gia công dừng đột ngột "
          "và hiển thị mã lỗi AL-214 trên màn hình điều khiển."),
    ("h2", "Mô tả sự cố"),
    ("p", "Người vận hành nghe tiếng kêu bất thường từ trục chính trước khi máy dừng. Kiểm tra sơ bộ cho "
          "thấy nhiệt độ ổ bi trục chính tăng lên 78 độ C, vượt ngưỡng cho phép là 65 độ C."),
    ("h2", "Các bước đã xử lý"),
    ("table", {
        "headers": ["Thời gian", "Hành động", "Người thực hiện"],
        "rows": [
            ["09:45", "Ngắt nguồn và khóa máy", "Trưởng ca"],
            ["10:10", "Kiểm tra ổ bi trục chính", "Kỹ sư cơ khí"],
            ["11:30", "Thay ổ bi và bơm mỡ", "Kỹ thuật viên"],
            ["14:00", "Chạy thử không tải", "Kỹ sư cơ khí"],
        ],
    }),
    ("h2", "Kết luận"),
    ("p", "Nguyên nhân là do thiếu mỡ bôi trơn ổ bi. Đề xuất bổ sung hạng mục kiểm tra mỡ trục chính "
          "vào lịch bảo dưỡng hàng tuần."),
]

DOC_VAN_HANH = [
    ("h1", "HƯỚNG DẪN VẬN HÀNH MÁY NÉN KHÍ"),
    ("p", "Máy nén khí trục vít cung cấp khí nén cho toàn bộ xưởng lắp ráp. Áp suất làm việc tiêu chuẩn "
          "là 7 bar, áp suất tối đa không vượt quá 8 bar."),
    ("h2", "Khởi động máy"),
    ("p", "Kiểm tra mức dầu qua kính thăm dầu, mức dầu phải nằm giữa hai vạch đỏ. Mở van xả nước ngưng "
          "trong bình chứa, sau đó đóng lại. Nhấn nút khởi động và theo dõi áp suất tăng dần trên đồng hồ."),
    ("h2", "Bảng thông số giám sát"),
    ("table", {
        "headers": ["Thông số", "Giá trị bình thường", "Ngưỡng cảnh báo"],
        "rows": [
            ["Áp suất đầu ra", "7 bar", "Trên 8 bar"],
            ["Nhiệt độ dầu", "80 độ C", "Trên 105 độ C"],
            ["Dòng điện động cơ", "45 A", "Trên 52 A"],
            ["Chênh áp lọc tách dầu", "0.3 bar", "Trên 1 bar"],
        ],
    }),
    ("h2", "Dừng máy"),
    ("p", "Nhấn nút dừng, máy sẽ chạy không tải khoảng 30 giây trước khi dừng hẳn. Không ngắt nguồn điện "
          "chính khi máy đang trong chu trình dừng."),
]


def _styles():
    return {
        "h1": ParagraphStyle("h1", fontName="VN-Bold", fontSize=15, leading=20, spaceAfter=8),
        "h2": ParagraphStyle("h2", fontName="VN-Bold", fontSize=12, leading=16, spaceBefore=6, spaceAfter=4),
        "p": ParagraphStyle("p", fontName="VN", fontSize=10.5, leading=15, spaceAfter=6),
        "cell": ParagraphStyle("cell", fontName="VN", fontSize=9.5, leading=12),
        "head": ParagraphStyle("head", fontName="VN-Bold", fontSize=9.5, leading=12),
    }


def build_pdf(content, path: Path, page_size=A4, bottom_margin=20 * mm):
    st = _styles()
    story = []
    for kind, val in content:
        if kind == "table":
            data = [[Paragraph(h, st["head"]) for h in val["headers"]]]
            rows = [list(r) for r in val["rows"]]
            for col, r0, r1 in val.get("spans", []):  # merged cell shows its text only once
                for r in range(r0 + 1, r1 + 1):
                    rows[r][col] = ""
            data += [[Paragraph(c, st["cell"]) for c in r] for r in rows]
            style = [("GRID", (0, 0), (-1, -1), 0.8, colors.black),
                     ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                     ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6"))]
            style += [("SPAN", (c, r0 + 1), (c, r1 + 1)) for c, r0, r1 in val.get("spans", [])]
            width = page_size[0] - 40 * mm
            table = Table(data, colWidths=[width / len(val["headers"])] * len(val["headers"]),
                          repeatRows=1, style=TableStyle(style))
            story += [table, Spacer(1, 6)]
        else:
            story.append(Paragraph(val, st[kind]))
    SimpleDocTemplate(str(path), pagesize=page_size, leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=20 * mm, bottomMargin=bottom_margin).build(story)


def make_scan(src_pdf: Path, out_pdf: Path, dpi=200, seed=0):
    """Rasterise each page, add skew + noise + blur + JPEG artefacts, rebuild an image-only PDF."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    src = pymupdf.open(src_pdf)
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        img = img.rotate(rng.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=False, fillcolor=255)
        arr = np.asarray(img, dtype=np.float32)
        arr = arr * 0.92 + 12 + nrng.normal(0, 9, arr.shape)  # grey paper + sensor noise
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.getvalue())
    out.save(out_pdf, deflate=True)


def ground_truth_from_content(content) -> dict:
    gt = {"paragraphs": [], "tables": []}
    for kind, val in content:
        if kind == "table":
            gt["tables"].append({"headers": val["headers"], "rows": val["rows"]})
        else:
            gt["paragraphs"].append(val)
    return gt


# --------------------------------------------------------------------------- excel
ERROR_CODES = [  # component (merged vertically), code, cause, fix
    ["Cảm biến nhiệt độ", "E01", "Đứt dây cảm biến", "Kiểm tra và nối lại dây, thay cảm biến nếu cần"],
    ["Cảm biến nhiệt độ", "E02", "Nhiệt độ vượt ngưỡng", "Kiểm tra quạt làm mát và bộ gia nhiệt"],
    ["Động cơ servo", "E10", "Quá tải động cơ", "Giảm tải, kiểm tra cơ cấu truyền động"],
    ["Động cơ servo", "E11", "Lỗi encoder", "Kiểm tra cáp encoder, reset driver"],
    ["Động cơ servo", "E12", "Quá áp DC bus", "Kiểm tra điện trở xả"],
    ["Bơm thủy lực", "H20", "Áp suất thấp", "Kiểm tra mức dầu và lọc hút"],
    ["Bơm thủy lực", "H21", "Nhiệt độ dầu cao", "Kiểm tra bộ làm mát dầu"],
    ["Bộ điều khiển PLC", "P30", "Mất kết nối mạng", "Kiểm tra cáp Ethernet và địa chỉ IP"],
]
ROBOT_CODES = [
    ["R-001", "Va chạm", "Dừng khẩn cấp", "Kiểm tra vùng làm việc, chạy lại chương trình"],
    ["R-002", "Mất điểm gốc", "Robot không di chuyển", "Thực hiện lại thao tác về gốc"],
    ["R-015", "Pin encoder yếu", "Cảnh báo trên teach pendant", "Thay pin trong vòng 7 ngày"],
]
PRESS_SPECS = [  # model, voltage, frequency, power, clamp force, install date
    ["MX-200", 380, 50, 22, 200, dt.date(2021, 5, 14)],
    ["MX-350", 380, 50, 37, 350, dt.date(2022, 8, 2)],
    ["MX-500", 380, 50, 55.5, 500, dt.date(2023, 1, 20)],
]
CNC_SPECS = [
    ["CNC-01", "Phay đứng", 12000, "800 x 500 x 500", "Fanuc 0i-MF"],
    ["CNC-02", "Tiện", 4500, "Ø300 x 600", "Fanuc 0i-TF"],
    ["CNC-03", "Phay đứng", 15000, "1000 x 600 x 600", "Siemens 828D"],
]


def make_excels(out: Path) -> dict:
    gts = {}
    bold = Font(bold=True)
    # ---- bang_ma_loi.xlsx
    wb = Workbook()
    ws = wb.active
    ws.title = "Máy ép"
    ws["A1"] = "BẢNG MÃ LỖI MÁY ÉP NHỰA MX-200"
    ws["A1"].font = Font(bold=True, size=14)
    ws.merge_cells("A1:D1")
    headers = ["Tên linh kiện", "Mã lỗi", "Nguyên nhân", "Cách xử lý"]
    ws.append([])
    ws.append(headers)
    for c in ws[3]:
        c.font = bold
    start = 4
    for r in ERROR_CODES:
        ws.append(r)
    # merge identical component names vertically, keeping only the first value
    r = 0
    while r < len(ERROR_CODES):
        r2 = r
        while r2 + 1 < len(ERROR_CODES) and ERROR_CODES[r2 + 1][0] == ERROR_CODES[r][0]:
            r2 += 1
        if r2 > r:
            ws.merge_cells(start_row=start + r, start_column=1, end_row=start + r2, end_column=1)
            ws.cell(start + r, 1).alignment = Alignment(vertical="center")
        r = r2 + 1
    ws2 = wb.create_sheet("Robot")
    robot_headers = ["Mã lỗi", "Tên lỗi", "Hiện tượng", "Cách xử lý"]
    ws2.append(robot_headers)
    for row in ROBOT_CODES:
        ws2.append(row)
    wb.save(out / "bang_ma_loi.xlsx")
    gts["bang_ma_loi.xlsx"] = {"paragraphs": [], "tables": [
        {"sheet": "Máy ép", "first_row": start, "headers": headers, "rows": ERROR_CODES},
        {"sheet": "Robot", "first_row": 2, "headers": robot_headers, "rows": ROBOT_CODES},
    ]}

    # ---- thong_so_may.xlsx (two-level header on sheet 1)
    wb = Workbook()
    ws = wb.active
    ws.title = "Máy ép nhựa"
    ws.append(["Model", "Thông số điện", None, None, "Lực kẹp (tấn)", "Ngày lắp đặt"])
    ws.append([None, "Điện áp (V)", "Tần số (Hz)", "Công suất (kW)", None, None])
    for rng in ("A1:A2", "B1:D1", "E1:E2", "F1:F2"):
        ws.merge_cells(rng)
    for row in PRESS_SPECS:
        ws.append(row)
    ws2 = wb.create_sheet("Máy CNC")
    cnc_headers = ["Mã máy", "Loại máy", "Tốc độ trục chính (vòng/phút)", "Hành trình (mm)", "Bộ điều khiển"]
    ws2.append(cnc_headers)
    for row in CNC_SPECS:
        ws2.append(row)
    wb.save(out / "thong_so_may.xlsx")
    press_headers = ["Model", "Thông số điện - Điện áp (V)", "Thông số điện - Tần số (Hz)",
                     "Thông số điện - Công suất (kW)", "Lực kẹp (tấn)", "Ngày lắp đặt"]
    gts["thong_so_may.xlsx"] = {"paragraphs": [], "tables": [
        {"sheet": "Máy ép nhựa", "first_row": 3, "headers": press_headers,
         "rows": [[str(v) if not isinstance(v, float) else f"{v:g}" for v in r[:5]] + [r[5].isoformat()]
                  for r in PRESS_SPECS]},
        {"sheet": "Máy CNC", "first_row": 2, "headers": cnc_headers,
         "rows": [[str(v) for v in r] for r in CNC_SPECS]},
    ]}
    return gts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    out = Path(args.out)
    gt_dir = out / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    _register_fonts()

    gts = {}
    build_pdf(DOC_BAO_TRI, out / "huong_dan_bao_tri.pdf", page_size=A5)  # A5 -> page breaks inside table/paragraphs
    gts["huong_dan_bao_tri.pdf"] = ground_truth_from_content(DOC_BAO_TRI)
    build_pdf(DOC_KIEM_TRA, out / "quy_trinh_kiem_tra.pdf")
    gts["quy_trinh_kiem_tra.pdf"] = ground_truth_from_content(DOC_KIEM_TRA)

    tmp = gt_dir / "_tmp.pdf"
    for i, (name, content) in enumerate([("scan_bien_ban_su_co.pdf", DOC_SU_CO),
                                         ("scan_huong_dan_van_hanh.pdf", DOC_VAN_HANH)]):
        # small bottom margin on purpose for the 2nd doc -> content spills onto 2 pages
        build_pdf(content, tmp, bottom_margin=(20 if i == 0 else 175) * mm)
        make_scan(tmp, out / name, seed=i)
        gts[name] = ground_truth_from_content(content)
    tmp.unlink()

    gts.update(make_excels(out))
    for name, gt in gts.items():
        (gt_dir / f"{name}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(gts)} files to {out}/ (ground truth in {gt_dir}/)")


if __name__ == "__main__":
    main()
