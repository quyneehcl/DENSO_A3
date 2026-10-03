"""Compare data/processed/ outputs with data/raw/ground_truth/ (criteria of spec section 7).

  * text accuracy   for each ground-truth paragraph: word recall against the output text
                    (order-aware, difflib); a paragraph counts as "found" when recall >= 0.9
  * table accuracy  for each ground-truth row: is there a chunk equal to the expected structured
                    sentence (exact), and the best character similarity otherwise; for Excel the
                    location (sheet, row number) must match too
  * speed           average processing time per file: digital PDF vs scanned PDF vs Excel
                    (from _report.json written by run_ingest.py)

Usage: python scripts/evaluate.py [--raw data/raw] [--processed data/processed]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest.schema import detect_language, validate_chunks  # noqa: E402
from ingest.table_utils import row_to_sentence  # noqa: E402

FOUND_THRESHOLD = 0.9


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", _norm(text))


def word_recall(reference: str, output_words: list[str]) -> float:
    ref = _words(reference)
    if not ref:
        return 1.0
    sm = SequenceMatcher(None, ref, output_words, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(ref)


def evaluate_file(gt: dict, chunks: list[dict]) -> dict:
    table_chunks = [c for c in chunks if c["doc_type"] == "excel_table" or ", bảng " in c["location"]]
    text_words = _words(" ".join(c["text"] for c in chunks if c not in table_chunks))

    res: dict = {"format_errors": validate_chunks(chunks)}
    paras = gt.get("paragraphs", [])
    if paras:
        recalls = [word_recall(p, text_words) for p in paras]
        res["paragraphs"] = len(paras)
        res["paragraphs_found"] = sum(r >= FOUND_THRESHOLD for r in recalls)
        res["text_word_recall"] = round(sum(recalls) / len(recalls), 4)
        res["missing_paragraphs"] = [p[:60] for p, r in zip(paras, recalls) if r < FOUND_THRESHOLD]

    rows_total = rows_exact = 0
    sims, wrong = [], []
    for t in gt.get("tables", []):
        for i, row in enumerate(t["rows"]):
            expected = row_to_sentence(t["headers"], row, detect_language(" ".join(t["headers"] + row)))
            exp_loc = f"{t['sheet']}, hàng {t['first_row'] + i}" if "sheet" in t else None
            rows_total += 1
            hit = any(_norm(c["text"]) == _norm(expected) and (exp_loc is None or c["location"] == exp_loc)
                      for c in table_chunks)
            rows_exact += hit
            best = max((SequenceMatcher(None, _norm(expected), _norm(c["text"])).ratio()
                        for c in table_chunks), default=0.0)
            sims.append(best)
            if not hit:
                wrong.append(expected[:70])
    if rows_total:
        res["table_rows"] = rows_total
        res["table_rows_exact"] = rows_exact
        res["table_row_similarity"] = round(sum(sims) / len(sims), 4)
        res["table_rows_wrong"] = wrong
        res["extra_table_chunks"] = max(0, len(table_chunks) - rows_total)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--processed", default="data/processed")
    args = ap.parse_args()
    gt_dir, out_dir = Path(args.raw) / "ground_truth", Path(args.processed)

    report = json.loads((out_dir / "_report.json").read_text(encoding="utf-8")) if (out_dir / "_report.json").exists() else {}
    meta = {r["file"]: r for r in report.get("files", [])}
    results = {}
    for gt_file in sorted(gt_dir.glob("*.json")):
        name = gt_file.name[:-5]
        out_file = out_dir / f"{name}.json"
        last = meta.get(name, {})
        if last.get("errors") and "seconds" not in last:
            print(f"!! {name}: FAILED in the last run_ingest.py run -> {last['errors'][0]}")
            continue
        if not out_file.exists():
            print(f"!! {name}: no output (run scripts/run_ingest.py first)")
            continue
        r = evaluate_file(json.loads(gt_file.read_text(encoding="utf-8")),
                          json.loads(out_file.read_text(encoding="utf-8")))
        r["category"] = meta.get(name, {}).get("category")
        r["seconds"] = meta.get(name, {}).get("seconds")
        results[name] = r

    print(f"{'file':<30} {'type':<9} {'paragraphs':>11} {'word recall':>12} {'table rows':>11} "
          f"{'row sim':>8} {'time(s)':>8}  format")
    for name, r in results.items():
        para = f"{r['paragraphs_found']}/{r['paragraphs']}" if "paragraphs" in r else "-"
        rec = f"{r['text_word_recall']:.1%}" if "text_word_recall" in r else "-"
        rows = f"{r['table_rows_exact']}/{r['table_rows']}" if "table_rows" in r else "-"
        sim = f"{r['table_row_similarity']:.1%}" if "table_row_similarity" in r else "-"
        print(f"{name:<30} {str(r['category']):<9} {para:>11} {rec:>12} {rows:>11} {sim:>8} "
              f"{r['seconds'] if r['seconds'] is not None else '-':>8}  {'OK' if not r['format_errors'] else 'ERR'}")
        for p in r.get("missing_paragraphs", []):
            print(f"    missing paragraph: {p}...")
        for w in r.get("table_rows_wrong", []):
            print(f"    row not exact:     {w}...")

    summary = {}
    for cat in ("pdf_text", "pdf_scan", "pdf_mixed", "excel"):
        rs = [r for r in results.values() if r["category"] == cat]
        if not rs:
            continue
        s = {"files": len(rs)}
        if any("paragraphs" in r for r in rs):
            s["paragraphs_found"] = f"{sum(r.get('paragraphs_found', 0) for r in rs)}/{sum(r.get('paragraphs', 0) for r in rs)}"
        if any("table_rows" in r for r in rs):
            s["table_rows_exact"] = f"{sum(r.get('table_rows_exact', 0) for r in rs)}/{sum(r.get('table_rows', 0) for r in rs)}"
        times = [r["seconds"] for r in rs if r["seconds"] is not None]
        if times:
            s["avg_seconds"] = round(sum(times) / len(times), 3)
        summary[cat] = s
    print("\nSummary by type:")
    for cat, s in summary.items():
        print(f"  {cat:<9} " + ", ".join(f"{k}={v}" for k, v in s.items()))
    if report.get("ocr_backend"):
        print(f"  (OCR backend: {report['ocr_backend']})")
    (out_dir / "_evaluation.json").write_text(json.dumps({"summary": summary, "files": results},
                                                         ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if any(r["format_errors"] for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
