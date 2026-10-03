"""Ingest every PDF / Excel file of data/raw/ into data/processed/.

Outputs
  data/processed/<file name>.json   list of chunks for one source file
  data/processed/all_chunks.jsonl   every chunk, one JSON object per line (input of the next stage)
  data/processed/_manifest.json     file name -> doc_id (kept stable between runs)
  data/processed/_report.json       per-file category, chunk count, processing time

Usage: python scripts/run_ingest.py [--input data/raw] [--output data/processed]
                                    [--ocr-backend auto|paddle_vl|tesseract|none] [--dpi 300]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest.ocr_pipeline import get_backend  # noqa: E402
from ingest.pipeline import SUPPORTED_EXT, file_category, ingest_file  # noqa: E402
from ingest.schema import source_name, validate_chunks, write_chunks  # noqa: E402


def short_error(e: Exception) -> str:
    """One readable line for the report (native libraries put pages of C++ traceback in the message)."""
    lines = [l.strip() for l in str(e).splitlines() if l.strip()]
    useful = [l for l in lines if len(l) > 20 and not l.startswith(("---", "C++", "(at "))
              and re.search(r"error|memory|not found|cannot|no module|undefined", l, re.I)]
    msg = useful[-1] if useful else (lines[0] if lines else "")
    return f"{type(e).__name__}: {msg[:300]}"


def load_manifest(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def assign_doc_ids(files: list[Path], manifest: dict[str, str]) -> dict[str, str]:
    used = {int(v[3:]) for v in manifest.values() if v.startswith("doc") and v[3:].isdigit()}
    nxt = max(used, default=0) + 1
    for f in files:
        if source_name(f) not in manifest:
            manifest[source_name(f)] = f"doc{nxt:03d}"
            nxt += 1
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/raw")
    ap.add_argument("--output", default="data/processed")
    ap.add_argument("--ocr-backend", default="auto", choices=["auto", "paddle_vl", "tesseract", "none"])
    ap.add_argument("--dpi", type=int, default=300, help="render resolution for scanned pages")
    ap.add_argument("--max-chars", type=int, default=800, help="max characters of a text chunk")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    src, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
    if not files:
        print(f"No PDF/Excel files in {src}/")
        return 1
    manifest = assign_doc_ids(files, load_manifest(out / "_manifest.json"))

    ocr_name = None
    report, all_chunks, failed = [], [], 0
    for f in files:
        name = source_name(f)
        entry = {"file": name, "doc_id": manifest[name]}
        try:
            entry["category"] = file_category(f)
            if entry["category"] in ("pdf_scan", "pdf_mixed") and ocr_name is None:
                backend = get_backend(args.ocr_backend)
                ocr_name = backend.name if backend else "none"
            t0 = time.perf_counter()
            chunks = [c.to_dict() for c in ingest_file(f, manifest[name], ocr_backend=args.ocr_backend,
                                                       max_chars=args.max_chars, dpi=args.dpi)]
            entry["seconds"] = round(time.perf_counter() - t0, 3)
            entry["chunks"] = len(chunks)
            entry["errors"] = validate_chunks(chunks)
            if entry["category"] != "excel":
                entry["ocr_backend"] = ocr_name if entry["category"] != "pdf_text" else None
            write_chunks(chunks, out / f"{name}.json")
            all_chunks += chunks
        except Exception as e:  # keep going with the other files
            logging.exception("failed: %s", f.name)
            entry["errors"] = [short_error(e)]
            (out / f"{name}.json").unlink(missing_ok=True)  # never leave a stale output behind
        failed += bool(entry["errors"])
        report.append(entry)
        print(f"{entry['doc_id']}  {name:<35} {entry.get('category', '?'):<10} "
              f"{entry.get('chunks', 0):>4} chunks  {entry.get('seconds', 0):>7.2f}s"
              + ("  ERRORS: " + "; ".join(entry["errors"][:3]) if entry["errors"] else ""))

    with (out / "all_chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {}
    for cat in ("pdf_text", "pdf_scan", "pdf_mixed", "excel"):
        times = [r["seconds"] for r in report if r.get("category") == cat and "seconds" in r]
        if times:
            summary[cat] = {"files": len(times), "avg_seconds": round(sum(times) / len(times), 3)}
    (out / "_report.json").write_text(json.dumps({"ocr_backend": ocr_name, "summary": summary, "files": report},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nAverage time per file:")
    for cat, s in summary.items():
        print(f"  {cat:<10} {s['avg_seconds']:>7.2f}s  ({s['files']} file(s))")
    print(f"\n{len(all_chunks)} chunks -> {out}/  ({failed} file(s) with errors)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
