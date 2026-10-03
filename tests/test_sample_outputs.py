"""Every JSON output (reference samples + whatever is in data/processed/) must follow the
intermediate format exactly."""
import json
from pathlib import Path

import pytest

from ingest.schema import CHUNK_FIELDS, VALID_DOC_TYPES, validate_chunks

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "tests" / "sample_outputs").glob("*.json")) + \
    sorted(p for p in (ROOT / "data" / "processed").glob("*.json") if not p.name.startswith("_"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_output_format(path):
    chunks = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(chunks, list) and chunks
    assert validate_chunks(chunks) == []
    for c in chunks:
        assert tuple(c) == CHUNK_FIELDS          # same fields, same order
        assert c["doc_type"] in VALID_DOC_TYPES
        assert c["source_file"] == path.name[:-5]
    ids = [c["chunk_id"] for c in chunks]
    doc = ids[0].split("_chunk")[0]
    assert ids == [f"{doc}_chunk{i:03d}" for i in range(1, len(ids) + 1)]


def test_jsonl_matches_files():
    jsonl = ROOT / "data" / "processed" / "all_chunks.jsonl"
    if not jsonl.exists():
        pytest.skip("run scripts/run_ingest.py first")
    lines = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines()]
    assert validate_chunks(lines) == []
