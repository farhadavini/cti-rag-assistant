"""Tests for document loading and chunking."""
import json

import pytest

# These modules pull in langchain; skip locally if it is not installed.
pytest.importorskip("langchain_text_splitters")
pytest.importorskip("langchain_community")

import ingest  # noqa: E402


def test_load_json_prefers_text_field(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"text": "hello world", "other": 1}), encoding="utf-8")
    assert ingest.load_json(p) == "hello world"


def test_load_json_list_of_strings(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps(["one", "two"]), encoding="utf-8")
    assert ingest.load_json(p) == "one\ntwo"


def test_load_json_bad_file_returns_none(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert ingest.load_json(p) is None


def test_chunk_documents_splits_and_keeps_metadata():
    docs = [{"content": "word " * 500, "filename": "f.txt", "filepath": "/f.txt"}]
    chunks = ingest.chunk_documents(docs, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c["filename"] == "f.txt" for c in chunks)
    assert chunks[0]["chunk_index"] == 0
