#!/usr/bin/env python3
"""Small evaluation harness for the RAG assistant.

Runs a list of questions through the pipeline and checks whether the answer
contains the expected keywords and whether sources were returned. This makes
retrieval/answer quality measurable instead of eyeballed.

    python eval.py eval_questions.example.json

Question file format (JSON list):
    [{"question": "...", "expected": ["apt29", "phishing"]}, ...]
`expected` is optional; without it a case passes if any source was retrieved.
"""
import argparse
import json
import sys

from query import answer


def run(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)

    total = len(cases)
    passed = 0
    with_sources = 0

    for i, case in enumerate(cases, 1):
        question = case["question"]
        expected = [e.lower() for e in case.get("expected", [])]

        res = answer(question, include_sources=True)
        if isinstance(res, dict):
            text = res.get("answer", "")
            sources = res.get("sources", [])
        else:
            text, sources = res, []

        if expected:
            ok = all(kw in text.lower() for kw in expected)
        else:
            ok = bool(sources)

        passed += ok
        with_sources += bool(sources)
        print(f"[{i:>3}/{total}] {'PASS' if ok else 'FAIL'}  "
              f"sources={len(sources)}  {question[:70]}")

    print("-" * 60)
    print(f"Passed:         {passed}/{total}")
    print(f"With sources:   {with_sources}/{total}")
    return 0 if passed == total else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the CTI RAG assistant.")
    ap.add_argument("questions", help="JSON file with questions (see module docstring)")
    args = ap.parse_args(argv)
    return run(args.questions)


if __name__ == "__main__":
    sys.exit(main())
