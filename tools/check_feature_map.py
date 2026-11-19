#!/usr/bin/env python3
"""Verify every `path:line` anchor in docs/feature-map.md.

The feature map is a plain table. Each anchor cell holds one or more
`path:line` references, each optionally followed by the symbol that must
appear on that exact line:

    `src/serve/openai_api.cpp:1234` `handle_chat`

This script opens the file, checks the line exists and is not blank, and
when a symbol is given, checks the symbol text is present on that line.
It also checks that any path mentioned in the 验证 column exists.

Usage:
    python3 -I tools/check_feature_map.py [docs/feature-map.md]
Exit code 0 means every anchor resolved.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ANCHOR = re.compile(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+):(\d+)`(?:\s*`([^`]+)`)?")
# test reference in the 验证 column: path:line, optionally wrapped in backticks
TEST_REF = re.compile(r"([A-Za-z0-9_./-]+\.[A-Za-z0-9]+):(\d+)")
PLAIN_PATH = re.compile(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`")

COLUMNS = ["功能", "入口", "核心实现", "协议/契约", "验证", "状态"]
ANCHOR_COLUMNS = {"入口", "核心实现"}

_line_cache = {}


def lines_of(path):
    if path not in _line_cache:
        try:
            _line_cache[path] = (ROOT / path).read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError as exc:
            _line_cache[path] = exc
    return _line_cache[path]


def split_row(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def check_anchor(path, number, symbol):
    rows = lines_of(path)
    if isinstance(rows, OSError):
        return False, f"missing file {path}"
    line_no = int(number)
    if line_no < 1 or line_no > len(rows):
        return False, f"{path}:{line_no} out of range (file has {len(rows)} lines)"
    text = rows[line_no - 1]
    if not text.strip():
        return False, f"{path}:{line_no} is blank"
    if symbol and symbol not in text:
        return False, f"{path}:{line_no} does not contain {symbol!r}: {text.strip()[:90]}"
    return True, f"{path}:{line_no} {text.strip()[:70]}"


def main(argv):
    doc = Path(argv[1]) if len(argv) > 1 else ROOT / "docs/feature-map.md"
    if not doc.is_absolute():
        doc = ROOT / doc
    if not doc.exists():
        print(f"FAIL: {doc} does not exist")
        return 1

    failures = []
    checked = 0
    rows_seen = 0
    header = None
    for lineno, raw in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.startswith("|"):
            header = None
            continue
        cells = split_row(raw)
        if set(cells) & set(COLUMNS) and len(cells) == len(COLUMNS):
            header = cells
            continue
        if header is None:
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        if len(cells) != len(header):
            failures.append(f"line {lineno}: {len(cells)} cells, expected {len(header)}")
            continue
        rows_seen += 1
        row = dict(zip(header, cells))
        label = row.get("功能", "?")[:40]

        for column in ANCHOR_COLUMNS:
            cell = row.get(column, "")
            found = ANCHOR.findall(cell)
            if not found:
                failures.append(f"line {lineno} [{label}] {column}: no `path:line`")
                continue
            for path, number, symbol in found:
                checked += 1
                ok, detail = check_anchor(path, number, symbol)
                if not ok:
                    failures.append(f"line {lineno} [{label}] {column}: {detail}")

        verify_cell = row.get("验证", "")
        for path in PLAIN_PATH.findall(verify_cell):
            if path in ("无",):
                continue
            checked += 1
            if not (ROOT / path).exists():
                failures.append(f"line {lineno} [{label}] 验证: missing {path}")
        for path, number in TEST_REF.findall(verify_cell):
            checked += 1
            ok, detail = check_anchor(path, number, None)
            if not ok:
                failures.append(f"line {lineno} [{label}] 验证: {detail}")

    print(f"feature-map: {doc.relative_to(ROOT) if doc.is_relative_to(ROOT) else doc}")
    print(f"rows checked: {rows_seen}")
    print(f"references checked: {checked}")
    for failure in failures:
        print(f"FAIL {failure}")
    print("RESULT:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
