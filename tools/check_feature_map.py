#!/usr/bin/env python3
"""Verify every `path:line` anchor in docs/feature-map.md against a baseline commit.

The feature map is a plain table. Each anchor cell holds one or more
`path:line` references, each optionally followed by the symbol that must
appear on that exact line:

    `src/serve/openai_api.cpp:1234` `handle_chat`

The document declares its baseline SHA (`基线 commit `<40 hex>``). Lines are
read from that commit's blob via `git show`, not from the working tree, so the
result does not depend on how far main has advanced. Each reference must
resolve: the file exists in the baseline, the line is in range, is not blank,
and contains the quoted symbol. Paths and line numbers in the 验证 column are
checked the same way.

Usage:
    python3 -I tools/check_feature_map.py [docs/feature-map.md] [--ref SHA]
Exit code 0 means every anchor resolved at the baseline.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ANCHOR = re.compile(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+):(\d+)`(?:\s*`([^`]+)`)?")
# test reference in the 验证 column: path:line, optionally wrapped in backticks
TEST_REF = re.compile(r"([A-Za-z0-9_./-]+\.[A-Za-z0-9]+):(\d+)")
PLAIN_PATH = re.compile(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`")
BASELINE = re.compile(r"基线 commit `([0-9a-f]{40})`")

COLUMNS = ["功能", "入口", "核心实现", "协议/契约", "验证", "状态"]
ANCHOR_COLUMNS = {"入口", "核心实现"}

_blob_cache = {}


def git(*args):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True
    )


def blob_lines(sha, path):
    key = (sha, path)
    if key not in _blob_cache:
        result = git("show", f"{sha}:{path}")
        _blob_cache[key] = (
            result.stdout.splitlines() if result.returncode == 0 else None
        )
    return _blob_cache[key]


def split_row(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def check_anchor(sha, path, number, symbol):
    rows = blob_lines(sha, path)
    if rows is None:
        return False, f"{path} missing at {sha[:7]}"
    line_no = int(number)
    if line_no < 1 or line_no > len(rows):
        return False, f"{path}:{line_no} out of range (blob has {len(rows)} lines)"
    text = rows[line_no - 1]
    if not text.strip():
        return False, f"{path}:{line_no} is blank at {sha[:7]}"
    if symbol and symbol not in text:
        return False, (
            f"{path}:{line_no} does not contain {symbol!r} at {sha[:7]}: "
            f"{text.strip()[:90]}"
        )
    return True, f"{path}:{line_no} {text.strip()[:70]}"


def main(argv):
    doc = ROOT / "docs/feature-map.md"
    ref = None
    args = argv[1:]
    while args:
        arg = args.pop(0)
        if arg == "--ref" and args:
            ref = args.pop(0)
        else:
            doc = Path(arg)
            if not doc.is_absolute():
                doc = ROOT / doc
    if not doc.exists():
        print(f"FAIL: {doc} does not exist")
        return 1

    text = doc.read_text(encoding="utf-8")
    match = BASELINE.search(text)
    if ref is None:
        if not match:
            print("FAIL: document has no `基线 commit `<40-hex>`` line")
            return 1
        ref = match.group(1)
    if git("cat-file", "-e", f"{ref}^{{commit}}").returncode != 0:
        print(f"FAIL: baseline commit {ref} is not in this repository")
        return 1

    failures = []
    checked = 0
    rows_seen = 0
    header = None
    for lineno, raw in enumerate(text.splitlines(), 1):
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
            found = ANCHOR.findall(row.get(column, ""))
            if not found:
                failures.append(f"line {lineno} [{label}] {column}: no `path:line`")
                continue
            for path, number, symbol in found:
                checked += 1
                ok, detail = check_anchor(ref, path, number, symbol)
                if not ok:
                    failures.append(f"line {lineno} [{label}] {column}: {detail}")

        verify_cell = row.get("验证", "")
        for path in PLAIN_PATH.findall(verify_cell):
            checked += 1
            if blob_lines(ref, path) is None:
                failures.append(f"line {lineno} [{label}] 验证: {path} missing at {ref[:7]}")
        for path, number in TEST_REF.findall(verify_cell):
            checked += 1
            ok, detail = check_anchor(ref, path, number, None)
            if not ok:
                failures.append(f"line {lineno} [{label}] 验证: {detail}")

    print(f"feature-map: {doc.relative_to(ROOT) if doc.is_relative_to(ROOT) else doc}")
    print(f"baseline: {ref}")
    print(f"rows checked: {rows_seen}")
    print(f"references checked: {checked}")
    for failure in failures:
        print(f"FAIL {failure}")
    print("RESULT:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
