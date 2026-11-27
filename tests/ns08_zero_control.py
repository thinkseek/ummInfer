#!/usr/bin/env python3
"""Produce a compact paired-arm NS-08 zero-control engineering report.

This is not the full NS-08 gate. Browser mode measures the real default DOM in
Chromium; real macOS, physical audio, perceptual quality, and human walkthrough
remain NOT_RUN. All evidence is written beneath a required external directory.
"""

import argparse
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys

CONTROL_GROUPS = {
    "visible_modality_control_count": ("image", "audio", "video"),
    "visible_routing_control_count": ("narrative", "visualMode", "maxSlots", "allowTextOnly",
        "narrativeContinuity", "styleAnchor", "fastVideo"),
    "visible_sampling_control_count": ("maxTokens", "temperature", "topP", "seed",
        "imageSteps", "videoSteps", "videoGuidance"),
    "visible_frame_control_count": ("videoFramePolicy", "videoFrameBudget",
        "videoMotionThreshold", "videoSceneThreshold"),
    "visible_voice_routing_control_count": ("voiceToggle", "voice", "realtimeEchoSuppression"),
}
ALL_CONTROLS = tuple(control for group in CONTROL_GROUPS.values() for control in group)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


class DefaultVisibility(HTMLParser):
    """Static fallback matching native details/hidden and accepted CSS rules."""
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
                 "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, css_hides_developer_controls):
        super().__init__(convert_charrefs=True)
        self.css_hides = css_hides_developer_controls
        self.stack = []
        self.visible = set()
        self.advanced_open = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        hidden = any(self.stack)
        if "hidden" in values:
            hidden = True
        if tag == "details" and values.get("id") == "advanced":
            self.advanced_open = "open" in values
            if not self.advanced_open:
                hidden = True
        classes = set(values.get("class", "").split())
        if self.css_hides and "developer-control" in classes:
            hidden = True
        if tag not in self.VOID_TAGS:
            self.stack.append(hidden)
        identity = values.get("id")
        if identity in ALL_CONTROLS and not hidden:
            self.visible.add(identity)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID_TAGS and self.stack:
            self.stack.pop()

    def handle_endtag(self, tag):
        if self.stack:
            self.stack.pop()


def static_controls(source_root):
    page = (source_root / "web" / "index.html").read_text(encoding="utf-8")
    parser = DefaultVisibility("#settings label.developer-control { display: none; }" in page)
    parser.feed(page)
    if parser.advanced_open is not False:
        raise AssertionError("Web Advanced is missing or initially open")
    return {name: sum(control in parser.visible for control in controls)
            for name, controls in CONTROL_GROUPS.items()}, sorted(parser.visible)


def run_suite(source_root, relative, log_path):
    path = source_root / relative
    if not path.is_file():
        return {"driver": relative, "status": "NOT_COVERED", "exit_code": None,
                "log_sha256": None}
    with log_path.open("wb") as log:
        result = subprocess.run([sys.executable, "-I", relative], cwd=source_root,
                                stdout=log, stderr=subprocess.STDOUT, timeout=900,
                                check=False)
    return {"driver": relative, "status": "PASS" if result.returncode == 0 else "FAIL",
            "exit_code": result.returncode, "log_sha256": sha_file(log_path),
            "log_bytes": log_path.stat().st_size}


def browser_check(source_root, evidence, chromium, playwright_core):
    if not chromium or not playwright_core:
        raise ValueError("browser mode requires caller-selected --chromium and --playwright-core")
    result_path = evidence / "browser-result.json"
    command = [os.environ.get("NODE", "node"),
               str(Path(__file__).resolve().with_name("ns08_browser_driver.cjs")),
               str(source_root / "web" / "index.html"), str(result_path),
               str(evidence / "browser-default.png"), str(playwright_core),
               str(chromium), json.dumps(list(ALL_CONTROLS))]
    completed = subprocess.run(command, cwd=source_root, capture_output=True, text=True,
                               timeout=120, check=False)
    (evidence / "browser-driver.log").write_text(completed.stdout + completed.stderr,
                                                 encoding="utf-8")
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError("Chromium driver failed; see external browser-driver.log")
    return json.loads(result_path.read_text(encoding="utf-8"))


def manifest(root):
    path = root / "SHA256SUMS"
    rows = []
    for item in sorted(root.rglob("*")):
        if item.is_file() and item != path:
            rows.append(f"{sha_file(item)}  {item.relative_to(root).as_posix()}")
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return {"file_count": len(rows), "sha256": sha_file(path), "self_excluded": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("before", "after"))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--chromium", type=Path)
    parser.add_argument("--playwright-core", type=Path)
    args = parser.parse_args(argv)
    source = args.source_root.resolve()
    evidence = args.evidence_dir.resolve()
    if evidence.exists() and any(evidence.iterdir()):
        raise SystemExit("evidence directory must be absent or empty")
    evidence.mkdir(parents=True, exist_ok=True)
    if len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit):
        raise SystemExit("source commit must be an exact lowercase SHA-1")
    static_counts, static_visible = static_controls(source)
    suites = [
        run_suite(source, "tests/web_ui_contract_test.py", evidence / "web-contract.log"),
        run_suite(source, "tests/macos_zero_control_test.py", evidence / "macos-static.log"),
    ]
    browser = {"status": "NOT_RUN", "checks_failed": 0, "checks_passed": 0,
               "visible_controls": static_visible, "visible_counts": static_counts,
               "checks": []}
    if args.browser:
        browser = browser_check(source, evidence, args.chromium, args.playwright_core)
        dump(evidence / "browser-result.json", browser)
    contract_failed = sum(suite["status"] == "FAIL" for suite in suites)
    counts = browser["visible_counts"]
    export_exact = int(suites[0]["status"] == "PASS")
    report = {
        "schema": "umminfer-ns08-zero-control-arm/v1",
        "case_id": "NS-08",
        "arm": args.arm,
        "engineering_status": "CONTRACT_PASS" if contract_failed == 0 and
                                args.browser and browser["status"] == "PASS" else
                                "FAIL" if contract_failed or args.browser else "NOT_RUN",
        "production_status": "NOT_RUN",
        "full_gate_status": "NOT_RUN",
        "metrics": {
            **{name: int(counts[name]) for name in CONTROL_GROUPS},
            "contract_checks_failed": contract_failed,
            "browser_checks_failed": int(browser["checks_failed"]),
            "export_import_exact": export_exact,
            "reconnect_exact": export_exact,
            "source_bound_revision": export_exact,
            "cancellation_retry_exact": export_exact,
        },
        "hard_gates": {
            "advanced_initially_collapsed": True,
            "zero_default_developer_controls": sum(counts.values()) == 0,
            "web_contracts": suites[0]["status"],
            "macos_static": suites[1]["status"],
            "real_macos": "NOT_RUN",
            "human_walkthrough": "NOT_RUN",
        },
        "failures": ([f"{suite['driver']}:{suite['status']}" for suite in suites
                      if suite["status"] == "FAIL"] +
                     [row["name"] for row in browser["checks"] if not row["ok"]]),
        "limitations": [
            "Chromium default-DOM and deterministic replay checks are not a human NS journey.",
            "Real macOS ummmlx execution, microphone/speaker acoustics, and perceptual quality are NOT_RUN.",
            "Full NS-08 PASS is not permitted from this arm report.",
        ],
        "source_identity": {
            "commit": args.source_commit,
            "web_sha256": sha_file(source / "web" / "index.html"),
            "app_swift_sha256": sha_file(source / "clients/ummmlx/Sources/ummmlx/App.swift"),
            "chat_model_sha256": sha_file(source / "clients/ummmlx/Sources/ummmlx/ChatModel.swift"),
            "adapter_sha256": sha_file(Path(__file__).resolve()),
        },
        "binary_identities": {"macos_executable": "NOT_RUN", "browser_executable_sha256":
                              sha_file(args.chromium) if args.browser and args.chromium else "NOT_RECORDED"},
        "model_identities": {},
        "config_identity": {"registry": "benchmarks/north-star/cases.json",
                            "advanced_default": "collapsed", "keyboard_action_network_requests":
                            len(browser["network_requests"])},
        "coverage": {"web_contract": suites[0]["status"], "chromium_default_dom": browser["status"],
                     "macos_static": suites[1]["status"], "macos_real": "NOT_RUN",
                     "human_walkthrough": "NOT_RUN", "physical_audio": "NOT_RUN"},
        "raw_evidence": {"browser_result_sha256": sha_file(evidence / "browser-result.json")
                         if args.browser else None,
                         "default_screenshot_sha256": sha_file(evidence / "browser-default.png")
                         if args.browser else None,
                         "contract_logs": [{key: value for key, value in suite.items()
                                            if key != "driver"} | {"driver": suite["driver"]}
                                           for suite in suites]},
    }
    dump(evidence / "NS-08.json", report)
    seal = manifest(evidence)
    print(json.dumps({"case_id": "NS-08", "arm": args.arm,
                      "engineering_status": report["engineering_status"],
                      "metrics": report["metrics"], "manifest": seal}, sort_keys=True))
    return 0 if report["engineering_status"] in {"CONTRACT_PASS", "FAIL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
