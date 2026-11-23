#!/usr/bin/env python3
"""Offline checks for the deployment scripts and the service description.

Loads no model, starts no server, opens no CUDA device. Every shell script is
run against a throw-away root under the system temp directory so no shared
state is touched.

Usage: python3 -I tests/script_hygiene_and_dryrun_test.py
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SHELL_SCRIPTS = [
    "service/native-service.sh",
    "service/concurrency-stack.sh",
    "docs/cases/0001_repro.sh",
    "tools/publish-release.sh",
    "release/tools/verify.sh",
    "clients/ummmlx/Scripts/build.sh",
]

# The tokens are assembled at run time so the literal spellings never appear
# in this file; the check is that they never appear in the service description.
FORBIDDEN_TOKENS = ["h" + "20b", "lin" + "yu", "172." + "17.0.2"]


def run(args, cwd, env=None, timeout=120):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        args, cwd=str(cwd), env=merged, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


class NativeServiceDryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="umm-native-dry-"))
        self.root = self.tmp / "root"
        self.ipc = self.tmp / "ipc"
        self.script = ROOT / "service" / "native-service.sh"
        self.env = {
            "UMM_NATIVE_ROOT": str(self.root),
            "UMM_NATIVE_IPC": str(self.ipc),
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_status_reports_every_service_down(self):
        result = run(["sh", str(self.script), "status"], ROOT, self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        for name in ("server", "asr", "qwen38", "image", "tts"):
            self.assertIn(f"{name} down", lines)

    def test_stop_leaves_unrelated_processes_alone(self):
        sentinel = subprocess.Popen(["sleep", "60"])
        try:
            result = run(["sh", str(self.script), "stop"], ROOT, self.env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("stopped", result.stdout)
            self.assertIsNone(sentinel.poll(), "stop signaled an unrelated pid")
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=10)

    def test_invalid_argument_exits_two_with_usage(self):
        result = run(["sh", str(self.script), "bogus"], ROOT, self.env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr.lower())


class ConcurrencyStackDryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="umm-concurrency-dry-"))
        self.root = self.tmp / "root"
        self.ipc = self.tmp / "ipc"
        self.script = ROOT / "service" / "concurrency-stack.sh"
        self.env = {
            "UMM_CONCURRENCY_ROOT": str(self.root),
            "UMM_CONCURRENCY_IPC": str(self.ipc),
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_status_exits_clean(self):
        result = run(["sh", str(self.script), "status"], ROOT, self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("server", "asr", "qwen38", "image", "tts"):
            self.assertIn(f"{name} down", result.stdout.splitlines())

    def test_stop_does_not_touch_external_processes(self):
        sentinel = subprocess.Popen(["sleep", "60"])
        try:
            result = run(["sh", str(self.script), "stop"], ROOT, self.env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("stopped", result.stdout)
            self.assertIsNone(sentinel.poll(), "stop signaled an unrelated pid")
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=10)

    def test_invalid_argument_exits_two_with_usage(self):
        result = run(["sh", str(self.script), "bogus"], ROOT, self.env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr.lower())


class ShellSyntax(unittest.TestCase):
    def test_scripts_parse(self):
        for rel in SHELL_SCRIPTS:
            with self.subTest(script=rel):
                path = ROOT / rel
                self.assertTrue(path.is_file(), f"missing {rel}")
                result = run(["sh", "-n", str(path)], ROOT, timeout=30)
                self.assertEqual(result.returncode, 0,
                                 f"{rel}: {result.stderr}")


class ServiceDescription(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "service" / "native-service.json"
        self.raw = self.path.read_text(encoding="utf-8")
        self.data = json.loads(self.raw)

    def test_valid_json_and_required_keys(self):
        for key in ("native_stages", "vlm_stage", "health_path",
                    "ready_path", "backoff_s"):
            self.assertIn(key, self.data, f"missing key {key}")
        self.assertIsInstance(self.data["native_stages"], list)
        self.assertTrue(self.data["native_stages"])

    def test_host_is_specific_loopback(self):
        host = self.data["host"]
        self.assertNotIn(host, ("", "0.0.0.0", "::", "[::]"))
        self.assertTrue(host.startswith("127."), f"host is {host!r}")

    def test_no_internal_tokens(self):
        lowered = self.raw.lower()
        for token in FORBIDDEN_TOKENS:
            self.assertNotIn(token, lowered)


class CudaBenchGuidance(unittest.TestCase):
    def test_non_cuda_build_reports_guidance(self):
        if shutil.which("zig") is None:
            self.skipTest("zig not on PATH")
        with tempfile.TemporaryDirectory(prefix="umm-zig-") as tmp:
            result = run(
                ["zig", "build", "cuda-bench",
                 "--cache-dir", str(Path(tmp) / "cache"),
                 "--global-cache-dir", str(Path(tmp) / "global"),
                 "--prefix", str(Path(tmp) / "prefix")],
                ROOT, timeout=600)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cuda-bench needs -Dcuda",
                      result.stdout + result.stderr)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(
        __import__(__name__))
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if outcome.wasSuccessful():
        print("PASS!")
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
