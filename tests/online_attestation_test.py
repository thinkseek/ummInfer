#!/usr/bin/env python3
"""CPU-only server HTTP attestation tests with deterministic generic IPC stages.

Usage: python3 -I tests/online_attestation_test.py zig-out/bin/umm-infer
No model, GPU, optimizer, credential, or live service is used.
"""

import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
import umminfer_online_rollout as online  # noqa: E402

FIXTURES = ROOT / "tests/online_attestation_canonical_fixtures.json"
MODEL_ID = "attestation-fixture-model"
POLICY_REVISION = "fixture-policy-revision-001"
MODEL_REVISION = "fixture-native-revision-001"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(port, body, headers=None, path="/v1/chat/completions"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    payload = body if isinstance(body, bytes) else json.dumps(
        body, ensure_ascii=False, indent=2).encode("utf-8")
    conn.request("POST", path, body=payload,
                 headers={"Content-Type": "application/json", **(headers or {})})
    response = conn.getresponse()
    raw = response.read()
    result = (response.status,
              {key.lower(): value for key, value in response.getheaders()}, raw)
    conn.close()
    return result


def duplicate_contract_request(port, body, headers):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest("POST", "/v1/chat/completions")
    conn.putheader("Content-Type", "application/json")
    for key, value in headers.items():
        conn.putheader(key, value)
    contract = online.RESPONSE_HEADERS["contract"]
    conn.putheader(contract, "1")
    payload = canonical(body)
    conn.putheader("Content-Length", str(len(payload)))
    conn.endheaders(payload)
    response = conn.getresponse()
    raw = response.read()
    result = (response.status,
              {key.lower(): value for key, value in response.getheaders()}, raw)
    conn.close()
    return result


def make_plan(template=None):
    messages = [[{
        "role": "user",
        "content": [
            {"type": "text", "text": "中文 é 🪐 fresh generation"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,AA=="},
             "nested": {"media": ["video/mp4", {"reference": "slot-0"}],
                        "floats": [0.0, -0.0, 1e-7, 0.0001,
                                   1000000000000000.0, 1e16,
                                   0.30000000000000004]}},
        ],
    }], [{
        "role": "user",
        "content": "second trusted prompt group",
    }]]
    return {
        "schema": online.PLAN_SCHEMA,
        "batch_id": "server-attestation-fixture-batch-v1",
        "behavior_policy": {
            "policy_id": "fixture-server-policy",
            "revision": POLICY_REVISION,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_sha256": "",
            "model_config_sha256": "",
        },
        "sampling": {"max_tokens": 32, "temperature": 0.25, "top_p": 0.9},
        "request_template": template if template is not None else {
            "response_format": {"type": "text"},
            "stream_options": {"include_usage": False, "include_asr_partials": False},
        },
        "reward": {"spec": "fixture_reward:null_reward"},
        "split_seed": "server-attestation-split-v1",
        "eval_ratio": 0.0,
        "transport": {
            "timeout_s": 2.0, "max_attempts": 3,
            "retry_statuses": [503], "retry_transport": False,
            "retry_timeouts": False,
        },
        "groups": [
            {"prompt_id": "unicode-media", "messages": messages[0],
             "expected_visual_modalities": [], "seeds": [-42, 43]},
            {"prompt_id": "second-group", "messages": messages[1],
             "expected_visual_modalities": [], "seeds": [-201, 202]},
        ],
    }


class Fixture:
    def __init__(self, root, template=None):
        self.root = Path(root)
        self.model = self.root / "model"
        self.model.mkdir(parents=True)
        config_bytes = b'{"model_type":"fake-contract","hidden_size":1}\n'
        weight_bytes = b"fake native weight bytes\x00\xff\n"
        (self.model / "config.json").write_bytes(config_bytes)
        (self.model / "weights.safetensors").write_bytes(weight_bytes)
        self.identity = {
            "schema": "umminfer-native-stage-identity-v1",
            "stage_name": "vlm",
            "state": "loaded",
            "loaded_at_epoch_ms": int(time.time() * 1000),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_root": "model",
            "config": {"path": "config.json", "sha256": sha(config_bytes),
                       "bytes": len(config_bytes)},
            "weight_files": {"weights.safetensors": {
                "sha256": sha(weight_bytes), "bytes": len(weight_bytes)}},
            "model_sha256": "",
        }
        self.identity["model_sha256"] = sha(canonical(self.identity["weight_files"]))
        self.plan = make_plan(template)
        self.plan["behavior_policy"]["model_sha256"] = self.identity["model_sha256"]
        self.plan["behavior_policy"]["model_config_sha256"] = \
            self.identity["config"]["sha256"]
        self.plan = online.prepare_plan(self.plan)
        self.schedule, _ = online.build_schedule(self.plan)
        self.config = {
            "schema": "umminfer-online-rollout-server-v1",
            "plan_path": "plan.json",
            "plan_sha256": "",
            "native_identity_path": "native-identity.json",
            "native_identity_sha256": "",
            "native_stage_name": "vlm",
            "max_identity_age_ms": 300000,
        }
        self.rewrite_plan()
        self.rewrite_identity()

    def rewrite_plan(self):
        payload = canonical(self.plan) + b"\n"
        (self.root / "plan.json").write_bytes(payload)
        self.config["plan_sha256"] = sha(payload)
        self.rewrite_config()

    def rewrite_identity(self):
        payload = canonical(self.identity) + b"\n"
        (self.root / "native-identity.json").write_bytes(payload)
        self.config["native_identity_sha256"] = sha(payload)
        self.rewrite_config()

    def rewrite_config(self):
        (self.root / "server.json").write_bytes(canonical(self.config) + b"\n")

    @property
    def config_path(self):
        return self.root / "server.json"


class RunningStack:
    def __init__(self, binary, root, fixture=None):
        self.binary = str(binary)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fixture = fixture
        self.port = free_port()
        self.processes = []
        self.log = (self.root / "stack.log").open("wb")
        ipc = self.root / "ipc"
        completion = f"ipc://{ipc}/completion"
        abort = f"ipc://{ipc}/abort"
        for name in ("asr", "vlm"):
            self.processes.append(subprocess.Popen([
                self.binary, "stage", "--name", name,
                "--recv", f"ipc://{ipc}/{name}",
                "--coordinator", completion, "--abort", abort,
            ], stdout=self.log, stderr=subprocess.STDOUT))
        command = [
            self.binary, "serve", "--host", "127.0.0.1", "--port", str(self.port),
            "--model", MODEL_ID, "--entry", "asr",
            "--entry-endpoint", f"ipc://{ipc}/asr",
            "--vlm-entry", "vlm", "--vlm-entry-endpoint", f"ipc://{ipc}/vlm",
            "--completion", completion, "--abort", abort,
        ]
        if fixture is not None:
            command += ["--online-rollout-config", str(fixture.config_path)]
        self.processes.append(subprocess.Popen(
            command, stdout=self.log, stderr=subprocess.STDOUT))
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if self.processes[-1].poll() is not None:
                    raise AssertionError("server exited before readiness")
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1)
                    conn.request("GET", "/ready")
                    response = conn.getresponse()
                    response.read()
                    conn.close()
                    if response.status == 200:
                        return
                except OSError:
                    pass
                time.sleep(0.02)
            raise AssertionError("server did not become ready")
        except Exception as exc:
            self.log.flush()
            details = (self.root / "stack.log").read_text(
                encoding="utf-8", errors="replace")[-4000:]
            self.close()
            raise AssertionError(f"{exc}\n{details}") from exc

    def close(self):
        for process in reversed(self.processes):
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in reversed(self.processes):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def server_command(binary, fixture, root):
    ipc = Path(root) / "ipc"
    return [
        str(binary), "serve", "--host", "127.0.0.1", "--port", str(free_port()),
        "--model", MODEL_ID, "--entry", "asr",
        "--entry-endpoint", f"ipc://{ipc}/asr",
        "--vlm-entry", "vlm", "--vlm-entry-endpoint", f"ipc://{ipc}/vlm",
        "--completion", f"ipc://{ipc}/completion", "--abort", f"ipc://{ipc}/abort",
        "--online-rollout-config", str(fixture.config_path),
    ]


def attestation(item, plan, status, attempt=0):
    expected = online._expected_attestation(
        item, plan, http_status=status, behavior_evidence="not_exposed",
        attempt=attempt)
    return {name: str(expected[key]) for key, name in online.RESPONSE_HEADERS.items()}


class OnlineAttestationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(sys.argv[1]).resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix="umm-oa-", dir="/tmp")
        cls.root = Path(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def case_dir(self, name):
        path = self.root / name
        path.mkdir()
        return path

    def test_01_python_canonical_fixtures(self):
        document = json.loads(FIXTURES.read_text(encoding="utf-8"))
        self.assertEqual(document["schema"],
                         "umminfer-online-attestation-canonical-fixtures-v1")
        for case in document["cases"]:
            parsed = json.loads(case["input"])
            self.assertEqual(canonical(parsed).decode("utf-8"), case["canonical"])
            self.assertEqual(sha(canonical(parsed)), case["sha256"])

    def test_02_success_rejections_and_default_parity(self):
        fixture = Fixture(self.case_dir("success"))
        item = fixture.schedule[0]
        with RunningStack(self.binary, self.case_dir("success-stack"), fixture) as stack:
            headers = online._request_headers(item, fixture.plan, 0)
            status, response_headers, raw = request(
                stack.port, json.dumps(item["body"], ensure_ascii=False,
                                       indent=1).encode("utf-8"), headers)
            self.assertEqual(status, 200, raw)
            for key, header in online.RESPONSE_HEADERS.items():
                self.assertEqual(response_headers[header],
                                 attestation(item, fixture.plan, 200)[header], key)
            self.assertEqual(response_headers[online.RESPONSE_HEADERS["behavior_evidence"]],
                             "not_exposed")
            leaked = "\n".join(response_headers[key] for key in
                               online.RESPONSE_HEADERS.values())
            self.assertNotIn(str(fixture.root), leaked)
            self.assertNotIn("127.0.0.1", leaked)

            status, duplicate_headers, raw = request(stack.port, item["body"], headers)
            self.assertEqual(status, 409, raw)
            for key, header in online.RESPONSE_HEADERS.items():
                self.assertEqual(duplicate_headers[header],
                                 attestation(item, fixture.plan, 409)[header], key)

            marked_off = copy.deepcopy(headers)
            marked_off[online.RESPONSE_HEADERS["generation"]] = "cached"
            status, stale_headers, raw = request(stack.port, fixture.schedule[1]["body"],
                                                 marked_off)
            self.assertEqual(status, 400, raw)
            self.assertFalse(set(online.RESPONSE_HEADERS.values()) & stale_headers.keys())

            cross = online._request_headers(fixture.schedule[1], fixture.plan, 0)
            status, cross_headers, raw = request(stack.port, fixture.schedule[2]["body"],
                                                 cross)
            self.assertEqual(status, 400, raw)
            self.assertFalse(set(online.RESPONSE_HEADERS.values()) & cross_headers.keys())

            drifted = copy.deepcopy(fixture.schedule[1]["body"])
            drifted["temperature"] = 0.5
            status, drift_headers, raw = request(
                stack.port, drifted,
                online._request_headers(fixture.schedule[1], fixture.plan, 0))
            self.assertEqual(status, 400, raw)
            self.assertFalse(set(online.RESPONSE_HEADERS.values()) & drift_headers.keys())

            forged = online._request_headers(fixture.schedule[1], fixture.plan, 0)
            forged[online.RESPONSE_HEADERS["model_sha256"]] = "ab" * 32
            status, forged_headers, raw = request(
                stack.port, fixture.schedule[1]["body"], forged)
            self.assertEqual(status, 400, raw)
            self.assertFalse(set(online.RESPONSE_HEADERS.values()) & forged_headers.keys())

            exposed = online._request_headers(fixture.schedule[1], fixture.plan, 0)
            exposed[online.RESPONSE_HEADERS["behavior_evidence"]] = "exposed"
            status, exposed_headers, raw = request(
                stack.port, fixture.schedule[1]["body"], exposed)
            self.assertEqual(status, 400, raw)
            self.assertFalse(set(online.RESPONSE_HEADERS.values()) & exposed_headers.keys())

            status, duplicate_wire_headers, raw = duplicate_contract_request(
                stack.port, fixture.schedule[1]["body"],
                online._request_headers(fixture.schedule[1], fixture.plan, 0))
            self.assertEqual(status, 400, raw)
            self.assertFalse(set(online.RESPONSE_HEADERS.values()) &
                             duplicate_wire_headers.keys())

            status, ordinary_headers, raw = request(stack.port, fixture.schedule[1]["body"])
            self.assertEqual(status, 200, raw)
            self.assertFalse(set(online.RESPONSE_HEADERS.values()) & ordinary_headers.keys())
            enabled_error = request(stack.port, {"messages": []})

        with RunningStack(self.binary, self.case_dir("default-stack")) as stack:
            disabled_error = request(stack.port, {"messages": []})
            self.assertEqual(enabled_error[0], disabled_error[0])
            self.assertEqual(enabled_error[2], disabled_error[2])
            self.assertFalse(set(online.RESPONSE_HEADERS.values()) & disabled_error[1].keys())
            status, disabled_headers, raw = request(
                stack.port, fixture.schedule[1]["body"],
                online._request_headers(fixture.schedule[1], fixture.plan, 0))
            self.assertEqual(status, 400, raw)
            self.assertFalse(set(online.RESPONSE_HEADERS.values()) & disabled_headers.keys())

    def test_03_error_response_and_fresh_process(self):
        fixture = Fixture(self.case_dir("error"), template={"stop": ["unsupported"]})
        item = fixture.schedule[0]
        observed = []
        for index in range(2):
            with RunningStack(self.binary, self.case_dir(f"error-stack-{index}"),
                              fixture) as stack:
                status, headers, raw = request(
                    stack.port, item["body"], online._request_headers(item, fixture.plan, 0))
                self.assertEqual(status, 400, raw)
                expected = attestation(item, fixture.plan, 400)
                for key, header in online.RESPONSE_HEADERS.items():
                    self.assertEqual(headers[header], expected[header], key)
                observed.append({key: headers[key] for key in
                                 online.RESPONSE_HEADERS.values()})
        self.assertEqual(observed[0], observed[1])

    def test_04_startup_identity_fail_closed(self):
        cases = (
            "missing_identity", "stale_identity", "future_identity", "not_loaded",
            "stage_mismatch", "model_mismatch", "weight_drift", "weight_set_mismatch",
            "plan_pin_mismatch", "identity_pin_mismatch", "config_schema",
            "plan_model_config_mismatch",
        )
        for name in cases:
            with self.subTest(name=name):
                root = self.case_dir("reject-" + name)
                fixture = Fixture(root)
                if name == "missing_identity":
                    (root / "native-identity.json").unlink()
                elif name == "stale_identity":
                    fixture.identity["loaded_at_epoch_ms"] -= 300001
                    fixture.rewrite_identity()
                elif name == "future_identity":
                    fixture.identity["loaded_at_epoch_ms"] += 60001
                    fixture.rewrite_identity()
                elif name == "not_loaded":
                    fixture.identity["state"] = "registered"
                    fixture.rewrite_identity()
                elif name == "stage_mismatch":
                    fixture.identity["stage_name"] = "other-vlm"
                    fixture.rewrite_identity()
                elif name == "model_mismatch":
                    fixture.identity["model_id"] = "other-model"
                    fixture.rewrite_identity()
                elif name == "weight_drift":
                    with (root / "model/weights.safetensors").open("ab") as handle:
                        handle.write(b"drift")
                elif name == "weight_set_mismatch":
                    fixture.identity["model_sha256"] = "00" * 32
                    fixture.rewrite_identity()
                elif name == "plan_pin_mismatch":
                    (root / "plan.json").write_bytes(canonical(fixture.plan))
                elif name == "identity_pin_mismatch":
                    (root / "native-identity.json").write_bytes(canonical(fixture.identity))
                elif name == "config_schema":
                    fixture.config["schema"] = "unknown"
                    fixture.rewrite_config()
                elif name == "plan_model_config_mismatch":
                    fixture.plan["behavior_policy"]["model_config_sha256"] = "11" * 32
                    fixture.rewrite_plan()
                with (root / "reject.log").open("wb") as log:
                    process = subprocess.Popen(server_command(self.binary, fixture, root),
                                               stdout=log, stderr=subprocess.STDOUT)
                    try:
                        code = process.wait(timeout=5)
                        self.assertNotEqual(code, 0, name)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: online_attestation_test.py /path/to/umm-infer")
    unittest.main(argv=[sys.argv[0]])
