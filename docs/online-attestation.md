# Server-side online-rollout HTTP attestation

## Scope and evidence status

`umm-infer serve --online-rollout-config PATH` implements the server side of the
accepted `umminfer-online-grpo-plan-v1` transport contract. It binds fresh
HTTP/SSE responses to a trusted plan, a unique scheduled attempt, exact canonical
request JSON, behavior-policy and sampling hashes, and independently verified
native-model identity evidence.

The v1 tests are **CPU-only transport/configuration tests**. They use small fake
weight files and deterministic generic IPC stages, not model inference evidence.
The v2 implementation additionally binds a real native post-load document to the
live process and its `stage_ready` registration. Real rollout evidence is recorded
separately; no online optimization or model-quality claim is made. Behavior
coverage is `exposed` only for a complete validated final five-field block, and
default/truncated/cancelled/unsupported responses remain `not_exposed`.

The trainer contract remains defined by
`python/umminfer_online_rollout.py::RESPONSE_HEADERS` and
`_expected_attestation`. The server adds no response body fields and no
`attestation_sha256` HTTP header. The trainer continues to recompute that digest
using the observed HTTP status.

## Trust boundaries

1. The operator-supplied server config is the local root of trust. It pins the
   exact bytes of a rollout plan and a native-stage identity manifest by SHA-256.
2. The server parses the pinned plan and derives group ids, rollout ids, request
   bodies, request hashes, policy hashes, and sampling hashes itself. Client
   headers can only select and confirm one derived binding. They are never copied
   into a response as proof.
3. The server independently reads and hashes the declared model config and every
   available weight file. It checks byte lengths, per-file hashes, the canonical
   weight-set hash, model id/revision, stage name, loaded state, and evidence age.
   A missing, changed, stale, future-dated, mismatched, malformed, or non-loaded
   identity fails startup before the HTTP listener accepts traffic.
4. The configured native stage name must equal the attached unified VLM stage.
   Fixed-upstream, non-unified, and `--demo` modes cannot enable attestation.
   Supervisor deployments pass the flag through the existing `server_args` list;
   no trainer or optimizer configuration is imported by the server.
5. A `(rollout_id, attempt)` slot is consumed atomically before request
   validation or generation. Reusing it returns 409 rather than replaying cached
   output. Attempts are zero-based and bounded by the plan's `max_attempts`.
6. Local model paths are evidence inputs only. They never enter canonical
   request bytes or response headers. Evidence files must be regular non-symlink
   files; model-file paths are normalized relative paths beneath `model_root`.
7. With server schema v2, the native Qwen process itself hashes the model config,
   every available `.safetensors` file, and its executable only after the real
   engine load returns. It writes a new identity document atomically and carries
   that exact document's SHA-256 in every `stage_ready` heartbeat. The gateway
   independently rehashes the same files, checks pid/start-time/executable and
   freshness, and requires the attached VLM's current ready digest to match on
   every scheduled request. A generic or stale ready frame cannot retain an old
   identity. Restart and re-pin configuration after a model/stage reload.

Unmarked requests do not enter the attestation path and receive no attestation
headers. Marked requests on an unconfigured server fail without proof. A valid
scheduled request receives headers on its 200 response and on subsequent HTTP
errors after binding, including 400, 409, and 503. Malformed, stale, cross-group,
sampling-drifting, duplicate-header, or otherwise unbound requests fail without
proof. Cancellation retains the original 200 SSE attestation and the ordinary
cancelled terminal. Authentication and transport failures before route dispatch
are not attested and are incompatible with the accepted rollout client.

## Exact server config schema

All fields are required; unknown fields are rejected. Schema
`umminfer-online-rollout-server-v1` retains the deterministic generic-stage
contract tests. Schema `umminfer-online-rollout-server-v2` requires the v2 native
identity below, a serial Qwen stage, and the live ready-digest match. Paths in
this document are relative to the config file (absolute local evidence paths are
also accepted). `plan_sha256` and `native_identity_sha256` cover the **exact file
bytes**, including any trailing newline, not canonicalized JSON.

```json
{
  "schema": "umminfer-online-rollout-server-v1",
  "plan_path": "plan.json",
  "plan_sha256": "<64-lowercase-hex>",
  "native_identity_path": "native-identity.json",
  "native_identity_sha256": "<64-lowercase-hex>",
  "native_stage_name": "vlm",
  "max_identity_age_ms": 300000
}
```

`max_identity_age_ms` is in `[1, 604800000]`. `loaded_at_epoch_ms` may be at most
60 seconds in the future and otherwise must be within this age at startup. The
plan uses the existing strict online-plan schema, including equal-size groups,
unique signed-int64 seeds, and `max_attempts` in `[1, 8]`.

## Exact native identity schema

All fields are required; unknown fields are rejected. `model_root` may be
absolute or relative to the identity file. `config.path` and every
`weight_files` key are normalized relative paths below that root. Each declared
file is opened without following a final symlink and independently re-hashed.
The weight-set digest is SHA-256 of canonical JSON for the entire `weight_files`
object, matching the Stage C available-weight convention.

```json
{
  "schema": "umminfer-native-stage-identity-v1",
  "stage_name": "vlm",
  "state": "loaded",
  "loaded_at_epoch_ms": 1795670000000,
  "model_id": "served-model-id",
  "model_revision": "operator-recorded-revision",
  "model_root": "model",
  "config": {
    "path": "config.json",
    "sha256": "<64-lowercase-hex>",
    "bytes": 1234
  },
  "weight_files": {
    "model-00001-of-00002.safetensors": {
      "sha256": "<64-lowercase-hex>",
      "bytes": 1000000
    },
    "model-00002-of-00002.safetensors": {
      "sha256": "<64-lowercase-hex>",
      "bytes": 900000
    }
  },
  "model_sha256": "<sha256-of-canonical-weight_files-object>"
}
```

The plan behavior-policy model id/revision/config hash/weight-set hash must match
this identity, and the model id must equal `--model` (or the configured served
model name). `state` must be exactly `loaded`; a readiness-only declaration is
not sufficient.

## Process-bound native identity v2

The real stage is started with `--identity-out`, `--model-id`, and
`--model-revision` together. Its schema is
`umminfer-native-stage-identity-v2`; it contains every v1 field plus:

```json
{
  "process": {
    "pid": 12345,
    "start_time_ticks": 678901234,
    "executable_path": "/local/build/umm-native-stage-qwen38",
    "executable_sha256": "<64-lowercase-hex>",
    "executable_bytes": 3609224
  },
  "stage_config": {
    "kind": "qwen38",
    "gpu": 7,
    "batch_size": 1,
    "max_new_tokens": 512,
    "max_seq": 8192
  }
}
```

The document is written only after tokenizer and engine construction completes,
with no existing-file overwrite. The server schema must be
`umminfer-online-rollout-server-v2`. The process must remain the same pid/start
instance and executable, the identity must remain within `max_identity_age_ms`,
and the coordinator's currently ready VLM must advertise the exact identity
document digest. Missing, mismatched, generic, stale, or disconnected identity
fails before a `(rollout_id, attempt)` slot is consumed.

## Canonical request JSON

The hash is over semantic JSON, not the HTTP byte framing: compact UTF-8, sorted
object keys, no ASCII escaping of non-ASCII text, and no NaN/Infinity. Formatting,
member order, and insignificant input whitespace do not change it. Integer and
floating-point types remain distinct; signed zero is preserved. Unicode is not
normalized, so NFC and NFD strings remain different. Nested media references are
part of the request object and therefore part of the hash.

Double formatting follows Python `json.dumps(..., sort_keys=True,
separators=(",", ":"), ensure_ascii=False, allow_nan=False)`: shortest
round-trip digits, fixed notation for decimal exponents `[-4, 16)`, scientific
notation otherwise, explicit exponent sign, and at least two exponent digits.
The C++ implementation deliberately does not rely on nlohmann's different
fixed/scientific threshold. Canonical nesting is capped at 256 levels.

`tests/online_attestation_canonical_fixtures.json` is generated and independently
rechecked with Python. Both the C++ tests and fresh-process HTTP tests cover
Unicode, combining text, non-BMP characters, nested media, signed-int64 seeds,
negative zero, subnormal and boundary-exponent doubles, and exact SHA-256. They
also assert that temporary paths and listener addresses do not leak into proof.

## Deterministic verification

```sh
python3 -I tests/online_attestation_test.py zig-out/bin/umm-infer
python3 -I tests/online_rollout_test.py
zig build online-attestation-test \
  -Ddeps-prefix=/path/to/deps -Dcurl-prefix=/path/to/curl
cmake -S native/stage_cli -B <host-build> -DCMAKE_PREFIX_PATH=<deps>
cmake --build <host-build> --target native_identity_test
<host-build>/native_identity_test
umm-test-local
```

The fresh-process suite starts the real gateway and generic deterministic IPC
stages. It verifies 200, attested 400, overload/retry behavior through the C++
fake-stage suite, cancellation, duplicate 409, fresh-process response identity,
and default error-body byte parity. It also launches failing server processes
for missing/stale/future/mismatched/unloaded identities and model-byte drift.
The unified C++ suite separately proves that missing/mismatched live ready
digests reject without client-echoed proof and a matching digest admits the same
not-yet-consumed attempt. `native_identity_test` checks exact document/file/process
hashes and no-overwrite behavior on tiny CPU fixtures. No CPU test contacts a live
service or uses a GPU.

## Real native-stage verification boundary

`tests/online_rollout_live.py` drives the real isolated stack through the actual
trainer client, a frozen two-group by three-seed plan, real overload retry,
separate default/truncated/cancelled/unsupported cases, and byte-identical fresh
replay. `tests/online_rollout_live_verify.py` independently rehashes every model
file and repeats the accepted-bundle replay. Fresh native processes recompute
sampled ids, full log-probabilities, and clean-target masks; this is distinct from
the offline source-span validator.

The real visual artifact is a CPU contract fixture, not generated-image or
perceptual-quality evidence. No online optimizer or model-quality evaluation is
part of this contract. Exact run outcomes, hashes, failed attempts, and cleanup
are recorded separately in [the live evidence document](online-rollout-live.md).
