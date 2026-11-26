# Server-side online-rollout HTTP attestation

## Scope and evidence status

`umm-infer serve --online-rollout-config PATH` implements the server side of the
accepted `umminfer-online-grpo-plan-v1` transport contract. It binds fresh
HTTP/SSE responses to a trusted plan, a unique scheduled attempt, exact canonical
request JSON, behavior-policy and sampling hashes, and independently verified
native-model identity evidence.

This is a **CPU-only transport/configuration implementation**. The tests use
small fake weight files and deterministic generic IPC stages. They are not model
inference evidence. Real native Qwen loaded-model verification, real online
rollout generation, online optimization, sampled behavior token/logprob
correctness, and model-quality outcomes remain **NOT_RUN**. Behavior evidence is
always `not_exposed`; no logprobs, masks, or token ids are invented.

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
7. Startup validation does not prove continuing liveness of a native process or
   cryptographically authenticate its manifest. Restart and re-pin configuration
   after a model/stage reload. The future real-stage gate below must prove the
   producer-to-process binding.

Unmarked requests do not enter the attestation path and receive no attestation
headers. Marked requests on an unconfigured server fail without proof. A valid
scheduled request receives headers on its 200 response and on subsequent HTTP
errors after binding, including 400, 409, and 503. Malformed, stale, cross-group,
sampling-drifting, duplicate-header, or otherwise unbound requests fail without
proof. Cancellation retains the original 200 SSE attestation and the ordinary
cancelled terminal. Authentication and transport failures before route dispatch
are not attested and are incompatible with the accepted rollout client.

## Exact server config schema

All fields are required; unknown fields are rejected. Paths in this document are
relative to the config file. `plan_sha256` and `native_identity_sha256` cover the
**exact file bytes**, including any trailing newline, not canonicalized JSON.

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
umm-test-local
```

The fresh-process suite starts the real gateway and generic deterministic IPC
stages. It verifies 200, attested 400, overload/retry behavior through the C++
fake-stage suite, cancellation, duplicate 409, fresh-process response identity,
and default error-body byte parity. It also launches failing server processes
for missing/stale/future/mismatched/unloaded identities and model-byte drift.
No test contacts a live service or uses a GPU.

## Future real native-stage requirements

A real gate must make the native stage write identity evidence only after the
model and all weights have loaded successfully; bind that evidence to the exact
stage process and a fresh gateway configuration; independently re-hash the actual
loaded model config and every weight file; and prove that the attached unified
VLM stage is that same process. It must then run the accepted rollout client
against isolated real generation, including 200/error/retry/cancel paths and
fresh-process replay, while keeping the resident service untouched.

That gate must separately prove output/model identity and raw-SSE binding. Before
any `exposed` behavior claim, it must also produce and validate the accepted final
behavior block and prove tokenizer, sampled token-id, logprob, and mask alignment
against the behavior model. This task does not implement or claim any of those
real-stage results; all remain **NOT_RUN**.
