# Observability

Four surfaces: a Prometheus endpoint, per-GPU gauges sampled over NVML, an
opt-in structured request trace, and opt-in request-native allocator envelopes.
The first three live in the `umm-infer` server process; allocator ownership is
observed in instrumented native stages and projected by the server. No new
server dependency is linked: the text form is assembled by hand and NVML is
opened with `dlopen`.

```
GET /metrics        Prometheus text exposition (version 0.0.4)
UMM_TRACE=1         one JSON line per finished request
```

`/metrics` sits behind the same `--api-key` check as every other route. A
scraper must send `Authorization: Bearer <key>` when a key is configured.

## Metrics

All names are prefixed `umm_`. Labels are fixed identifier sets only: `stage`,
`leg`, `request_type`, `status`, `source`, `queue`, `route`, `index`, `name`.
No label ever carries prompt text, request bodies, keys or file paths.

### Per-stage

The coordinator counts a request against the stage it was submitted to. For a
unified chat request that means the ASR stage gets the entry submission and the
VLM stage gets the child submission, so the per-stage numbers are a real
breakdown rather than one total.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `umm_stage_requests_total` | counter | `stage`, `request_type`, `status` | Submissions (`submitted`) and terminals (`completed`, `failed`, `aborted`) |
| `umm_stage_in_flight` | gauge | `stage` | Submitted and not yet terminal |
| `umm_stage_duration_seconds` | histogram | `stage`, `request_type` | Submit to terminal, as the coordinator observed it |

`request_type` is derived from the payload shape: `chat`, `transcription` or
`other`.

### Engine and legs

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `umm_request_total` | counter | `request_type`, `status` | Unified engine terminals |
| `umm_request_duration_seconds` | histogram | `request_type`, `status` | Submit to terminal, end to end |
| `umm_leg_requests_total` | counter | `leg`, `request_type`, `status` | Terminals per leg (`asr`, `vlm`, `image`, `tts`) |
| `umm_leg_duration_seconds` | histogram | `leg`, `request_type` | One leg, submit to terminal |
| `umm_queue_depth` | gauge | `queue` | `unified_admission` is the backpressure queue; `coordinator_pending` is the backlog waiting for a stage to become ready |
| `umm_aborts_total` | counter | `source` | Abort broadcasts: `client`, `timeout`, `shutdown` |
| `umm_abort_latency_seconds` | histogram | – | Submission to observed abort |

The leg durations are measured by the server. The next four are phase times the
worker measured itself and reported in the stage result. A worker that does not
report a phase records nothing; the histogram stays at zero samples rather than
counting a fabricated zero.

| Metric | Type | Meaning |
|---|---|---|
| `umm_vlm_prefill_seconds` | histogram | VLM prompt prefill (`prefill_ms` in the stage result) |
| `umm_vlm_decode_seconds` | histogram | VLM token decode (`decode_ms`) |
| `umm_tts_first_frame_seconds` | histogram | Time to the first generated codec frame (`first_frame_ms`), reported by the TTS stage regardless of whether `audio.stream=true` PCM16 chunk streaming is enabled. |
| `umm_tts_total_seconds` | histogram | TTS synthesis (`total_ms`) |
| `umm_image_denoise_step_seconds` | histogram | `denoise_ms / steps` per image request |

### HTTP

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `umm_http_requests_total` | counter | `route`, `status` | Responses at the HTTP boundary, every early return included |

### Process

| Metric | Type | Meaning |
|---|---|---|
| `umm_build_info{component="umm-infer"}` | gauge | Always 1; scrape target marker |
| `umm_trace_enabled` | gauge | 1 when `UMM_TRACE=1` |

## GPU metrics

Sampled on each scrape, read-only. The server never writes a device setting.

| Metric | Labels |
|---|---|
| `umm_gpu_utilization_percent` | `index`, `name` |
| `umm_gpu_memory_used_bytes` | `index` |
| `umm_gpu_memory_total_bytes` | `index` |
| `umm_gpu_temperature_celsius` | `index` |
| `umm_gpu_power_watts` | `index` |
| `umm_gpu_sm_clock_mhz` | `index` |
| `umm_gpu_memory_clock_mhz` | `index` |
| `umm_gpu_nvml_available` | – |

`umm_gpu_nvml_available` is 1 when `libnvidia-ml.so.1` (or `libnvidia-ml.so`)
loaded and `nvmlInit` succeeded. When it is 0 the endpoint still answers and no
per-GPU sample is emitted; the rest of the metrics are unaffected. This is the
normal state on a host without an NVIDIA driver. (The standalone unit suite
`tests/metrics_test.cpp` was removed in `ab51cf5`; `/metrics` scraping is
exercised by `tests/soak_native.py`.)

Every field except `name` also needs one `nvmlDeviceGet*` call to succeed for
that device; a device that refuses a reading emits 0 for that field.

## Request-native allocator telemetry

Off by default. `POST /v1/chat/completions` accepts
`telemetry: {"native_allocator": true}` and returns a terminal `telemetry`
envelope only for that request. Serial instrumented Qwen3.8 and Wan2.2 stages
report allocated/reserved byte peaks for explicit native CUDA blocks first owned
by a new request epoch. Batches, concurrent/shared execution, uninstrumented
stages, external proxies, and malformed/missing data remain `NOT_COVERED` with
null peaks. No request-only value is inferred from another request or a resident
baseline.

This is a different scope from every `umm_gpu_*` gauge and from controlled
`measurement.resources`. Those remain whole-device NVML samples; request peaks
live only under `request_scoped_telemetry` in ablation reports. Epoch reset means
a new ownership log, not a global CUDA counter reset. The projection is bounded,
allow-listed, and excludes model/internal paths. Internal
`request_telemetry` result keys are stripped from nonstream artifact descriptors;
only the terminal envelope is public. See
[native-service.md](native-service.md#opt-in-request-native-allocator-telemetry)
for field definitions, lifecycle behavior, and the SM90 gate. Real serial CUDA
measurements passed on 2026-11-26 for Qwen3.8 and Wan2.2 on isolated physical
GPU 0, including off/on output parity, cache repeats, lifecycle boundaries, and
independent 50 ms whole-device NVML samples. Batched, concurrent/shared, and
other-backend request attribution remains `NOT_COVERED`; host/fake-allocator
tests alone are not GPU evidence.

## Request tracing

Off by default. Set `UMM_TRACE=1` to enable and `UMM_TRACE_FILE` to choose the
output path (default `./umm-trace.jsonl`, which is the service root because the
launcher changes into it).

One line per finished request, one JSON object:

```json
{"request_id":"req-3","event":"request_end","request_type":"chat","stage":"vlm",
 "start_epoch_ms":1763370000123,"end_epoch_ms":1763370001456,"status":"completed",
 "duration_ms":1333.2,"asr_ms":210.4,"vlm_ms":1105.7}
```

A leg that did not run is omitted, never reported as `-1`. The record type has
no field that can hold request content.

The file is capped at 64 KiB. When the next line would cross the cap the active
file is renamed to `<path>.1` (replacing the previous one) and writing continues
at zero, so both files stay at or below 64 KiB. A line larger than the cap is
dropped rather than truncated.

Verified behaviour: with `UMM_TRACE` unset no file is created at all.

## Scrape example

```sh
curl -s http://127.0.0.1:43301/metrics | grep -E 'umm_stage_requests_total|umm_queue_depth'
```

Check that the counters move:

```sh
curl -s http://127.0.0.1:43301/metrics | grep 'umm_request_total'   # before
curl -s -X POST http://127.0.0.1:43301/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-native","messages":[{"role":"user","content":"hello"}]}'
curl -s http://127.0.0.1:43301/metrics | grep 'umm_request_total'   # after
```

Useful queries once scraped:

```promql
histogram_quantile(0.95, sum by (le) (rate(umm_request_duration_seconds_bucket[5m])))
sum by (stage) (rate(umm_stage_requests_total{status="failed"}[5m]))
sum by (leg) (rate(umm_leg_duration_seconds_sum[5m])) / sum by (leg) (rate(umm_leg_duration_seconds_count[5m]))
umm_stage_in_flight
umm_gpu_memory_used_bytes / umm_gpu_memory_total_bytes
```

## What is not covered

Stages run as separate processes. The server sees what crosses the coordinator
boundary: submissions, terminals and timings. It cannot see a stage's internal
scheduler queue, so `umm_queue_depth` reports the server-side queues
(`unified_admission`, `coordinator_pending`) and `umm_stage_in_flight` reports
requests outstanding at each stage boundary. A stage-side queue depth would
need an endpoint in each stage process and is out of scope here.
