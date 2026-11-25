# Stage C Ablation #3: Voice Collaboration Strategy

Realizes Row 3 of [vision.md](vision.md#32-完整消融实验体系ablation-matrix) (`语音协同策略: 整段生成后播报 vs 句级流式 TTS`) as a standalone, pure-stdlib, single-variable ablation harness over the public OpenAI-compatible `POST /v1/chat/completions` SSE contract.

## 1. Public Request Contract & Single-Variable Rule

Both arms send `POST /v1/chat/completions` with `stream: true`, `modalities: ["text", "audio"]`, fixed model ID, fixed messages, fixed sampling parameters (`max_tokens: 256`, `seed: 42`, `temperature: 0.0`, `top_p: 1.0`), and fixed audio parameters (`voice: "vivian"`, `format: "wav"`, `sample_rate: 24000`).

**Only `audio.stream` changes across arms:**

| Arm (`voice_strategy`) | `audio.stream` | Public Engine Behavior (`src/pipeline/unified.cpp`) |
| --- | --- | --- |
| `full_response_tts` | `false` | Waits for VLM completion, then synthesizes the full text as one WAV artifact (`mime_type: "audio/wav"`). |
| `sentence_streaming_tts` | `true` | Flushes completed sentences mid-stream to TTS, emitting incremental PCM16 chunks (`mime_type: "audio/pcm"`, `format: "pcm16"`, `stream_index: 0, 1, ...`). |

### Invariance Enforcement

- `fixed_request_sha256` hashes the canonical JSON request with **only** `audio.stream` removed.
- `fixed_matrix_sha256` hashes the ordered per-prompt `fixed_request_sha256` digests for each arm.
- The runner fails closed (nonzero exit) if:
  - `fixed_request_sha256` or `fixed_matrix_sha256` differs across arms,
  - full `request_sha256` fails to differ across arms,
  - any pinned `dataset` or `replay` SHA-256 mismatches,
  - any replay case's `fixed_request_sha256`, `request_sha256`, or `response_sha256` mismatches, or
  - any audio artifact fails strict base64 decode, byte-count check, SHA-256 verification, or WAV/PCM16 framing validation.

## 2. Versioned Inputs

- [`benchmarks/ablations/voice-strategy.json`](../benchmarks/ablations/voice-strategy.json): ablation schema v1 config pinning the dataset SHA-256, replay SHA-256, fixed sampling/audio settings, and `stall_threshold_ms: 500.0`.
- [`benchmarks/ablations/voice-strategy-sm90.json`](../benchmarks/ablations/voice-strategy-sm90.json): ad-hoc SM90 live config (`UNCONTROLLED_LIVE`).
- [`benchmarks/ablations/voice-strategy-controlled-sm90.json`](../benchmarks/ablations/voice-strategy-controlled-sm90.json): controlled SM90 config (`model.id: "qwen38"`, same pinned dataset SHA-256 and fixed sampling/audio settings, plus `controlled_measurement` schema v1 with `warmup_runs: 1`, `repetitions: 3`, `schedule_mode: "interleaved_alternating"`, `require_exclusive_gpu: true`, `max_failed_cases: 0`).
- [`benchmarks/ablations/voice-strategy-prompts.json`](../benchmarks/ablations/voice-strategy-prompts.json): 3 fixed prompts covering distinct utterance lengths:
  1. `short_factual` (single concise factual sentence)
  2. `physics_explanation` (3-sentence physics explanation)
  3. `vivid_narrative` (4-sentence vivid scene)
- [`benchmarks/ablations/voice-strategy-replay.json`](../benchmarks/ablations/voice-strategy-replay.json): 6 deterministic canned SSE responses with recorded `latency_us` contract observations (`CONTRACT_REPLAY`, `REAL_MODEL_EVIDENCE: NOT_RUN`).

## 3. Metrics

Computed per prompt case and aggregated per arm (`median`, `p95`, `min`, `max`, plus explicit `sample_count`/`total_count` and `coverage`; latency and gap metrics `ttft_ms`, `ttfa_ms`, `ttfa_after_ttft_ms`, `terminal_ms`, and `inter_chunk_gap_p95_ms` also report dispersion via `iqr` = Q3 − Q1 and `mad` = median absolute deviation):

| Metric | Definition |
| --- | --- |
| `ttft_ms` | Arrival time of the first non-empty text delta (`vlm_delta` / `choices[0].delta.content`). |
| `ttfa_ms` | Arrival time of the first audio chunk (`stage == "tts"` / `media_type == "audio"`). |
| `ttfa_after_ttft_ms` | `ttfa_ms - ttft_ms` (audio startup lag after first text token). |
| `terminal_ms` | Arrival time of the terminal `complete` event. |
| `audio_chunk_count` | Number of audio chunks received in the SSE stream. |
| `audio_bytes_total` | Total verified decoded audio bytes across all audio artifacts (each artifact base64-decoded and SHA-256 verified). |
| `inter_chunk_gap_p95_ms` | 95th percentile of consecutive inter-audio-chunk arrival gaps (`null` / `NOT_APPLICABLE` when `audio_chunk_count < 2`). |
| `stall_count` / `stall_rate` | Count (and fraction) of inter-audio-chunk gaps strictly exceeding `stall_threshold_ms` (default `500 ms`), directly measuring playback stall rate (断流率). |
| `ttfa_speedup_x` | Cross-arm ratio `full_response_tts.ttfa_ms.median / sentence_streaming_tts.ttfa_ms.median`. |

## 4. Commands

### Deterministic Offline Contract Replay (CPU-only)

```bash
python3 -I python/umminfer_voice_ablation.py \
  --config benchmarks/ablations/voice-strategy.json \
  --replay-only \
  --out /tmp/umm-voice-ablation-replay
```

Outputs written to `--out`:
- `report.json` (top-level summary, provenance, single-variable verification, per-arm aggregates, cross-arm comparison)
- `arms/full_response_tts/report.json`
- `arms/sentence_streaming_tts/report.json`
- `SHA256SUMS` (sorted relative paths; byte-identical across repeated replay runs)

### Live Ad-Hoc Server Run

```bash
python3 -I python/umminfer_voice_ablation.py \
  --config benchmarks/ablations/voice-strategy.json \
  --base-url http://127.0.0.1:43301 \
  --out /tmp/umm-voice-ablation-live
```

Plain HTTP is restricted to loopback addresses by default (`--allow-remote-http` is required for non-loopback plain HTTP). Running `--base-url` without `--controlled-live`—even against a config containing `controlled_measurement`—always records `UNCONTROLLED_LIVE`.

### Controlled Multi-Repetition Live Run (`CONTROLLED_LIVE`)

```bash
python3 -I python/umminfer_voice_ablation.py \
  --config benchmarks/ablations/voice-strategy-controlled-sm90.json \
  --base-url http://127.0.0.1:43301 \
  --controlled-live \
  --provenance-json /path/to/provenance.json \
  --out /tmp/umm-voice-ablation-controlled
```

Controlled-live contract (`controlled_measurement` schema v1, fail-closed):
- **Warmup exclusion**: `warmup_runs` full passes execute live requests (verifying HTTP 200, SSE framing, terminal `complete`, and audio artifact digests) before measured repetitions begin; warmup observations are recorded in `warmup_summary` and strictly excluded from measured arm aggregates.
- **Interleaved alternating schedule**: `schedule_mode: "interleaved_alternating"` alternates arm order across repetitions (`rep 0: [full_response_tts, sentence_streaming_tts]`, `rep 1: [sentence_streaming_tts, full_response_tts]`, `rep 2: [full_response_tts, sentence_streaming_tts]`, …) and records per-repetition breakdowns in `repetitions` and full step order in `schedule_execution`.
- **Zero-failure gate**: `max_failed_cases: 0` fails closed with nonzero exit if any warmup or measured request errors or fails audio validation.
- **Provenance manifest (`--provenance-json`)**: required when `require_exclusive_gpu: true`. Must provide:
  - `commit_sha`: 7–40 lowercase hex git commit SHA;
  - `host_profile`: non-empty object describing host/OS/driver/CUDA/GPU architecture;
  - `binary_sha256`: map containing 64-hex SHA-256 digests for at least `gateway`, `vlm`, and `tts`;
  - `model_manifests`: map containing at least `vlm` and `tts` entries, each with 64-hex `weights_sha256` and positive integer `file_count` (or `shard_count`); when config `model.weights_sha256` is non-null, `model_manifests.vlm.weights_sha256` must match;
  - `gpus`: non-empty list of GPU objects (`index`, `name`, `pre_check`, `post_check`) where every GPU's `pre_check.exclusive` and `post_check.exclusive` (and optional top-level `exclusive`) are strictly `true` and any reported `other_compute_processes` is `0`.

### Unit & Contract Tests

```bash
python3 -m unittest tests/voice_ablation_test.py
```

## 5. Evidence & Claim Boundaries

| Mode | `evidence_class` | `real_model_evidence` | Permitted Claims | Forbidden Claims |
| --- | --- | --- | --- | --- |
| `--replay-only` | `CONTRACT_REPLAY` | `NOT_RUN` | Protocol conformance, single-variable invariance, tamper detection, deterministic metric computation & byte-identical outputs. | Real model latency/TTFA speedup, hardware stall rate, or perceptual audio quality. |
| `--base-url` | `UNCONTROLLED_LIVE` | `UNCONTROLLED_LIVE` | Uncontrolled live server observations at the measured endpoint. | Isolated controlled benchmark claims (without `--controlled-live` and verified provenance) or perceptual audio quality. |
| `--base-url --controlled-live --provenance-json` | `CONTROLLED_LIVE` | `CONTROLLED_LIVE` | Controlled interleaved multi-repetition live latency/TTFA/stall comparison with verified exclusive GPU & binary/weight provenance (`controlled_measurement_status: "COMPLETED"`). | Perceptual audio quality or full-duplex barge-in latency. |

- Full-duplex barge-in latency and native audio token policies from Row 3 of `docs/vision.md` §3.2 are intentionally scoped outside this two-arm `/v1/chat/completions` HTTP SSE ablation (see [full-duplex.md](full-duplex.md) for the WebSocket Realtime barge-in contract).

## 6. Measured SM90 Live Observation (`UNCONTROLLED_LIVE`) — 2026-11-26

Executed against the live 5-GPU SM90 stack (`Qwen3.8-27B` CUDA Graph on GPU 3 + native `Qwen3-TTS-12Hz-1.7B` dual CUDA Graph on GPU 5) via `benchmarks/ablations/voice-strategy-sm90.json` (`fixed_matrix_sha256=1fa3d84619dd54cd...`, `SHA256SUMS` 3/3 OK, `evidence_class=UNCONTROLLED_LIVE`):

| Policy (`audio.stream`) | Completed | TTFT median (ms) | TTFA median / p95 (ms) | TTFA-after-TTFT median (ms) | Inter-chunk gap p95 (ms) | Stalls (`>500 ms`) | Audio chunks / bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_response_tts` (`false`) | 3/3 | 714.3 | 6,261.0 / 11,296.6 | 5,546.7 | 372.3 | 0 / 1 (0.0%) | 4 / 5,426,052 |
| `sentence_streaming_tts` (`true`) | 3/3 | 712.0 | **825.0 / 826.4** | **112.8** | **212.3** | **0 / 229 (0.0%)** | 232 / 6,149,280 |

- Cross-arm TTFA speedup: **`7.59×`** (`-5,436.0 ms` median TTFA), **`49.15×`** faster post-TTFT audio onset (`5,546.7 -> 112.8 ms`), **`0 / 229` stalls (`0.0%`)**.
- Scope boundary: this is an ad-hoc `UNCONTROLLED_LIVE` run against the resident service; perceptual audio quality remains `NOT_COVERED`.

## 7. Controlled SM90 Live Run (`CONTROLLED_LIVE / COMPLETED`) — 2026-11-26

Executed on isolated SM90 GPUs (**GPU 1**: `Qwen3.8-27B` native CUDA Graph with `UMM_QWEN38_PREFIX_CACHE=0`, `--batch-size 1`, `--max-new-tokens 256`; **GPU 7**: `Qwen3-TTS-12Hz-1.7B-CustomVoice` native dual CUDA Graph + idle required entry `Qwen3-ASR-1.7B`) over ephemeral loopback (`127.0.0.1:43911`, resident 5-GPU stack on GPUs 2–6 untouched) via [`benchmarks/ablations/voice-strategy-controlled-sm90.json`](../benchmarks/ablations/voice-strategy-controlled-sm90.json):

- **Schedule & Isolation**: `warmup_runs: 1` (`6` warmup requests excluded from measured aggregates), `repetitions: 3`, `schedule_mode: "interleaved_alternating"` (`18` measured requests = `9` per arm, `0` failures), `elapsed=172.2 s`. Pre-run and post-teardown GPUs 1 & 7 verified `3 MiB / 0` compute processes (`other_compute_processes=0`, post-run `/health` `in_flight=0`).
- **Provenance & Integrity**:
  - `fixed_matrix_sha256`: `c8744b2e2595a59910cc6e760caa6e996662f0955a0266f14be284ea067b519f`
  - `provenance_json_sha256`: `9603f72d34f6377614cf691f929e9081a197beda3a30da7dae5efc430af792d6`
  - `SHA256SUMS`: `3/3 OK` (`report.json` `915ae4c2...`, `full_response_tts` `4021f20f...`, `sentence_streaming_tts` `e1dce512...`)
  - Binaries: gateway `723bce45...`, `qwen38` `0c4f2d4f...`, `tts` `9289696d...`, idle `asr` `bd81c546...`
  - Weights: `Qwen3.8-27B` 18-shard manifest `3ebc83ee...`, `Qwen3-TTS-12Hz-1.7B-CustomVoice` `38b1d597...`, idle `Qwen3-ASR-1.7B` `4b421cb7...`

| Policy (`audio.stream`) | Completed | TTFT median / p95 / IQR (ms) | TTFA median / p95 / IQR / MAD (ms) | TTFA-after-TTFT median / p95 / IQR (ms) | Inter-chunk gap p95 / IQR (ms) | Stalls (`>500 ms`) | Audio chunks / bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_response_tts` (`false`) | 9/9 | 153.4 / 153.7 / 1.6 | 5,285.4 / 10,834.5 / 8,017.1 / 2,594.8 | 5,132.0 / 10,680.8 / 8,015.5 | 29.7 / 0.4 | 0 / 3 (0.0%) | 12 / 16,209,036 |
| `sentence_streaming_tts` (`true`) | 9/9 | 153.3 / 153.7 / 1.5 | **265.8 / 266.6 / 1.9 / 0.7** | **112.5 / 113.1 / 0.2** | **210.6 / 0.1** | **0 / 687 (0.0%)** | 696 / 18,447,840 |

- **Cross-arm Controlled Comparison**:
  - Median TTFA speedup: **`19.88×`** (`-5,019.5 ms`, `5,285.4 -> 265.8 ms`, with sub-millisecond `MAD=0.7 ms` and `IQR=1.9 ms`).
  - Post-TTFT audio onset speedup: **`45.62×`** (`5,132.0 -> 112.5 ms`, `IQR=0.2 ms`).
  - Playback stall rate: **`0 / 687` gaps (`0.0%`)** exceeding `500 ms` (`pooled_inter_chunk_gap_p95_ms = 210.6 ms`).
- **Remaining Claim Boundaries**: perceptual audio quality (`MOS` / listening test) and physical-device full-duplex barge-in remain `NOT_COVERED` / `NOT_RUN`.
