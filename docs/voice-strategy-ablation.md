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
- [`benchmarks/ablations/voice-strategy-prompts.json`](../benchmarks/ablations/voice-strategy-prompts.json): 3 fixed prompts covering distinct utterance lengths:
  1. `short_factual` (single concise factual sentence)
  2. `physics_explanation` (3-sentence physics explanation)
  3. `vivid_narrative` (4-sentence vivid scene)
- [`benchmarks/ablations/voice-strategy-replay.json`](../benchmarks/ablations/voice-strategy-replay.json): 6 deterministic canned SSE responses with recorded `latency_us` contract observations (`CONTRACT_REPLAY`, `REAL_MODEL_EVIDENCE: NOT_RUN`).

## 3. Metrics

Computed per prompt case and aggregated per arm (`median`, `p95`, `min`, `max`, plus explicit `sample_count`/`total_count` and `coverage`):

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

Plain HTTP is restricted to loopback addresses by default (`--allow-remote-http` is required for non-loopback plain HTTP).

### Unit & Contract Tests

```bash
python3 -m unittest tests/voice_ablation_test.py
```

## 5. Evidence & Claim Boundaries

| Mode | `evidence_class` | `real_model_evidence` | Permitted Claims | Forbidden Claims |
| --- | --- | --- | --- | --- |
| `--replay-only` | `CONTRACT_REPLAY` | `NOT_RUN` | Protocol conformance, single-variable invariance, tamper detection, deterministic metric computation & byte-identical outputs. | Real model latency/TTFA speedup, hardware stall rate, or perceptual audio quality. |
| `--base-url` | `UNCONTROLLED_LIVE` | `UNCONTROLLED_LIVE` | Uncontrolled live server observations at the measured endpoint. | Isolated controlled benchmark claims (without a dedicated controlled measurement manifest) or perceptual audio quality. |

- Full-duplex barge-in latency and native audio token policies from Row 3 of `docs/vision.md` §3.2 are intentionally scoped outside this two-arm `/v1/chat/completions` HTTP SSE ablation (see [full-duplex.md](full-duplex.md) for the WebSocket Realtime barge-in contract).

## 6. Measured SM90 Live Observation (`UNCONTROLLED_LIVE`) — 2026-11-26

Executed against the live 5-GPU SM90 stack (`Qwen3.8-27B` CUDA Graph on GPU 3 + native `Qwen3-TTS-12Hz-1.7B` dual CUDA Graph on GPU 5) via `benchmarks/ablations/voice-strategy-sm90.json` (`fixed_matrix_sha256=1fa3d84619dd54cd...`, `SHA256SUMS` 3/3 OK, `evidence_class=UNCONTROLLED_LIVE`):

| Policy (`audio.stream`) | Completed | TTFT median (ms) | TTFA median / p95 (ms) | TTFA-after-TTFT median (ms) | Inter-chunk gap p95 (ms) | Stalls (`>500 ms`) | Audio chunks / bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_response_tts` (`false`) | 3/3 | 714.3 | 6,261.0 / 11,296.6 | 5,546.7 | 372.3 | 0 / 1 (0.0%) | 4 / 5,426,052 |
| `sentence_streaming_tts` (`true`) | 3/3 | 712.0 | **825.0 / 826.4** | **112.8** | **212.3** | **0 / 229 (0.0%)** | 232 / 6,149,280 |

- Cross-arm TTFA speedup: **`7.59×`** (`-5,436.0 ms` median TTFA), **`49.15×`** faster post-TTFT audio onset (`5,546.7 -> 112.8 ms`), **`0 / 229` stalls (`0.0%`)**.
- Scope boundary: this is an ad-hoc `UNCONTROLLED_LIVE` run against the resident service; isolated `CONTROLLED_LIVE` multi-repetition execution and perceptual audio quality remain `NOT_RUN` / `NOT_COVERED`.
