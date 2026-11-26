# Real public-SSE capture and replay gate (SM90, 2026-11-26)

## Scope and provenance

This gate passed against source base
`c3bb5d2cdbf02e70f86a16bff85c815612f7311d` (`a30e32a` is an ancestor) with
accepted raw-SSE replay change
`4206e308a8e357ecd2578e82d5879fcd8edc60f1` integrated. The gateway and Qwen
stage were rebuilt from the task source tree in an isolated directory:

| Object | SHA-256 |
| --- | --- |
| `umm-infer` | `807401e2808ce555bb9760d1be037ff7ad7244520390d666829155d2b564675e` |
| `umm-native-stage-qwen38` | `5dea2e9a4d347b3d4476f5418cfc641dca48e77484d064e2b504cde2b4b7850d` |
| Qwen `config.json` | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| `model.safetensors.index.json` | `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df` |

The served model id was `qwen3-native`; the config declares
`Qwen3_5ForConditionalGeneration` / `qwen3_5`, and its revision remains
`UNKNOWN`. The complete hashes for all 18 available Safetensors shards, 11
configuration/tokenizer files, replay source files, requests, and output CAS
objects are in [the hash-only live summary](evidence/sse-replay-live-sm90.json).
No model-quality or generalized-performance conclusion is drawn from these
transport observations.

Only explicitly idle physical GPU 0 was used, with an ephemeral loopback
endpoint. The resident service on port `43301` and every other GPU were left
untouched. Qwen3.8 produced the text and processed the visual input. Its public
unified route requires a distinct legacy entry stage, so a CPU-only fixture was
registered for readiness but received no request. The interleaved output image
came from `umminfer_flux2_stage.py --backend mock` with model id
`contract-fixture-image-not-model-output`, with CUDA visibility disabled. Both
visual input and output bytes are therefore **contract media, not real image
model generation or perceptual-quality evidence**.

## Captured public streams

Each request used `POST /v1/chat/completions` with `stream=true`. Arrival ticks
came from `python.time.perf_counter_ns`, relative to capture start. The full
uncommitted evidence preserves exact raw `.sse` bytes, actual transport chunk
boundaries and per-chunk hashes/ticks, blank-line-delimited SSE frames and
`[DONE]`, per-event source frame indices/hashes/ticks, canonical JSONL, and CAS
input/output media.

| Case | Raw bytes / SHA-256 | Transport chunks | SSE frames (`[DONE]` index) | Normalized events | Unique terminal |
| --- | --- | ---: | ---: | ---: | --- |
| Text-only | 1,654 / `ef5cac415431de682fac40ef19c312a842a11da78b4d0f4bc07f4704daff55b7` | 7 | 7 (6) | 5 | `completed` at event 4 |
| Interleaved visual contract | 235,713 / `b526a215861ea1cdcf75827c23f17ea2f871d40a676124efe71823f246818afd` | 27 | 22 (21) | 20 | `completed` at event 19 |
| Real in-flight `/abort` | 725 / `c2fc93843319b032f26970fd40de0b2addc70f28bac75a2d649518b992a6cb52` | 4 | 4 (3) | 2 | `cancelled` at event 1 |

The abort was issued only after the first public text delta, returned HTTP 200,
and the captured stream subsequently carried one final unified `cancelled`
event followed by `[DONE]`. HTTP 200 itself was not treated as proof of model
cancellation; the terminal SSE event was required.

The visual input PNG CAS was
`1d99634be29f5339715ac8df08f7fa4d86ec41c51106e4b85d8b82594ddb519e` (192
bytes). The contract output PNG CAS was
`e3f2f427686a9e91a370fd99cb2df0162e2c15475cec6eb7b673772c5122532c`
(171,803 bytes). These are byte-integrity fixtures only.

## Replay, framing, and rejection results

A fresh `python3 -I python/umminfer_sse_replay.py` process produced
byte-identical canonical JSONL:

- Source and replay JSONL: 26,040 bytes, SHA-256
  `b63b60dc025a2d438100995a49b0891f4a673015ebeac9256aaab4b5b75ebb1d`.
- All five source/destination CAS files matched byte-for-byte: three raw SSE
  sidecars and two PNGs.
- Strict on-disk audit verified one input media reference, one output media
  reference, and all three raw replay references.
- Every evidence file passed `sha256sum -c SHA256SUMS` (33 files), including
  the replay and tamper-case checksums.
- Split reads (1/2/3/5/8/13-byte pattern) and one coalesced read preserved raw
  bytes, frame offsets/kinds/hashes, normalized event content, artifacts, and
  the unique terminal for all three accepted streams. Their new chunk/tick
  descriptors were intentionally treated as new observations; the original
  accepted capture was not modified.
- Flipping a byte in a source event frame made fresh-process validation exit 2
  with a SHA-256 mismatch.
- Incrementing an event source tick **and recomputing its normalized event
  digest** still made fresh-process validation exit 2 with `source tick`
  mismatch, proving rejection is not merely detection of a stale secondary
  digest.

The focused harness is `tests/sse_replay_live.py`; it requires explicit
`UMMINFER_LIVE_OPT_IN=1`, rejects non-loopback URLs and nonzero `--gpu`, and
opens no socket without the opt-in. `tests/sse_replay_test.py` additionally
covers a real chunked HTTP response so the live helper cannot accidentally
consume a successful response body before capture.

## Cleanup and retained limits

Before shutdown, the isolated `/health` endpoint returned
`{"in_flight":0,"status":"ok"}`. Only the four recorded isolated PIDs were
signaled. They exited (the container now shows zombie entries awaiting init
reaping), the ephemeral endpoint refuses connections, and GPU 0 returned to
`3 MiB`; GPUs 1 and 7 also remained at `3 MiB`. Raw evidence and cleanup status
were preserved outside the tracked tree.

This is a recorded-transport replay result only. It does not establish sampled
token-id recovery, deterministic live inference across hosts, real image/video
generation quality, model improvement, online rollout training, or a
generalized latency/throughput benchmark.
