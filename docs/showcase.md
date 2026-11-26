# North-Star Native Multimodal Showcase (`tools/umminfer_showcase.py`)

`tools/umminfer_showcase.py` is the pure-stdlib verification and showcase runner for ummInfer's five native multimodal journeys against either an in-process deterministic loopback mock server (`--self-check` / `--mock`) or a live OpenAI-compatible ummInfer server (`--url`):

1. **Interleaved Narrative + Global Streaming Voice** — `POST /v1/chat/completions` (`stream=true`, `narrative.enabled=true`, `modalities=[text,image,video,audio]`, `audio.stream=true`).
2. **In-Place Image-to-Video Animation** — `POST /v1/visual/render` (`kind=video`, PNG data URL input).
3. **3-Turn Multi-Turn Native Video Understanding (Prefix KV-Cache)** — `POST /v1/chat/completions` (`video_url` anchored on Turn 0 across 3 turns).
4. **Autonomous Modality Routing Contrast (`visual_mode=auto`)** — pure-text reasoning prompt vs. rich visual scene prompt under identical narrative settings.
5. **Native Cross-Slot Storyboard I2V Chain (`continuity_mode=i2v_chain`)** — single-stream keyframe image + I2V video continuation + streaming voice.

Every run writes `showcase_report.json`, `showcase_report.md`, and a deterministic sorted `SHA256SUMS` file to `--out-dir`.

## Quick Start

Offline deterministic verification (all 5 journeys):

```sh
python3 -I tools/umminfer_showcase.py --mock --out-dir /tmp/showcase-check
```

Live server run (all 5 journeys):

```sh
python3 -I tools/umminfer_showcase.py \
  --url http://127.0.0.1:43301 \
  --model qwen3-native \
  --journey all \
  --out-dir /tmp/umminfer-showcase
```

Select specific journeys with `--journey` (or `--journeys`), e.g. `--journey 3` or `--journey 1,2,3`.

## Journey 3: Opt-In Temporal Video Frame Selection

Journey 3 supports the same opt-in `video_url.frame_selection` CLI flags and fail-closed validation contract as `umm-infer chat` (`src/client/cli.cpp`) and `/v1/chat/completions` (`src/serve/openai_chat.cpp`):

| Flag | Values | Default | Notes |
| --- | --- | --- | --- |
| `--video-frame-policy` | `uniform` \| `temporal` | omitted (`None`) | When both policy and budget are omitted, Turn 1 `video_url` stays byte-compatible (`{"url": "data:video/mp4;base64,..."}` with no `frame_selection` key) and `showcase_report.json` / `showcase_report.md` omit `video_frame_selection`. |
| `--video-frame-budget` | `4` \| `8` \| `16` | omitted (`None`) | Sets `video_url.frame_selection.max_frames`. Defaults to `policy="uniform"` when `--video-frame-policy` is omitted, and is omitted from JSON when unset so default runs remain byte-identical. |
| `--video-motion-threshold` | finite float in `[0.0, 1.0]` | omitted (`None`) | Requires `--video-frame-policy temporal`. Omitted from JSON when unset (never serialized as `null`). |
| `--video-scene-threshold` | finite float in `[0.0, 1.0]` | omitted (`None`) | Requires `--video-frame-policy temporal`. Omitted from JSON when unset (never serialized as `null`). |

### Fail-Closed Validation Rules

- Passing `--video-motion-threshold` or `--video-scene-threshold` without `--video-frame-policy temporal` (including when omitted, paired with `uniform`, or paired only with `--video-frame-budget`) exits with CLI code `2` before any HTTP request or output directory creation.
- Passing `--video-frame-budget` with booleans, floats, non-integer strings, `0`, negative numbers, or any integer outside `{4, 8, 16}` exits with CLI code `2` before any HTTP request or output directory creation.
- Non-finite (`NaN`, `inf`, `-inf`) or out-of-range (`< 0.0` or `> 1.0`) threshold values exit with a non-zero CLI error.
- Unknown `--video-frame-policy` values exit with a non-zero CLI error.
- The offline mock server (`--mock` / `--self-check`) enforces the same `video_url.frame_selection` schema as the native server and rejects malformed objects, unknown fields, thresholds under `uniform`, non-integer or out-of-allowlist `max_frames`, or out-of-range/non-numeric thresholds with HTTP `400`.

### Examples

Offline mock verification of Journey 3 with uniform frame budget (`max_frames=4`):

```sh
python3 -I tools/umminfer_showcase.py \
  --self-check \
  --video-frame-budget 4 \
  --out-dir /tmp/showcase-budget-check
```

Offline mock verification of Journey 3 with temporal frame selection and frame budget (`max_frames=16`):

```sh
python3 -I tools/umminfer_showcase.py \
  --mock \
  --journey 3 \
  --video-frame-policy temporal \
  --video-frame-budget 16 \
  --video-motion-threshold 0.2 \
  --video-scene-threshold 0.35 \
  --out-dir /tmp/showcase-temporal-check
```

Live Journey 3 run against an explicit MP4 with temporal frame selection and budget:

```sh
python3 -I tools/umminfer_showcase.py \
  --url http://127.0.0.1:43301 \
  --model qwen3-native \
  --journey 3 \
  --video-path /tmp/sample.mp4 \
  --video-frame-policy temporal \
  --video-frame-budget 16 \
  --video-motion-threshold 0.02 \
  --video-scene-threshold 0.25 \
  --out-dir /tmp/showcase-j3-temporal
```

When enabled, `video_frame_selection` (including `max_frames` when `--video-frame-budget` is set) is attached to Turn 1's `video_url` object (preserved in the conversation history for Turns 2–3), recorded under `journeys.journey_3.video_frame_selection` in `showcase_report.json`, and displayed in both the Executive Summary table and the Journey 3 section of `showcase_report.md`.

## Optional Native Allocator Telemetry

Pass `--native-allocator-telemetry` to send `"telemetry": {"native_allocator": true}` on `/v1/chat/completions` requests and strictly validate/report request-native allocator telemetry (`COVERED`, `PARTIAL`, or `NOT_COVERED` / `N/C`). Default OFF omits the `telemetry` field completely.

## Optional Behavior Logprobs Evidence

Pass `--behavior-logprobs` (default OFF) to opt into sampled behavior-token logprobs evidence on `/v1/chat/completions` journeys:

- **Default OFF (strict wire compatibility)**: when omitted, requests never include `"behavior"` and `showcase_report.json` / `showcase_report.md` omit `behavior_logprobs` completely.
- **Opt-in ON**: sends `"behavior": {"logprobs": true}` with `stream: true` on `/v1/chat/completions` across Journeys 1, 3, 4, and 5 (Journey 2 uses `/v1/visual/render`, so it is left unchanged and reported honestly as `N/A`).
- **Fail-closed honest reporting (zero fabrication)**: captures the case-insensitive `X-UmmInfer-Behavior-Evidence` HTTP response header (`exposed`, `not_exposed`, or `absent`) and the final SSE `"behavior"` block (`umminfer-behavior-logprobs-v1`). When `exposed`, strictly validates `format`, `tokenizer_sha256`, `token_ids`, finite `logprobs`, and `0`/`1` `mask` of matching non-zero length and renders `Behavior: N tok (M sup)`. When `not_exposed` or `absent`, rejects stray payloads and renders `Behavior: not_exposed` or `Behavior: absent` with `null` counts.

```sh
python3 -I tools/umminfer_showcase.py --self-check --behavior-logprobs --out-dir /tmp/showcase-behavior-check
```

## Optional GUI Session Exports

Pass `--export-gui-sessions` (default OFF) to export each completed journey as a clean, base64-free `umminfer-session-export-v1` JSON file under `<out-dir>/gui_sessions/<journey_id>.session.json` that can be imported directly into both the Web UI (`web/index.html`) and macOS `ummmlx` (`clients/ummmlx`):

- **Default OFF (strict output invariance)**: when omitted, no `gui_sessions/` directory is created and `showcase_report.json`, `showcase_report.md`, and `SHA256SUMS` are unchanged.
- **Opt-in ON (self-check & live modes)**: writes deterministic canonical JSON files (sorted keys, trailing newline) for every completed journey (`journey_1`..`journey_5`), records `gui_sessions` metadata (`journey_id`, `path`, `sha256`, `bytes`, `turn_count`) in `showcase_report.json` and the `## GUI Session Exports` Markdown section, and includes all exported session files in `SHA256SUMS`.
- **Base64-free & honest telemetry**: omits all binary/base64 media payloads (preserving slot metadata, byte counts, SHA-256 digests, and I2V continuity anchors), preserves multi-turn structure (e.g., Journey 3 emits 3 turns; Journey 4 emits 2 turns contrasting pure-text abstention vs. visual scene), and leaves unobserved timing/VRAM/behavior fields as `null` (or honest `NOT_COVERED` / `not_exposed` / `absent`) without fabricating numbers.
- **Journey 3 video frame selection restore**: omits top-level `video_frame_selection` on default runs, `uniform` runs without `--video-frame-budget`, and all other journeys (`journey_1`, `journey_2`, `journey_4`, `journey_5`). When Journey 3 is run with `--video-frame-budget` and/or `--video-frame-policy temporal`, `journey_3.session.json` includes canonical top-level `video_frame_selection` (populating the canonical `0.02` / `0.25` temporal thresholds if omitted) so importing it restores the exact policy, frame budget, and thresholds in both the Web UI (`web/index.html`) and macOS `ummmlx`.

```sh
python3 -I tools/umminfer_showcase.py \
  --self-check \
  --export-gui-sessions \
  --native-allocator-telemetry \
  --behavior-logprobs \
  --out-dir /tmp/showcase-gui-export
```
