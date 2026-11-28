# Interactive Storyboard Web candidate and raw-replay gate (Task #77, 2026-11-27)

## NS-06/NS-08 paired revision (current)

The owner-required paired gate now uses the exact frozen natural NS-06 café
reference/audio input and no preset, modality, routing, sampling, or frame-budget
interaction. Existing developer controls are hidden unless Advanced is opened;
Advanced stayed closed. The default JSON action is lossless, with compact legacy
JSON explicitly nondefault.

| Dimension | Before: official main `91310b36` | After: `655e906b` |
| --- | --- | --- |
| Script labels detected in initial raw stream | 2 | 3 |
| Natural keyframes | 1 | 3 |
| Click-to-animate I2V previews | 1 | 3 |
| Clean narrative without directive leakage | FAIL | PASS |
| Source-bound Revise/new visual | NOT_IMPLEMENTED | PASS |
| Cancellation, fresh retry, gateway reconnect | PASS | PASS |
| Default export schema | compact v1 | raw-SSE/media replay v1 |
| Exact text/history restored | FAIL | PASS |
| Omitted media after import | 2 | 0 |
| Browser checks | 14 PASS / 4 FAIL | 19 PASS / 0 FAIL |
| Hard North-Star status | NOT_MET | **Engineering improvement only** |

All four preregistered engineering gates are MET: NS-06 counts/revision/cancel/
retry, NS-08 default export/import/reconnect, no After regression, and paired raw/
binary/model identity. Both arms used the same GPU 0, native binaries, model
identities, 512x512 reference PNG SHA-256
`21f3092c4e2ca7dc115478a6110f61083e4c2c1074ad3872d1ff6ac8dd855173`, and
14.62-second spoken input SHA-256
`f3ab51c2d5d0ff9f3b834ea4ac143eabaf61ab24009f9a36c43ed478bd549749`.
The After revision's VLM/edit source was exactly
`9f6273e4bbe97df4f0a685d81d518e28ce93703a567d3170539ff8e287e826f5`.

The final After archive has four turns, 671 events, 309 raw artifacts, three
direct-rendered I2V results, and 7,783,680 PCM bytes. Independent raw verification
passed, and the 332-file self-excluded external manifest passed complete coverage.

- Preregistration: `benchmarks/storyboard/ns06-ns08-preregistration.json`,
  SHA-256 `0bbecc117e6f4d759667ed0128d62c851b978adacd75ba2d21cd8b2af6da550b`.
- Machine scorecard: [storyboard-ns06-ns08-scorecard.json](evidence/storyboard-ns06-ns08-scorecard.json),
  SHA-256 `1c84fb4f8d0730e3b242bfa1b7eb32ecdcf68ead6605df2da56b7def4c50b72f`.
- Final archive: SHA-256
  `1baa2f66c7f9adc9a8ce9f8c0a5254a3203133569f367df3583c4ba65985cc50`.
- Independent archive verification: `9ea1fe799bd2270b219e77c1827dead7ca1d9f53696bd7dc11f7ed3b4cbf0888`.
- External manifest: `7ecd32455f9ebe88c48150a088d9431583f698d8306f709decce77472cba7c86`;
  fresh raw+manifest verification: `29e420077c79e47626c4cdf0ca154b69f1677fe6f841bf634585f1a8f8d348ae`.

**Hard-gate limits are unchanged:** human character/style consistency is NOT_RUN,
so NS-06 is NOT_MET; macOS and human walkthrough are NOT_RUN, so NS-08 is NOT_RUN.
There is no full NS-06/NS-08 PASS, scoreboard promotion, production-readiness,
deployment, or resident-readiness claim. `scoreboard_update_allowed=false`.

The first scorer attempt selected a non-ASR response and returned NOT_PROMOTED;
that report is retained externally. The corrected independent scorer selects the
first raw stream containing the actual ASR final and the observed initial slot
count, yielding the unchanged-evidence result above. Earlier browser attempts
also exposed and repaired direct-Animate source binding, retained variants,
automatic import restoration, transport-only cancellation, and stops before any
response bytes. Every negative attempt remains in the external bundle.

## Historical revision-0 isolated journey (not an NS gate)

The record below used a keyboard-activated preset and explicit frame controls.
It is retained as real engineering evidence, but does not satisfy the natural
zero-manual-control NS journey and is not used for NS-06/NS-08 promotion.

## Result and scope

**ISOLATED_CANDIDATE_PASS** for the executed Web workflow on one explicitly idle
NVIDIA H20 SM90 physical GPU 0. This is a safe candidate/readiness record only:
there was no deployment, resident-service contact or upgrade, and no claim of
resident readiness.

The run started from official main
`a4320e7678748a2336f74e3c2477c21b30805fb8`, fast-forwarded through accepted
Task #73 `e3b97ac6df001e0d5f0b4ca189995a1be8f7cbca`, and executed final
candidate source `1176aa83dbb3038bbb44a1f5778b9d816d6efd5a`.
Task #75/#76 branches and GPUs 1/7 were not used. No scheduler, native model,
or serving implementation was changed by the Task #77 fixes.

The compact machine-readable record is
[storyboard-web-sm90.json](evidence/storyboard-web-sm90.json). Screenshots,
raw SSE, media, request bodies, harnesses, negative attempts, model/file
manifests, and logs remain outside Git.

## Actual user journey

Chrome for Testing 156.0.8078.4 was automated with Playwright 1.65.1 against
the real `web/index.html` served by the isolated `umm-infer` gateway. The
browser exercised actual focus, keyboard, DOM, network, Blob media, and Web
Audio paths.

1. Activated **Storyboard I2V Chain** with keyboard Enter and sent with
   Ctrl+Enter. The real Qwen3.8 stream produced clean prose, a FLUX.2 image,
   explicit `source=slot-0` Wan I2V video, and native Qwen3-TTS PCM. The page
   contained no narrative directives/source suffixes; both visual artifacts
   decoded and video playback reached nonzero dimensions.
2. Clicked **Revise** on the existing slot-0 image and supplied a user-directed
   cobalt-blue revision. The outgoing request bound exactly the same source
   bytes to both VLM `image_url` context and FLUX `image.image` editing:
   `212083df654daf894d07ec0a7b220f1289a047cbae553a1872ca1bb22aa06cb0`.
   A new image and explicitly anchored video completed.
3. Exported compact JSON and **Replay JSON**, reset with Alt+N, re-imported the
   archive, and selected **Replay raw SSE**. Import verified raw bytes,
   narrative events, terminal legs, slot identities, continuity references,
   CAS media and PCM before mutation. Local replay reconstructed media/audio
   with zero new chat/network requests; re-export preserved exact raw bytes.
4. Started another real generation and pressed Escape while it was in flight.
   The turn ended Stopped, the source slot became aborted rather than
   shimmering, and Replay JSON retained `outcome=cancelled` with 17 events and
   no falsely completed media. The cancelled prose was not added to retry
   history. A fresh story then completed with new artifacts and no starts on a
   closed AudioContext.
5. New chat/reset stopped active resources and cleared the prompt, attachment,
   edit-source and revision state. Real page reload/reconnect also completed
   later requests without cross-session artifacts.
6. Ran real native video-input requests for default omission, explicit Uniform
   4/8/16, and Temporal 16 (`0.02`/`0.25`). Every request reached a terminal;
   default omitted `frame_selection`, while explicit budgets were exact JSON
   integers.
7. Ran a separate non-narrative image + manual-audio turn with `audio.stream`
   omitted. The browser decoded a real 602,924-byte, 12.56-second WAV and its
   image; Replay JSON re-import and local replay restored both controls.

The final story/revision archive has 137 events, 63 artifacts, and
1,470,720 bytes of PCM. The cancellation archive has no completed PCM artifact:
this run does **not** claim active-audio barge-in. Physical playback, listening,
and realtime microphone/barge-in remain separate hardware/protocol gates.

## Candidate fixes demonstrated by the Showcase

All fixes are focused in the Web client and deterministic verification:

- source-bound **Revise** action, including clearing any unrelated manually
  selected edit image;
- cross-session/request rejection, duplicate artifact/chunk suppression, and
  immutable artifact attribution;
- terminal settlement for pending/running/missing/failed visual slots, with
  unresolved responses never presented as cleanly complete;
- preserved HTTP cancellation outcomes and exclusion of cancelled prose from
  follow-up history;
- clean render/microphone/media/object-URL cleanup on reset/import;
- bounded 96 MiB raw capture per response and 128 MiB replay-import ceiling;
- opt-in `umminfer-web-session-replay-v1` archives containing exact raw SSE,
  transport chunks, narrative events, media metadata/digests, visual revision
  references, terminal legs, and outcomes while legacy JSON stays base64-free;
- pre-mutation verification of raw/artifact hashes, event projections, terminal
  kind/legs/request identity, slot/artifact binding, and explicit earlier-image
  anchors;
- replay-stop/error restoration of the historical outcome, plus parsed PCM
  detection independent of JSON whitespace.

The independent stdlib verifier is
[tests/storyboard_replay_verify.py](../tests/storyboard_replay_verify.py). It
does not import the producer or the existing SSE replay implementation.

## Raw evidence and independent verification

| Evidence | SHA-256 |
| --- | --- |
| Final story/revision Replay JSON | `ebc9c915d920ff8d8ba39429d7bc62cfc98a85471f55de8fc239ece01438297d` |
| Final cancellation Replay JSON | `cf9ce7a9b558dd1e6466ba87e1fa131c1001e4060139195fb983933131f662f1` |
| Final WAV Replay JSON | `7b002ac58c0812d023d0c1b94c5d67f18683caac504563b1e4c683236deb47f0` |
| Story independent verification | `b0887f3feb1d29a265d9849d45dd014da6736581434447bbc6a072f876df9b98` |
| Cancellation independent verification | `b35520d50d7df0ef63ffe95638eb4fea47e38e8fa1c55371f4a37ff66463c26a` |
| WAV independent verification | `5899c5a5c605acb1e3a09b47207dee628783848c6070770d187ca88f3d36e7ea` |
| External 109-file self-excluded `SHA256SUMS` | `a28b2c01122df3c671113f008c0692d80ec008a905564d5e10f56bec5f05e0bb` |
| Fresh raw-archive + complete-manifest verification | `ee166904c5deccc7c514937f98cb1a28479a48c418e73665eba5dc57c82acee6` |
| Final server lifecycle trace | `8e9041fa4e36bdc0ed052873494be9d5f29028110321e59692381d3f20d8885a` |
| Final stack/model/process report | `9cddc34d1756f6ff574fa492e7a21c6f13b28860fc518268bd57f2fe7b17f0b2` |

The manifest excludes itself and covers exactly all other 109 external evidence
files. The verifier rejected event, raw-media, terminal/outcome, chunk-attribution,
manifest-coverage, unresolved-slot, and cross-session tampering in deterministic
fresh-process tests. Eight verifier tests passed.

### Source, binary, and model identity

| Object | SHA-256 |
| --- | --- |
| Web source | `08ea1cec41482836b028f47c731c73452dcaec43adfc843e712597a34dd2b568` |
| Gateway binary | `c7898fc0f7c22958cac2da2cd0393939a6b7e55bf54a3b216c4dd99ef0e9f18e` |
| Qwen3.8 stage binary | `9c2f692b7f7b69d32fe31f36bde67c1b455361677ae23fbd5c3c54ebc6e3a9ef` |
| Qwen3-TTS stage binary | `b311ed10a230008c6e43d8cafb7654ac203374a03774c58ce850343d4979d456` |
| Wan stage binary | `19ab38c8b83e274e0762d220daea18d31818606c2faa7738d31e6760179d9b2a` |
| Qwen3.8-27B local model tree | `22e638404d0e413007064c47d5c98ad1696a26a0e980eb2fce83cbbfcca0a304` |
| FLUX.2-klein-4B local model tree | `10c1b2d7096cd6c2798fa9d2b9aecfab5163117cbf3d66de6f32f9a2e8f17997` |
| Qwen3-TTS-1.7B-CustomVoice local model tree | `69c90c52d9d88390f4b4f540b09acf40f5ead7080650585ff20d2e144aa5e69d` |
| Wan2.2-TI2V-5B-Diffusers local model tree | `b118ac07b533cf7d68a4806ec70bdf39041515673e44535afeda5cfefe761616` |

Model revisions are `UNKNOWN`; exact local tree/file hashes are authoritative.
The serial stack kept Qwen3.8 resident while switching exactly one of FLUX.2,
native TTS, and native Wan at a time. GPU 0 began at 3 MiB / 0% with no compute
process. The supervisor's final sample was 3 MiB / 1%; a separate later sample
was **3 MiB / 0% with no compute process**. Only recorded task-owned processes
were stopped. The local browser tunnel was closed.

## Failed attempts retained

- The initial external supervisor missed its trace trigger. The request reached
  the server's 600-second deadline and aborted; the raw trace/report and cleanup
  were retained. The corrected supervisor used the actual FLUX completion log
  plus unified lifecycle traces and completed repeated serial handoffs.
- One browser harness incorrectly accepted only `finish_reason=stop` and
  rejected `Completed · length`. That harness assertion was corrected without
  weakening any media, slot, terminal, or replay requirement.
- Independent review reproduced revision-source, terminal/slot-binding,
  chunk-attribution, historical-outcome, resource-cleanup, raw-capture-bound,
  and JSON-whitespace PCM defects. They were repaired and the complete real and
  deterministic journeys rerun on the final source.

## Reproduction checks

```sh
python3 -I tests/web_ui_contract_test.py
python3 -I tests/storyboard_replay_verify_test.py
python3 -I tests/sse_replay_test.py
python3 -I tests/narrative_slot_reference_live.py --self-check
python3 -I tests/public_speech_pcm_test.py
python3 -I tests/showcase_test.py
python3 -I tests/client_media_cli_e2e.py zig-out/bin/umm-infer
umm-test-local
```

Final results: Web 24/24 checks; verifier 8/8 fresh-process tests; raw replay
10/10; public PCM 21/21; Showcase 28/28; client media 13/13; full
`umm-test-local` PASS. The 23-check deterministic Chromium journey also passed,
but is contract-fixture evidence, not a substitute for the real model run.

To verify a retained archive independently:

```sh
python3 -I tests/storyboard_replay_verify.py \
  --archive /path/to/live-story.replay.json \
  --out /path/to/independent-report.json
```

Add `--manifest /path/to/SHA256SUMS --manifest-root /path/to/evidence` to require
exact, self-excluded external evidence coverage.

## Explicit limits

- **NOT_RUN:** physical microphone, physical speaker/audible playback,
  active-audio barge-in, perceptual/acoustic listening, and macOS execution.
- **NOT_RUN on this final Web candidate:** realtime WebSocket reconnect and
  playback-ACK cursor capture. HTTP page reload/reconnect passed. Accepted
  Task #70/#73 server/network records remain separate evidence, not a substitute
  for this unrun final-Web dimension.
- **NOT_RUN:** generalized performance, model-quality improvement, deployment,
  resident-service validation, and any resident upgrade.
- Browser media decode, event ordering, exact bytes, terminal states, and digital
  audio scheduling are verified; visual or acoustic quality is not claimed.

## Task #84 — first-turn retention through export/import/reconnect

The accepted Task #83 parent is `84f82ee879efe384f1eea6a7ee2544d9d3b7b274`;
the executed integration candidate is `3ec56a24796a89e99f03c1cb9365b8f9140e264b`,
which merges official main `9c02225451252001c11021b389a7792caf03d503` before
final verification. Before uses that integration's server/native binaries with the
official legacy Web client; After uses the same runtime with the fixed client.
The focused Web change is to send the existing `session_history: true` field from
the first default stateful turn, rather than waiting until local history already
contains a completed exchange. The server, prompt, native model, and public schema
are unchanged. Direct API omission or explicit `session_history: false` remains
stateless.

A paired real Qwen3.8/TTS Chromium journey used zero manual modality, frame, or
seed controls on isolated physical GPU 1. Both arms completed **Replay JSON export
→ Alt+N clear → verified import → local raw replay → HTTP reload/reconnect →
natural follow-up**. The re-import after reload uses the retained archive against
the same still-running server; this is not a claim that a new or restarted server
can recover its empty process-local store.

| Metric | Before | After |
| --- | --- | --- |
| Setup retained by server | false | true |
| Follow-up history messages after leading system | 1 (current user only) | 3 (setup user + exact setup assistant + current user) |
| Follow-up context terms `solar` / `water` | false / true | true / true |
| Raw PCM preserved in setup archive | 32,770,560 B / 1,079 artifacts | identical |
| Whole original → import → re-export file | byte-identical | byte-identical |
| Explicit stateless read/write isolation | PASS | PASS |
| Cross-session and cancelled-turn exclusion | PASS | PASS |
| Native prompt replay | 6/6 PASS | 6/6 PASS |

Before/After whole-file SHA-256 are respectively
`cbb21d03e72b0a01076ee89bde4e0a88240e25d3b8205773510cde48b5740b2a` and
`c56a4ef8a7023718a5786e84a652612ab86fdb1cfb5170e932ad0ba7a378e4ad`;
each immediate re-export has the identical digest. The complete self-excluded
26-file arm manifests have SHA-256
`407aedfb2a3fe297a2a25e1d8aa3109459168ea40f69ab0c0304edebd8ec7664` and
`8266065eab290759d990be6d7cbf91cd4ff8cfcaa95fb9772b91f74890a4c466`.
The fresh paired independent report is `PASS`, with SHA-256
`f617e335c928a2407e1bb486c8632e1c9c497b1aeaaa11e970c7c7fe4427fd94`.
The independent verifier also passes 13/13 deterministic tests, including a
transport-only cancellation with zero public bytes and no fabricated request id.

Accepted Task #77 story/cancellation/WAV archives were separately re-verified with
its complete external manifest: 137 events + 63 artifacts + 1,470,720 PCM bytes,
17 cancelled events, and 602,924 WAV bytes. Its historical whole-file roundtrip
differed only in `exported_at`; the new live comparison fixes the browser clock and
compares complete bytes, not just raw SSE. Exact current answers, native prompts,
metrics, and hashes are in the
[Task #84 scorecard](evidence/ns08-first-turn-history-scorecard.json).

This is a **paired engineering improvement**, not full NS-08 PASS. macOS execution,
human walkthrough, physical microphone/speaker, active-audio barge-in, perceptual
quality, new/restarted-server restoration, realtime WebSocket ACK capture, and
resident deployment remain `NOT_RUN`/`NOT_CLAIMED` as applicable. All failed
external-driver attempts were excluded from measured arms and retained externally;
no model output or historical scorecard was edited to remove a failure.
