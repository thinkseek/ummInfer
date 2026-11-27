# Cross-slot visual continuity: image → image edit → native I2V (Task #76, 2026-11-27)

## Result and scope

**CONTROLLED_LIVE / PASS** on one explicitly idle NVIDIA Hopper SM90 H20,
physical **GPU 7**, starting from official main
`a4320e7678748a2336f74e3c2477c21b30805fb8`. Accepted Task #74 geometry work was
merged before execution. The final merged runtime/live source is commit
`a37a84a7464c3e31ad9762bde46b6d64cf09a215`, which also merges official Wave 99
through `91310b36184c86a1c147debb27db8643cc0cb3e4`; the earlier verifier-only
correction is `4f40318d9c4e384f5adf8ca18ce6d8a26ff8664a`. The final run repeated
the pre-merge capture with byte-identical media and reference/PCM joins.

One real public VLM narrative established an image in `slot-0`, edited that exact
image in `slot-1`, and generated native Wan2.2 I2V from the edited image in
`slot-2`, with streaming speech. Separate real controls covered a wrong/self
reference, cancellation with three finalized dependents, native wrong-hash
rejection, and valid recovery. Exact raw public/native records and an independent
fresh-process verifier passed remotely and after a fresh local copy.

This proves **explicit content-addressed reference propagation**, not perceptual
character similarity. Human/perceptual character or style review is **NOT_RUN**.
No learned router, training, optimizer, deployment, or generalized quality claim
is part of this task.

## Preregistered paired NS-06 / NS-08 review follow-up

After supervisor review, the same frozen sketch, real generated voice request, text
plan, and six-slot auto narrative were executed once per arm on GPU 7, using the
untouched parent `91310b3` and candidate `a37a84a`. Preregistration SHA-256 is
`25dc4e5798d73294edfcae43c12923a7b41fd9d6a79cbf8133766da5b2efe38c`; the accepted
harness is `8dd14321de4b6a0bd0a50e1490fd1494f8985d28`. No prompt or reference plan
changed after model output was observed.

| Frozen measure | Parent Before | Candidate After |
| --- | --- | --- |
| Completed keyframes / I2V previews | 1 / 1 | **3 / 3** |
| Exact image-edit links | 0 | **2** |
| Exact keyframe→I2V links | 1 | **3** |
| Main terminal / reconnect | complete / `READY` | complete / `READY` |
| Clean prose, ASR transcript, PCM | identical to After | identical to Before |
| Native TTS legs versus frozen target | 4 / 3 — **NOT_MET** | 4 / 3 — **NOT_MET** |

Candidate per-shot chains are:

1. slot 0 PNG `6d7c77f39a955e2f6fe25eca00a96bd0cdc7d4c47a61e2bfe4c2118b90ff5713`
   → slot 1 MP4 `817a0744d6a17e0a3f098b8bdca3d6a52b1d29da7687d39e0c7dce0ce5b85851`,
   first-frame/motion MAE `3.825130 / 27.079818`;
2. slot 0 → exact FLUX edit slot 2
   `5f405f3c718b5148c913c1e6cd7304f2c3469a794c417c784de2c9e0db45a4bf`
   → slot 3 MP4 `4da5754685585c898c71450a6015f7b7abefe9bc9bce7439dec51f9256fc8105`,
   MAE `4.049870 / 7.072396`;
3. slot 0 → exact FLUX edit slot 4
   `65e04f3c46e20050b62295a0e6206b665a324779d5ef8e8e6e781cc0a68e4592`
   → slot 5 MP4 `dd45f0336f364f31a8b8f1bf183e5babd861468c6819b762a093e14c87719cc1`,
   MAE `4.124740 / 24.247396`.

All three outputs decode as 704×480 H.264/yuv420p, 29 frames at 16 fps; all
keyframes decode as 512×512. The parent shares the same slot-0/slot-1 media bytes
but fails slots 2–5 without generation. This is the explicit promotable technical
reference gain, not a perceptual-similarity result.

For NS-08, unmodified Web production helpers performed export JSON/Markdown →
actual `newChat` clear → parse/apply import with network forbidden → byte-exact
re-export → real text-only reconnect from imported history. Both arms preserve
text, artifact hashes/bytes, and resolved reference metadata, and a fresh Node
replay is byte-identical. The existing schema still omits raw media and PCM;
`importedMediaOmitted` remains visible, so full restoration and human default-UI
acceptance are **NOT_RUN**, not claimed. No NS-08 regression was observed.

The final paired report is `1e00192fcbf33bffd6c5b9f55a694049e76aa5d75b9b6b68ba7320ae3bcc74b0`;
305-file self-excluded manifest is
`593bb5da128f434ec9db893e35442081e61766212d9e45aa6702b76d24dcd532`.
Remote fresh check `54988bd98e67939b55f784bf1ddf1497d1f35931fd28a82c2b3bdda27047141f`
and local fresh raw/client replay
`421a60c4223519c10dc641bdfb26f1b590ca43bd40df1220e6792f46849d7b22` both pass.
See the [compact NS scorecard](evidence/ns06-paired-76-sm90.json) for exact native
submits, per-leg speech joins, provenance, retained harness failures, and limits.
**Neither full NS-06 nor full NS-08 PASS is claimed, and neither score changes.**

## Focused runtime contract

The existing `narrative.continuity_mode="i2v_chain"` and public slot/anchor fields
are reused. No public request field was added and the director prompt is
unchanged:

```text
[[image: establish character and style]]
[[image: preserve that character and style at night | source=slot-0]]
[[video: the character turns as the camera orbits | source=slot-1]]
```

- An explicit source must be an earlier **image** slot in the same unified request.
  Image dependents perform a FLUX.2 edit; video dependents perform native I2V.
- Pending/running/failed/aborted, missing, self, forward, duplicate, malformed,
  wrong-modality, conflicting-input, and cross-session references fail closed.
  There is no text-only, wrong-artifact, stale-session, or unconditioned fallback.
- Before dispatch, `UnifiedEngine` recomputes SHA-256 over immutable settled source
  bytes. FLUX receives exact MessagePack binary `inputs.image`, source slot, and
  digest; Wan additionally receives `anchor_crop=center_cover`.
- FLUX validates explicit image bytes/index/digest before invoking Diffusers.
  Legacy data-URL edits and native Wan's existing anchor checks are preserved.
- A failed intermediate source recursively finalizes transitive dependents.
  Cancellation and primary failure finalize pending **and running** slots.
  Optional-output cancelled/error terminals now include those finalized `legs`;
  default text-only terminal bytes remain unchanged.
- Raw-SSE/schema-v2 validation accepts resolved image or video dependents only
  when the selected earlier image's completed CAS SHA-256 matches. Reconstruction
  restores the explicit suffix for either visual modality. No training ran.

Focused C++, FLUX, replay, and reconstruction tests failed before the runtime fix:
image references were rejected, FLUX lacked binary/hash validation, and replay
accepted only video dependents. The same controls pass after the fix.

## Exact real reference chain and decoded media

All requests used seed 42 and temperature 0. The valid public request was
`http-3`; it emitted one final `complete` and ended with `in_flight=0`.

| Slot | Real operation | Exact input source | Output bytes / SHA-256 | Decoded output |
| --- | --- | --- | --- | --- |
| 0 | FLUX.2 establishment | none | 301,844 / `01af1c29f95aa2362b57067bd6592a032617b8c46d91509e7b946d7b0d4e150b` | PNG, 512×512 |
| 1 | FLUX.2 edit | slot 0, exact 301,844 bytes, SHA `01af1c29…150b` | 336,884 / `1eddde7e82a79401e3710f4e6750c68df81c0c8b0da76f5c78c40958bf3e45b3` | PNG, 512×512 |
| 2 | native Wan2.2 I2V | slot 1, exact 336,884 bytes, SHA `1eddde7e…45b3` | 137,640 / `1866e18d5bbe035ebd5be123fcebb0396c54731c781bb7e465a4469c83131dcf` | H.264/yuv420p, 704×480, 29 frames, 16 fps |

The retained native FLUX edit submit is 302,599 bytes, SHA-256
`0df64ea9f5c5b600b7ef932f24a4ee5f1e3326c8bb8df272bfa7a4592e8e99e7`.
The native Wan submit is 337,638 bytes, SHA-256
`46bd6e2322ada5fe1b42f3ee790e3625572e1d7d72045c32bc95f38315e29775`.
Independent MessagePack decoding proved byte equality with the selected public
CAS source, not merely equal metadata. The Wan log has exactly one
`backend=native-cuda-wan`, `i2v=1` request for `http-3:slot:2`, with `i2v=0`
absent. FLUX stopped before Wan loaded; Wan start/readiness followed public
slot-1 completion.

### Raw public stream and terminal legs

The valid stream has **139 unified events**, **68 artifacts in 68 chunks**:
67 VLM deltas, three visual slots, 68 artifact events, and one complete. Final raw
SSE is 3,354,341 bytes, SHA-256
`9e507cfdd7abadcad9f85820fa303603798fbb0dd3b4fdabf97d86ac04a6b15c`.
The three request bodies' combined canonical SHA-256 is
`ad800a642827181bc28ec6410425b6c398591fad9dd2d1d5bb811cf637af6ebe`.

The terminal visual legs are all `completed`; slot 1 carries source index 0 /
`01af1c29…150b`, and slot 2 carries source index 1 / `1eddde7e…45b3`.
Clean prose SHA-256 is
`f5e70a918ef6a6877f6cdcc75a4549e212c3422ff8d3febaf1a0ef9b1e4563c1`.
No directive or source suffix appears in prose, render prompts, or TTS input.

### Exact speech/text joins

Nine native TTS legs produced **65 public PCM chunks, 1,693,440 bytes** total.
The ordered public PCM concatenation SHA-256 is
`b41e6085d206287664de082362c34e12cd26d47beeb94d1b6421fc8b08f6c5c4`.
The joined native TTS text SHA-256 is
`3ae12f4527ca8c25a667c8d005c0ed1adf2282c8c4ce4e877a7125b458ef7dd3`.
Its linguistic content equals clean prose; only documented sentence/clause span
whitespace is normalized. Every public PCM chunk equals its native chunk, and
each leg's native/public chunk concatenation equals its terminal WAV PCM exactly.
All 65 chunks are non-silent 24 kHz mono PCM16. Per-leg PCM and terminal-WAV hashes
are retained in the compact machine record.

## Numeric continuity/motion proxies

Using the unchanged `center-cover-64x40-rgb8-frame0-frame28-v1` metric, source
slot 1 versus decoded video frame 0 MAE is **2.218620** (gate ≤48); frame 0 versus
frame 28 motion MAE is **11.511328** (gate ≥1). Source/recovery PNGs and all video
RGB buffers were independently decoded and hashed.

These are numeric continuity/motion proxies only. They are **not** a perceptual
similarity metric and no file-hash-derived quality score is claimed.

## Negative controls and valid recovery

| Control | Raw result |
| --- | --- |
| Wrong/self `slot-0` video reference with no earlier image | Public request `http-1` completed with one failed slot, exact error `source slot slot-0 is not an earlier slot in this request`, zero artifacts and zero native video submits |
| Real cancellation after all three slots opened | `http-2` produced one `cancelled`; all three visual legs `aborted`; zero visual artifacts, one native source submit, zero dependent/video submits; 25 audio artifacts were strictly pre-terminal |
| FLUX source with correct bytes but 64-zero declared hash | `native-76-wrong-hash` failed before Diffusers with `image does not match anchor_sha256`; no successful generation log |
| Valid native retry with exact source/digest | `native-76-valid-retry` returned a decoded 512×512 PNG, 317,716 bytes, SHA `dc568834635d847d8a708e64b7521d6693f8d0b6dd4e29fff10c3aba31c43e3c` |
| Valid public retry after cancellation | `http-3` completed the full image/image/I2V/TTS chain above |

The cancellation reached FLUX at denoising step 20/50. Raw capture/replay rejects
post-terminal unified events or late visual artifacts. Deterministic tests retain
forward, duplicate, missing, malformed, wrong-modality, cross-session,
failed-source, cancelled-source, tampered-source, wrong-hash, and valid-record
recovery controls.

## Provenance and independent verification

| Object | SHA-256 |
| --- | --- |
| `umm-infer` server (merged) | `241681951bd2cde69032c84d30984ea2713428fa38d8f2280710207c53862509` |
| Qwen3.8 stage (merged) | `472dbe3758df7eaa830cd460df2cdc58dc3981966468206ca6e1f63e6da76a25` |
| Qwen3-TTS stage (merged) | `0a2b3af3ce14063918ee73d04b7fe5eb43a1584ec811755882201c036c257997` |
| Native Wan stage (merged) | `0f4b5df70061b1bbe092996abc12a585e69cebeeec7bedf6fa1d53b13c0c9284` |
| FLUX stage source | `ab88ae88f3a9207151d4774e4bb000fd385353724e22e7720cf94345a922ef74` |
| Live harness at accepted execution | `a9e8f9f1b64577fc7ced9af4d2fd200fed522403dd09e559d39782c56e8f54d2` |
| Shared live helpers at accepted execution | `895db1cb1943b277d197abffc4e731e00c49b2d895aec017f8fc5a3ad68cacdf` |
| Producer verifier (before correction) | `ea265611d620752af7f3040e233d176c92db40e65583be3dc5600828adc1dee0` |
| Corrected independent verifier | `cc712a000a0703a70504db38df1e1f46ef1624d06eebca4718520adc68823b98` |

Exact model tree identities are Qwen3.8 `22e63840…a304` (29 files), FLUX.2
`10c1b2d7…7997` (25), CustomVoice TTS `69c90c52…69d` (14), and Wan2.2
`b118ac07…616` (30). Model revisions remain `UNKNOWN`; exact local file/tree
hashes are the identity, not an upstream revision claim.

The independent verifier imports neither the live driver nor replay producer. It
parses raw SSE/CAS/MessagePack, reconstructs all public artifacts, verifies every
reference and control, joins all PCM/WAV/text, decodes media/RGB, checks native
logs and process timing, and rehashes supplied binaries/model files remotely.

| Sealed object | Identity |
| --- | --- |
| Final merged `report.json` | `ee21cc780c9ea77486ead5f6b46996b3c00df2d23ee01dbc5ba69ebb9b7fa883` |
| Final `independent-verification.json` | `86f6ef553b63f53b5736ceccc853efedba6ce9b8b934c2079ff42d774b4074b0` |
| Final `summary.json` | `ded3b562e67784b4dd2b44a4a77044c90d286fdf0f7d4300617a034c6d5fda2d` |
| Final `SHA256SUMS` | 439 files / `b135ba7f664d3dc74043ffc9c32ead23a2aa81f3ab4472967fdc1c0fd1c0c29f` |
| Remote fresh manifest check | `be4550e05093faf7d781ad064b68733c7a66697bfd35d03265586c7fdb28a6e1` |
| Local fresh copied-evidence check | `c400c31366190699d2c6e6bb5859bf42a84be88c381a258b83d084d069917c57` |

The manifest covers the exact regular-file set and excludes only itself plus the
task-owned `run/` socket directory. A second remote fresh process checked
all 439 files and external provenance. A fresh local process then checked the same
manifest/raw evidence and decoded the media again; external remote binary/model
rehashing is explicitly marked skipped locally because those paths are absent.
Raw media, per-event records, native frames, PCM, RGB, and logs remain outside Git.

## Retained failed attempts and corrections

No failed attempt is counted as a passing gate:

1. `b0d9683` and `539f5e2` eagerly evaluated `response.read()` in an HTTP-200
   assertion message, consuming the wrong-reference SSE before capture. They ran
   no visual generation. `089e31c` fixed the harness and retained raw failures.
2. `089e31c` produced a valid real cancellation but counted broadcast abort frames
   as extra submits. `07a4b9b` separated submit/abort evidence.
3. `07a4b9b` completed real FLUX/native-Wan work, but the TTS tap incorrectly
   required WAV bytes on a cancelled terminal and stopped forwarding later TTS
   terminals. The task-owned harness was interrupted and cleaned; `04e9f3c`
   retained unsuccessful terminals without media and passed the complete gate.
4. The accepted `04e9f3c` raw capture initially failed verifier-only decoded
   dimension bookkeeping (`KeyError: width`). Commit `4f40318` retains dimensions
   and distinct producer/corrected-verifier hashes. No media was regenerated for
   the correction.
5. Review-only portable-path correction `665e217` removes both task-specific
   absolute output defaults. Live mode now requires a caller-selected `--out-dir`;
   offline self-checks still need no output path. Executed source hashes remain
   distinct from the corrected portable sources, and both fresh local raw replays
   pass without GPU execution or any measured-result change.

## Tests, reproduction, cleanup

Offline/focused results:

- `umm-test-local`: **PASS**, including the new raw/manifest self-checks.
- FLUX contract remote: **16/16 PASS, zero skips**.
- Native video focused CTest: **4/4 PASS**; native public contract-stub
  completed/cancelled/failed/recovery **PASS**, final `in_flight=0` (not real media).
- Narrative reference, raw-SSE replay, SFT reconstruction, Stage C compatibility,
  Wan cancellation, and Task #74 geometry checks: **PASS**.
- Feature-map/documentation validation and hand-maintained AOCI baseline hashes
  are finalized with the integration documentation commit; no AOCI CLI ran.

With exact local assets, an empty output directory, and explicitly idle GPU 7:

```sh
UMMINFER_LIVE_OPT_IN=1 /path/to/venv/bin/python -I \
  tests/cross_slot_visual_continuity_live.py \
  --python /path/to/venv/bin/python \
  --server-binary /path/to/umm-infer \
  --vlm-binary /path/to/umm-native-stage-qwen38 \
  --tts-binary /path/to/umm-native-stage-tts \
  --wan-binary /path/to/umm-native-stage-video \
  --flux-stage python/umminfer_flux2_stage.py \
  --vlm-model /path/to/qwen3.8-27b \
  --flux-model /path/to/FLUX.2-klein-4B \
  --tts-model /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --wan-model /path/to/Wan2.2-TI2V-5B-Diffusers \
  --out-dir "<fresh-output-dir>" \
  --base-commit a4320e7678748a2336f74e3c2477c21b30805fb8 \
  --execution-commit a37a84a7464c3e31ad9762bde46b6d64cf09a215
```

All six task-owned processes in the accepted run returned 0. Final GPU 7 state is
**3 MiB, 0%, zero compute processes**. GPUs 0, 1, and 2–6 were not used; the
resident service was not contacted, restarted, or validated.

## Explicit limits

Perceptual quality, human character/style review, generalized quality,
production deployment, resident-service validation, training/optimization, and
performance improvement remain **NOT_RUN**, **NOT_COVERED**, or unclaimed as
applicable. The supported claim is the measured operational reference-propagation
and lifecycle contract above.
