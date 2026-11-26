# Real local Qwen tokenizer export gate

Date: 2026-11-26. Scope: CPU-only tokenization and offline trajectory/SFT
validation. No GPU, model inference, serving process, network model download,
allocator telemetry, or online rollout work was used or changed.

## Identity and runtime

| Field | Exact value |
| --- | --- |
| Declared tokenizer id | `Qwen/Qwen3.8-27B` |
| Upstream revision | `UNKNOWN` (the existing local model directory has no verified Git revision) |
| Loaded class | `Qwen2Tokenizer`, `is_fast=true` |
| Token-id space | `248077` |
| Unknown-token id | `null` |
| Runtime | `transformers 5.6.0`, `tokenizers 0.22.2`, Python 3 in the existing container |
| Loading | `AutoTokenizer.from_pretrained(..., local_files_only=True, use_fast=True, trust_remote_code=False)` |
| Tokenizer model | BPE with NFC normalizer, ByteLevel pre-tokenizer/decoder, no prefix-space/trim/regex rewrite |
| Complete selected-file-bundle digest | `b74ec186875f88cc00e2655347693099c8c2b211c0f8eb777b7dff9ca55d20f9` |

The selected tokenizer-only directory contains exact copies of these six files
from the existing model directory. Model weights are not part of this bundle and
were not read or hashed for this gate.

| Relative path | Bytes | SHA-256 |
| --- | ---: | --- |
| `chat_template.jinja` | 9004 | `af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375` |
| `config.json` | 4312 | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| `merges.txt` | 3353259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `tokenizer.json` | 12809320 | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `tokenizer_config.json` | 9984 | `207c03c41a06a9c3b0cc527d8a56115159c1eac6b59a4ffae85b646e773d074e` |
| `vocab.json` | 6722759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |

`UNKNOWN` is not replaced by a fabricated upstream commit. Reproducibility is
bound to the explicit id, runtime versions, complete file list, sizes, and hashes.

## Why format v2 was necessary

Direct probing before the implementation showed both legitimate incompatibilities
with strict v1:

- Emoji byte fallback emits several token ids for one source codepoint with
  repeated offsets; independent `decode([id])` calls return U+FFFD for partial
  UTF-8 bytes. V1 correctly rejects this as non-byte-exact per-token decode.
- NFC maps `e` + U+0301 to `é`; real offsets include `[0,1]` followed by a source
  gap, and per-token decode returns normalized rather than original bytes. V1
  correctly rejects the gap and normalization.

No v1 rule was weakened. The accepted linked-fixture v1 export remains pinned at
canonical record SHA-256
`ea53770a67a602db250cf0cb8247752a181921544eee0bfc187e87a3eb712bd8`
and export-block SHA-256
`929339094274e0084ea564ba811893e782c7ccc3dc9cbde5a79e5b43916c167e`.
The default export remains format 1. Format 2 is opt-in and limited to verified
NFC + ByteLevel BPE. It stores exact normalized per-token bytes and complete
source-unit mappings; it does not claim that each byte-fallback token is valid
standalone UTF-8 or that original token ranges are disjoint. SFT weights use the
real tokenizer's raw original offsets and match the existing adapter exactly.
See [rl.md](rl.md#format-v2-nfc-normalization-and-byte-fallback-provenance).

## Command and deterministic gate

The following ran in the existing CPU-capable container against the local
six-file bundle. The same command is reproducible with any equivalent local path;
no remote model resolution is possible under these settings.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES="" \
python3 -I tests/token_export_test.py \
  --real-tokenizer-dir /tmp/umminfer-task37-qwen38-tokenizer \
  --real-tokenizer-id Qwen/Qwen3.8-27B \
  --real-tokenizer-revision UNKNOWN \
  --real-tokenization-version transformers-5.6.0_tokenizers-0.22.2
```

The real gate performs deterministic re-export, exact-id and raw-offset
re-encoding, full-sequence decode, per-token normalized-byte reconstruction,
original-source span/hash checks, complete file rebinding, SFT label/weight and
annotation equivalence, worker/SFT integrity validation, and two new CLI
processes for export and strong validation.

| Case | Covered content | Clean tokens | SFT-target tokens | Export SHA-256 | Result |
| --- | --- | ---: | ---: | --- | --- |
| Multilingual/emoji/combining/spoken/visual | Multilingual prose, 🌅/emoji, decomposed `e` + U+0301, `<spoken>`, image slot/prompt | 18 | 28 | `bad2303b3b0c321ea696dd6a9cface0785c972835ec71c0953c32218136101ef` | **PASS** |
| Explicit anchor | Interleaved image/video, verified `source=slot-0`, raw source events/frames | 8 | 33 | `a2620fdeb690a28809c8f3ac06a06e6b03dac2d0049e65a1051162035415cc9f` | **PASS** |
| Cancelled | Cancelled clean prefix and canonical visual target | 8 | 33 | `65cbbf934808ed2887355f2b4e2f39b75d4ab3b4ad243c4778f62b41d4df872f` | **PASS** |

All three cases use available-files digest
`b74ec186875f88cc00e2655347693099c8c2b211c0f8eb777b7dff9ca55d20f9`.
Fresh-process export reports `status=PASS`, `records=3`, `exported=3`, and
`legacy_preserved=0`. A separate fresh process reports `status=PASS`,
`records=3`, `validated=3`, `tokenizer_reencoded=3`, and `legacy_or_absent=0`.

The ordinary local command remains:

```sh
python3 -I tests/token_export_test.py
```

It uses no ML package and honestly prints `REAL_TOKENIZER_FIXTURE: NOT_RUN` when
no real tokenizer is supplied. Its deterministic v1/v2 tests cover golden v1
bytes, NFC/byte fallback, source/offset/mask/hash tampering, unsupported
normalizer/backend behavior, cancellation, and rejection of malformed structural
input before any `transformers` import.

## Limitations

- Evidence applies to the exact six-file bundle and runtime above, not to other
  Qwen revisions, tokenizers, `transformers`/`tokenizers` versions, or weights.
- V2 rejects non-NFC normalization, non-ByteLevel vocabularies, cross-cluster NFC
  that cannot be mapped exactly, backend/Python Unicode disagreement, empty raw
  offsets, unknown ids, and full-sequence decode mismatch. It is not a generic
  normalizing-tokenizer fallback.
- Original source ranges may overlap by design. Normalized token bytes are the
  gap-free sequence; SFT masks deliberately use raw original offsets to avoid
  double-counting shared byte-fallback/composition units.
- Exported ids are re-encoded supervision ids, not sampled model decode ids.
  Tokenization correctness does not establish model quality, training quality,
  online-rollout parity, or serving behavior.
- No GPU or real model was run, and the resident service was not contacted.
