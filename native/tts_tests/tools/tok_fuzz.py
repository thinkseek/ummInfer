#!/usr/bin/env python3
"""Fuzz parity: native C++ BPE encoder vs HuggingFace tokenizers 0.22.2 with
the fix_mistral_regex=True patched regex P. Test-only; no runtime role.

Usage: tok_fuzz.py CXX_DRIVER MODEL_DATA_DIR [N]
"""
import json
import random
import subprocess
import sys

from tokenizers import Regex, Tokenizer
from tokenizers import decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers import AddedToken

PATCH = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def build_ref(model_dir: str) -> Tokenizer:
    vocab = json.load(open(f"{model_dir}/vocab.json"))
    merges = [tuple(l.rstrip("\n").split(" ", 1))
              for l in open(f"{model_dir}/merges.txt", encoding="utf-8") if l.strip()]
    cfg = json.load(open(f"{model_dir}/tokenizer_config.json"))
    tok = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token=None,
                        fuse_unk=False, byte_fallback=False))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATCH), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.add_tokens([AddedToken(s["content"], special=s["special"],
                               lstrip=s["lstrip"], rstrip=s["rstrip"],
                               normalized=s["normalized"])
                    for s in cfg["added_tokens_decoder"].values()])
    return tok


def main() -> int:
    driver, model_dir = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
    ref = build_ref(model_dir)

    rng = random.Random(99)
    pools = ["abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789", " ",
             ".,!?;:'-/\\()[]{}@#%^&*+", "éèêëàâîôùçñÉÈÇ", "̀́̂̃̄̆̈̋̌̍",
             "日本語テキスト", "한국어", "αβγδε", "\r\n\t",
             "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
             "<|im_start|><|im_end|><tts_pad>"]
    cases = ["".join(rng.choice(rng.choice(pools)) for _ in range(rng.randint(1, 40)))
             for _ in range(n)]
    cases += [
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
        "Hello, welcome to the native inference engine test.",
        "<|im_start|>assistant\nHi<|im_end|>\n<|im_start|>assistant\n",
        "a<|im_start|>b<tts_pad>c", "  spaced  text \n\n", "1234567890",
        "camelCase HTTPConnection", "a/b/c 10/20", "line\r\nbreak", "emoji 😀 x",
    ]

    proc = subprocess.run([driver, model_dir], input="\0".join(cases) + "\0",
                          text=True, capture_output=True, check=True)
    cpp_lines = proc.stdout.splitlines()
    assert len(cpp_lines) == len(cases), (len(cpp_lines), len(cases))

    mism = 0
    for s, line in zip(cases, cpp_lines):
        want = ref.encode(s).ids
        got = [int(x) for x in line.split()] if line else []
        if got != want:
            mism += 1
            if mism <= 5:
                print("MISMATCH", repr(s), "\n ref ", want, "\n cpp ", got)
    print(f"fuzz mismatches: {mism} / {len(cases)}")
    return 1 if mism else 0


if __name__ == "__main__":
    sys.exit(main())
