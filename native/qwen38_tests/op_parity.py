#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-layer linear-attention operator parity.

For every one of the 48 linear_attention layers, feed the *same* input (the
native trajectory's pre-layer hidden state, after the layer's input
RMSNorm) into both the native kernel (its dumped mixer output) and the
transformers module, then compare.  This isolates the operator from the
accumulated residual-stream drift, so it answers the criterion "each linear
attention layer agrees with HF in BF16 tolerance".

Gate: relative RMS error <= 1e-2 for every layer.
"""
import argparse
from pathlib import Path

import numpy as np


def read_bf16(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(b.astype(np.float64) ** 2)) + 1e-12)
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(np.sqrt(np.mean(d ** 2)) / rms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--native", required=True, help="native dump dir")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    nat = Path(args.native)
    meta = dict(line.split("=", 1) for line in (nat / "meta.txt").read_text().split())
    seq = int(meta["prompt_tokens"])

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    lm = model.model.language_model

    worst = (0.0, -1)
    ok = True
    with torch.no_grad():
        for i in range(64):
            if i % 4 == 3:
                continue
            src = "embed" if i == 0 else f"layer_{i-1}"
            x = torch.from_numpy(read_bf16(nat / f"{src}.bf16")).view(
                1, seq, 5120).to("cuda", dtype=torch.bfloat16)
            y = lm.layers[i].linear_attn(
                lm.layers[i].input_layernorm(x)).float().reshape(-1).cpu().numpy()
            m = read_bf16(nat / f"mixer_{i}.bf16")
            err = rel_rms(y, m)
            ok &= err <= 1e-2
            if err > worst[0]:
                worst = (err, i)
            print(f"layer {i:2d} same-input rel_rms={err:.3e}")
    print(f"OPERATOR parity: {'PASS' if ok else 'FAIL'} "
          f"(worst layer {worst[1]}, {worst[0]:.3e}, gate 1e-2)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
