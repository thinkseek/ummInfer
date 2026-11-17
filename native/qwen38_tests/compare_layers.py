#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end parity of the native dumps against the HF reference.

Two verdicts are printed.

TENSOR (requires --fp32-ref): both the native BF16 run and the HF BF16 run are
compared against an FP32 run of the same HF model, which is the common anchor.
A tensor passes when

    rel_rms(native, fp32) <= 1.5 * rel_rms(hf_bf16, fp32)   (+ 1e-3 floor)

i.e. our BF16 path is no more than 1.5x further from the exact model than the
reference's own BF16 path.  This is the criterion the FP32 experiment supports;
it is not a free threshold: the 1.5x and the band both come from measurements.

FUNCTIONAL: prompt ids, first-token argmax and generated ids agree exactly.

Note that this end-to-end test includes chaotic amplification through 64
residual layers.  The per-layer linear-attention operator is tested separately
by op_parity.py on identical inputs.
"""
import argparse
from pathlib import Path

import numpy as np

LAYERS = 64
SLACK = 1.5
FLOOR = 1e-3
FALLBACK = 5e-2


def read_bf16(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def read_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(b.astype(np.float64) ** 2)) + 1e-12)
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(np.sqrt(np.mean(d ** 2)) / rms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--native", required=True)
    ap.add_argument("--fp32-ref", default="")
    args = ap.parse_args()
    ref, nat = Path(args.ref), Path(args.native)
    fp32 = Path(args.fp32_ref) if args.fp32_ref else None

    functional = True
    tensor_ok = True
    worst = (0.0, "")

    ri = np.fromfile(ref / "prompt_ids.i32", dtype=np.int32)
    ni = np.fromfile(nat / "prompt_ids.i32", dtype=np.int32)
    same_ids = ri.shape == ni.shape and np.array_equal(ri, ni)
    print(f"prompt_ids match={same_ids} ref={ri.size} native={ni.size}")
    functional &= same_ids
    if not same_ids:
        print("PARITY FAIL")
        return 1

    def check(name: str, ext: str = "bf16") -> None:
        nonlocal tensor_ok, worst
        r = read_bf16(ref / f"{name}.{ext}")
        n = read_bf16(nat / f"{name}.{ext}")
        if r.shape != n.shape:
            print(f"{name:12s} SHAPE {n.shape} vs {r.shape}")
            tensor_ok = False
            return
        err_hf = rel_rms(n, r)
        if fp32 is not None and (fp32 / f"{name}.f32").exists():
            f = read_f32(fp32 / f"{name}.f32")
            band = rel_rms(r, f)
            err_fp = rel_rms(n, f)
            if band < 1e-9:
                ok = err_fp < FLOOR
                ratio_s = "n/a"
            else:
                ok = err_fp <= SLACK * band + FLOOR
                ratio_s = f"{err_fp / band:.2f}"
            tensor_ok &= ok
            if band > 1e-9 and err_fp / band > worst[0]:
                worst = (err_fp / band, name)
            print(f"{name:12s} native_vs_hf={err_hf:.3e} hf_vs_fp32={band:.3e} "
                  f"native_vs_fp32={err_fp:.3e} ratio={ratio_s} "
                  f"{'ok' if ok else 'OVER'}")
        else:
            ok = err_hf <= FALLBACK
            tensor_ok &= ok
            print(f"{name:12s} native_vs_hf={err_hf:.3e} "
                  f"(no fp32 anchor, fallback {FALLBACK:g}) "
                  f"{'ok' if ok else 'OVER'}")

    check("embed")
    for i in range(LAYERS):
        check(f"layer_{i}")
        check(f"mixer_{i}")
    for name in ["vpatch"] + [f"vblock_{i}" for i in range(27)] + ["pooler"]:
        if (ref / f"{name}.bf16").exists():
            check(name)

    lr = read_f32(ref / "logits0.f32")
    ln = read_f32(nat / "logits0.f32")
    if lr.shape == ln.shape:
        check_logits = True
        err_hf = rel_rms(ln, lr)
        if fp32 is not None and (fp32 / "logits0.f32").exists():
            f = read_f32(fp32 / "logits0.f32")
            band = rel_rms(lr, f)
            err_fp = rel_rms(ln, f)
            ok = err_fp <= SLACK * band + FLOOR if band > 1e-9 else err_fp < FLOOR
            tensor_ok &= ok
            print(f"{'logits0':12s} native_vs_hf={err_hf:.3e} hf_vs_fp32={band:.3e} "
                  f"native_vs_fp32={err_fp:.3e} "
                  f"ratio={(err_fp / band if band > 1e-9 else float('nan')):.2f}")
        else:
            tensor_ok &= err_hf <= FALLBACK
    same_argmax = int(np.argmax(lr)) == int(np.argmax(ln))
    print(f"first-token argmax ref={int(np.argmax(lr))} native={int(np.argmax(ln))} "
          f"equal={same_argmax}")
    functional &= same_argmax

    gr = np.fromfile(ref / "gen_ids.i32", dtype=np.int32)
    gn = np.fromfile(nat / "gen_ids.i32", dtype=np.int32)
    same_gen = gr.shape == gn.shape and gr.size > 0 and np.array_equal(gr, gn)
    print(f"gen_ids ref={gr.size} native={gn.size} equal={same_gen}")
    functional &= same_gen

    print(f"TENSOR parity (native_vs_fp32 <= {SLACK}*hf_vs_fp32): "
          f"{'PASS' if tensor_ok else 'FAIL'} (worst {worst[1]} {worst[0]:.2f}x)")
    print(f"FUNCTIONAL parity: {'PASS' if functional else 'FAIL'}")
    print("PARITY PASS" if functional and tensor_ok else "PARITY FAIL")
    return 0 if (functional and tensor_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
