#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FP32 reference dumps (development only).

Runs the reference model in float32 on CPU -- one 96 GB GPU cannot hold 112 GB
of FP32 weights -- and writes raw float32 tensors that compare_layers.py uses
as the common anchor for the BF16 tolerance band.

  python fp32_dump.py --model DIR --prompt TEXT [--image PATH] --out DIR
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()
torch.set_num_threads(48)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model)
    content = []
    images = None
    if args.image:
        content.append({"type": "image", "image": args.image})
        images = [Image.open(args.image).convert("RGB")]
    content.append({"type": "text", "text": args.prompt})
    tpl = proc.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    inputs = proc(text=[tpl], images=images, return_tensors="pt")

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, device_map={"": "cpu"}).eval()
    lm = model.model.language_model
    vis = model.model.visual
    os.makedirs(args.out, exist_ok=True)

    def save(name, t):
        t.detach().float().reshape(-1).numpy().astype(np.float32).tofile(
            os.path.join(args.out, name + ".f32"))

    handles = [lm.layers[0].register_forward_pre_hook(
        lambda m, a: save("embed", a[0][0]))]
    for i, layer in enumerate(lm.layers):
        handles.append(layer.register_forward_hook(
            lambda m, a, out, i=i: save(
                f"layer_{i}", out[0] if isinstance(out, tuple) else out)))
        mixer = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
        handles.append(mixer.register_forward_hook(
            lambda m, a, out, i=i: save(
                f"mixer_{i}", out[0] if isinstance(out, tuple) else out)))
    if args.image:
        handles.append(vis.blocks[0].register_forward_pre_hook(
            lambda m, a: save("vpatch", a[0])))
        for i, blk in enumerate(vis.blocks):
            handles.append(blk.register_forward_hook(
                lambda m, a, out, i=i: save(
                    f"vblock_{i}", out[0] if isinstance(out, tuple) else out)))
        handles.append(vis.merger.register_forward_hook(
            lambda m, a, out: save("pooler",
                                   out[0] if isinstance(out, tuple) else out)))

    with torch.no_grad():
        out = model(**inputs, use_cache=False, logits_to_keep=1)
    save("logits0", out.logits[0, -1])
    for h in handles:
        h.remove()
    print("saved", len(os.listdir(args.out)), "tensors to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
