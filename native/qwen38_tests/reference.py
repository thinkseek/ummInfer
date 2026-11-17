#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Development-only HF reference: prompt ids, per-layer activations, text.

Run with the cu124 venv that has transformers>=5.6.  GPU is selected through
CUDA_VISIBLE_DEVICES before torch is imported.

Example:
  CUDA_VISIBLE_DEVICES=6 python reference.py --model /root/models/qwen3.8-27b \
      --prompt "..." --image /root/test.png --max-new-tokens 64 --out /root/ref
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main() -> int:
    args = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model)
    content = []
    if args.image:
        content.append({"type": "image", "image": args.image})
    content.append({"type": "text", "text": args.prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=args.thinking,
    )
    images = [Image.open(args.image).convert("RGB")] if args.image else None
    device = "cuda" if args.device == "cuda" else "cpu"
    inputs = processor(text=[text], images=images, return_tensors="pt").to(device)

    ids = inputs["input_ids"][0].tolist()
    np.asarray(ids, dtype=np.int32).tofile(out / "prompt_ids.i32")

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16,
        device_map=("cuda:0" if device == "cuda" else {"": "cpu"}),
    )
    model.eval()
    lm = model.model.language_model
    n_img = int(inputs["image_grid_thw"].prod()) // 4 if args.image else 0
    grid = inputs["image_grid_thw"][0].tolist() if args.image else [0, 0, 0]

    def dump(name: str, tensor: torch.Tensor) -> None:
        # numpy has no bfloat16; write the raw 16-bit pattern.
        b = tensor.detach().to("cpu", torch.bfloat16)
        b.view(torch.int16).numpy().tofile(out / name)

    def dump_f32(name: str, tensor: torch.Tensor) -> None:
        tensor.detach().float().cpu().numpy().tofile(out / name)

    def pre_hook(_module, args_):
        hs = args_[0]
        if hs.shape[1] > 1:
            dump("embed.bf16", hs[0])

    def make_layer_hook(idx):
        def hook(_module, _args, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] > 1:
                dump(f"layer_{idx}.bf16", hs[0])
        return hook

    def make_mixer_hook(idx):
        def hook(_module, _args, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] > 1:
                dump(f"mixer_{idx}.bf16", hs[0])
        return hook

    def merger_hook(_module, _args, output):
        hs = output[0] if isinstance(output, tuple) else output
        dump("pooler.bf16", hs)

    handles = [lm.layers[0].register_forward_pre_hook(pre_hook)]
    for i, layer in enumerate(lm.layers):
        handles.append(layer.register_forward_hook(make_layer_hook(i)))
        mixer = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
        handles.append(mixer.register_forward_hook(make_mixer_hook(i)))
    if args.image:
        vis = model.model.visual

        def vpatch_hook(_module, args_):
            hs = args_[0]
            dump("vpatch.bf16", hs)

        handles.append(vis.blocks[0].register_forward_pre_hook(vpatch_hook))
        for i, block in enumerate(vis.blocks):
            handles.append(
                block.register_forward_hook(
                    lambda _m, _a, out, i=i: dump(f"vblock_{i}.bf16", out[0] if isinstance(out, tuple) else out)
                )
            )
        handles.append(vis.merger.register_forward_hook(merger_hook))

    with torch.no_grad():
        fwd = model(**inputs, use_cache=False, logits_to_keep=1)
    logits = fwd.logits[0, -1].float()
    dump_f32("logits0.f32", logits)
    first = int(torch.argmax(logits).item())

    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            use_cache=True,
        )
    gen_ids = gen[0, inputs["input_ids"].shape[1]:].tolist()
    # The native executor stops before appending EOS; match that convention.
    while gen_ids and gen_ids[-1] in (248046, 248044):
        gen_ids.pop()
    np.asarray(gen_ids, dtype=np.int32).tofile(out / "gen_ids.i32")
    text_out = processor.batch_decode([gen_ids], skip_special_tokens=True)[0]
    (out / "text.txt").write_text(text_out)

    meta = (
        f"prompt_tokens={len(ids)}\n"
        f"image_tokens={n_img}\n"
        f"grid_h={grid[1]}\n"
        f"grid_w={grid[2]}\n"
        f"first_token={first}\n"
    )
    (out / "meta.txt").write_text(meta)
    for h in handles:
        h.remove()
    print(f"wrote reference to {out}; first_token={first}; text={text_out!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
