#!/usr/bin/env python3
"""Contract test for the MLX stage's terminal result counts.

The stage must forward the prompt and completion counts MLX-LM reported and
must not invent one it did not observe. Loads no model and starts no server.

Usage: python3 -I tests/mlx_stage_result_test.py
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent


def load_stage_module():
    path = ROOT / "python" / "umminfer_mlx_stage.py"
    spec = importlib.util.spec_from_file_location("umminfer_mlx_stage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response(**fields):
    base = {"text": "", "generation_tokens": 3, "finish_reason": "stop"}
    base.update(fields)
    return SimpleNamespace(**base)


def main():
    module = load_stage_module()
    stage = module.MLXStage("vlm", "ipc:///tmp/unused-in",
                            "ipc:///tmp/unused-out", "ipc:///tmp/unused-abort",
                            ROOT / "unused-model")

    observed = response(prompt_tokens=7)
    message = stage.complete_message(
        "req-1", True, "hello", generated_tokens=3, finish_reason="stop",
        prompt_tokens=stage.prompt_token_count(observed))
    assert message["result"] == {
        "text": "hello", "prompt_tokens": 7,
        "generated_token_count": 3, "finish_reason": "stop",
    }, message["result"]

    # MLX-LM without a prompt count, and values that are not a count, leave the
    # key out entirely.
    for missing in (response(), response(prompt_tokens=True),
                    response(prompt_tokens=-1),
                    response(prompt_tokens="7")):
        assert stage.prompt_token_count(missing) is None, missing
        result = stage.complete_message(
            "req-2", True, "x", generated_tokens=3, finish_reason="stop",
            prompt_tokens=stage.prompt_token_count(missing))["result"]
        assert "prompt_tokens" not in result, result

    assert stage.prompt_token_count(None) is None
    print("mlx stage result: prompt/completion counts forwarded truthfully")


if __name__ == "__main__":
    main()
