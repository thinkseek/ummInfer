#!/usr/bin/env python3
"""Paired output check for decode batching.

Eight different prompts go to a stack that decodes each request alone and to a
stack that decodes them in one batch. The response text must be byte-identical
for every prompt.

Usage:
  batch_parity.py collect --url http://172.17.0.2:43304 --out ref.json
  batch_parity.py collect --url http://172.17.0.2:43304 --out batched.json
  batch_parity.py compare ref.json batched.json
"""
import argparse
import json
import sys
import threading
import urllib.request

PROMPTS = [
    "Name the capital of France.",
    "Explain why the sky is blue in one sentence.",
    "What is 17 multiplied by 23?",
    "List two prime numbers above 20.",
    "Write one sentence about the moon.",
    "What does HTTP stand for?",
    "Give a synonym for the word quick.",
    "State the boiling point of water at sea level in Celsius.",
]


def ask(url, prompt, max_tokens, timeout):
    body = {
        "model": "qwen3-native",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read())
    return parsed["choices"][0]["message"]["content"]


def collect(args):
    results = [None] * len(PROMPTS)

    def worker(index):
        try:
            results[index] = {"prompt": PROMPTS[index],
                              "text": ask(args.url, PROMPTS[index],
                                          args.max_tokens, args.timeout)}
        except Exception as error:  # noqa: BLE001 - the record is the evidence
            results[index] = {"prompt": PROMPTS[index], "error": str(error)}

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(len(PROMPTS))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with open(args.out, "w", encoding="utf-8") as sink:
        json.dump(results, sink, ensure_ascii=False, indent=1)
        sink.write("\n")
    print(json.dumps({"records": len(results),
                      "errors": sum(1 for r in results if "error" in r),
                      "out": args.out}))
    return 0


def compare(args):
    with open(args.reference, encoding="utf-8") as sink:
        reference = json.load(sink)
    with open(args.candidate, encoding="utf-8") as sink:
        candidate = json.load(sink)
    if len(reference) != len(candidate):
        print("record count differs", file=sys.stderr)
        return 1
    failures = 0
    for ref, new in zip(reference, candidate):
        same = ref.get("text") == new.get("text") and "error" not in ref \
            and "error" not in new and ref.get("text")
        print(json.dumps({"prompt": ref["prompt"][:40],
                          "same": bool(same),
                          "chars": len(ref.get("text") or "")}))
        if not same:
            failures += 1
    print(json.dumps({"compared": len(reference), "mismatches": failures}))
    return 0 if failures == 0 else 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    take = sub.add_parser("collect")
    take.add_argument("--url", required=True)
    take.add_argument("--out", required=True)
    take.add_argument("--max-tokens", type=int, default=64)
    take.add_argument("--timeout", type=float, default=900)
    take.set_defaults(func=collect)
    diff = sub.add_parser("compare")
    diff.add_argument("reference")
    diff.add_argument("candidate")
    diff.set_defaults(func=compare)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
