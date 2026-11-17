#!/bin/sh
# Live acceptance for umm-native-stage-qwen38 on the H20B deployment.
# Runs inside container linyu-dev. Writes raw evidence under $OUT.
set -u
BASE=${BASE:-http://172.17.0.2:43301}
IMG=${IMG:-/root/umm-native-service/inputs/man_ironing.png}
OUT=${OUT:-/root/umm-native-service/evidence-qwen38}
mkdir -p "$OUT"

echo "== health =="
curl -s "$BASE/health" | tee "$OUT/health.json"; echo
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "$OUT/gpu_before.txt"

echo "== 1. text, non-stream =="
curl -sS -o "$OUT/text.json" -w 'http=%{http_code} seconds=%{time_total}\n' \
  "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-native","max_tokens":64,"messages":[{"role":"user","content":"Reply with one short sentence about the sea."}]}'
cat "$OUT/text.json"; echo

echo "== 2. image, non-stream =="
B64=$(base64 -w0 "$IMG")
printf '{"model":"qwen3-native","max_tokens":96,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}},{"type":"text","text":"Describe what the person is doing in one sentence."}]}]}' "$B64" > "$OUT/image_request.json"
curl -sS -o "$OUT/image.json" -w 'http=%{http_code} seconds=%{time_total}\n' \
  "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  --data-binary @"$OUT/image_request.json"
cat "$OUT/image.json"; echo

echo "== 3. text, stream=true (delta frames) =="
curl -sS -N -D "$OUT/stream_headers.txt" \
  "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-native","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"Count from one to twenty in words."}]}' \
  > "$OUT/stream.sse"
echo "delta_frames=$(grep -c '"delta":{"content"' "$OUT/stream.sse" || true)"
echo "nan_delta_frames=$(grep -o 'delta":{[^}]*}' "$OUT/stream.sse" | grep -vc '"role":"assistant"' || true)"
grep -o 'delta":{"content":"[^"]\{0,40\}' "$OUT/stream.sse" | head -5
tail -2 "$OUT/stream.sse"

echo "== 3b. repeatability: the same request twice must give the same text =="
for n in 1 2; do
  curl -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-native","max_tokens":48,"messages":[{"role":"user","content":"Name three colors, comma separated."}]}' \
    > "$OUT/repeat_$n.json"
done
A=$(sed 's/.*"content":"//; s/".*//' "$OUT/repeat_1.json")
B=$(sed 's/.*"content":"//; s/".*//' "$OUT/repeat_2.json")
[ "$A" = "$B" ] && echo "repeatable=yes text=$A" || { echo "repeatable=NO"; echo "1=$A"; echo "2=$B"; }

echo "== 4. abort during decode =="
python3 -I - "$BASE" "$OUT" <<'PY'
import json, sys, time, urllib.request
base, out = sys.argv[1], sys.argv[2]
body = json.dumps({"model": "qwen3-native", "stream": True, "max_tokens": 96,
                   "messages": [{"role": "user",
                                 "content": "List the integers from 1 to 200 as "
                                            "digits, separated by commas, and "
                                            "write nothing else."}]}).encode()
req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
frames = []
terminal = None
abort_status = None
sent = received = 0
with urllib.request.urlopen(req, timeout=300) as resp:
    def next_frame():
        """Returns the next SSE JSON frame, or None at [DONE] / EOF."""
        pending = []
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("data:"):
                pending.append(line[5:].lstrip())
                continue
            if line or not pending:
                continue
            payload = "\n".join(pending)
            pending = []
            if payload == "[DONE]":
                return None
            return json.loads(payload)
        return None

    request_id = None
    started = time.time()
    while True:
        frame = next_frame()
        if frame is None:
            break
        request_id = frame.get("id", request_id)
        frames.append(frame)
        if request_id and time.time() - started > 1.0:
            break
    sent = int(time.time() * 1000)
    abort_req = urllib.request.Request(
        base + "/abort",
        data=json.dumps({"request_id": request_id}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(abort_req, timeout=30) as r:
        abort_status = r.status
    # Drain to the terminal event and stamp the moment it arrives.
    while True:
        frame = next_frame()
        if frame is None:
            break
        frames.append(frame)
        unified = frame.get("unified") or {}
        if unified.get("kind") in ("cancelled", "complete", "error"):
            terminal = unified
            break
    received = int(time.time() * 1000)
deltas = sum(1 for f in frames
             if (f.get("choices") or [{}])[0].get("delta", {}).get("content"))
record = {"request_id": request_id, "abort_status": abort_status,
          "abort_epoch_ms": sent, "terminal_epoch_ms": received,
          "abort_to_terminal_ms": received - sent,
          "deltas_before_abort": deltas,
          "terminal": terminal}
open(out + "/abort.json", "w").write(json.dumps(record, indent=1) + "\n")
print(json.dumps(record))
PY

echo "== 5. unsupported audio input to qwen38 stage (direct submit is not exposed; HTTP-side check) =="
curl -sS -o "$OUT/audio_part.json" -w 'http=%{http_code}\n' \
  "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-native","stream":true,"max_tokens":16,"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}},{"type":"text","text":"hi"}]}]}'
cat "$OUT/audio_part.json"; echo

echo "== 6. stage log: abort boundary and abort->exit ms =="
LOG=/root/umm-native-service/logs/qwen38.log
tail -c 4000 "$LOG" | tr -d '\000' | grep 'aborted' | tail -2
python3 -I - "$LOG" <<'PY'
import re, sys
data = open(sys.argv[1], "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
lines = [l for l in data.splitlines() if "aborted" in l]
if not lines:
    print("no abort line"); raise SystemExit(0)
last = lines[-1]
m = re.search(r'abort_epoch_ms=(\d+) exit_epoch_ms=(\d+)', last)
print(last)
if m:
    print("stage_abort_to_exit_ms=%d" % (int(m.group(2)) - int(m.group(1))))
PY

echo "== 7. after abort =="
curl -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-native","max_tokens":16,"messages":[{"role":"user","content":"Say OK."}]}' \
  | tee "$OUT/after_abort.json"; echo
curl -s "$BASE/health" | tee "$OUT/health_after.json"; echo
