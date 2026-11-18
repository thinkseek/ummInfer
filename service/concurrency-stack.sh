#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Independent stack used for the concurrency and capacity measurements.
#
#   server 172.17.0.2:43302
#   asr    GPU 2      qwen38 GPU 2      image GPU 6      tts GPU 7
#
# It never touches the resident service on 43300/43301 or GPUs 0/1/3/4/5.
# The image and TTS legs take stage-level workload caps so a case stays
# bounded: IMAGE_STEPS (default 8) and TTS_FRAMES (default 200). The admission
# cap is the server flag MAX_IN_FLIGHT (default 32).
#
# usage: concurrency-stack.sh start|stop|status
set -eu

ROOT=${UMM_CONCURRENCY_ROOT:-/root/umm-concurrency-20261117}
IPC=${UMM_CONCURRENCY_IPC:-/tmp/umminfer-concurrency-ipc}
HOST=${UMM_CONCURRENCY_HOST:-172.17.0.2}
PORT=${UMM_CONCURRENCY_PORT:-43302}
BIN=$ROOT/bin
IMAGE_STEPS=${IMAGE_STEPS:-8}
TTS_FRAMES=${TTS_FRAMES:-200}
MAX_IN_FLIGHT=${MAX_IN_FLIGHT:-32}
ASR_MODEL=${UMM_CONCURRENCY_ASR_MODEL:-/root/omniinfer-model-validation-20261111-k3/models/asr}
QWEN38_MODEL=${UMM_CONCURRENCY_QWEN38_MODEL:-/root/models/qwen3.8-27b}
IMAGE_MODEL=${UMM_CONCURRENCY_IMAGE_MODEL:-/root/umm-mixed-output-20261116/models/Qwen-Image-2512}
TTS_MODEL=${UMM_CONCURRENCY_TTS_MODEL:-/root/umm-mixed-output-20261116/models/qwen3-tts}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/root/umm-c6rt-ckl/deps/lib:/root/umm-c6rt-ckl/curl-prefix/lib}

mkdir -p "$ROOT/run" "$ROOT/logs" "$IPC"

start() {
  rm -f "$IPC"/*
  nohup "$BIN/umm-infer" serve --host "$HOST" --port "$PORT" --model qwen3-native \
    --entry asr --entry-endpoint "ipc://$IPC/asr" \
    --vlm-entry qwen38 --vlm-entry-endpoint "ipc://$IPC/qwen38" \
    --image-entry image --image-entry-endpoint "ipc://$IPC/image" \
    --tts-entry tts --tts-entry-endpoint "ipc://$IPC/tts" \
    --max-in-flight "$MAX_IN_FLIGHT" \
    --completion "ipc://$IPC/done" --abort "ipc://$IPC/abort" \
    >"$ROOT/logs/server.log" 2>&1 &
  echo $! >"$ROOT/run/server.pid"
  i=0
  until curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 100 ]; then
      echo "server health timeout" >&2
      return 1
    fi
    sleep 0.2
  done
  stage asr asr "$ASR_MODEL" 2 ""
  stage qwen38 qwen38 "$QWEN38_MODEL" 2 "--max-new-tokens 96"
  stage image image "$IMAGE_MODEL" 6 "--steps $IMAGE_STEPS"
  stage tts tts "$TTS_MODEL" 7 "--max-frames $TTS_FRAMES"
  i=0
  until curl -sf "http://$HOST:$PORT/ready" >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 900 ]; then
      echo "ready timeout" >&2
      return 1
    fi
    sleep 0.5
  done
  echo READY
}

# start_stage NAME KIND MODEL GPU EXTRA
stage() {
  name=$1 kind=$2 model=$3 gpu=$4 extra=$5
  # shellcheck disable=SC2086
  nohup "$BIN/umm-native-stage-$kind" --kind "$kind" --name "$name" \
    --model "$model" --gpu "$gpu" $extra \
    --recv-endpoint "ipc://$IPC/$name" \
    --coordinator-endpoint "ipc://$IPC/done" \
    --abort-endpoint "ipc://$IPC/abort" \
    >"$ROOT/logs/$name.log" 2>&1 &
  echo $! >"$ROOT/run/$name.pid"
  echo "started $name pid $! gpu $gpu $extra"
}

stop() {
  for name in server asr qwen38 image tts; do
    [ -f "$ROOT/run/$name.pid" ] || continue
    pid=$(cat "$ROOT/run/$name.pid")
    kill "$pid" 2>/dev/null || true
    i=0
    while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 100 ]; do
      i=$((i + 1))
      sleep 0.1
    done
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$ROOT/run/$name.pid"
  done
  echo stopped
}

status() {
  for name in server asr qwen38 image tts; do
    if [ -f "$ROOT/run/$name.pid" ] &&
       kill -0 "$(cat "$ROOT/run/$name.pid")" 2>/dev/null; then
      echo "$name running $(cat "$ROOT/run/$name.pid")"
    else
      echo "$name down"
    fi
  done
}

case ${1:-} in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
