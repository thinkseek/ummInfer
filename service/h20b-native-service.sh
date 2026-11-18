#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Native-only deployment on H20B (container linyu-dev). Five processes:
# one umm-infer server and one native stage per model. No Python, PyTorch or
# SGLang on the serving path.
#
#   ASR    GPU 2, stage name asr,    ipc:///tmp/umminfer-native-ipc/asr
#   Qwen38 GPU 3, stage name qwen38, ipc:///tmp/umminfer-native-ipc/qwen38
#   Image  GPU 4, stage name image,  ipc:///tmp/umminfer-native-ipc/image
#   TTS    GPU 5, stage name tts,    ipc:///tmp/umminfer-native-ipc/tts
#
# The omni stage is retained but not started. Set UMM_NATIVE_OMNI=1 to run it
# as the VLM stage instead of qwen38; only one of the two may hold GPU 3.
#   Server 172.17.0.2:43301, completion/abort under the same ipc dir.
#
# GPU0/1 hold the existing SGLang backends and are never touched. The port is
# distinct from the phase7 service on 43300.
#
# usage: h20b-native-service.sh start|stop|status
set -eu

ROOT=${UMM_NATIVE_ROOT:-/root/umm-native-service}
RUN_DIR=$ROOT/run
LOG_DIR=$ROOT/logs
IPC_DIR=${UMM_NATIVE_IPC:-/tmp/umminfer-native-ipc}
HOST=${UMM_NATIVE_HOST:-172.17.0.2}
PORT=${UMM_NATIVE_PORT:-43301}
MODELS=${UMM_NATIVE_MODELS:-/root/omniinfer-model-validation-20261111-k3/models}
ASR_MODEL=${UMM_NATIVE_ASR_MODEL:-$MODELS/asr}
QWEN38_MODEL=${UMM_NATIVE_QWEN38_MODEL:-/root/models/qwen3.8-27b}
OMNI_MODEL=${UMM_NATIVE_OMNI_MODEL:-$MODELS/omni}
if [ "${UMM_NATIVE_OMNI:-0}" = "1" ]; then VLM_STAGE=omni; else VLM_STAGE=qwen38; fi
VLM_MODEL=$([ "$VLM_STAGE" = omni ] && echo "$OMNI_MODEL" || echo "$QWEN38_MODEL")
TTS_MODEL=${UMM_NATIVE_TTS_MODEL:-/root/umm-mixed-output-20261116/models/qwen3-tts}
IMAGE_MODEL=${UMM_NATIVE_IMAGE_MODEL:-/root/umm-mixed-output-20261116/models/Qwen-Image-2512}
BIN=$ROOT/bin

started() {
  [ -f "$RUN_DIR/$1.pid" ] && kill -0 "$(cat "$RUN_DIR/$1.pid")" 2>/dev/null
}

start() {
  mkdir -p "$RUN_DIR" "$LOG_DIR" "$IPC_DIR"
  if started server; then
    echo "already running (server pid $(cat "$RUN_DIR/server.pid"))"
    return 0
  fi
  rm -f "$IPC_DIR"/*
  ( cd "$ROOT" && nohup "$BIN/umm-infer" serve --host "$HOST" --port "$PORT" \
      --model qwen3-native \
      --entry asr --entry-endpoint "ipc://$IPC_DIR/asr" \
      --vlm-entry "$VLM_STAGE" --vlm-entry-endpoint "ipc://$IPC_DIR/$VLM_STAGE" \
      --image-entry image --image-entry-endpoint "ipc://$IPC_DIR/image" \
      --tts-entry tts --tts-entry-endpoint "ipc://$IPC_DIR/tts" \
      --completion "ipc://$IPC_DIR/completion" \
      --abort "ipc://$IPC_DIR/abort" \
      >"$LOG_DIR/server.log" 2>&1 & echo $! >"$RUN_DIR/server.pid" )
  # The Coordinator must be bound before a stage sends stage_ready.
  tries=0
  until curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; do
    tries=$((tries + 1))
    if [ "$tries" -gt 50 ]; then
      echo "server did not answer /health" >&2
      return 1
    fi
    sleep 0.2
  done
  VLM_EXTRA="--max-new-tokens 96"
  if [ "$VLM_STAGE" = "qwen38" ]; then
    VLM_EXTRA="$VLM_EXTRA --batch-size ${UMM_QWEN38_BATCH_SIZE:-8}"
  fi
  start_stage asr asr "$BIN/umm-native-stage-asr" "$ASR_MODEL" 2 ""
  start_stage "$VLM_STAGE" "$VLM_STAGE" "$BIN/umm-native-stage-$VLM_STAGE" "$VLM_MODEL" 3 "$VLM_EXTRA"
  start_stage image image "$BIN/umm-native-stage-image" "$IMAGE_MODEL" 4 ""
  start_stage tts tts "$BIN/umm-native-stage-tts" "$TTS_MODEL" 5 ""
}

# start_stage NAME KIND BINARY MODEL GPU EXTRA
start_stage() {
  name=$1 kind=$2 binary=$3 model=$4 gpu=$5 extra=$6
  # shellcheck disable=SC2086
  ( cd "$ROOT" && nohup "$binary" --kind "$kind" --name "$name" \
      --model "$model" --gpu "$gpu" $extra \
      --recv-endpoint "ipc://$IPC_DIR/$name" \
      --coordinator-endpoint "ipc://$IPC_DIR/completion" \
      --abort-endpoint "ipc://$IPC_DIR/abort" \
      >"$LOG_DIR/$name.log" 2>&1 & echo $! >"$RUN_DIR/$name.pid" )
  echo "started $name pid $(cat "$RUN_DIR/$name.pid") gpu $gpu"
}

stop() {
  for name in asr qwen38 omni image tts server; do
    [ -f "$RUN_DIR/$name.pid" ] || continue
    pid=$(cat "$RUN_DIR/$name.pid")
    kill "$pid" 2>/dev/null || true
    tries=0
    while kill -0 "$pid" 2>/dev/null && [ "$tries" -lt 100 ]; do
      tries=$((tries + 1))
      sleep 0.1
    done
    # A stage inside a long generation drains only when the request ends.
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$RUN_DIR/$name.pid"
  done
  echo stopped
}

status() {
  for name in server asr "$VLM_STAGE" image tts; do
    if started "$name"; then
      echo "$name running pid $(cat "$RUN_DIR/$name.pid")"
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
