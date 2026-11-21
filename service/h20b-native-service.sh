#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Native-only deployment on H20B (GPU container). Five processes:
# one umm-infer server and one native stage per model. No Python, PyTorch or
# SGLang on the serving path.
#
# Binary origin. Two build systems, not one:
#   umm-infer                    zig build -Doptimize=ReleaseFast
#                                -> zig-out/bin/umm-infer
#   umm-native-stage-*           cmake -S native/stage_cli -B build
#                                  -DUMM_STAGE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
#                                cmake --build build -j
#                                -> build/umm-native-stage-{asr,omni,qwen38,image,tts}
# zig build does not produce the stage binaries, and the CMake project does not
# produce umm-infer. Install both into $BIN:
#   install -m755 zig-out/bin/umm-infer build/umm-native-stage-* "$BIN/"
#
#   ASR    GPU 2, stage name asr,    ipc:///tmp/umminfer-native-ipc/asr
#   Qwen38 GPU 3, stage name qwen38, ipc:///tmp/umminfer-native-ipc/qwen38
#   Image  GPU 4, stage name image,  ipc:///tmp/umminfer-native-ipc/image
#   TTS    GPU 5, stage name tts,    ipc:///tmp/umminfer-native-ipc/tts
# Override the GPU of each stage with UMM_NATIVE_GPU_ASR, UMM_NATIVE_GPU_VLM,
# UMM_NATIVE_GPU_IMAGE and UMM_NATIVE_GPU_TTS. The defaults above assume the
# whole box is yours; a host that already runs a service on 2-5 needs the
# overrides.
#
# The omni stage is retained but not started. Set UMM_NATIVE_OMNI=1 to run it
# as the VLM stage instead of qwen38; only one of the two may hold GPU_VLM.
#   Server ${UMM_NATIVE_HOST:-127.0.0.1}:${UMM_NATIVE_PORT:-43301},
#   completion/abort under the same ipc dir.
# Set UMM_NATIVE_HOST=<container-private-ip> when binding a container bridge
# address, and override UMM_NATIVE_ROOT / UMM_NATIVE_MODELS / UMM_NATIVE_*_MODEL
# to match your local install and weight directories.
#
# The port is distinct from the retired phase7 service on 43300.
#
# usage: h20b-native-service.sh start|stop|status
set -eu

ROOT=${UMM_NATIVE_ROOT:-/opt/umminfer/native-service}
RUN_DIR=$ROOT/run
LOG_DIR=$ROOT/logs
IPC_DIR=${UMM_NATIVE_IPC:-/tmp/umminfer-native-ipc}
HOST=${UMM_NATIVE_HOST:-127.0.0.1}
PORT=${UMM_NATIVE_PORT:-43301}
MODELS=${UMM_NATIVE_MODELS:-$ROOT/models}
ASR_MODEL=${UMM_NATIVE_ASR_MODEL:-$MODELS/asr}
QWEN38_MODEL=${UMM_NATIVE_QWEN38_MODEL:-$MODELS/qwen3.8-27b}
OMNI_MODEL=${UMM_NATIVE_OMNI_MODEL:-$MODELS/omni}
# GPU per stage. Only one process may hold a card. Override when the defaults
# are held by another service on the same host.
GPU_ASR=${UMM_NATIVE_GPU_ASR:-2}
GPU_VLM=${UMM_NATIVE_GPU_VLM:-3}
GPU_IMAGE=${UMM_NATIVE_GPU_IMAGE:-4}
GPU_TTS=${UMM_NATIVE_GPU_TTS:-5}
if [ "${UMM_NATIVE_OMNI:-0}" = "1" ]; then VLM_STAGE=omni; else VLM_STAGE=qwen38; fi
VLM_MODEL=$([ "$VLM_STAGE" = omni ] && echo "$OMNI_MODEL" || echo "$QWEN38_MODEL")
TTS_MODEL=${UMM_NATIVE_TTS_MODEL:-$MODELS/qwen3-tts}
IMAGE_MODEL=${UMM_NATIVE_IMAGE_MODEL:-$MODELS/Qwen-Image-2512}
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
  start_stage asr asr "$BIN/umm-native-stage-asr" "$ASR_MODEL" "$GPU_ASR" ""
  start_stage "$VLM_STAGE" "$VLM_STAGE" "$BIN/umm-native-stage-$VLM_STAGE" "$VLM_MODEL" "$GPU_VLM" "$VLM_EXTRA"
  start_stage image image "$BIN/umm-native-stage-image" "$IMAGE_MODEL" "$GPU_IMAGE" ""
  start_stage tts tts "$BIN/umm-native-stage-tts" "$TTS_MODEL" "$GPU_TTS" ""
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
