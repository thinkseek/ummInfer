#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Produces the before/after evidence committed in this directory.
#
#   ./run_measurement.sh host   # Mac/Linux host: cmake + ctest, no GPU
#   ./run_measurement.sh gpu    # H20B GPU host: baseline, optimised, GPU tests
#
# Every command is recorded with its command line, its real exit code, and its
# full stdout+stderr.  Each output file is hard-capped at 64 KiB: the command's
# own output is piped through `head -c` (so a runaway command is bounded) and
# the log is re-capped after every append.
#
# The GPU mode expects:
#   $NEW, $BENCH   the optimised build (default /tmp/omni-perf/build/omni_cli/...)
#   $BASE          the pre-optimisation binary
#                  (built from base commit 2b6177f with -DUMM_OMNI_CUDA=ON)
#   $MODEL         the Qwen3-Omni checkpoint directory
#   $IMAGE         the accepted fixture, sha256 e0691718...060e
set -u

CAP=65536
MODE=${1:-gpu}
HERE=$(cd "$(dirname "$0")" && pwd)

capfile() { head -c "$CAP" "$1" > "$1.cap" && mv "$1.cap" "$1"; }

run() {  # run <logfile> <cmd...>
  local log=$1; shift
  "$@" 2>&1 | head -c "$CAP" > /tmp/_ev_cap.out
  local rc=${PIPESTATUS[0]}
  {
    printf '$ %s\n' "$*"
    printf 'exit=%d\n' "$rc"
    cat /tmp/_ev_cap.out
    printf '\n'
  } >> "$log"
  capfile "$log"
  rm -f /tmp/_ev_cap.out
  return "$rc"
}

if [ "$MODE" = host ]; then
  LOG="$HERE/40_host_tests.txt"
  : > "$LOG"
  run "$LOG" uname -a
  run "$LOG" c++ --version
  rm -rf /tmp/omni-ev-build
  run "$LOG" cmake -S "$HERE/../../omni" -B /tmp/omni-ev-build
  run "$LOG" cmake --build /tmp/omni-ev-build -j 8
  run "$LOG" ctest --test-dir /tmp/omni-ev-build --output-on-failure
  echo "wrote $LOG"
  exit 0
fi

NEW=${NEW:-/tmp/omni-perf/build/omni_cli/umm-omni-native}
BENCH=${BENCH:-/tmp/omni-perf/build/omni_cli/umm-omni-bench}
BASE=${BASE:-/tmp/umm-omni-vlm-verify-8252c10/build/omni_cli/umm-omni-native}
MODEL=${MODEL:-/root/omniinfer-model-validation-20261111-k3/models/omni}
IMAGE=${IMAGE:-/tmp/man_ironing.png}
PROMPT="What is the person in this image doing? Answer in one short sentence."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}

ENVLOG="$HERE/00_env.txt"
BASELOG="$HERE/10_baseline.txt"
OPTLOG="$HERE/20_optimised.txt"
TESTLOG="$HERE/30_gpu_tests.txt"
: > "$ENVLOG"; : > "$BASELOG"; : > "$OPTLOG"; : > "$TESTLOG"

# ---- environment ---------------------------------------------------------
run "$ENVLOG" nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total \
                          --format=csv
run "$ENVLOG" nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 6
run "$ENVLOG" nvcc --version
run "$ENVLOG" sha256sum "$BASE" "$NEW" "$BENCH"
run "$ENVLOG" sha256sum "$IMAGE"
run "$ENVLOG" ldd "$NEW"
run "$ENVLOG" cmake --build /tmp/omni-perf/build -j 8
run "$TESTLOG" ctest --test-dir /tmp/omni-perf/build -R omni_moe_down_test \
                     --output-on-failure
run "$TESTLOG" /tmp/omni-perf/build/omni_cli/omni_moe_down_test

# ---- baseline (unmodified binary, serial, GPU6) --------------------------
run "$BASELOG" "$BASE" --model "$MODEL" --prompt "$PROMPT" \
                       --image "$IMAGE" --max-new-tokens 1
run "$BASELOG" "$BASE" --model "$MODEL" --prompt "$PROMPT" --image "$IMAGE"
run "$BASELOG" "$BASE" --model "$MODEL" --prompt "$PROMPT"

# ---- optimised -----------------------------------------------------------
run "$OPTLOG" "$BENCH" --model "$MODEL" --image "$IMAGE" --gpu 0
run "$OPTLOG" "$NEW" --model "$MODEL" --prompt "$PROMPT" --image "$IMAGE" \
                     --max-new-tokens 24
run "$OPTLOG" "$NEW" --model "$MODEL" --prompt "$PROMPT" --max-new-tokens 24
run "$ENVLOG" nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 6

echo "wrote $ENVLOG $BASELOG $OPTLOG $TESTLOG"
