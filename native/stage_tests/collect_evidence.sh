#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Collects the deployment evidence for the native stages. Runs inside the
# H20B container (linyu-dev) with the service already started.
# Every file is capped at 64 KiB.
set -u
ROOT=${UMM_NATIVE_ROOT:-/root/umm-native-service}
OUT=${1:-$ROOT/evidence}
IPC=/tmp/umminfer-native-ipc
mkdir -p "$OUT"
CAP='head -c 65536'

run() {
  # Prints the command, its stdout+stderr and its exit code.
  name=$1
  shift
  {
    echo "\$ $*"
    "$@" 2>&1
    echo "exit=$?"
  } | $CAP >"$OUT/$name.txt"
}

run env date
run kernel uname -a
run nvidia_smi_driver nvidia-smi --query-gpu=driver_version --format=csv,noheader
run nvidia_smi_gpu nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
run binaries sha256sum "$ROOT/bin/umm-infer" "$ROOT"/bin/umm-native-stage-asr \
  "$ROOT"/bin/umm-native-stage-omni "$ROOT"/bin/umm-native-stage-tts
run ldd_server ldd "$ROOT/bin/umm-infer"
run ldd_asr ldd "$ROOT/bin/umm-native-stage-asr"
run ldd_omni ldd "$ROOT/bin/umm-native-stage-omni"
run ldd_tts ldd "$ROOT/bin/umm-native-stage-tts"
run python_torch_grep sh -c "ldd $ROOT/bin/umm-native-stage-* | grep -Ei 'python|torch|sglang' || echo 'no python/torch/sglang libraries'"
run processes sh -c "ps -eo pid,ppid,etimes,comm,args | grep -E 'umm-infer|umm-native-stage' | grep -v grep"
run models sh -c "du -sh /root/omniinfer-model-validation-20261111-k3/models/asr \
  /root/omniinfer-model-validation-20261111-k3/models/omni \
  /root/umm-mixed-output-20261116/models/qwen3-tts; \
  sha256sum /root/umm-mixed-output-20261116/models/qwen3-tts/config.json \
  /root/omniinfer-model-validation-20261111-k3/models/asr/config.json \
  /root/omniinfer-model-validation-20261111-k3/models/omni/config.json"
run model_revision sh -c "for d in /root/omniinfer-model-validation-20261111-k3/models/asr \
  /root/omniinfer-model-validation-20261111-k3/models/omni \
  /root/umm-mixed-output-20261116/models/qwen3-tts; do \
  echo \"\$d\"; (cd \$d && git rev-parse HEAD 2>/dev/null || echo UNKNOWN); done"
run health curl -s http://172.17.0.2:43400/health
run ready curl -s -o /dev/null -w '%{http_code}\n' http://172.17.0.2:43400/ready
run phase7_health curl -s http://172.17.0.2:43300/health
# A stage log opened with truncation can stay sparse after an in-place
# truncation, so strip NUL padding before keeping the tail.
{
  echo '$ for f in server asr omni tts; do tail -c 40000 logs/$f.log | tr -d NUL; done'
  for f in server asr omni tts; do
    echo "== $f =="
    tail -c 40000 "$ROOT/logs/$f.log" | tr -d '\000'
  done
  echo "exit=0"
} | $CAP >"$OUT/stage_logs.txt"
echo "evidence written to $OUT"
