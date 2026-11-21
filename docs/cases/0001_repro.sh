#!/bin/sh
# One-command reproduction of docs/cases/0001-end-to-end-native.md.
#
#   SSH_OPTS="-o ControlPath=$HOME/.ssh/cm/tn-h20b" sh docs/cases/0001_repro.sh
#
# Needs: git checkout of the recorded commit, ssh access to H20B, docker
# container linyu-dev, the 16 kHz WAV fixture below. The script stops the
# resident service at the end and leaves every stage GPU released.
set -eu

TARGET=${TARGET:-h20b}
CONTAINER=${CONTAINER:-linyu-dev}
AUDIO=${AUDIO:-/tmp/umminfer-phase4-bench-evidence/inputs/asr_en_pcm16.wav}
SSH_OPTS=${SSH_OPTS:-}
ROOT=/root/umm-native-service
CASE=/root/case0001

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

git archive --format=tar.gz HEAD > "$work/src.tgz"
cp docs/cases/0001_e2e_client.py "$work/0001_e2e_client.py"
cp "$AUDIO" "$work/input.wav"

cat > "$work/inner.sh" <<'INNER'
set -eu
ROOT=/root/umm-native-service
CASE=/root/case0001
cd "$ROOT"
if [ -f run/supervisor.pid ]; then
  python3 -I python/umminfer_service.py --config config.json stop || true
fi
rm -rf src
mkdir -p src
tar xzf "$CASE/src.tgz" -C src
cd src
rm -rf .zig-cache zig-out build
/root/umm-c6rt-ckl/zig/zig build -Doptimize=ReleaseFast \
  -Dcurl-prefix=/root/umm-c6rt-ckl/curl-prefix \
  -Ddeps-prefix=/root/umm-c6rt-ckl/deps
cmake -S native/stage_cli -B build -DUMM_STAGE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
cd "$ROOT"
install -m755 src/zig-out/bin/umm-infer bin/umm-infer
for kind in asr omni image tts; do
  install -m755 "src/build/umm-native-stage-$kind" "bin/umm-native-stage-$kind"
done
cp src/python/umminfer_service.py python/umminfer_service.py
cp src/service/h20b-service.json config.json
sha256sum bin/*
python3 -I python/umminfer_service.py --config config.json start
tries=0
until curl -sf http://172.17.0.2:43301/ready >/dev/null 2>&1; do
  tries=$((tries + 1))
  if [ "$tries" -gt 300 ]; then echo "ready timeout" >&2; exit 1; fi
  sleep 0.5
done
rm -rf "$CASE/out"
mkdir -p "$CASE/out"
python3 -I "$CASE/0001_e2e_client.py" run --url http://172.17.0.2:43301 \
  --audio "$CASE/input.wav" --out "$CASE/out/run1" --timeout 1200
sha256sum "$CASE"/out/run1/art-*
printf 'answer: '; cat "$CASE/out/run1/text.txt"; echo
python3 -I python/umminfer_service.py --config config.json stop
INNER

ssh $SSH_OPTS "$TARGET" "mkdir -p $CASE"
scp $SSH_OPTS "$work/src.tgz" "$work/0001_e2e_client.py" "$work/input.wav" \
  "$work/inner.sh" "$TARGET:$CASE/" >/dev/null
ssh $SSH_OPTS "$TARGET" "docker cp $CASE $CONTAINER:$CASE"
ssh $SSH_OPTS "$TARGET" "docker exec $CONTAINER sh $CASE/inner.sh"
ssh $SSH_OPTS "$TARGET" "docker exec $CONTAINER tar czf $CASE/out.tgz -C $CASE out"
ssh $SSH_OPTS "$TARGET" "docker cp $CONTAINER:$CASE/out.tgz $CASE/"

mkdir -p evidence
scp $SSH_OPTS "$TARGET:$CASE/out.tgz" "evidence/case0001-out.tgz" >/dev/null
echo "evidence/case0001-out.tgz"
