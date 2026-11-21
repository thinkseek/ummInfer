#!/bin/sh
# One-command reproduction of docs/cases/0001-end-to-end-native.md.
#
#   TARGET=<h20b-ssh-host> CONTAINER=<gpu-container> \
#   SERVICE_HOST=<container-private-ip> \
#   ZIG_BIN=/opt/umminfer/zig/zig CURL_PREFIX=/opt/umminfer/curl-prefix \
#   DEPS_PREFIX=/opt/umminfer/deps \
#     sh docs/cases/0001_repro.sh
#
# Needs: git checkout of the recorded commit, ssh access to the H20B host
# (TARGET), a running GPU docker container (CONTAINER), and the 16 kHz WAV
# fixture (AUDIO). Optional: SSH_OPTS, SERVICE_HOST, SERVICE_PORT, ROOT, CASE,
# ZIG_BIN, CURL_PREFIX, DEPS_PREFIX. The script stops the resident service at
# the end and leaves every stage GPU released.
set -eu

TARGET=${TARGET:?set TARGET to the H20B SSH host}
CONTAINER=${CONTAINER:?set CONTAINER to the GPU container name}
AUDIO=${AUDIO:-/tmp/umminfer-phase4-bench-evidence/inputs/asr_en_pcm16.wav}
SSH_OPTS=${SSH_OPTS:-}
ROOT=${ROOT:-/opt/umminfer/native-service}
CASE=${CASE:-/tmp/umm-case0001}
SERVICE_HOST=${SERVICE_HOST:-127.0.0.1}
SERVICE_PORT=${SERVICE_PORT:-43301}
ZIG_BIN=${ZIG_BIN:-zig}
CURL_PREFIX=${CURL_PREFIX:-/opt/umminfer/curl-prefix}
DEPS_PREFIX=${DEPS_PREFIX:-/opt/umminfer/deps}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

git archive --format=tar.gz HEAD > "$work/src.tgz"
cp docs/cases/0001_e2e_client.py "$work/0001_e2e_client.py"
cp "$AUDIO" "$work/input.wav"

cat > "$work/inner.sh" <<'INNER'
set -eu
ROOT=${ROOT:-/opt/umminfer/native-service}
CASE=${CASE:-/tmp/umm-case0001}
SERVICE_HOST=${SERVICE_HOST:-127.0.0.1}
SERVICE_PORT=${SERVICE_PORT:-43301}
ZIG_BIN=${ZIG_BIN:-zig}
CURL_PREFIX=${CURL_PREFIX:-/opt/umminfer/curl-prefix}
DEPS_PREFIX=${DEPS_PREFIX:-/opt/umminfer/deps}
cd "$ROOT"
if [ -f run/supervisor.pid ]; then
  python3 -I python/umminfer_service.py --config config.json stop || true
fi
rm -rf src
mkdir -p src
tar xzf "$CASE/src.tgz" -C src
cd src
rm -rf .zig-cache zig-out build
"$ZIG_BIN" build -Doptimize=ReleaseFast \
  -Dcurl-prefix="$CURL_PREFIX" \
  -Ddeps-prefix="$DEPS_PREFIX"
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
until curl -sf "http://$SERVICE_HOST:$SERVICE_PORT/ready" >/dev/null 2>&1; do
  tries=$((tries + 1))
  if [ "$tries" -gt 300 ]; then echo "ready timeout" >&2; exit 1; fi
  sleep 0.5
done
rm -rf "$CASE/out"
mkdir -p "$CASE/out"
python3 -I "$CASE/0001_e2e_client.py" run --url "http://$SERVICE_HOST:$SERVICE_PORT" \
  --audio "$CASE/input.wav" --out "$CASE/out/run1" --timeout 1200
sha256sum "$CASE"/out/run1/art-*
printf 'answer: '; cat "$CASE/out/run1/text.txt"; echo
python3 -I python/umminfer_service.py --config config.json stop
INNER

ssh $SSH_OPTS "$TARGET" "mkdir -p $CASE"
scp $SSH_OPTS "$work/src.tgz" "$work/0001_e2e_client.py" "$work/input.wav" \
  "$work/inner.sh" "$TARGET:$CASE/" >/dev/null
ssh $SSH_OPTS "$TARGET" "docker cp $CASE $CONTAINER:$CASE"
ssh $SSH_OPTS "$TARGET" "docker exec \
  -e ROOT='$ROOT' -e CASE='$CASE' \
  -e SERVICE_HOST='$SERVICE_HOST' -e SERVICE_PORT='$SERVICE_PORT' \
  -e ZIG_BIN='$ZIG_BIN' -e CURL_PREFIX='$CURL_PREFIX' -e DEPS_PREFIX='$DEPS_PREFIX' \
  $CONTAINER sh $CASE/inner.sh"
ssh $SSH_OPTS "$TARGET" "docker exec $CONTAINER tar czf $CASE/out.tgz -C $CASE out"
ssh $SSH_OPTS "$TARGET" "docker cp $CONTAINER:$CASE/out.tgz $CASE/"

mkdir -p evidence
scp $SSH_OPTS "$TARGET:$CASE/out.tgz" "evidence/case0001-out.tgz" >/dev/null
echo "evidence/case0001-out.tgz"
