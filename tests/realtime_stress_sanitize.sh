#!/usr/bin/env bash
# Build production realtime code with GCC ASan+UBSan; no GPU or model is used.
set -euo pipefail

if [[ $# -lt 1 || ( "$1" != "voice" && "$1" != "unified" ) ]]; then
  printf 'usage: %s voice|unified [-- test-args...]\n' "$0" >&2
  exit 2
fi
mode=$1
shift
if [[ $# -gt 0 && "$1" == "--" ]]; then shift; fi

root=$(cd -- "$(dirname -- "$0")/.." && pwd)
deps_prefix=${UMM_DEPS_PREFIX:-"$HOME/.local/opt/umm-deps"}
curl_prefix=${UMM_CURL_PREFIX:-"$HOME/.local/opt/curl-ws"}
build_dir=${UMM_SANITIZE_BUILD_DIR:-"${TMPDIR:-/tmp}/umminfer-realtime-sanitize"}
mkdir -p -- "$build_dir"
binary="$build_dir/realtime_${mode}_asan_ubsan"

common=(
  -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -pthread
  -I"$root/include" -I"$root/src" -I"$curl_prefix/include" -I"$deps_prefix/include"
)
voice_sources=(
  "$root/tests/realtime_stress_test.cpp"
  "$root/src/client/voice.cpp"
  "$root/src/client/websocket.cpp"
  "$root/src/client/http.cpp"
  "$root/src/client/audio_device.cpp"
  "$root/src/util/codec.cpp"
)
unified_sources=(
  "$root/tests/unified_contract_test.cpp"
  "$root/src/util/codec.cpp"
  "$root/src/proto/wire.cpp"
  "$root/src/proto/messages.cpp"
  "$root/src/obs/metrics.cpp"
  "$root/src/obs/request_telemetry.cpp"
  "$root/src/relay/relay.cpp"
  "$root/src/comm/comm.cpp"
  "$root/src/scheduling/simple_scheduler.cpp"
  "$root/src/pipeline/pipeline.cpp"
  "$root/src/pipeline/session_store.cpp"
  "$root/src/pipeline/unified_schema.cpp"
  "$root/src/pipeline/narrative_parser.cpp"
  "$root/src/pipeline/unified.cpp"
  "$root/src/serve/upstream_proxy.cpp"
  "$root/src/serve/openai_helpers.cpp"
  "$root/src/serve/openai_audio.cpp"
  "$root/src/serve/openai_chat.cpp"
  "$root/src/serve/openai_api.cpp"
  "$root/src/serve/realtime_api.cpp"
  "$root/src/audio/audio.cpp"
  "$root/src/client/http.cpp"
  "$root/src/client/openai.cpp"
  "$root/src/client/asr.cpp"
  "$root/src/client/websocket.cpp"
  "$root/src/client/realtime.cpp"
  "$root/src/client/audio_device.cpp"
  "$root/src/client/voice.cpp"
  "$root/src/client/cli.cpp"
)

if [[ $mode == voice ]]; then
  sources=("${voice_sources[@]}")
  libraries=(-lcurl -ldl)
else
  sources=("${unified_sources[@]}")
  libraries=(-lzmq -lmsgpackc -lwebsockets -lcurl -ldl)
fi

"${CXX:-g++}" "${common[@]}" "${sources[@]}" \
  -L"$deps_prefix/lib" -L"$curl_prefix/lib" \
  -Wl,-rpath,"$deps_prefix/lib" -Wl,-rpath,"$curl_prefix/lib" \
  "${libraries[@]}" -o "$binary"

ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=1:halt_on_error=1} \
UBSAN_OPTIONS=${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1} \
  "$binary" "$@"
