#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
CONFIG=${1:-release}
cd "$ROOT"
swift build -c "$CONFIG"
BIN=$(swift build -c "$CONFIG" --show-bin-path)
APP="$ROOT/build/ummmlx.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN/ummmlx" "$APP/Contents/MacOS/ummmlx"
cp "$ROOT/Support/Info.plist" "$APP/Contents/Info.plist"
cp "$ROOT/../../LICENSE" "$APP/Contents/Resources/LICENSE"
cp "$ROOT/ThirdParty/Voxt/NOTICE" "$APP/Contents/Resources/Voxt-NOTICE"
# Ad-hoc signature, cdhash-only: the app must be signed for TCC to consider
# speech and microphone access at all. Every rebuild changes the cdhash, so a
# grant may be asked again after Scripts/build.sh runs. No certificate,
# entitlement, or notarization.
codesign --force --sign - --identifier dev.umminfer.ummmlx "$APP" >/dev/null
printf '%s\n' "$APP"
