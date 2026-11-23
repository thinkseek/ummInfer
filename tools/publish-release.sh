#!/usr/bin/env bash
# Build release artifacts from a git ref and publish them to the public
# release repository.
#
# Usage: tools/publish-release.sh v0.2.0
#
# Required on a macOS or Linux build host:
#   DEPS_PREFIX   prefix with include/ and lib/ for zmq, msgpack, nlohmann-json
#   CURL_PREFIX   prefix of a libcurl build with WebSocket support
#
# Stage binaries need CUDA. Pick one source:
#   STAGE_BUILD=local    build on this host (Linux + CUDA toolkit)
#   STAGE_BUILD=remote   build on STAGE_SSH, optionally inside STAGE_CONTAINER
# Default STAGE_BUILD=auto: local on Linux, remote when STAGE_SSH is set.
#
# See docs/release.md for the full environment.
set -euo pipefail

VERSION="${1:-}"
case "$VERSION" in
  v[0-9]*.[0-9]*.[0-9]*) ;;
  *) echo "usage: tools/publish-release.sh vX.Y.Z" >&2; exit 2 ;;
esac

SOURCE_REF="${SOURCE_REF:-main}"
RELEASE_REPO="${RELEASE_REPO:-git@github.com:thinkseek/ummInfer-releases.git}"
DEPS_PREFIX="${DEPS_PREFIX:-}"
CURL_PREFIX="${CURL_PREFIX:-}"
ZIG="${ZIG:-zig}"
JOBS="${JOBS:-8}"
STAGE_BUILD="${STAGE_BUILD:-auto}"
STAGE_SSH="${STAGE_SSH:-}"
STAGE_SSH_OPTS="${STAGE_SSH_OPTS:-}"
STAGE_CONTAINER="${STAGE_CONTAINER:-}"
STAGE_DEPS_PREFIX="${STAGE_DEPS_PREFIX:-}"
STAGE_CURL_PREFIX="${STAGE_CURL_PREFIX:-}"
STAGE_ZIG="${STAGE_ZIG:-zig}"
FORCE="${FORCE:-0}"

log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -n "$DEPS_PREFIX" ] || die "DEPS_PREFIX is not set"
[ -d "$DEPS_PREFIX/include" ] || die "DEPS_PREFIX has no include/: $DEPS_PREFIX"
[ -n "$CURL_PREFIX" ] || die "CURL_PREFIX is not set"

# Build from a clean tree so the published artifacts match the commit.
git diff --quiet && git diff --cached --quiet \
  || die "working tree is not clean; commit or revert first"
COMMIT="$(git rev-parse --verify "$SOURCE_REF^{commit}")"

HOST_OS="$(uname -s)"
HOST_ARCH="$(uname -m)"
case "$HOST_OS-$HOST_ARCH" in
  Darwin-arm64) PLATFORM=darwin-arm64 ;;
  Linux-x86_64) PLATFORM=linux-x86_64 ;;
  *) die "unsupported build host: $HOST_OS-$HOST_ARCH" ;;
esac

if [ "$STAGE_BUILD" = auto ]; then
  if [ -n "$STAGE_SSH" ]; then
    STAGE_BUILD=remote
  elif [ "$HOST_OS" = Linux ]; then
    STAGE_BUILD=local
  else
    die "stage binaries need Linux + CUDA: set STAGE_SSH or STAGE_BUILD=local"
  fi
fi
case "$STAGE_BUILD" in
  local|remote) ;;
  *) die "STAGE_BUILD must be local or remote" ;;
esac

WORK="$(mktemp -d "${TMPDIR:-/tmp}/umm-release.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# The temporary release clone has no identity of its own. Reuse the identity
# of this repository so the published commits carry the same author.
AUTHOR_NAME="$(git config user.name)"
AUTHOR_EMAIL="$(git config user.email)"
[ -n "$AUTHOR_NAME" ] && [ -n "$AUTHOR_EMAIL" ] \
  || die "git user.name and user.email must be set"

log "archiving $SOURCE_REF ($COMMIT)"
mkdir -p "$WORK/src" "$WORK/dist/$VERSION"
git archive --format=tar "$COMMIT" | tar -x -C "$WORK/src"

CURL_ARG=()
[ -n "$CURL_PREFIX" ] && CURL_ARG=(-Dcurl-prefix="$CURL_PREFIX")

log "building umm-infer on $PLATFORM"
(
  cd "$WORK/src"
  "$ZIG" build -Doptimize=ReleaseFast -Ddeps-prefix="$DEPS_PREFIX" \
    ${CURL_ARG[@]+"${CURL_ARG[@]}"}
)
mkdir -p "$WORK/dist/$VERSION/$PLATFORM"
install -m755 "$WORK/src/zig-out/bin/umm-infer" "$WORK/dist/$VERSION/$PLATFORM/umm-infer"

make_remote_script() {
  local out="$1"
  cat > "$out" <<EOF
#!/bin/bash
set -eu
R=/opt/umminfer/umm-release-$VERSION
rm -rf "\$R"; mkdir -p "\$R"
tar xzf /opt/umminfer/umm-release-$VERSION-src.tar.gz -C "\$R"
cd "\$R"
export LD_LIBRARY_PATH="$STAGE_DEPS_PREFIX/lib:$STAGE_CURL_PREFIX/lib"
"$STAGE_ZIG" build -Doptimize=ReleaseFast \\
  -Ddeps-prefix="$STAGE_DEPS_PREFIX" -Dcurl-prefix="$STAGE_CURL_PREFIX"
cmake -S native/stage_cli -B build -DUMM_STAGE_CUDA=ON \\
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$STAGE_DEPS_PREFIX"
cmake --build build -j "$JOBS"
mkdir -p out/lib
install -m755 zig-out/bin/umm-infer out/
for s in asr omni qwen38 tts image; do
  install -m755 "build/umm-native-stage-\$s" out/
done
# The server needs a libcurl with WebSocket support. Ubuntu's libcurl does
# not provide it, so ship the build prefix copy next to the binary.
patchelf --set-rpath '\$ORIGIN/lib' out/umm-infer
cp -L "$STAGE_CURL_PREFIX/lib/libcurl.so.4" out/lib/libcurl.so.4
EOF
}

if [ "$STAGE_BUILD" = remote ]; then
  [ -n "$STAGE_SSH" ] || die "STAGE_BUILD=remote needs STAGE_SSH"
  [ -n "$STAGE_DEPS_PREFIX" ] || die "STAGE_BUILD=remote needs STAGE_DEPS_PREFIX"
  [ -n "$STAGE_CURL_PREFIX" ] || die "STAGE_BUILD=remote needs STAGE_CURL_PREFIX"

  log "building stages on $STAGE_SSH${STAGE_CONTAINER:+:$STAGE_CONTAINER}"
  git archive --format=tar.gz -o "$WORK/src.tar.gz" "$COMMIT"
  make_remote_script "$WORK/stage-build.sh"
  scp -q "$WORK/src.tar.gz" "$STAGE_SSH:/tmp/umm-release-$VERSION-src.tar.gz"
  scp -q "$WORK/stage-build.sh" "$STAGE_SSH:/tmp/umm-release-$VERSION-build.sh"
  ssh "$STAGE_SSH" "VERSION='$VERSION' CONTAINER='$STAGE_CONTAINER' bash -s" <<'REMOTE'
set -eu
OUT=/tmp/umm-release-$VERSION-out
if [ -n "$CONTAINER" ]; then
  docker exec "$CONTAINER" mkdir -p /opt/umminfer
  docker cp "/tmp/umm-release-$VERSION-src.tar.gz" "$CONTAINER:/opt/umminfer/umm-release-$VERSION-src.tar.gz"
  docker cp "/tmp/umm-release-$VERSION-build.sh" "$CONTAINER:/opt/umminfer/umm-release-$VERSION-build.sh"
  docker exec "$CONTAINER" bash "/opt/umminfer/umm-release-$VERSION-build.sh"
  rm -rf "$OUT"; mkdir -p "$OUT"
  docker cp "$CONTAINER:/opt/umminfer/umm-release-$VERSION/out/." "$OUT/"
else
  mkdir -p /opt/umminfer
  cp "/tmp/umm-release-$VERSION-src.tar.gz" "/opt/umminfer/umm-release-$VERSION-src.tar.gz"
  bash "/tmp/umm-release-$VERSION-build.sh"
  OUT="/opt/umminfer/umm-release-$VERSION/out"
fi
rm -f "/tmp/umm-release-$VERSION-out.tgz"
tar czf "/tmp/umm-release-$VERSION-out.tgz" -C "$OUT" .
REMOTE
  scp -q "$STAGE_SSH:/tmp/umm-release-$VERSION-out.tgz" "$WORK/stages.tgz"
  mkdir -p "$WORK/stage-out"
  tar xzf "$WORK/stages.tgz" -C "$WORK/stage-out"
else
  STAGE_DEPS_PREFIX="${STAGE_DEPS_PREFIX:-$DEPS_PREFIX}"
  STAGE_CURL_PREFIX="${STAGE_CURL_PREFIX:-$CURL_PREFIX}"
  [ -n "$STAGE_CURL_PREFIX" ] || die "STAGE_BUILD=local needs STAGE_CURL_PREFIX or CURL_PREFIX"
  log "building stages locally"
  (
    cd "$WORK/src"
    export LD_LIBRARY_PATH="$STAGE_DEPS_PREFIX/lib:$STAGE_CURL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    cmake -S native/stage_cli -B build -DUMM_STAGE_CUDA=ON \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$STAGE_DEPS_PREFIX"
    cmake --build build -j "$JOBS"
  )
  mkdir -p "$WORK/stage-out/lib"
  install -m755 "$WORK/src/zig-out/bin/umm-infer" "$WORK/stage-out/umm-infer"
  for s in asr omni qwen38 tts image; do
    install -m755 "$WORK/src/build/umm-native-stage-$s" "$WORK/stage-out/"
  done
  patchelf --set-rpath '$ORIGIN/lib' "$WORK/stage-out/umm-infer"
  cp -L "$STAGE_CURL_PREFIX/lib/libcurl.so.4" "$WORK/stage-out/lib/libcurl.so.4"
fi

log "assembling dist/$VERSION"
mkdir -p "$WORK/dist/$VERSION/linux-x86_64"
cp -a "$WORK/stage-out/." "$WORK/dist/$VERSION/linux-x86_64/"
if [ "$PLATFORM" = linux-x86_64 ]; then
  # The local server build is the same file the stage build just produced.
  rm -f "$WORK/dist/$VERSION/$PLATFORM/umm-infer"
  rmdir "$WORK/dist/$VERSION/$PLATFORM" 2>/dev/null || true
fi

DIST="$WORK/dist/$VERSION"
if command -v sha256sum >/dev/null 2>&1; then
  SHA_CMD="sha256sum"
else
  SHA_CMD="shasum -a 256"
fi
(
  cd "$DIST"
  find . -type f ! -name SHA256SUMS ! -name MANIFEST.md -print0 \
    | sort -z | xargs -0 $SHA_CMD | sed 's|  \./|  |'
) > "$DIST/SHA256SUMS"

{
  echo "# Release $VERSION"
  echo
  echo "- Source commit: \`$COMMIT\` (branch \`$SOURCE_REF\`)"
  echo "- Published: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- Server build host: $HOST_OS-$HOST_ARCH, Zig $("$ZIG" version)"
  if [ "$STAGE_BUILD" = remote ]; then
    echo "- Stage build host: $STAGE_SSH${STAGE_CONTAINER:+ (container $STAGE_CONTAINER)}, CUDA 12.4"
  else
    echo "- Stage build host: local"
  fi
  echo
  echo "## Build commands"
  echo
  echo '```sh'
  echo "git archive $COMMIT | tar -x -C src && cd src"
  echo "zig build -Doptimize=ReleaseFast -Ddeps-prefix=\$DEPS_PREFIX ${CURL_PREFIX:+-Dcurl-prefix=\$CURL_PREFIX}"
  echo "cmake -S native/stage_cli -B build -DUMM_STAGE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=\$DEPS_PREFIX"
  echo "cmake --build build -j $JOBS"
  echo '```'
  echo
  echo "## Files"
  echo
  echo '| File | Bytes | SHA256 |'
  echo '| --- | ---: | --- |'
  while read -r sum path; do
    echo "| \`$path\` | $(wc -c < "$DIST/$path" | tr -d ' ') | \`$sum\` |"
  done < "$DIST/SHA256SUMS"
  echo
  echo "## Reproducibility"
  echo
  echo "A rebuild from the same commit produces the same artifact set, not the"
  echo "same bytes. The macOS binary embeds a per-run linker UUID, and nvcc gives"
  echo "each anonymous-namespace kernel symbol a per-compilation suffix. The"
  echo "input commit, build commands and file list are reproducible; these"
  echo "SHA256 values pin the published bytes."
} > "$DIST/MANIFEST.md"

log "syncing release repository"
git clone -q "$RELEASE_REPO" "$WORK/release"
cd "$WORK/release"
if git rev-parse -q --verify "refs/tags/$VERSION" >/dev/null; then
  [ "$FORCE" = 1 ] || die "tag $VERSION already exists in the release repository"
  git tag -d "$VERSION"
fi

rm -rf docs tools
mkdir -p docs tools
cp "$WORK/src/release/README.md" README.md
cp "$WORK/src/LICENSE" LICENSE
cp "$WORK/src/release/THIRD_PARTY_NOTICES.md" THIRD_PARTY_NOTICES.md
cp -a "$WORK/src/release/docs/." docs/
cp "$WORK/src/docs/release.md" docs/release.md
cp -a "$WORK/src/release/tools/." tools/
rm -rf "dist/$VERSION"
mkdir -p dist
cp -a "$DIST" "dist/$VERSION"

git add -A
git -c user.name="$AUTHOR_NAME" -c user.email="$AUTHOR_EMAIL" commit -q \
  -m "Publish $VERSION

Source commit: $COMMIT
Artifacts: dist/$VERSION (SHA256SUMS and MANIFEST.md included)"
git -c user.name="$AUTHOR_NAME" -c user.email="$AUTHOR_EMAIL" \
  tag -a "$VERSION" -m "$VERSION"
git push origin HEAD:main
if [ "$FORCE" = 1 ]; then
  git push --force origin "$VERSION"
else
  git push origin "$VERSION"
fi

log "published $VERSION"
printf '%s\n' "$DIST/SHA256SUMS"
