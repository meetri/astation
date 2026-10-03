#!/usr/bin/env bash
# Atomically activate one validated immutable plugin release for the shared
# plugin path and every non-hidden profile. Restores all changed paths on error.
set -euo pipefail

[ "$#" -eq 3 ] || { echo "usage: $0 DATA PLUGIN_NAME SOURCE_SHA" >&2; exit 2; }
DATA=$1
PLUGIN_NAME=$2
SOURCE_SHA=$3
RELEASE_ROOT="$DATA/plugin-deployments/$PLUGIN_NAME"
RELEASE="$RELEASE_ROOT/$SOURCE_SHA"
SHARED_LINK="$DATA/plugins/$PLUGIN_NAME"
SHARED_TARGET="../plugin-deployments/$PLUGIN_NAME/$SOURCE_SHA"
PROFILE_TARGET="../../../plugins/$PLUGIN_NAME"
TOKEN="$$"

for required in plugin.yaml __init__.py dashboard/plugin_api.py gateway/api/bootstrap.py .bundle.sha256; do
  test -f "$RELEASE/$required"
done

# Refuse ambiguous/stale profile copies before changing anything. Existing
# canonical links need no mutation; absent links are staged and added below.
declare -a profile_links=()
declare -a missing_profile_links=()
declare -a staged_profile_links=()
for profile in "$DATA"/profiles/*/; do
  [ -d "$profile" ] || continue
  link="$profile/plugins/$PLUGIN_NAME"
  profile_links+=("$link")
  if [ -L "$link" ]; then
    [ "$(readlink "$link")" = "$PROFILE_TARGET" ] || {
      echo "REFUSING: unexpected profile plugin link: $link -> $(readlink "$link")" >&2
      exit 3
    }
  elif [ -e "$link" ]; then
    echo "REFUSING: profile plugin path is not a symlink: $link" >&2
    exit 3
  else
    missing_profile_links+=("$link")
  fi
done

shared_next="$DATA/plugins/.$PLUGIN_NAME.next.$TOKEN"
shared_previous=""
shared_switched=0
declare -a created_profile_links=()
declare -a created_profile_dirs=()

exchange_paths() {
  python3 - "$1" "$2" <<'PY'
import ctypes
import os
import sys

left, right = (os.fsencode(value) for value in sys.argv[1:])
libc = ctypes.CDLL(None, use_errno=True)
renameat2 = libc.renameat2
renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
renameat2.restype = ctypes.c_int
if renameat2(-100, left, -100, right, 2) != 0:  # AT_FDCWD, RENAME_EXCHANGE
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error))
PY
}

rollback() {
  status=$?
  trap - EXIT
  for link in "${created_profile_links[@]}"; do rm -f "$link"; done
  for next in "${staged_profile_links[@]}"; do rm -f "$next"; done
  for directory in "${created_profile_dirs[@]}"; do rmdir "$directory" 2>/dev/null || true; done
  if [ "$shared_switched" -eq 1 ]; then
    if [ -n "$shared_previous" ] && { [ -e "$shared_previous" ] || [ -L "$shared_previous" ]; }; then
      exchange_paths "$SHARED_LINK" "$shared_previous" || true
    else
      rm -f "$SHARED_LINK"
    fi
  fi
  rm -f "$shared_next"
  exit "$status"
}
trap rollback EXIT

# Only mutate after every existing profile path has passed preflight. All
# staged paths and newly created empty directories are covered by rollback.
mkdir -p "$DATA/plugins"
rm -f "$shared_next"
ln -s "$SHARED_TARGET" "$shared_next"
for link in "${missing_profile_links[@]}"; do
  directory=${link%/$PLUGIN_NAME}
  if [ ! -d "$directory" ]; then
    mkdir -p "$directory"
    created_profile_dirs+=("$directory")
  fi
  next="$directory/.$PLUGIN_NAME.next.$TOKEN"
  rm -f "$next"
  ln -s "$PROFILE_TARGET" "$next"
  staged_profile_links+=("$next")
done

# renameat2(RENAME_EXCHANGE) switches an existing directory or symlink with no
# missing-path window. The old path remains at shared_next until verification.
if [ -e "$SHARED_LINK" ] || [ -L "$SHARED_LINK" ]; then
  exchange_paths "$SHARED_LINK" "$shared_next"
  shared_previous="$shared_next"
else
  mv -T "$shared_next" "$SHARED_LINK"
fi
shared_switched=1

for next in "${staged_profile_links[@]}"; do
  link="${next%/.${PLUGIN_NAME}.next.${TOKEN}}/$PLUGIN_NAME"
  mv -T "$next" "$link"
  created_profile_links+=("$link")
done

test "$(readlink "$SHARED_LINK")" = "$SHARED_TARGET"
test "$(readlink -f "$SHARED_LINK")" = "$(readlink -f "$RELEASE")"
test -f "$SHARED_LINK/plugin.yaml"
for link in "${profile_links[@]}"; do
  test "$(readlink "$link")" = "$PROFILE_TARGET"
  test -f "$link/plugin.yaml"
done

# Commit: keep a displaced legacy directory as rollback evidence; an old
# release symlink is already recoverable by its immutable target and is removed.
if [ -n "$shared_previous" ]; then
  if [ -d "$shared_previous" ] && [ ! -L "$shared_previous" ]; then
    mv "$shared_previous" "$RELEASE_ROOT/legacy-$(date +%Y%m%dT%H%M%S)"
  else
    rm -f "$shared_previous"
  fi
fi
trap - EXIT
printf '%s\n' "${#profile_links[@]}"
