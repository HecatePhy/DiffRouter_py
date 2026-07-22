#!/bin/bash
# Clone Potter at the pinned upstream commit, apply the DiffRouter GR-guidance
# patch, and build. Potter is a separate BSD-3 project (github.com/diriLin/Potter);
# we vendor only our patch, not their source.
#
# Usage:
#   potter/setup_potter.sh [dest_dir]        # default: ./third_party/Potter
#
# Then:
#   <dest>/build/route -i <design>_unrouted.phys -o routed.phys \
#       -d data/xcvu3p.device -t 32 -r -g <guide> --guide_penalty 0.5
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/third_party/Potter}"
UPSTREAM="https://github.com/diriLin/Potter.git"
PINNED="19b898ab77c13fc88646992e789aa2211626a123"   # commit the patch applies to
PATCH="$ROOT/potter/0001-diffrouter-gr-guidance.patch"

echo "==> Potter -> $DEST (pinned $PINNED)"
if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone "$UPSTREAM" "$DEST"
fi
cd "$DEST"

# Pin, so the patch always applies to the source it was written against.
git fetch --all --tags 2>/dev/null || true
if git cat-file -e "$PINNED^{commit}" 2>/dev/null; then
  git checkout -q "$PINNED"
else
  echo "WARNING: pinned commit $PINNED not found (shallow clone?); using default branch."
  echo "         If the patch fails to apply, upstream has moved -- rebase the patch."
fi

echo "==> applying $PATCH"
if git apply --check "$PATCH" 2>/dev/null; then
  git apply "$PATCH"
  echo "    applied."
elif git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "    already applied, skipping."
else
  echo "ERROR: patch does not apply. Upstream may have changed; rebase"
  echo "       potter/0001-diffrouter-gr-guidance.patch against $(git rev-parse --short HEAD)."
  exit 1
fi

echo "==> building (needs cmake, a C++17 compiler, zlib, boost-serialization)"
make build

# The Python wirelength tools (wa.py for CPWL, scripts/total_wirelength.py for total WL)
# import the Cap'n Proto schema from $DEST/fpga-interchange-schema/interchange. Some Potter
# checkouts only ship it under libs/interchange/definition; make the expected path resolve
# either way so CPWL / total-WL don't fail with "No module named PhysicalNetlist_capnp".
if [ ! -f "$DEST/fpga-interchange-schema/interchange/PhysicalNetlist.capnp" ] \
   && [ -f "$DEST/libs/interchange/definition/PhysicalNetlist.capnp" ]; then
  echo "==> linking interchange schema for the wirelength tools"
  mkdir -p "$DEST/fpga-interchange-schema"
  ln -sf ../libs/interchange/definition "$DEST/fpga-interchange-schema/interchange"
fi

echo "==> done: $DEST/build/route"
echo "    new options:  -g <guide file>   --guide_penalty <float>"
echo
echo "Potter also needs the device file; symlink the one in data/:"
echo "    ln -sf $ROOT/data/xcvu3p.device $DEST/xcvu3p.device"
