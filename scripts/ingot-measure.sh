#!/bin/bash
# Once the Ingot 27B pack exists, capture its V vectors. Loading the engine is the proof the pack is good; the
# capture is the measurement the 2B numbers came from, one size up.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PACK=/b/models/packs/ingot-27b
LOG=/tmp/ingot_capture.log
: > "$LOG"

for _ in $(seq 1 480); do                       # up to 12 h at 90 s a poll
  [ -f "$PACK/kindi_pack.json" ] && break
  sleep 90
done
[ -f "$PACK/kindi_pack.json" ] || { echo "no pack appeared at $PACK" >> "$LOG"; exit 1; }

cd "$REPO" || exit 1
echo "pack present: $(du -sh "$PACK" | cut -f1)" >> "$LOG"
python -m tools.kvq.capture --pack "$PACK" --tokens 96 --per-layer \
    --out /b/models/packs/ingot-27b-vcap.bin >> "$LOG" 2>&1
echo "capture exit $?" >> "$LOG"
