#!/bin/bash
# Wait for the Ingot 27B download to finish, then convert it to a Kindi pack.
# Run detached: the download takes hours on this link, and the conversion must not depend on a tool call still being open.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GGUF=/b/models/ingot-27b/Ingot-XXIV-27B-PTQ1_0.gguf
PART=$GGUF.part
SIZE=5946648928
LOG=/tmp/ingot_convert.log
: > "$LOG"

# 1. the bytes. No pgrep here: this shell has no procps, so "is curl done" has to be read off the file itself.
while :; do
  n=$(stat -c %s "$PART" 2>/dev/null || echo 0)
  [ "$n" -ge "$SIZE" ] && break
  sleep 30
done

# 2. curl holds the handle for a moment after the last byte; wait for the size to stop moving, then rename with retries
prev=-1
while :; do
  n=$(stat -c %s "$PART" 2>/dev/null || echo 0)
  [ "$n" = "$prev" ] && break
  prev=$n
  sleep 30
done
for _ in $(seq 1 40); do
  [ -f "$PART" ] || break
  mv "$PART" "$GGUF" 2>/dev/null && break
  sleep 15
done
[ -f "$GGUF" ] || { echo "could not rename $PART" >> "$LOG"; exit 1; }
echo "downloaded: $(stat -c %s "$GGUF") bytes" >> "$LOG"

# 3. convert (the profile (64, 5120, 17408, 24, 4, 16) is in convert.PROFILES)
cd "$REPO" || exit 1
python -m kindi convert "$GGUF" --out /b/models/packs/ingot-27b >> "$LOG" 2>&1
echo "convert exit $?" >> "$LOG"
