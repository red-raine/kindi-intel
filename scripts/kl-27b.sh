#!/bin/bash
# Bonsai 27B vs Ingot 27B: finish the downloads, convert both, then price the refit.
#
# The link gives ~0.3 MB/s to one connection and *less to each* of several (measured: one stream 0.56 MB/s, four in
# parallel 0.22 MB/s aggregate, three 0.27 MB/s), so the two 6 GB files are fetched strictly one at a time: Ingot's curl
# already holds the link, and Bonsai resumes the moment Ingot's file is complete, overlapping Ingot's conversion.
#
# Every step is guarded on its output, so re-running the script resumes rather than repeating: a file that is already
# finalised is not downloaded again, and a pack that already has its manifest is not converted again. Downloads retry
# with `-C -` on the same `.part`, and a transfer that stalls is aborted (`--speed-limit`) so the retry can resume it -
# otherwise an 8 h unattended job would sit forever in a wait loop on a curl that died.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SIZE=5946648928                                  # both files, from Content-Length on the CDN object
INGOT_DIR=/b/models/ingot-27b
INGOT_GGUF=$INGOT_DIR/Ingot-XXIV-27B-PTQ1_0.gguf
INGOT_PART=$INGOT_GGUF.part
INGOT_URL="https://huggingface.co/dendriteholdings/Ingot-XXIV-27B-Q2/resolve/main/Ingot-XXIV-27B-PTQ1_0.gguf"
INGOT_CURL=5547                                  # the curl already streaming Ingot; only its liveness is assumed
BONSAI_DIR=/b/models/bonsai-27b
BONSAI_GGUF=$BONSAI_DIR/Ternary-Bonsai-2-27B-PTQ1_0.gguf
BONSAI_PART=$BONSAI_GGUF.part
BONSAI_URL="https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
INGOT_PACK=/b/models/packs/ingot-27b
BONSAI_PACK=/b/models/packs/bonsai-27b
OUT=/b/models/packs
LOG=/tmp/kl27.log
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

size_of() { stat -c %s "$1" 2>/dev/null || echo 0; }

# rename a complete .part, retrying because the writing curl holds a Windows handle for a moment after the last byte
finalize() {
  local part=$1 gguf=$2
  [ -f "$part" ] || return 0
  for _ in $(seq 1 40); do
    mv "$part" "$gguf" 2>/dev/null && break
    sleep 15
  done
  [ -f "$gguf" ] && log "finalised $gguf ($(size_of "$gguf") bytes)" || { log "could not rename $part"; return 1; }
}

# download to .part until it reaches the full size, resuming across as many sessions as the link needs
fetch() {
  local gguf=$1 part=$2 url=$3 n attempt
  [ -f "$gguf" ] && { log "already downloaded: $gguf"; return 0; }
  for attempt in $(seq 1 40); do
    n=$(size_of "$part")
    [ "$n" -ge "$SIZE" ] && break
    log "download $(basename "$gguf") attempt $attempt from $n of $SIZE"
    curl -L -sS -C - --retry 20 --retry-delay 5 --retry-all-errors \
         --speed-limit 2048 --speed-time 300 -o "$part" "$url" >> "$LOG" 2>&1
    [ "$(size_of "$part")" -ge "$SIZE" ] && break
    sleep 30
  done
  [ "$(size_of "$part")" -ge "$SIZE" ] || { log "gave up on $part at $(size_of "$part") of $SIZE"; return 1; }
  finalize "$part" "$gguf"
}

convert() {
  local gguf=$1 pack=$2 name=$3
  if [ -f "$pack/kindi_pack.json" ]; then log "$name already converted"; return 0; fi
  log "converting $name -> $pack"
  ( cd "$REPO" && python -m kindi convert "$gguf" --out "$pack" >> "$LOG" 2>&1 )
  local rc=$?
  log "convert $name exit $rc"
  return $rc
}

cd "$REPO" || exit 1

# 1. Ingot: wait while the external curl is alive, then let fetch() finish it (it returns at once if already complete)
if [ ! -f "$INGOT_GGUF" ]; then
  while :; do
    [ "$(size_of "$INGOT_PART")" -ge "$SIZE" ] && break
    kill -0 "$INGOT_CURL" 2>/dev/null || break
    sleep 60
  done
fi
fetch "$INGOT_GGUF" "$INGOT_PART" "$INGOT_URL" || log "ingot download incomplete"

# 2. Bonsai, in the background so its download overlaps the Ingot conversion
( fetch "$BONSAI_GGUF" "$BONSAI_PART" "$BONSAI_URL" ) &
BONSAI_PID=$!

# 3. convert Ingot
[ -f "$INGOT_GGUF" ] && convert "$INGOT_GGUF" "$INGOT_PACK" ingot

# 4. wait for Bonsai, convert it
wait "$BONSAI_PID"
[ -f "$BONSAI_GGUF" ] && convert "$BONSAI_GGUF" "$BONSAI_PACK" bonsai

# 5. the tool's own end-to-end check on a real 27B file: a file against itself must differ in nothing
if [ -f "$INGOT_GGUF" ]; then
  log "self-comparison of the Ingot GGUF (expect all zeros)"
  python -m tools.kl blocks --a "$INGOT_GGUF" --b "$INGOT_GGUF" >> "$LOG" 2>&1
  log "self-comparison exit $?"
fi

# 6. what changed in the bytes: the ternary codes and the fp16 block scales, group by group
if [ -f "$BONSAI_GGUF" ] && [ -f "$INGOT_GGUF" ]; then
  log "blocks: bonsai vs ingot"
  python -m tools.kl blocks --a "$BONSAI_GGUF" --b "$INGOT_GGUF" --json "$OUT/kl-blocks-bonsai-ingot.json" >> "$LOG" 2>&1
  log "blocks exit $?"
fi

# 7. what that did to the distributions: KL both ways on the same tokens, plus each model's own NLL
if [ -f "$BONSAI_PACK/kindi_pack.json" ] && [ -f "$INGOT_PACK/kindi_pack.json" ]; then
  log "logits: bonsai vs ingot, 96 tokens"
  python -m tools.kl logits --a "$BONSAI_PACK" --b "$INGOT_PACK" --tokens 96 --ctx 256 \
    --json "$OUT/kl-logits-bonsai-ingot.json" >> "$LOG" 2>&1
  log "logits exit $?"
fi

log "chain done"
