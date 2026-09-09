#!/usr/bin/env bash
# Keep GB10 page cache small during model load so NVRM can allocate the KV slab.
# Runs for 25 min max, flushes whenever Cached > 40 GiB. NVIDIA KB 5776 remedy.
#
# Source: tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark cache_flusher.sh @ 050081d, adopted 2026-09-08.
# LEO: sudo -n (drop_caches is the one whitelisted sudo command on this pair), a pidfile so a relaunch replaces
# LEO: the previous instance instead of stacking, one log line per flush, DURATION / CACHED_GIB knobs.
# LEO: Started on both nodes by `make cache-flusher` (every launch target depends on it); `make stop-cache-flusher` ends it early.
set -u
DURATION="${DURATION:-1500}"; LIMIT="${CACHED_GIB:-40}"
PIDFILE="$HOME/.cache_flusher.pid"; LOG="$HOME/bench/cache-flusher-$(hostname).log"
mkdir -p "$(dirname "$LOG")"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then kill "$(cat "$PIDFILE")" 2>/dev/null; sleep 1; fi
echo $$ > "$PIDFILE"; trap 'rm -f "$PIDFILE"' EXIT
echo "$(date '+%F %T') start pid $$: ${DURATION}s, flush when Cached > ${LIMIT} GiB" >> "$LOG"
end=$((SECONDS+DURATION)); n=0
while [ $SECONDS -lt $end ]; do
  c=$(awk '/^Cached:/{print int($2/1048576)}' /proc/meminfo)
  if [ "${c:-0}" -gt "$LIMIT" ]; then
    sync
    if echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; then
      n=$((n+1)); echo "$(date '+%F %T') flushed: Cached was ${c} GiB, MemAvailable now $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB" >> "$LOG"
    else
      echo "$(date '+%F %T') drop_caches failed (sudo -n not allowed?)" >> "$LOG"; sleep 60
    fi
  fi
  sleep 5
done
echo "$(date '+%F %T') done: $n flushes" >> "$LOG"; rm -f "$PIDFILE"
