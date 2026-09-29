#!/bin/bash
# JSpark3 refuses to start over the previous boot's receipts (work/rankN/evidence)
# and its OPERATIONS.md says to archive them after a stop. Move each rank's evidence
# aside (kernel/compile caches in the work root stay, so the next boot stays warm).
set -euo pipefail
ts=$(date +%Y%m%dT%H%M%S)
for h in 127.0.0.1 "$@"; do
  ssh -o BatchMode=yes "$h" "for d in /home/leo/jspark3/work/rank*/evidence; do [ -d \"\$d\" ] && mv \"\$d\" \"\$d-stopped-$ts\" && echo \"\$(hostname): archived \$d -> evidence-stopped-$ts\"; done; true"
done
