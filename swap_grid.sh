#!/usr/bin/env bash
# The window grid under the corrected design: length-matched phrase swap, 45
# frames, run per factor on the eight samples whose ceiling effect is largest.
#
# The ceiling screen (`swap_screen.sh`) established that there is something to
# localise: appearance moves the video toward the swapped caption in 14 of 16
# samples (p = 0.002) and owns the luminance metric 2.05x over the other two
# factors; dynamics owns the flow metric 2.34x. Geometry passes the sign test but
# at a sixth of appearance's size and owns no pixel metric, so it goes last.
#
# Selecting on ceiling effect size is the fix for the first pass, where 5 of 7
# samples had no effect at all and diluted every window average.
#
#   ./swap_grid.sh                 # A and D
#   FACTORS="G" ./swap_grid.sh     # geometry afterwards
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

OUT=${OUT:-./runs/swapgrid}
GPUS=${GPUS:-"0 1 2 7"}
FACTORS=${FACTORS:-"A D"}
PER=${PER:-2}
mkdir -p "$OUT"
read -ra G <<< "$GPUS"

specs=()
for f in $FACTORS; do
  mapfile -t S < <($PY -c "
import json; print('\n'.join(json.load(open('runs/swap45/top8.json'))['$f']))")
  for ((k = 0; k < ${#S[@]}; k += PER)); do
    sel=""
    for ((j = k; j < k + PER && j < ${#S[@]}; j++)); do sel+="${S[$j]},"; done
    specs+=("$f|${sel%,}|$k")
  done
done
echo "[plan] ${#specs[@]} jobs, ${#G[@]} GPUs, $PER samples each, 7 windows, 45 frames"

i=0
while [ $i -lt ${#specs[@]} ]; do
  pids=()
  for g in "${G[@]}"; do
    [ $i -ge ${#specs[@]} ] && break
    IFS='|' read -r f sel k <<< "${specs[$i]}"
    i=$((i + 1))
    tag="g${f}_o${k}"
    if [ -f "$OUT/swap_${f}_${tag}.json" ]; then
      echo "[skip] $tag"
      continue
    fi
    echo "[run] $f  $sel  gpu $g"
    CUDA_VISIBLE_DEVICES=$g nohup $PY -u swap_probe.py \
      --factor "$f" --only "$sel" --frames 45 --steps 20 --no_ceiling \
      --out "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
    pids+=($!)
    sleep 4
  done
  for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" || echo "[fail] $p"; done
done
echo "[all done]"
ls "$OUT"/swap_*.json | wc -l
