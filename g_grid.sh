#!/usr/bin/env bash
# Geometry's window grid, on the eight samples whose future-topology swap beat the
# length-matched filler by the most (struct ratio x hist ratio, from runs/grecond
# vs runs/gfiller).
#
# The ceiling is measured inside this run rather than reused from runs/grecond:
# pair assignment is now keyed on the sample's position in the clean list, which
# is stable across shards but differs from the shard-local assignment that earlier
# run used, so its ceiling would normalise a different caption pair.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
OUT=${OUT:-./runs/ggrid}
GPUS=${GPUS:-"0 1 2 5 7"}
PER=${PER:-2}
mkdir -p "$OUT"
read -ra G <<< "$GPUS"
mapfile -t S < <($PY -c "
import json; print('\n'.join(json.load(open('runs/grecond/top8.json'))['G']))")
echo "[samples] ${#S[@]}: ${S[*]}"
specs=()
for ((k = 0; k < ${#S[@]}; k += PER)); do
  sel=""
  for ((j = k; j < k + PER && j < ${#S[@]}; j++)); do sel+="${S[$j]},"; done
  specs+=("${sel%,}|$k")
done
i=0
while [ $i -lt ${#specs[@]} ]; do
  pids=()
  for g in "${G[@]}"; do
    [ $i -ge ${#specs[@]} ] && break
    IFS='|' read -r sel k <<< "${specs[$i]}"
    i=$((i + 1))
    tag="gg_o$k"
    [ -f "$OUT/swap_G_${tag}.json" ] && { echo "[skip] $tag"; continue; }
    echo "[run] $sel  gpu $g"
    CUDA_VISIBLE_DEVICES=$g nohup $PY -u swap_probe.py \
      --factor G --pairs g_pairs.json --only "$sel" \
      --frames 45 --steps 20 \
      --out "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
    pids+=($!)
    sleep 4
  done
  for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" || echo "[fail] $p"; done
done
echo "[all done]"
