#!/usr/bin/env bash
# Coarse-to-fine: split each factor's winning four-block window into two-block
# halves and see which half carries it.
#
# The four-block grid (`swap_grid.sh`) localised appearance to L8-11 (luminance
# and colour argmax there in 4 of 8 samples, p = 0.018, with L12-19 rejected at
# 0-1 of 8) and dynamics to L12-15 (flow argmax there in 7 of 8, p = 0.000, and
# the L12-19 band winning 8 of 8, p = 0.0000). Halving the window is the next
# honest step: going straight to single layers would put eight sites under FDR
# with eight samples each, and the earlier causal work showed single-layer
# injection is highly redundant, so a per-layer null is likely underpowered.
#
# Appearance gets L4-11 halved four ways because its profile is broad (L4-7 55%,
# L8-11 60%); dynamics gets L12-19, where its own metric peaks.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

OUT=${OUT:-./runs/swapfine}
GPUS=${GPUS:-"0 1 5 7"}
PER=${PER:-2}
mkdir -p "$OUT"
read -ra G <<< "$GPUS"

declare -A WIN=(
  [A]="4-5,6-7,8-9,10-11"
  [D]="12-13,14-15,16-17,18-19"
)
FACTORS=${FACTORS:-"A D"}

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
echo "[plan] ${#specs[@]} jobs, ${#G[@]} GPUs, $PER samples each, 4 two-block sites"

i=0
while [ $i -lt ${#specs[@]} ]; do
  pids=()
  for g in "${G[@]}"; do
    [ $i -ge ${#specs[@]} ] && break
    IFS='|' read -r f sel k <<< "${specs[$i]}"
    i=$((i + 1))
    tag="f${f}_o${k}"
    if [ -f "$OUT/swap_${f}_${tag}.json" ]; then
      echo "[skip] $tag"
      continue
    fi
    echo "[run] $f  ${WIN[$f]}  $sel  gpu $g"
    CUDA_VISIBLE_DEVICES=$g nohup $PY -u swap_probe.py \
      --factor "$f" --only "$sel" --windows "${WIN[$f]}" \
      --frames 45 --steps 20 --no_ceiling \
      --out "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
    pids+=($!)
    sleep 4
  done
  for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" || echo "[fail] $p"; done
done
echo "[all done]"
ls "$OUT"/swap_*.json | wc -l
