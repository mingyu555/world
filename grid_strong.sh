#!/usr/bin/env bash
# The window grid, run only on the samples where the factor's effect reaches the
# pixels at all. `runs/screen/selected.json` lists them: the ceiling screen over
# 59 clean samples found a real dynamics effect in 10, an appearance effect in 3
# and a geometry effect in 1, so an unselected grid is mostly measuring samples
# where there is nothing to localise.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
OUT=${OUT:-./runs/grid}
GPUS=${GPUS:-"0 3 4 5 7"}
mkdir -p "$OUT"
read -ra G <<< "$GPUS"
mapfile -t S < <($PY -c "
import json; print('\n'.join(json.load(open('runs/screen/selected.json'))['strong_D']))")
echo "[samples] ${#S[@]}: ${S[*]}"
pids=()
for i in "${!G[@]}"; do
  sel=""
  for ((j=i; j<${#S[@]}; j+=${#G[@]})); do sel+="${S[$j]},"; done
  sel=${sel%,}
  [ -z "$sel" ] && continue
  tag="g$i"
  [ -f "$OUT/lpi_$tag.json" ] && { echo "[skip] $tag"; continue; }
  echo "[run] gpu ${G[$i]} <- $sel"
  CUDA_VISIBLE_DEVICES=${G[$i]} nohup $PY -u layer_prompt_image_probe.py \
    --only "$sel" --no_ceiling --skip_control --steps 20 \
    --out "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
  pids+=($!)
  sleep 5
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" || echo "[fail] $p"; done
echo "[all done]"
