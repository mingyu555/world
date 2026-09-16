#!/usr/bin/env bash
# Ceiling screen: for every clean sample, how much does each factor caption
# actually change the video when injected into all 28 blocks?
#
# The window grid was diluted because only 2 of 7 samples had a real dynamics
# effect at the ceiling -- for the rest the D caption describes the same motion
# as the base, so there is nothing for a window to localise. Screening first
# costs 4 generations a sample instead of 26.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
OUT=${OUT:-./runs/screen}
GPUS=${GPUS:-"0 3 4 5 7"}
PER=${PER:-13}
mkdir -p "$OUT"
read -ra G <<< "$GPUS"
pids=()
for i in "${!G[@]}"; do
  off=$((i * PER))
  tag="sc$i"
  [ -f "$OUT/lpi_$tag.json" ] && { echo "[skip] $tag"; continue; }
  echo "[run] samples $off..$((off+PER-1)) on gpu ${G[$i]}"
  CUDA_VISIBLE_DEVICES=${G[$i]} nohup $PY -u layer_prompt_image_probe.py \
    --samples "$PER" --sample_offset "$off" --sample_stride 1 \
    --windows "" --skip_control --steps 20 \
    --out "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
  pids+=($!)
  sleep 5
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" || echo "[fail] $p"; done
echo "[all done]"
