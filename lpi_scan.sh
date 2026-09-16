#!/usr/bin/env bash
# One sample per GPU: B-LoRA's block analysis over Cosmos' seven windows.
# Each job is 1 + 1 + 8*3 = 26 generations (reference, control, 7 windows +
# ceiling x 3 factors); at ~75 s a generation that is ~35 min per sample.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
OUT=${OUT:-./runs/lpi}
GPUS=${GPUS:-"0 3 4 7"}
STEPS=${STEPS:-20}
mkdir -p "$OUT"
read -ra G <<< "$GPUS"
pids=()
for i in "${!G[@]}"; do
  tag="s$i"
  [ -f "$OUT/lpi_$tag.json" ] && { echo "[skip] $tag"; continue; }
  echo "[run] sample offset $i on gpu ${G[$i]} -> $tag"
  CUDA_VISIBLE_DEVICES=${G[$i]} nohup $PY -u layer_prompt_image_probe.py \
    --samples 1 --sample_offset "$i" --steps "$STEPS" \
    --out "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
  pids+=($!)
  sleep 5
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" || echo "[fail] $p"; done
echo "[all done]"; ls "$OUT"/lpi_*.json
