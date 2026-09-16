#!/usr/bin/env bash
# B-LoRA style scan: confine LoRA to one four-block window and see whether the
# factors separate on their own.
#
# B-LoRA (ECCV 2024) restricted LoRA to two of SDXL's eleven blocks and found that
# style and content landed in different blocks with no supervision telling them
# to. The analogue here: seven windows of four blocks over Cosmos' 28, and four
# captions per window.
#
# The four captions of one agd_dataset sample are a real-data minimal pair --
#
#   base  "The bus is approaching the intersection."
#   A     base + appearance phrase      "... on a cloudy day with overcast lighting."
#   G     base + geometry phrase        "... from the right lane, with two lanes ..."
#   D     base + dynamics phrase        "... drives forward and speeds up ..."
#
# -- so with one seed fixed across the four runs (LoRA init, sample order, sigma
# sequence and noise draws are all seeded before the adapter is built), the only
# thing that differs between them is the caption. dW_f - dW_base is therefore an
# exact paired difference attributable to the factor phrase, which matters because
# dW's raw direction is ~93% seed-determined (cos 0.074 across seeds vs 0.741
# across factors) and so unusable unpaired.
#
# attn1 is included deliberately: it takes no text, so any factor structure
# appearing there is an artifact, not a finding.
#
#   ./blora_scan.sh            # 7 windows x 4 captions on 4 GPUs, ~85 min
#   GPUS="0 3" ./blora_scan.sh # fewer GPUs, proportionally longer
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

OUT=${OUT:-./runs/blora}
GPUS=${GPUS:-"0 3 4 7"}
SEED=${SEED:-0}
STEPS=${STEPS:-252}
MODULES=${MODULES:-attn1,attn2,ff}
WINDOWS=${WINDOWS:-"0-3 4-7 8-11 12-15 16-19 20-23 24-27"}
FACTORS=${FACTORS:-"base A G D"}

mkdir -p "$OUT"
read -ra GPU_ARR <<< "$GPUS"

for w in $WINDOWS; do
  i=0
  pids=()
  for f in $FACTORS; do
    g=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}
    tag="w${w}_${f}_s${SEED}"
    if [ -f "$OUT/lora_${tag}.pt" ]; then
      echo "[skip] $tag"
      i=$((i + 1))
      continue
    fi
    echo "[run] window $w  factor $f  gpu $g  -> $tag"
    CUDA_VISIBLE_DEVICES=$g nohup $PY -u lora_train.py \
      --factor "$f" --seed "$SEED" --steps "$STEPS" \
      --layers "$w" --modules "$MODULES" \
      --out "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
    sleep 3
  done
  for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" || echo "[fail] pid $p"; done
  echo "[done] window $w"
done

echo "[all done]"
ls -la "$OUT"/lora_*.pt | wc -l
