#!/usr/bin/env bash
# Train the three experts on the layers the wm_dataset analysis picked.
#
# The first round used placements measured on agd_dataset (2 fps keyframes,
# generated at 45 frames / fps 16). Redoing the same caption-swap analysis in the
# regime the experts are actually trained and judged in (wm_dataset, 29 frames,
# fps 12) moved two of the three:
#
#   A  L4-5   -> L8-11   (d_lum argmax there in 5 of 8 clips, q = 0.014)
#   G  L12-13 -> L4-7    (d_hist and d_flowdir 5 of 8, q = 0.007)
#   D  L12-13 -> L12-15  (d_struct/d_flow/d_flowdir 6 of 8, q = 0.000) -- unchanged
#
# and that explains the first round's outcome: appearance on L4-5 lost to
# youngjae's L3,9,11,12,17, whose L9/L11/L12 sit inside the corrected band, while
# dynamics on L12-13 won. The three corrected bands are mutually disjoint, so the
# experts can be merged by a plain union of state dicts.
#
# Budget is held at youngjae's runs_wm/*_analysis level (5 layers x rank 8 =
# 1.39 M): 4 layers x rank 10 is the same 1.39 M, alpha/r kept at 4.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

TRAINER="$YJ/cosmos_exp/train_lora.py"
CACHE="$YJ/cosmos_exp/cache_wm"
OUT=${OUT:-./runs_train}
STEPS=${STEPS:-800}
GPUS=${GPUS:-"0 1 2"}
mkdir -p "$OUT"
read -ra G <<< "$GPUS"

specs=("wm_A|A|8-11" "wm_G|G|4-7" "wm_D|D|12-15")

pids=()
i=0
for spec in "${specs[@]}"; do
  IFS='|' read -r tag fac lay <<< "$spec"
  g=${G[$((i % ${#G[@]}))]}
  i=$((i + 1))
  [ -d "$OUT/$tag/final" ] && { echo "[skip] $tag"; continue; }
  echo "[run] $tag  factor $fac  layers $lay  gpu $g"
  CUDA_VISIBLE_DEVICES=$g nohup $PY "$TRAINER" \
    --factor "$fac" --layers "$lay" --rank 10 --alpha 40 \
    --steps "$STEPS" --save_every 200 --log_every 50 \
    --cache "$CACHE" --out "$OUT/$tag" > "$OUT/$tag.log" 2>&1 &
  pids+=($!)
  sleep 5
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" || echo "[fail] $p"; done
echo "[all done]"
for spec in "${specs[@]}"; do
  IFS='|' read -r tag _ _ <<< "$spec"
  [ -f "$OUT/$tag/history.json" ] && $PY -c "
import json;h=json.load(open('$OUT/$tag/history.json'))
v=[x for x in h if 'val_loss' in x]
print('$tag', ' '.join(f\"{x['step']}:{x['val_loss']:.4f}\" for x in v))"
done
