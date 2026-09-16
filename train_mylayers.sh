#!/usr/bin/env bash
# Train A/D experts on the layers this repo's pixel-level analysis picked, as a
# crossover against each other's site.
#
# The claim under test: appearance acts at L4-5 and dynamics at L12-13 (B-LoRA
# style caption-swap injection, 45-frame generations, q = 0.055 / 0.000). If that
# is real, each factor should train better on its own site than on the other's --
# a self-contained test that does not depend on agreeing with any earlier
# analysis.
#
#   mg_A       A @ L4,5     own site
#   mg_A_ctrl  A @ L12,13   the other factor's site
#   mg_D       D @ L12,13   own site
#   mg_D_ctrl  D @ L4,5     the other factor's site
#
# Budget is matched to youngjae's runs_wm/*_analysis (5 layers x rank 8 = 1.39 M):
# 2 layers x rank 20 is the same 1.39 M, so only placement differs. alpha/r is
# kept at 4, as in those runs.
#
# Everything is read out of youngjae's tree (trainer, cache, weights) and written
# into this one.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

TRAINER="$YJ/cosmos_exp/train_lora.py"
CACHE="$YJ/cosmos_exp/cache_wm"
OUT=${OUT:-./runs_train}
STEPS=${STEPS:-800}
GPUS=${GPUS:-"4 5 6 7"}
mkdir -p "$OUT"
read -ra G <<< "$GPUS"

specs=(
  "mg_A|A|4,5"
  "mg_A_ctrl|A|12,13"
  "mg_D|D|12,13"
  "mg_D_ctrl|D|4,5"
)

pids=()
i=0
for spec in "${specs[@]}"; do
  IFS='|' read -r tag fac lay <<< "$spec"
  g=${G[$((i % ${#G[@]}))]}
  i=$((i + 1))
  if [ -f "$OUT/$tag/final/adapter_model.safetensors" ] || [ -d "$OUT/$tag/final" ]; then
    echo "[skip] $tag"
    continue
  fi
  echo "[run] $tag  factor $fac  layers $lay  gpu $g"
  CUDA_VISIBLE_DEVICES=$g nohup $PY "$TRAINER" \
    --factor "$fac" --layers "$lay" --rank 20 --alpha 80 \
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
