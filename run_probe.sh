#!/bin/bash
# 코드는 이 폴더, 가중치/데이터는 /mnt/ssd4/youngjae/cvpr2027_yj 에서 읽기만 한다.
set -e
cd "$(dirname "$0")"
source ./env.sh
SAMPLE=${SAMPLE:-$AGD/scene-0014_f16}
OUT=${OUT:-./runs/$(basename "$SAMPLE")}
CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -u factor_layer_probe.py \
    --sample "$SAMPLE" --out "$OUT" \
    --frames ${FRAMES:-29} --group ${GROUP:-4} --step_every ${STEP_EVERY:-7} \
    --tag ${TAG:-probe} "$@"
$PY analyze_factor_layer.py "$OUT" --tag ${TAG:-probe}
