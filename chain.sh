#!/bin/bash
# 축 2->3->4->5 를 A축이 끝난 뒤 순서대로 실행한다.
cd "$(dirname "$0")"
source ./env.sh
YJC=/mnt/ssd4/youngjae/cvpr2027_yj/cosmos_exp
OUT=${OUT:-./runs/scene-0626_f24}
SAMPLE=${SAMPLE:-$AGD/scene-0626_f24}
G=${GPU:-1}
F=${FRAMES:-45}

echo "### wait for axis A"
while pgrep -f "factor_layer_probe.py" >/dev/null; do sleep 20; done
echo "### A done at $(date +%H:%M:%S)"

echo "### 2/5  write_f  (cross-attn span decomposition, youngjae's script)"
CUDA_VISIBLE_DEVICES=$G $PY -u $YJC/factor_attention.py \
    --sample "$SAMPLE" --out "$OUT" --cond AGD --frames $F --no_video --tag AGD \
  && $PY -u $YJC/analyze_factor_attention.py "$OUT" --tag AGD
echo "### 2 exit=$?  $(date +%H:%M:%S)"

echo "### 3/5  local_div  (branch-resolved layer-local divergence)"
CUDA_VISIBLE_DEVICES=$G $PY -u local_div_probe.py \
    --sample "$SAMPLE" --out "$OUT" --frames $F
echo "### 3 exit=$?  $(date +%H:%M:%S)"

echo "### 4/5  grad  (factor leverage on weights)"
CUDA_VISIBLE_DEVICES=$G $PY -u grad_probe.py \
    --sample "$SAMPLE" --out "$OUT" --frames $F
echo "### 4 exit=$?  $(date +%H:%M:%S)"

echo "### 5/5  compare axes"
$PY -u compare_axes.py "$OUT"
echo "### chain finished $(date +%H:%M:%S)"
