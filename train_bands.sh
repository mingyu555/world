#!/bin/bash
cd "$(dirname "$0")"; source ./env.sh
for F in A D base; do
  for BAND in "lo 0 0.3" "hi 0.3 100"; do
    set -- $BAND
    echo "=== $F band=$1 [$2,$3]  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=${GPU:-3} $PY -u lora_train.py --factor $F --seed 0 \
      --sigma_min $2 --sigma_max $3 --out /mnt/ssd1/mingyu_cvpr2027/lora_$1 \
      2>&1 | grep --line-buffered -E "^\[loss\]|^\[save\]|step +(0|250|251)/|Error|Traceback"
    echo "=== done $F/$1 $(date +%H:%M:%S)"
  done
done
echo "=== bands finished $(date +%H:%M:%S)"
