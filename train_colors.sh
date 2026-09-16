#!/bin/bash
cd "$(dirname "$0")"; source ./env.sh
for C in red green blue white nocolor; do
  if [ "$C" = "nocolor" ]; then T="The car drives along the road."; else T="The $C car drives along the road."; fi
  echo "=== $C : \"$T\"  $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -u lora_train.py --factor $C --template "$T" \
    --seed 0 --out /mnt/ssd1/mingyu_cvpr2027/lora_color \
    2>&1 | grep --line-buffered -E "^\[caption|^\[loss\]|^\[save\]|step +(0|251)/|Error|Traceback"
  echo "=== done $C $(date +%H:%M:%S)"
done
echo "=== colors finished $(date +%H:%M:%S)"
