#!/bin/bash
cd "$(dirname "$0")"; source ./env.sh
run(){ # $1=tag $2=template
  echo "=== $1 : \"$2\"  $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u lora_train.py --factor "$1" --template "$2" \
    --seed 0 --out /mnt/ssd1/mingyu_cvpr2027/lora_slot \
    2>&1 | grep --line-buffered -E "^\[loss\]|^\[save\]|Error|Traceback"
}
if [ "$SET" = "AD" ]; then
  for W in rainy foggy snowy wet; do run "A_$W" "The car drives along the $W road."; done
  for W in accelerates brakes idles cruises; do run "D_$W" "The car $W along the road."; done
else
  for W in left right center middle; do run "G_$W" "The car drives in the $W lane."; done
fi
echo "=== $SET finished $(date +%H:%M:%S)"
