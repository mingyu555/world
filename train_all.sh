#!/bin/bash
cd "$(dirname "$0")"; source ./env.sh
G=${GPU:-1}
for SEED in 0 1; do
  for F in A G D base; do
    echo "=== factor=$F seed=$SEED  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$G $PY -u lora_train.py --factor $F --seed $SEED \
        --out /mnt/ssd1/mingyu_cvpr2027/lora 2>&1 | grep --line-buffered -vE "^Loading|^Fetching|it/s\]|deprecat|warnings.warn|FutureWarning|UserWarning|config attributes|Couldn't connect|local cache"
    echo "=== done $F s$SEED exit=$? $(date +%H:%M:%S)"
  done
done
echo "=== all training finished $(date +%H:%M:%S)"
