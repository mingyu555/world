#!/usr/bin/env bash
# The caption-swap placement analysis, redone in the regime the experts are
# actually trained and judged in.
#
# The first pass ran on agd_dataset: 2 fps nuScenes keyframes, generated at
# 45 frames / fps 16. Training and evaluation use wm_dataset: native ~12 Hz,
# 29 frames, fps 12. That gap is the most likely reason the placement it found
# for appearance (L4-5) lost to youngjae's (L3,9,11,12,17) when both were
# trained -- youngjae's write-share analysis was run on wm_dataset at 29 frames.
#
# Ceiling screen first (the swapped caption in all 28 blocks): a window grid is
# only meaningful on clips where swapping the phrase changes the video at all.
# On agd_dataset that was 10 of 59 clips for dynamics and 3 of 59 for appearance.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

OUT=${OUT:-./runs/wm_screen}
GPUS=${GPUS:-"0 1 2 4 5 7"}
N=${N:-122}
PER=${PER:-11}
FACTORS=${FACTORS:-"D A G"}
mkdir -p "$OUT"
read -ra G <<< "$GPUS"

specs=()
for f in $FACTORS; do
  for ((k = 0; k < N; k += PER)); do specs+=("$f|$k"); done
done
echo "[plan] ${#specs[@]} jobs x $PER clips, ${#G[@]} GPUs, 29 frames @ fps 12"

i=0
while [ $i -lt ${#specs[@]} ]; do
  pids=()
  for g in "${G[@]}"; do
    [ $i -ge ${#specs[@]} ] && break
    IFS='|' read -r f k <<< "${specs[$i]}"
    i=$((i + 1))
    tag="${f}_o${k}"
    [ -f "$OUT/swap_${f}_${tag}.json" ] && { echo "[skip] $tag"; continue; }
    echo "[run] $f clips $k..$((k + PER - 1)) gpu $g"
    CUDA_VISIBLE_DEVICES=$g nohup $PY -u swap_probe.py \
      --factor "$f" --agd "$YJ/wm_dataset" --windows "" \
      --frames 29 --fps 12 --steps 20 \
      --samples "$PER" --sample_offset "$k" \
      --out "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
    pids+=($!)
    sleep 4
  done
  for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" || echo "[fail] $p"; done
done
echo "[all done]"
