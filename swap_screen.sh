#!/usr/bin/env bash
# Swap-design ceiling screen at the model's native 45 frames.
#
# Two corrections over the first pass:
#
#   the caption pair is a length-matched phrase swap, not base vs base+phrase.
#   The dataset's factor captions are base + 5..15 extra words, and the addition
#   elaborates rather than contradicts -- the base never claimed a weather, so
#   "on a clear day" gives the model nothing to change. Dynamics was the only
#   factor whose phrase conflicts with the base, and it was the only factor with
#   a measurable effect.
#
#   45 frames, not 13. The pipeline hard-overwrites the first latent frame with
#   the conditioning image at every step, so appearance and geometry can only
#   diverge as frames accumulate. Measured on one sample, going 13 -> 45 frames
#   took the last frame's luminance change under a day->night swap from 0.0009
#   to 0.1392, and the structural change from 0.030 to 0.667. The 13-frame
#   setting was inherited from the training config and is what made appearance
#   and geometry look absent.
#
#   ./swap_screen.sh
#   N=30 GPUS="0 1 2" ./swap_screen.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

OUT=${OUT:-./runs/swap45}
GPUS=${GPUS:-"0 1 2 7"}
N=${N:-16}
PER=${PER:-4}
FACTORS=${FACTORS:-"A G D"}
mkdir -p "$OUT"
read -ra G <<< "$GPUS"

jobs_spec=()
for f in $FACTORS; do
  for ((k = 0; k < N; k += PER)); do
    jobs_spec+=("$f:$k")
  done
done
echo "[plan] ${#jobs_spec[@]} jobs of $PER samples, ${#G[@]} GPUs, 45 frames"

i=0
while [ $i -lt ${#jobs_spec[@]} ]; do
  pids=()
  for g in "${G[@]}"; do
    [ $i -ge ${#jobs_spec[@]} ] && break
    spec=${jobs_spec[$i]}
    f=${spec%%:*}
    k=${spec##*:}
    tag="${f}_o${k}"
    i=$((i + 1))
    if [ -f "$OUT/swap_${f}_${tag}.json" ]; then
      echo "[skip] $tag"
      continue
    fi
    echo "[run] factor $f samples $k..$((k + PER - 1)) gpu $g -> $tag"
    CUDA_VISIBLE_DEVICES=$g nohup $PY -u swap_probe.py \
      --factor "$f" --windows "" --frames 45 --steps 20 \
      --samples "$PER" --sample_offset "$k" \
      --out "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
    pids+=($!)
    sleep 4
  done
  for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" || echo "[fail] $p"; done
done
echo "[all done]"
ls "$OUT"/swap_*.json | wc -l
