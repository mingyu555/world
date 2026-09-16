#!/usr/bin/env bash
# When does the text act? Ceiling injection confined to one third of the
# denoising trajectory at a time.
#
# Everything before this injected for the whole trajectory, so a window's measured
# effect is summed over all 20 steps and says nothing about timing. One smoke
# sample already shows the answer is not uniform: for appearance, injecting only
# during steps 0-6 reproduced 99.4% of the full-trajectory luminance change, and
# steps 14-19 gave 0.2%.
#
# s0-19 is kept as a control band -- it must reproduce the earlier full-trajectory
# ceiling exactly, which it did on the smoke sample (toward +0.0420, lum 0.0506,
# struct 0.6674 against the same three numbers from runs/swap45).
#
# Dynamics and geometry are expected to differ: AC3D found camera motion settled
# in the first 40% of the trajectory, and Characterizing Motion Encoding puts the
# motion/appearance boundary near t = 700-950 of 1000.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

OUT=${OUT:-./runs/stepband}
GPUS=${GPUS:-"0 1 2 5 7"}
BANDS=${BANDS:-"0-19,0-6,7-13,14-19"}
N=${N:-8}
PER=${PER:-2}
mkdir -p "$OUT"
read -ra G <<< "$GPUS"

specs=()
for f in A D; do
  mapfile -t S < <($PY -c "
import json; print('\n'.join(json.load(open('runs/swap45/top8.json'))['$f']))")
  for ((k = 0; k < N && k < ${#S[@]}; k += PER)); do
    sel=""
    for ((j = k; j < k + PER && j < ${#S[@]}; j++)); do sel+="${S[$j]},"; done
    specs+=("$f|${sel%,}|$k|inv")
  done
done
# geometry has no dataset inventory worth swapping -- it needs the hand-written
# future-topology pairs, so it runs in --pairs mode over the first N samples
for ((k = 0; k < N; k += PER)); do
  specs+=("G||$k|pairs")
done
echo "[plan] ${#specs[@]} jobs, ${#G[@]} GPUs, bands $BANDS"

i=0
while [ $i -lt ${#specs[@]} ]; do
  pids=()
  for g in "${G[@]}"; do
    [ $i -ge ${#specs[@]} ] && break
    IFS='|' read -r f sel k mode <<< "${specs[$i]}"
    i=$((i + 1))
    tag="sb${f}_o${k}"
    if [ -f "$OUT/swap_${f}_${tag}.json" ]; then
      echo "[skip] $tag"
      continue
    fi
    if [ "$mode" = "pairs" ]; then
      pick=(--pairs g_pairs.json --samples "$PER" --sample_offset "$k")
    else
      pick=(--only "$sel")
    fi
    echo "[run] $f  $mode  ${sel:-offset $k}  gpu $g"
    CUDA_VISIBLE_DEVICES=$g nohup $PY -u swap_probe.py \
      --factor "$f" "${pick[@]}" --windows "" --step_bands "$BANDS" \
      --frames 45 --steps 20 \
      --out "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
    pids+=($!)
    sleep 4
  done
  for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" || echo "[fail] $p"; done
done
echo "[all done]"
ls "$OUT"/swap_*.json | wc -l
