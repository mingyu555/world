#!/usr/bin/env bash
# Seven-window grid on wm_dataset, on the ten clips per factor whose ceiling
# effect is largest. Same protocol as the agd_dataset grid, but in the regime the
# experts are trained and judged in (29 frames, fps 12), so the layers it names
# can be compared with youngjae's write-share placement on equal footing.
#
# The ceiling screen over 121 clips found appearance moving toward the swapped
# caption in 106 of 121 (p < 1e-4), geometry in 86 of 118 (p < 1e-4) and dynamics
# carrying a median flow change of 0.72 -- all three are measurable here, unlike
# on agd_dataset where only dynamics was.
#
# The ceiling is regenerated inside this run rather than reused, so every window
# is normalised against a ceiling produced by the same code path and seed.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

OUT=${OUT:-./runs/wm_grid}
GPUS=${GPUS:-"0 1 2 4 5 7"}
PER=${PER:-2}
FACTORS=${FACTORS:-"D A G"}
mkdir -p "$OUT"
read -ra G <<< "$GPUS"

specs=()
for f in $FACTORS; do
  mapfile -t S < <($PY -c "
import json; print('\n'.join(json.load(open('runs/wm_screen/top10.json'))['$f']))")
  for ((k = 0; k < ${#S[@]}; k += PER)); do
    sel=""
    for ((j = k; j < k + PER && j < ${#S[@]}; j++)); do sel+="${S[$j]},"; done
    specs+=("$f|${sel%,}|$k")
  done
done
echo "[plan] ${#specs[@]} jobs x $PER clips x (1 ref + 7 windows + ceiling)"

i=0
while [ $i -lt ${#specs[@]} ]; do
  pids=()
  for g in "${G[@]}"; do
    [ $i -ge ${#specs[@]} ] && break
    IFS='|' read -r f sel k <<< "${specs[$i]}"
    i=$((i + 1))
    tag="${f}_o${k}"
    [ -f "$OUT/swap_${f}_${tag}.json" ] && { echo "[skip] $tag"; continue; }
    echo "[run] $f  $sel  gpu $g"
    CUDA_VISIBLE_DEVICES=$g nohup $PY -u swap_probe.py \
      --factor "$f" --agd "$YJ/wm_dataset" --only "$sel" \
      --frames 29 --fps 12 --steps 20 \
      --out "$OUT" --tag "$tag" > "$OUT/$tag.log" 2>&1 &
    pids+=($!)
    sleep 4
  done
  for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" || echo "[fail] $p"; done
done
echo "[all done]"
