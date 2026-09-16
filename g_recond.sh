#!/usr/bin/env bash
# Geometry, re-conditioned two ways at once.
#
# Why the dataset's geometry phrases failed: they describe the road as it is right
# now ("right lane of a two-lane road"), and the pipeline overwrites the first
# latent frame with the conditioning image at every denoising step, so that fact
# is already on screen. A caption claiming a different lane count contradicts the
# image rather than adding anything. Appearance escaped this because day -> night
# can be realised as drift over frames (measured: frame 0 changes by exactly 0,
# the last frame by 0.139); dynamics escaped it because one frame cannot say
# whether the car will accelerate.
#
# Arm 1 (cond)     hand-written pairs describing *future* road topology -- curves,
#                  intersections, merges, tunnels that are out of frame at t=0 and
#                  only arrive as the clip plays. Same escape dynamics had.
# Arm 2 (nocond)   the same pairs with the image conditioning removed. If geometry
#                  works here but not in arm 1, the image is the limiter and the
#                  conclusion is about the conditioning, not the model's text path.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

GPUS=${GPUS:-"0 1 5 7"}
N_COND=${N_COND:-18}
N_NOCOND=${N_NOCOND:-12}
PER=${PER:-3}
mkdir -p runs/grecond runs/grecond_nocond
read -ra G <<< "$GPUS"

specs=()
for ((k = 0; k < N_COND; k += PER)); do specs+=("cond|$k"); done
for ((k = 0; k < N_NOCOND; k += PER)); do specs+=("nocond|$k"); done
echo "[plan] ${#specs[@]} jobs of $PER samples, ${#G[@]} GPUs"

i=0
while [ $i -lt ${#specs[@]} ]; do
  pids=()
  for g in "${G[@]}"; do
    [ $i -ge ${#specs[@]} ] && break
    IFS='|' read -r arm k <<< "${specs[$i]}"
    i=$((i + 1))
    if [ "$arm" = "cond" ]; then
      out=runs/grecond
      extra=""
    else
      out=runs/grecond_nocond
      extra="--no_cond"
    fi
    tag="${arm}_o${k}"
    if [ -f "$out/swap_G_${tag}.json" ]; then
      echo "[skip] $tag"
      continue
    fi
    echo "[run] $arm samples $k..$((k + PER - 1)) gpu $g"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$g nohup $PY -u swap_probe.py \
      --factor G --pairs g_pairs.json --windows "" \
      --frames 45 --steps 20 --samples "$PER" --sample_offset "$k" $extra \
      --out "$out" --tag "$tag" > "$out/$tag.log" 2>&1 &
    pids+=($!)
    sleep 4
  done
  for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" || echo "[fail] $p"; done
done
echo "[all done]"
ls runs/grecond/swap_*.json runs/grecond_nocond/swap_*.json 2>/dev/null | wc -l
