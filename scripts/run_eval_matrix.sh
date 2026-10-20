#!/bin/bash
# Evaluate every trained DR-study policy in every evaluation world.
# Usage:  ~/agile/scripts/run_eval_matrix.sh            (all policies that exist)
#         ~/agile/scripts/run_eval_matrix.sh nominal a  (only these)
# Output: ~/agile/logs/dr_eval/<policy>/<world>/{metrics.json, trajectories/}

AGILE=~/agile
SCENARIO=$AGILE/agile/rl_env/tasks/dr_study/eval_squat_walk.yaml
WORLDS="NoTS Sat1p0 Sat1p5 Sat2p0 AGILE"
POLICIES=${@:-"nominal a b"}

for p in $POLICIES; do
  run_root=$AGILE/logs/rsl_rl/dr_study_g1_$p
  if [ ! -d "$run_root" ]; then echo "[skip] no runs for policy '$p'"; continue; fi
  run_dir=$(ls -d "$run_root"/*/ | sort | tail -1)
  ckpt=$(ls "$run_dir"model_*.pt | sort -V | tail -1)
  echo "=== policy: $p | checkpoint: $ckpt"
  for w in $WORLDS; do
    out=$AGILE/logs/dr_eval/$p/$w
    if [ -f "$out/metrics.json" ]; then echo "[done] $p / $w"; continue; fi
    mkdir -p "$out"
    echo "--- $p in world $w"
    $AGILE/run_agile.sh scripts/eval.py \
      --task DRStudy-G1-Eval-$w-v0 --num_envs 16 \
      --checkpoint "$ckpt" \
      --eval_config "$SCENARIO" \
      --run_evaluation --save_trajectories \
      --metrics_file "$out/metrics.json" > "$out/eval_log.txt" 2>&1
    if [ -f "$out/metrics.json" ]; then echo "    ok"; else echo "    FAILED (see $out/eval_log.txt)"; fi
  done
done
echo "Finished. Summarize with: ~/agile/run_agile.sh scripts/summarize_eval.py"
