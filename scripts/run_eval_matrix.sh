#!/bin/bash
# Evaluate every trained DR-study policy in every evaluation world x payload.
# Usage:  ~/agile/scripts/run_eval_matrix.sh              (policies: agile a b)
#         ~/agile/scripts/run_eval_matrix.sh agile a      (only these)
#         PAYLOADS="0 20" ~/agile/scripts/run_eval_matrix.sh
# Output: ~/agile/logs/dr_eval_v2/<policy>/<world>_P<kg>/{metrics.json, trajectories/, eval_log.txt}

AGILE=~/agile
SCENARIO=$AGILE/agile/rl_env/tasks/dr_study/eval_squat_walk.yaml
WORLDS="NoTS Sat2p0 Sat1p5 Sat1p0 AGILE"
PAYLOADS=${PAYLOADS:-"0 10 20"}
POLICIES=${@:-"agile a b"}
ROOT=$AGILE/logs/dr_eval_v2

for p in $POLICIES; do
  run_root=$AGILE/logs/rsl_rl/dr_study_g1_$p
  if [ ! -d "$run_root" ]; then echo "[skip] no runs for policy '$p'"; continue; fi
  run_dir=$(ls -d "$run_root"/*/ | sort | tail -1)
  ckpt=$(ls "$run_dir"model_*.pt | sort -V | tail -1)
  echo "=== policy: $p | checkpoint: $ckpt"
  for w in $WORLDS; do
    for kg in $PAYLOADS; do
      if [ "$kg" = "0" ]; then task=DRStudy-G1-Eval-$w-v0; else task=DRStudy-G1-Eval-$w-P$kg-v0; fi
      out=$ROOT/$p/${w}_P$kg
      if [ -f "$out/metrics.json" ]; then echo "[done] $p / ${w}_P$kg"; continue; fi
      rm -rf "$out"; mkdir -p "$out"
      echo "--- $p in $task"
      $AGILE/run_agile.sh scripts/eval.py \
        --task $task --num_envs 16 \
        --checkpoint "$ckpt" \
        --eval_config "$SCENARIO" \
        --run_evaluation --save_trajectories \
        --metrics_file "$out/metrics.json" > "$out/eval_log.txt" 2>&1
      if [ -f "$out/metrics.json" ]; then echo "    ok"; else echo "    FAILED (see $out/eval_log.txt)"; fi
    done
  done
done
echo "Finished. Summarize with: ~/agile/run_agile.sh scripts/summarize_eval.py"
