#!/usr/bin/env bash
# plan 13 Tracks B-D, local fallback, chained so nothing waits on a person: when the cache build ends, retry any study it
# skipped, train s6ft for the fixed 3 epochs on the RTX 4050, score gold_58 and apply the registered adoption rule.
# Every step is fixed in plan 13 / 13b. Logs: D:/rsna_work/s6ft_chain.log (+ each step's own log).
# Usage: bash src/emergency/s6ft_chain.sh <cache build log> <cache dir> <k>      (run in the background; k as the build)
set -u
cd /d/Coding/kaggle/RSNA
PY=/c/Users/nishk/anaconda3/envs/torch_gpu/python.exe
LOG=D:/rsna_work/s6ft_chain.log
CACHE=$2
K=$3
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "waiting for the cache build ($1 -> $CACHE, k $K)"
until grep -q '^exit ' "$1"; do sleep 60; done
say "cache build ended: $(tail -n 1 "$1")"
$PY -I src/emergency/s6ft_cache.py --gpu-workers --workers 5 --k "$K" --quality 90 --timeout 600 --out "$CACHE" \
    > D:/rsna_work/s6ft_cache_retry.log 2>&1
n=$(find "$CACHE/studies" -name '*.npz' | wc -l)       # no glob: 4,349 names overflow a Windows command line
say "retry pass done: $n cached studies"
if [ "$n" -lt 4300 ]; then say "STOP: fewer than 4,300 cached studies"; exit 1; fi

say "training s6ft (3 epochs)"
$PY -I src/emergency/s6ft_train.py --cache "$CACHE" --targets data/labels_v4/training_labels_v4_superior.csv \
    --init D:/rsna_work/public_weights/rsna-knee-s6fall-weights/E_prior_s42_fall.pt --code D:/rsna_work/public_code/s6_src \
    --out D:/rsna_work/s6ft_local --epochs 3 --batch 1 --accum 2 --workers 4 --mem-fraction 0.85 > D:/rsna_work/s6ft_local_train.log 2>&1
if ! grep -q S6FT_DONE D:/rsna_work/s6ft_local_train.log; then say "STOP: training did not finish (see s6ft_local_train.log)"; exit 1; fi
say "trained: $(grep S6FT_DONE D:/rsna_work/s6ft_local_train.log)"

say "scoring gold_58 with s6ft"
$PY -I src/emergency/task1_compact_preds.py --leg s6ft --only-set gold_58 --workers 3 > D:/rsna_work/s6ft_gold_preds.log 2>&1
$PY -I src/emergency/s6ft_gold_eval.py > D:/rsna_work/s6ft_gold_eval.log 2>&1
say "gold eval exit $?: output/emergency_research/task1/s6ft_gold_eval.json"
say "CHAIN_DONE"
