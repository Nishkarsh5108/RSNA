#!/usr/bin/env bash
# plan 14, the local GPU queue, chained so nothing waits on a person (one GPU job at a time on the 6 GB card):
#   1. wait for the R7 gold run (r7_gold_preds.py, log D:/rsna_work/r7_gold.log), then r7_gold_eval.py (14 B)
#   2. fidelity check of compact_infer_x.py on gold_58 (14 A): xpass_gold_check.py run + compare
#   3. S6 held-out teacher on the cache (14 C): oof_300 check (every Spearman >= 0.85, registered), then all studies
#   4. s6ft_v2 targets, training (2 epochs, alpha uniform, lr 2e-5 / 5e-5), gold_58 scoring, the plan-13 rule
# Logs: D:/rsna_work/s6ft_v2_chain.log (+ each step's own log).
# Usage: bash src/emergency/s6ft_v2_chain.sh      (run in the background)
set -u
cd /d/Coding/kaggle/RSNA
PY=/c/Users/nishk/anaconda3/envs/torch_gpu/python.exe
LOG=D:/rsna_work/s6ft_v2_chain.log
W=D:/rsna_work
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "waiting for the R7 gold run"
# the worker writes receipt.json last; the log's own exit line may carry a BOM (PowerShell append), so match it loosely
until [ -f output/emergency_research/task1/r7/receipt.json ] || grep -aqE 'exit -?[0-9]+' $W/r7_gold.log; do sleep 60; done
sleep 30
say "R7 run ended: $(tail -n 1 $W/r7_gold.log)"
$PY -I src/emergency/r7_gold_eval.py > $W/r7_gold_eval.log 2>&1
say "r7_gold_eval exit $?"

say "xpass fidelity run (R6 + in-pass s6fall on gold_58)"
$PY -I src/emergency/xpass_gold_check.py run > $W/xpass_run.log 2>&1
say "xpass run exit $?: $(grep -a X_DONE $W/xpass_run.log | tail -n 1)"
$PY -I src/emergency/xpass_gold_check.py compare > $W/xpass_compare.log 2>&1
say "xpass compare exit $?"
touch $W/XPASS_DONE.flag

say "S6 teacher: oof_300 check"
$PY -I src/emergency/s6_oof_cache.py --only oof300 > $W/s6_oof300.log 2>&1
ok=$($PY -c "import json;r=json.load(open('D:/Coding/kaggle/RSNA/output/emergency_research/track2/s6_oof_cache_check.json'));print(int(r['spearman_min']>=0.85))" 2>> "$LOG")
say "oof_300 check: min Spearman $(grep -o '"spearman_min": [0-9.]*' D:/Coding/kaggle/RSNA/output/emergency_research/track2/s6_oof_cache_check.json) -> go $ok"
if [ "$ok" != "1" ]; then say "STOP: teacher check failed (plan 14 C)"; exit 1; fi
$PY -I src/emergency/s6_oof_cache.py > $W/s6_oof_full.log 2>&1
say "S6 teacher full run exit $?: $(tail -n 1 $W/s6_oof_full.log)"
$PY -I src/emergency/s6ft_targets_v5.py > $W/s6ft_targets_v5.log 2>&1
say "targets v5 exit $?"
if [ ! -s data/labels_v4/training_labels_v5_clean.csv ]; then say "STOP: no v5 targets"; exit 1; fi

say "training s6ft_v2 (2 epochs, alpha uniform, lr 2e-5 / 5e-5)"
$PY -I src/emergency/s6ft_train.py --cache $W/s6ft_cache_all --targets data/labels_v4/training_labels_v5_clean.csv \
    --init $W/public_weights/rsna-knee-s6fall-weights/E_prior_s42_fall.pt --code $W/public_code/s6_src \
    --out $W/s6ft_v2_local --epochs 2 --lr 2e-5 --head-lr 5e-5 --alpha uniform --batch 1 --accum 2 --workers 4 \
    --mem-fraction 0.85 > $W/s6ft_v2_train.log 2>&1
if ! grep -q S6FT_DONE $W/s6ft_v2_train.log; then say "STOP: training did not finish (see s6ft_v2_train.log)"; exit 1; fi
say "trained: $(grep S6FT_DONE $W/s6ft_v2_train.log)"

say "scoring gold_58 with s6ft_v2"
S6FT_WEIGHTS=$W/s6ft_v2_local/s6ft_ema.pt $PY -I src/emergency/task1_compact_preds.py --leg s6ft --only-set gold_58 --workers 3 \
    --out-tag s6ft_v2 > $W/s6ft_v2_gold_preds.log 2>&1
$PY -I src/emergency/s6ft_gold_eval.py s6ft_v2 > $W/s6ft_v2_gold_eval.log 2>&1
say "gold eval exit $?: output/emergency_research/task1/s6ft_v2_gold_eval.json"
say "CHAIN_DONE"
