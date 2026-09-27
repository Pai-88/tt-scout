#!/bin/bash
# Run OpenTTGames clips through the pipeline as their files land: table + truth from the markup, then a full detection
# pass (eval_openttgames.py, white ball, default Config). Usage:
#   nohup scripts/process_openttgames.sh test_1 test_4 game_1 ... > data/process.log 2>&1 &
# SKIP_FIT=1 reuses an existing data/<clip>_table.json + _truth.json instead of refitting.
# Waits up to 6 h per clip for data/<clip>.mp4 (no .part left) and data/<clip>_markup/. Logs per clip:
#   data/<clip>_openttgames.log, out/<clip>_eval.log; outputs in out/<clip>/.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
mkdir -p out
for n in "$@"; do
  waited=0
  until [ -s "data/$n.mp4" ] && [ ! -e "data/$n.mp4.part" ] && [ -d "data/${n}_markup" ]; do
    sleep 30; waited=$((waited + 30))
    if [ $waited -ge 21600 ]; then echo "$(date +%H:%M:%S) $n: TIMEOUT waiting for files"; continue 2; fi
  done
  if [ -n "$SKIP_FIT" ] && [ -s "data/${n}_table.json" ] && [ -s "data/${n}_truth.json" ]; then
    echo "$(date +%H:%M:%S) $n: files present, using the existing calibration (SKIP_FIT)"
  else
    echo "$(date +%H:%M:%S) $n: files present, fitting table + truth"
    if ! $PY openttgames.py "data/$n" > "data/${n}_openttgames.log" 2>&1; then
      echo "$(date +%H:%M:%S) $n: FAIL table/truth ($(tail -1 "data/${n}_openttgames.log" | cut -c1-160))"; continue
    fi
    echo "$(date +%H:%M:%S) $n: $(head -1 "data/${n}_openttgames.log" | cut -c1-160)"
  fi
  echo "$(date +%H:%M:%S) $n: detection pass started"
  if $PY eval_openttgames.py "data/$n" > "out/${n}_eval.log" 2>&1; then
    echo "$(date +%H:%M:%S) $n: DONE $(grep -E '^bounces|^net crossings|^rallies' "out/${n}_eval.log" | tr '\n' ' ' | cut -c1-300)"
  else
    echo "$(date +%H:%M:%S) $n: FAIL detection ($(tail -1 "out/${n}_eval.log" | cut -c1-160))"
  fi
done
echo "$(date +%H:%M:%S) ALL PROCESSED: $*"
