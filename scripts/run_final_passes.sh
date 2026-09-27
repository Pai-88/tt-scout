#!/bin/bash
# Launch the final v0 passes on all 12 OpenTTGames videos as 6 detached workers.
# RUN THIS FROM A TERMINAL, not from a background shell: macOS schedules background-QoS processes on the efficiency
# cores only, which starved six passes to ~10 % of a core each on 2026-09-17.
cd "$(dirname "$0")/.." || exit 1
export SKIP_FIT=1 OPENCV_FOR_THREADS_NUM=2
echo "$(date +%H:%M:%S) === FINAL PASSES relaunched from Terminal at normal priority (SKIP_FIT=1), 6 workers ===" >> data/process.log
nohup scripts/process_openttgames.sh game_2 >> data/process.log 2>&1 &
nohup scripts/process_openttgames.sh game_1 test_6 >> data/process.log 2>&1 &
nohup scripts/process_openttgames.sh game_3 test_5 test_2 >> data/process.log 2>&1 &
nohup scripts/process_openttgames.sh game_5 test_3 >> data/process.log 2>&1 &
nohup scripts/process_openttgames.sh game_4 test_7 >> data/process.log 2>&1 &
nohup scripts/process_openttgames.sh test_4 test_1 >> data/process.log 2>&1 &
disown -a 2>/dev/null
sleep 2
echo "launched $(pgrep -f process_openttgames.sh | wc -l | tr -d ' ') workers at $(date +%H:%M:%S); watch with: tail -f data/process.log"
