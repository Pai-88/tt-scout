#!/bin/bash
# Keep at most N detection passes running at once (SIGSTOP/SIGCONT), because on a memory-starved machine more
# concurrent passes lower the aggregate frame rate. Usage: nohup scripts/pace_passes.sh 3 >> data/pace.log 2>&1 &
# Exits when no worker (scripts/process_openttgames.sh) is left.
N=${1:-3}
while true; do
  pgrep -f "process_openttgames.sh" > /dev/null || { echo "$(date +%H:%M:%S) no workers left, pacer exits"; exit 0; }
  running=(); stopped=()
  # only real passes: a python interpreter whose first argument is the pass script (a shell whose command line merely
  # mentions the script name was once paused by this loop)
  for p in $(ps -eo pid=,command= | awk '$2 ~ /python[0-9.]*$/ && $3 == "eval_openttgames.py" {print $1}'); do
    st=$(ps -o stat= -p "$p" 2>/dev/null | cut -c1)
    [ -z "$st" ] && continue
    if [ "$st" = "T" ]; then stopped+=("$p"); else running+=("$p"); fi
  done
  # too many running (a worker started its next clip): pause the youngest
  while [ ${#running[@]} -gt "$N" ]; do
    young=$(printf '%s\n' "${running[@]}" | sort -n | tail -1)          # highest pid = started last (macOS ps has no etimes)
    [ -z "$young" ] && break
    kill -STOP "$young" && echo "$(date +%H:%M:%S) paused $(ps -o command= -p "$young" | sed 's#.*data/##') ($young)"
    running=($(printf '%s\n' "${running[@]}" | grep -v "^$young$"))
  done
  # free slots: resume the oldest paused
  while [ ${#running[@]} -lt "$N" ] && [ ${#stopped[@]} -gt 0 ]; do
    old=$(printf '%s\n' "${stopped[@]}" | sort -n | head -1)            # lowest pid = started first
    [ -z "$old" ] && break
    kill -CONT "$old" && echo "$(date +%H:%M:%S) resumed $(ps -o command= -p "$old" | sed 's#.*data/##') ($old)"
    stopped=($(printf '%s\n' "${stopped[@]}" | grep -v "^$old$")); running+=("$old")
  done
  sleep 30
done
