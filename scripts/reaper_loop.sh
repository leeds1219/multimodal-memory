#!/usr/bin/env bash
# Every 10 min: kill Minecraft instances that no episode uses (older than 15 min).
while true; do
  date '+%m-%d %H:%M:%S'
  /opt/conda/envs/mcagent/bin/python "$(cd "$(dirname "$0")" && pwd)/reap_minecraft.py" --kill --min-age 15 | grep LEAKED || true
  sleep 600
done
