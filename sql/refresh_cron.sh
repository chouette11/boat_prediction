#!/bin/bash
# 毎朝 3:00 に cron などから呼ぶ想定
cd /opt/boatrace
source /path/to/venv/bin/activate
python etl.py >> logs/etl_$(date +%F).log 2>&1
