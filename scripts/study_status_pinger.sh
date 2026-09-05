#!/bin/bash
# Hourly Telegram status for an unattended study (owner 9/5: "send a telegram
# update with status every hour"). Reads the LAST line starting with
# "STATUS:" from the study's STATUS.md and posts it; stops when a line
# starting with "DONE" appears. Usage: study_status_pinger.sh STATUS_MD LABEL
set -u
cd /home/ec2-user/onemil || exit 1
STATUS_MD=${1:-research/orb_signal_study/STATUS.md}
LABEL=${2:-ORB SIGNAL STUDY}
while true; do
  line=$(grep -E '^STATUS:' "$STATUS_MD" 2>/dev/null | tail -1 | cut -c1-900)
  /usr/bin/python3 - "$LABEL" "$line" <<'EOF'
import os, sys
from dotenv import load_dotenv; load_dotenv('/home/ec2-user/onemil/.env')
sys.path.insert(0, '/home/ec2-user/onemil')
from notifications.telegram_notifier import TelegramNotifier
from datetime import datetime, timezone
label, line = sys.argv[1], sys.argv[2]
msg = f"[{label} — hourly {datetime.now(timezone.utc).strftime('%H:%M')} UTC]\n{line or 'no status line yet'}"
n = TelegramNotifier(os.getenv('TELEGRAM_BOT_TOKEN'), os.getenv('TELEGRAM_CHAT_ID'), enabled=True)
n.send_message_sync(msg, parse_mode=None)
EOF
  grep -qE '^DONE' "$STATUS_MD" 2>/dev/null && exit 0
  sleep 3600
done
