#!/usr/bin/env python3
"""One-shot OS-cron backstop for the 2026-08-17 ORB B+ GO session.

The Claude session that performs the GO flip is session-bound and died twice
on 8/16 (node freezes). This runs from the OS crontab at 11:35 UTC Monday —
15 minutes after the GO session should have flipped orb.yaml — and alarms
over Telegram if the flip never happened, so the owner knows to restart the
Claude session instead of silently missing the launch. Safe to delete after
8/17 (it fires only on Aug 17).
"""
import sys

sys.path.insert(0, '/home/ec2-user/onemil')

import yaml  # noqa: E402
from scripts.report_common import send_telegram  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load(open('/home/ec2-user/onemil/orb.yaml'))
    enabled = bool(cfg['strategy']['enabled'])
    if enabled:
        send_telegram('[BACKSTOP] GO flip verified: orb.yaml strategy.enabled=true — Claude GO session did its job.')
    else:
        send_telegram('[BACKSTOP] ⚠️ ORB GO session appears DEAD: it is 11:35 UTC Monday and orb.yaml strategy.enabled is still FALSE. '
                      'Restart the Claude session and tell it: "run the Monday GO session".')


if __name__ == '__main__':
    main()
