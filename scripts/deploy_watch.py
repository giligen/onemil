"""Deploy watch — Mon 7/20 + Tue 7/21 catalyst-veto/ignition-shadow ramp.

Owner is away; this is the ALWAYS-REPORTS mechanical layer (unlike
trader_watchdog which is silent-when-healthy). Four checkpoints via
crontab; each sends a Telegram status even when everything is green so
absence-of-message itself signals a broken watch layer.

Usage: deploy_watch.py --checkpoint {boot,open,midday,eod} [--no-telegram]
Playbook + green criteria: docs/deploy_watch_runbook_jul2026.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

GREEN, ORANGE, RED = 0, 1, 2
ICON = {GREEN: '🟢', ORANGE: '🟠', RED: '🔴'}


def sh(cmd: str) -> str:
    """Run a shell command, never raise."""
    try:
        return subprocess.run(['bash', '-c', cmd], capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f'__CMD_ERROR__ {e}'


def jgrep(pattern: str, since: str = '12:00') -> int:
    out = sh(f"journalctl -u onemil-trader --since '{since}' --no-pager "
             f"| grep -cE {pattern!r} || true")
    try:
        return int(out.splitlines()[-1])
    except (ValueError, IndexError):
        return -1


def jtail(pattern: str, n: int = 3, since: str = '12:00') -> str:
    out = sh(f"journalctl -u onemil-trader --since '{since}' --no-pager "
             f"| grep -E {pattern!r} | tail -{n}")
    return out[:500]


class Report:
    def __init__(self) -> None:
        self.status = GREEN
        self.lines: list[str] = []
        self.silent_ok = False   # storm checkpoint: healthy -> no send

    def add(self, level: int, text: str) -> None:
        self.status = max(self.status, level)
        self.lines.append(f"{ICON[level]} {text}")


def check_boot(r: Report) -> None:
    state = sh('systemctl is-active onemil-trader')
    if state != 'active':
        r.add(RED, f"service state='{state}' (expected active) — see "
                   f"runbook P1: fix or rollback BEFORE 13:15 UTC")
    else:
        r.add(GREEN, 'service active')
    tb = jgrep('Traceback')
    r.add(RED if tb > 0 else GREEN, f'tracebacks since 12:00: {tb}')
    gates = jtail('ORBEngine gates:', 1)
    if 'catalyst_veto=True' in gates:
        r.add(GREEN, 'gates line OK: veto armed')
    elif gates:
        r.add(RED, f'gates line anomalous: {gates[-200:]}')
    else:
        r.add(RED, 'gates line MISSING — engine did not init')
    if jgrep('IgnitionShadow ACTIVE') >= 1:
        r.add(GREEN, 'IgnitionShadow ACTIVE')
    else:
        r.add(ORANGE, 'IgnitionShadow init line missing (shadow off?)')


def check_open(r: Report) -> None:
    if sh('systemctl is-active onemil-trader') != 'active':
        r.add(RED, 'service NOT active during session')
    tb = jgrep('Traceback')
    r.add(RED if tb > 0 else GREEN, f'tracebacks: {tb}')
    n_news = jgrep('NEWS prefetch')
    r.add(GREEN if n_news >= 1 else ORANGE,
          f'news prefetch runs: {n_news}')
    n_scored = jgrep('ORB SCORED')
    r.add(GREEN if n_scored >= 1 else ORANGE,
          f'ORB SCORED lines: {n_scored}')
    n_veto = jgrep('CATALYST VETO')
    r.add(GREEN, f'catalyst vetoes: {n_veto}')
    if n_veto:
        r.lines.append(jtail('CATALYST VETO', 4))
    if n_scored > 0 and n_veto >= n_scored:
        r.add(ORANGE, 'veto hit EVERY scored candidate — spot-check '
                      'news snapshot (runbook P3, money-safe, no '
                      'mid-session action)')
    n_warm = jgrep('anchor pre-warm budget')
    if n_warm:
        r.add(ORANGE, f'anchor pre-warm budget warnings: {n_warm}')
    n_drop = jgrep('ignition-shadow: queue full')
    if n_drop:
        r.add(ORANGE, f'shadow queue-full drop warnings: {n_drop}')
    _shadow_journal(r, minimum_expected=0)


def check_midday(r: Report) -> None:
    if sh('systemctl is-active onemil-trader') != 'active':
        r.add(RED, 'service NOT active during session')
    tb = jgrep('Traceback')
    r.add(RED if tb > 0 else GREEN, f'tracebacks: {tb}')
    n_restart = jgrep('Started onemil-trader', since='12:20')
    if n_restart > 2:
        r.add(RED, f'{n_restart} service starts today — crash-loop '
                   f'(runbook P2)')
    _shadow_journal(r, minimum_expected=1)
    n_over = jgrep('CYCLE TIMING.*overrun|cycle overrun')
    if n_over > 20:
        r.add(ORANGE, f'scanner overruns high: {n_over}')
    n_wedge = jgrep('worker eval failed')
    if n_wedge:
        r.add(ORANGE, f'shadow worker eval failures: {n_wedge}')


def check_eod(r: Report) -> None:
    state = sh('systemctl is-active onemil-trader')
    r.add(GREEN if state != 'active' else ORANGE,
          f"service state='{state}' (inactive expected after 20:00)")
    tb = jgrep('Traceback')
    r.add(RED if tb > 0 else GREEN, f'tracebacks today: {tb}')
    day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    rep = sh(f"grep -c 'IGNITION SHADOW] {day}' "
             f"{ROOT}/logs/ignition_shadow_report.log || true")
    try:
        ran = int(rep.splitlines()[-1]) >= 1
    except (ValueError, IndexError):
        ran = False
    r.add(GREEN if ran else ORANGE,
          f"21:40 shadow report {'ran' if ran else 'DID NOT RUN'} "
          f"(runbook P5)")
    _shadow_journal(r, minimum_expected=0)
    n_veto = jgrep('CATALYST VETO')
    n_trades = jgrep('ORB.*fill|FILLED', since='13:30')
    r.add(GREEN, f'day counts: vetoes={n_veto} fill-log-lines={n_trades}')
    try:
        import sqlite3
        db = sqlite3.connect(f'file:{ROOT}/data/onemil.db?mode=ro',
                             uri=True)
        row = db.execute(
            'SELECT total_trades, gross_pnl FROM daily_trading_summary '
            'WHERE trade_date=?', (day,)).fetchone()
        if row:
            r.add(GREEN, f'daily summary: {row[0]} trades, '
                         f'gross P&L ${row[1]:+,.0f}')
        else:
            r.add(ORANGE, 'no daily_trading_summary row for today')
    except Exception as e:
        r.add(ORANGE, f'daily summary read failed: {e}')


def _shadow_journal(r: Report, minimum_expected: int) -> None:
    day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    p = ROOT / 'logs' / f'ignition_shadow_{day}.jsonl'
    if not p.exists():
        lvl = ORANGE if minimum_expected else GREEN
        r.add(lvl, 'shadow journal: no file yet')
        return
    verdicts: dict = {}
    try:
        for ln in p.read_text().splitlines():
            v = json.loads(ln).get('verdict', '?')
            verdicts[v] = verdicts.get(v, 0) + 1
    except Exception as e:
        r.add(ORANGE, f'shadow journal parse error: {e}')
        return
    r.add(GREEN, f'shadow journal: {sum(verdicts.values())} records '
                 f'{verdicts}')


def check_storm(r: Report) -> None:
    """Hourly error-storm backstop (silent when healthy, like
    trader_watchdog). Thresholds calibrated on Fri 7/17 baseline:
    0 ERROR, 0 Traceback, 2-11 WARNING per hour."""
    since = '65 minutes ago'
    tb = jgrep('Traceback', since=since)
    er = jgrep(r'\| ERROR', since=since)
    wa = jgrep(r'\| WARNING', since=since)
    state = sh('systemctl is-active onemil-trader')
    hh = datetime.now(timezone.utc).hour
    market = 12 <= hh < 20
    if market and state != 'active':
        r.add(RED, f"service state='{state}' during market hours")
    if tb >= 1:
        r.add(RED, f'{tb} traceback(s) in last 65min (baseline 0): '
                   + jtail('Traceback', 1, since=since))
    if er >= 5:
        r.add(RED, f'{er} ERROR lines in last 65min (baseline 0): '
                   + jtail(r'\| ERROR', 2, since=since))
    if wa >= 25:
        r.add(ORANGE, f'{wa} WARNING lines in last 65min '
                      f'(baseline max 11): '
                      + jtail(r'\| WARNING', 2, since=since))
    if not r.lines:
        r.add(GREEN, f'quiet: tb={tb} err={er} warn={wa} '
                     f"state={state}")
        r.silent_ok = True   # healthy -> no telegram


CHECKS = {'boot': check_boot, 'open': check_open,
          'midday': check_midday, 'eod': check_eod,
          'storm': check_storm}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, choices=CHECKS)
    ap.add_argument('--no-telegram', action='store_true')
    args = ap.parse_args()
    r = Report()
    try:
        CHECKS[args.checkpoint](r)
    except Exception as e:
        r.add(RED, f'WATCHDOG INTERNAL ERROR: {e}')
    day = datetime.now(timezone.utc).strftime('%a %m-%d %H:%M')
    msg = (f"{ICON[r.status]} <b>[DEPLOY WATCH {args.checkpoint}]</b> "
           f"{day} UTC\n" + '\n'.join(r.lines))
    print(msg)
    if not args.no_telegram and not r.silent_ok:
        try:
            import report_common as rc
            rc.send_telegram(msg)
        except Exception as e:
            print(f'telegram send failed: {e}')
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
