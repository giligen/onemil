# OneMil systemd units

All units are checked into the repo so they track with code. Installation
is **manual** (requires `sudo`) — do it once per machine, then `git pull`
updates propagate via `systemctl daemon-reload`.

## Units

| File | Schedule | Purpose |
|------|----------|---------|
| `onemil-trader.service` | continuous (already installed) | Live paper-trading process (bull flag + MACD wave + ORB) |
| `onemil-orb-backtest.service` + `.timer` | Mon–Fri 16:30 ET | Post-close ORB backtest refresh: updates cache.db daily/intraday bars, regenerates features CSV, runs the static-lock pipeline |

## Install `onemil-orb-backtest.timer`

One-time setup (Mon–Fri 16:30 ET post-close refresh):

```bash
# Symlink the repo-tracked units into systemd's unit directory so
# `git pull` updates propagate by touching a single file.
sudo ln -sf /home/ec2-user/onemil/systemd/onemil-orb-backtest.service \
            /etc/systemd/system/onemil-orb-backtest.service
sudo ln -sf /home/ec2-user/onemil/systemd/onemil-orb-backtest.timer \
            /etc/systemd/system/onemil-orb-backtest.timer

sudo systemctl daemon-reload
sudo systemctl enable --now onemil-orb-backtest.timer

# Sanity
systemctl list-timers onemil-orb-backtest.timer
systemd-analyze calendar 'Mon..Fri *-*-* 16:30:00 America/New_York'
```

### Manual run (ad-hoc)

Sometimes you want to run the backtest by hand — e.g. a fresh refresh
immediately after a data issue:

```bash
sudo systemctl start onemil-orb-backtest.service
# Then tail the journal:
journalctl -u onemil-orb-backtest.service -f
```

### Mid-day BT (provisional overlay)

The timer runs post-close only. If you want to run BT during market hours
and see today's trades, skip the timer and run the CLI directly with the
`--include-today-provisional` flag:

```bash
python3 orb_backtest.py --end $(date +%F) --slice $(date +%F) \
    --include-today-provisional
```

This writes today's still-open bar to the `daily_bars_provisional`
sidecar (never to the main `daily_bars` cache that live reads). The
sidecar is cleared at the start of each run so stale mid-day values
can't accumulate. Live never reads this sidecar.

## Invariants this schedule relies on

The `onemil-orb-backtest.service` omits `--include-today-provisional`
intentionally. At 16:30 ET:

- The 16:15 ET guard inside `persistence.database.save_daily_bars`
  allows today's row through (post-close).
- `fill_daily_bars_for_dates` fetches today's now-final bar and
  `INSERT OR REPLACE`s it into `daily_bars`. Live on the next morning
  reads the correct final close.
- No provisional overlay is needed or used.

If the schedule ever drifts earlier than 16:15 ET, the guard will drop
today's row and the refresh is a no-op — safe but wasteful.

## Logs

- `journalctl -u onemil-orb-backtest.service` — full output of each run.
- `journalctl -u onemil-orb-backtest.timer` — scheduler events
  (fired/skipped/failed).
- Features CSV lands in `analysis_results/orb_features_*.csv` and the
  run prints the path.

## Uninstall

```bash
sudo systemctl disable --now onemil-orb-backtest.timer
sudo rm /etc/systemd/system/onemil-orb-backtest.{service,timer}
sudo systemctl daemon-reload
```
