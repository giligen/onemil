#!/usr/bin/env python3
"""
Index ORB walker — implements research/index_orb/PREREG.md verbatim (frozen
spec, owner 2026-09-21). Cells 1,329-1,346.

Pipeline: fetch Alpaca 1-min SIP bars for QQQ/SPY (2016-01-01..2026-05-31,
raw, regular session only) -> parquet cache (resumable) -> walk the frozen
opening-range-breakout rule across 18 cells (2 symbols x 3 windows x
3 side-configs) -> research/index_orb/REPORT.md with per-cell TRAIN/VAL
tables and the pre-committed pass bar.

Usage:
    nohup python3 -u research/index_orb/walker.py \
        > research/index_orb/walker.log 2>&1 &

Resumable: the fetch stage keeps a per-symbol manifest of completed
(year, month) pairs and only fetches what's missing, appending to the
parquet cache after every month so a killed run loses no completed work.
"""
import json
import logging
import subprocess
import sys
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config import get_config  # noqa: E402
from data_sources.alpaca_client import AlpacaClient, AlpacaAPIError  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
REPORT_PATH = HERE / "REPORT.md"

NY = pytz.timezone("America/New_York")
UTC = pytz.utc

SYMBOLS = ["QQQ", "SPY"]
WINDOWS = [5, 15, 30]
SIDE_CONFIGS = ["long", "short", "both"]

DATA_START = date(2016, 1, 1)
DATA_END = date(2026, 5, 31)

TRAIN_START, TRAIN_END = date(2016, 1, 1), date(2024, 12, 31)
TRAIN_H1_END = date(2019, 12, 31)
TRAIN_H2_START = date(2020, 1, 1)
TRAIN_YEARS = list(range(2016, 2025))  # 9 years
VAL_START, VAL_END = date(2025, 1, 1), date(2026, 5, 31)

MIN_BARS_PER_DAY = 380
SIGNAL_CUTOFF = dtime(15, 0)
CLOSE_EXIT_TIME = dtime(15, 55)

MAX_RETRIES = 5
RETRY_BACKOFF_S = 5.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("walker")


# ==========================================================================
# Fetch / cache
# ==========================================================================

def month_bounds_utc(year, month):
    """UTC (start, end) datetimes spanning a full America/New_York calendar month."""
    start_ny = NY.localize(datetime(year, month, 1, 0, 0, 0))
    if month == 12:
        end_ny = NY.localize(datetime(year + 1, 1, 1, 0, 0, 0))
    else:
        end_ny = NY.localize(datetime(year, month + 1, 1, 0, 0, 0))
    return start_ny.astimezone(UTC), (end_ny - timedelta(seconds=1)).astimezone(UTC)


def month_list(start_d, end_d):
    """List of (year, month) tuples spanning start_d..end_d inclusive."""
    months = []
    y, m = start_d.year, start_d.month
    while (y, m) <= (end_d.year, end_d.month):
        months.append((y, m))
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return months


def manifest_path(symbol):
    return CACHE_DIR / f"{symbol}_1min.manifest.json"


def cache_path(symbol):
    return CACHE_DIR / f"{symbol}_1min.parquet"


def load_manifest(symbol):
    p = manifest_path(symbol)
    if p.exists():
        return set(tuple(x) for x in json.loads(p.read_text()))
    return set()


def save_manifest(symbol, done_months):
    manifest_path(symbol).write_text(json.dumps(sorted(list(m) for m in done_months)))


def fetch_symbol(client, symbol):
    """Fetch/resume monthly 1-min bars for `symbol` into the parquet cache.

    Filters each month's response to regular-session weekday bars
    (09:30-15:59 ET). Retries transient AlpacaAPIError with backoff.
    Cache + manifest are rewritten after every month, so an interrupted
    run resumes from the last completed month.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    done = load_manifest(symbol)
    combined = pd.read_parquet(cache_path(symbol)) if cache_path(symbol).exists() else pd.DataFrame()
    months = month_list(DATA_START, DATA_END)
    todo = [m for m in months if m not in done]
    log.info(f"{symbol}: {len(done)}/{len(months)} months already cached, {len(todo)} to fetch")

    for (y, m) in todo:
        start_utc, end_utc = month_bounds_utc(y, m)
        attempt = 0
        df = None
        while True:
            attempt += 1
            try:
                df = client.get_historical_1min_bars(symbol, start_utc, end_utc)
                break
            except AlpacaAPIError as e:
                if attempt >= MAX_RETRIES:
                    log.error(f"{symbol} {y}-{m:02d}: giving up after {attempt} attempts: {e}")
                    raise
                wait = RETRY_BACKOFF_S * attempt
                log.warning(f"{symbol} {y}-{m:02d}: attempt {attempt} failed ({e}), retry in {wait:.0f}s")
                import time as _time
                _time.sleep(wait)

        if df is None or df.empty:
            log.warning(f"{symbol} {y}-{m:02d}: 0 bars returned")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
            df = df[(df["timestamp"].dt.time >= dtime(9, 30)) & (df["timestamp"].dt.time <= dtime(15, 59))]
            df = df[df["timestamp"].dt.weekday < 5]
            combined = pd.concat([combined, df], ignore_index=True) if not combined.empty else df
            log.info(f"{symbol} {y}-{m:02d}: +{len(df)} regular-session bars")
        done.add((y, m))
        if not combined.empty:
            combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
            combined.to_parquet(cache_path(symbol), index=False)
        save_manifest(symbol, done)

    log.info(f"{symbol}: fetch complete, cache has {len(combined)} rows")
    return combined


# ==========================================================================
# Day-quality check
# ==========================================================================

def thin_days(df):
    """Session days with fewer than MIN_BARS_PER_DAY regular-session bars."""
    counts = df.groupby(df["timestamp"].dt.date).size()
    return counts[counts < MIN_BARS_PER_DAY]


# ==========================================================================
# Walk: one signal/entry/stop/exit trade per (symbol, day, window)
# ==========================================================================

def walk_symbol_window(df, symbol, W, excluded_days):
    """Walk the frozen ORB rule for one symbol and one range window W.

    Returns a list of trade dicts (both long and short trades; side config
    filtering happens downstream). One trade per instrument per day.
    """
    trades = []
    window_end = (datetime.combine(date(2000, 1, 1), dtime(9, 30)) + timedelta(minutes=W)).time()

    for day, day_df in df.groupby(df["timestamp"].dt.date):
        if day in excluded_days:
            continue
        day_df = day_df.sort_values("timestamp").reset_index(drop=True)
        win_bars = day_df[day_df["timestamp"].dt.time < window_end]
        if win_bars.empty:
            log.warning(f"{symbol} W{W} {day}: no opening-window bars, skipped")
            continue
        range_high = win_bars["high"].max()
        range_low = win_bars["low"].min()
        if not (range_high > range_low):
            log.warning(f"{symbol} W{W} {day}: degenerate range, skipped")
            continue

        post = day_df[day_df["timestamp"].dt.time >= window_end]
        signal_idx = None
        side = None
        for idx, bar in post.iterrows():
            if bar["timestamp"].time() > SIGNAL_CUTOFF:
                break
            if bar["close"] > range_high:
                side, signal_idx = "long", idx
                break
            if bar["close"] < range_low:
                side, signal_idx = "short", idx
                break
        if signal_idx is None:
            continue  # no signal today for this window

        pos = day_df.index.get_loc(signal_idx)
        if pos + 1 >= len(day_df):
            log.warning(f"{symbol} W{W} {day}: signal on last bar of day, no entry bar, skipped")
            continue
        entry_bar = day_df.iloc[pos + 1]
        entry_price = float(entry_bar["open"])
        stop_price = range_low if side == "long" else range_high
        R = abs(entry_price - stop_price)
        if R <= 0:
            log.warning(f"{symbol} W{W} {day}: zero R, skipped")
            continue

        after_entry = day_df.iloc[pos + 2:]
        stopped, gap_through = False, False
        exit_price, exit_bar_time = None, None
        for _, bar in after_entry.iterrows():
            if side == "long" and bar["low"] <= stop_price:
                exit_price = bar["open"] if bar["open"] <= stop_price else stop_price
                gap_through = bar["open"] <= stop_price
                stopped = True
                exit_bar_time = bar["timestamp"]
                break
            if side == "short" and bar["high"] >= stop_price:
                exit_price = bar["open"] if bar["open"] >= stop_price else stop_price
                gap_through = bar["open"] >= stop_price
                stopped = True
                exit_bar_time = bar["timestamp"]
                break

        held_to_close = not stopped
        if held_to_close:
            close_bars = day_df[day_df["timestamp"].dt.time >= CLOSE_EXIT_TIME]
            if close_bars.empty:
                # data problem: no 15:55+ bar this day; fall back to the last
                # available bar of the day and log it.
                log.warning(f"{symbol} W{W} {day}: no 15:55 bar, using last bar of day as close exit")
                exit_bar = day_df.iloc[-1]
            else:
                exit_bar = close_bars.iloc[0]
            exit_price = float(exit_bar["open"])
            exit_bar_time = exit_bar["timestamp"]

        gross_R = (exit_price - entry_price) / R if side == "long" else (entry_price - exit_price) / R
        cost_1bp_R = (entry_price * 0.0001 + exit_price * 0.0001) / R
        cost_3bp_R = (entry_price * 0.0003 + exit_price * 0.0003) / R
        net_1bp = gross_R - cost_1bp_R
        net_3bp = gross_R - cost_3bp_R

        trades.append({
            "date": str(day), "symbol": symbol, "W": W, "side": side,
            "entry": entry_price, "stop": stop_price, "exit": exit_price,
            "R_dollars": R, "gross_R": gross_R, "net_1bp_R": net_1bp, "net_3bp_R": net_3bp,
            "stopped": stopped, "held_to_close": held_to_close, "gap_through": gap_through,
        })

    return trades


# ==========================================================================
# Stats
# ==========================================================================

def t_stat(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 or np.std(x, ddof=1) == 0:
        return float("nan")
    return np.mean(x) / (np.std(x, ddof=1) / np.sqrt(n))


def weeks_span(start_d, end_d):
    return max((end_d - start_d).days / 7.0, 1e-9)


def max_drawdown_R(pnl_seq):
    """Max peak-to-trough drawdown of the cumulative R curve, in R."""
    if not pnl_seq:
        return 0.0
    cum = np.cumsum(pnl_seq)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max())


def longest_losing_streak_weeks(trades_df):
    """Longest run of consecutive ISO weeks with negative total net_1bp_R."""
    if trades_df.empty:
        return 0
    d = trades_df.copy()
    d["dt"] = pd.to_datetime(d["date"])
    d["iso_week"] = d["dt"].dt.strftime("%G-W%V")
    weekly = d.groupby("iso_week")["net_1bp_R"].sum().sort_index()
    streak = best = 0
    for v in weekly:
        if v < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def cell_trades(all_trades, symbol, W, side_config):
    d = all_trades[(all_trades["symbol"] == symbol) & (all_trades["W"] == W)]
    if side_config != "both":
        d = d[d["side"] == side_config]
    return d.sort_values("date")


def split_trades(d, start_d, end_d):
    return d[(d["date"] >= str(start_d)) & (d["date"] <= str(end_d))]


def ex_top_pct_mean(vals, pct=0.05):
    vals = sorted(vals)
    if not vals:
        return float("nan")
    k = int(np.ceil(len(vals) * pct))
    kept = vals[: len(vals) - k] if k > 0 else vals
    return float(np.mean(kept)) if kept else float("nan")


def capped_mean(vals, cap=5.0):
    if not vals:
        return float("nan")
    return float(np.mean([min(v, cap) for v in vals]))


def split_summary(d, start_d, end_d, label):
    s = split_trades(d, start_d, end_d)
    n = len(s)
    out = {"label": label, "n": n}
    if n == 0:
        return out
    out["trades_per_week"] = n / weeks_span(start_d, end_d)
    out["gross_mean"] = s["gross_R"].mean()
    out["net_1bp_mean"] = s["net_1bp_R"].mean()
    out["net_3bp_mean"] = s["net_3bp_R"].mean()
    out["sd_net_1bp"] = s["net_1bp_R"].std(ddof=1) if n > 1 else float("nan")
    out["iid_t"] = t_stat(s["net_1bp_R"].values)
    out["day_clustered_t"] = out["iid_t"]  # one trade/instrument/day -> coincide
    out["win_rate"] = (s["net_1bp_R"] > 0).mean()
    out["ex_top5_net_1bp"] = ex_top_pct_mean(list(s["net_1bp_R"]))
    out["capped5R_net_1bp"] = capped_mean(list(s["net_1bp_R"]))
    out["stopped_frac"] = s["stopped"].mean()
    out["held_to_close_frac"] = s["held_to_close"].mean()
    out["gap_through_frac"] = s["gap_through"].mean()
    out["mdd_R"] = max_drawdown_R(list(s["net_1bp_R"]))
    out["losing_streak_wk"] = longest_losing_streak_weeks(s)
    out["mde"] = (out["sd_net_1bp"] / np.sqrt(n) * 2) if n > 1 else float("nan")
    return out


def run_cadence_bar(trades_csv, split):
    """Invoke scripts/cadence_bar.py on a (date, pnl_R) trades csv. Returns (stdout, pass_flags)."""
    try:
        proc = subprocess.run(
            ["python3", str(REPO_ROOT / "scripts" / "cadence_bar.py"),
             "--trades", str(trades_csv), "--split", split],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
        )
        text = proc.stdout.strip() or proc.stderr.strip()
    except Exception as e:
        text = f"cadence_bar.py invocation failed: {e}"
    flags = {}
    for crit in ["C3", "C4", "C5"]:
        for line in text.splitlines():
            if line.strip().startswith(crit):
                flags[crit] = "[pass]" in line
                break
    return text, flags


# ==========================================================================
# Report
# ==========================================================================

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "N/A"
    return f"{x:.{nd}f}"


def build_report(all_trades, thin_day_report):
    lines = []
    lines.append("# Index ORB walk — REPORT (research/index_orb/PREREG.md)\n")
    lines.append(f"Generated {datetime.now(UTC).isoformat()}\n")
    lines.append("## Data quality\n")
    lines.append(thin_day_report + "\n")

    n_pass = 0
    cell_num = 1329
    for symbol in SYMBOLS:
        for W in WINDOWS:
            d_sw = cell_trades(all_trades, symbol, W, "both")
            csv_path = HERE / f"trades_{symbol}_W{W}.csv"
            d_sw.to_csv(csv_path, index=False)
            for side in SIDE_CONFIGS:
                d = cell_trades(all_trades, symbol, W, side)
                lines.append(f"\n## Cell {cell_num}: {symbol} W={W} {side}\n")
                cell_num += 1

                train = split_summary(d, TRAIN_START, TRAIN_END, "TRAIN 2016-2024")
                h1 = split_summary(d, TRAIN_START, TRAIN_H1_END, "TRAIN half 2016-2019")
                h2 = split_summary(d, TRAIN_H2_START, TRAIN_END, "TRAIN half 2020-2024")
                val = split_summary(d, VAL_START, VAL_END, "VAL 2025-2026/05")

                header = ("| split | n | tr/wk | gross | net1bp | net3bp | sd | t(iid=clust) | win% | "
                          "ex-top5% | cap5R | stop% | close% | gap% | MDD(R) | losestreak(wk) |")
                sep = "|---" * 15 + "|"
                lines.append(header)
                lines.append(sep)
                for s in (train, h1, h2, val):
                    if s["n"] == 0:
                        lines.append(f"| {s['label']} | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - |")
                        continue
                    lines.append(
                        f"| {s['label']} | {s['n']} | {fmt(s['trades_per_week'],2)} | "
                        f"{fmt(s['gross_mean'])} | {fmt(s['net_1bp_mean'])} | {fmt(s['net_3bp_mean'])} | "
                        f"{fmt(s['sd_net_1bp'])} | {fmt(s['iid_t'],2)} | {fmt(s['win_rate']*100,1)} | "
                        f"{fmt(s['ex_top5_net_1bp'])} | {fmt(s['capped5R_net_1bp'])} | "
                        f"{fmt(s['stopped_frac']*100,1)} | {fmt(s['held_to_close_frac']*100,1)} | "
                        f"{fmt(s['gap_through_frac']*100,1)} | {fmt(s['mdd_R'],2)} | {s['losing_streak_wk']} |"
                    )

                lines.append("\nTRAIN year-by-year (net 1bp mean R, total R):")
                year_pos_count = 0
                for yr in TRAIN_YEARS:
                    yd = split_trades(d, date(yr, 1, 1), date(yr, 12, 31))
                    if len(yd) == 0:
                        lines.append(f"- {yr}: n=0")
                        continue
                    ymean = yd["net_1bp_R"].mean()
                    ytot = yd["net_1bp_R"].sum()
                    if ytot > 0:
                        year_pos_count += 1
                    lines.append(f"- {yr}: n={len(yd)} mean={fmt(ymean)} total={fmt(ytot,2)}")

                # cadence bar on VAL
                cadence_text, cadence_flags = ("N/A (n=0)", {})
                if val["n"] > 0:
                    val_csv = HERE / f"_cadence_val_cell{cell_num-1}.csv"
                    vd = split_trades(d, VAL_START, VAL_END)[["date"]].copy()
                    vd["pnl_R"] = split_trades(d, VAL_START, VAL_END)["net_1bp_R"].values
                    vd.to_csv(val_csv, index=False)
                    cadence_text, cadence_flags = run_cadence_bar(val_csv, "VAL")
                lines.append("\n```\n" + cadence_text + "\n```")

                # pass bar
                p1 = val["n"] > 0 and val["net_1bp_mean"] >= 0.08 and val["day_clustered_t"] >= 2.0
                p2 = train["n"] > 0 and train["net_1bp_mean"] >= 0.08 and h1.get("net_1bp_mean", -1) > 0 and h2.get("net_1bp_mean", -1) > 0
                p3 = year_pos_count >= 6
                p4 = (train["n"] > 0 and val["n"] > 0 and
                      train["ex_top5_net_1bp"] > 0 and train["capped5R_net_1bp"] > 0 and
                      val["ex_top5_net_1bp"] > 0 and val["capped5R_net_1bp"] > 0)
                p5 = all(cadence_flags.get(c, False) for c in ["C3", "C4", "C5"]) if cadence_flags else False
                p6 = val["n"] > 0 and val["net_3bp_mean"] > 0
                overall = p1 and p2 and p3 and p4 and p5 and p6
                if overall:
                    n_pass += 1
                lines.append(
                    f"\n**Pass bar**: 1(VAL net1bp+t)={p1} 2(TRAIN+halves)={p2} "
                    f"3(>=6/9 yrs)={p3} 4(ex-top5/cap5R>0 both)={p4} "
                    f"5(cadence C3/C4/C5 VAL)={p5} 6(3bp VAL>0)={p6} -> **{'PASS' if overall else 'NO PASS'}**"
                )
                if not overall:
                    lines.append(f"MDE (VAL, sd/sqrt(n)*2) = {fmt(val.get('mde'))} R")

    lines.append(f"\n## Summary\n{n_pass} of 18 cells pass all 6 criteria.\n")
    REPORT_PATH.write_text("\n".join(lines))
    log.info(f"REPORT written: {REPORT_PATH} ({n_pass}/18 cells pass)")
    return n_pass


# ==========================================================================
# Main
# ==========================================================================

def main():
    log.info("Index ORB walker starting")
    cfg = get_config()
    client = AlpacaClient(api_key=cfg.alpaca_api_key, api_secret=cfg.alpaca_api_secret)

    all_bars = {}
    thin_lines = []
    for symbol in SYMBOLS:
        df = fetch_symbol(client, symbol)
        all_bars[symbol] = df
        td = thin_days(df)
        thin_lines.append(f"{symbol}: {len(td)} thin session days (<{MIN_BARS_PER_DAY} bars), excluded from walk")
        if len(td) > 0:
            sample = ", ".join(str(x) for x in list(td.index)[:30])
            thin_lines.append(f"  sample: {sample}{' ...' if len(td) > 30 else ''}")

    log.info("Fetch stage complete for all symbols, starting walk")

    all_trades = []
    for symbol in SYMBOLS:
        df = all_bars[symbol]
        excluded = set(thin_days(df).index)
        for W in WINDOWS:
            trades = walk_symbol_window(df, symbol, W, excluded)
            log.info(f"{symbol} W={W}: {len(trades)} trades")
            all_trades.extend(trades)

    all_trades_df = pd.DataFrame(all_trades)
    all_trades_df.to_csv(HERE / "trades_all.csv", index=False)
    log.info(f"Walk complete: {len(all_trades_df)} total trades across all symbol/window combos")

    n_pass = build_report(all_trades_df, "\n".join(thin_lines))
    log.info(f"DONE. {n_pass}/18 cells pass the pre-committed pass bar. See {REPORT_PATH}")


if __name__ == "__main__":
    main()
