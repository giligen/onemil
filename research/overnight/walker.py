"""
research/overnight/walker.py — PREREG cells 1,347-1,350: the overnight effect on
index ETFs (close-to-open), per research/overnight/PREREG.md.

Frozen rule: buy at the official close (MOC), sell at the next session's official
open (MOO). r_on = open_t / close_{t-1} - 1 (the overnight leg being tested).
r_id = close_t / open_t - 1 is the complement (reported beside it; the mechanism's
signature is r_on > 0 and r_id <= 0).

Pipeline: fetch Alpaca daily bars (feed=sip, raw/unadjusted) for SPY, QQQ, TQQQ,
UPRO, 2010-03-01..2024-12-31 -> cache to parquet -> compute per-cell, per-split
stats -> shell out to scripts/cadence_bar.py on a weekly pnl_R series -> write
research/overnight/REPORT.md and research/overnight/nights_<SYM>.csv.

TEST (>= 2025-01-01) is NOT fetched, per PREREG. Do not add symbols or splits
without re-running the PREREG process (multiplicity is pre-committed at 4 cells).

Usage:
    python research/overnight/walker.py
"""

import logging
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import Config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("overnight_walker")

SYMBOLS = ["SPY", "QQQ", "TQQQ", "UPRO"]
START = date(2010, 3, 1)
END = date(2024, 12, 31)
TRAIN_START, TRAIN_END = "2010-03-01", "2019-12-31"
TRAIN_H1_START, TRAIN_H1_END = "2010-03-01", "2014-12-31"
TRAIN_H2_START, TRAIN_H2_END = "2015-01-01", "2019-12-31"
VAL_START, VAL_END = "2020-01-01", "2024-12-31"

NOTIONAL = 66000.0
R_DOLLARS = 0.005 * NOTIONAL  # $330, cadence-bar risk unit (0.5% of notional/night)
COST_BP_SENSITIVITY = 1.0  # bp per side, sensitivity line

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
REPORT_PATH = BASE_DIR / "REPORT.md"
CADENCE_SCRIPT = BASE_DIR.parents[1] / "scripts" / "cadence_bar.py"


def fetch_daily_bars(symbol: str) -> pd.DataFrame:
    """Fetch (or load cached) raw daily bars for `symbol` from Alpaca SIP feed.

    Caches to CACHE_DIR/<symbol>_daily.parquet. Returns a DataFrame indexed by
    date with columns open, high, low, close, volume. Adjustment is RAW
    (unadjusted) per PREREG data spec.
    """
    cache_path = CACHE_DIR / f"{symbol}_daily.parquet"
    if cache_path.exists():
        logger.info(f"{symbol}: loading cached daily bars from {cache_path}")
        df = pd.read_parquet(cache_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    logger.info(f"{symbol}: fetching daily bars {START}..{END} from Alpaca (feed=sip, raw)")
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    cfg = Config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    start_dt = datetime(START.year, START.month, START.day, tzinfo=timezone.utc)
    end_dt = datetime(END.year, END.month, END.day, 23, 59, 59, tzinfo=timezone.utc)
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt,
        feed=DataFeed.SIP,
        adjustment=Adjustment.RAW,
    )
    bars_raw = client.get_stock_bars(request)
    bars_df = bars_raw.df
    if isinstance(bars_df.index, pd.MultiIndex):
        bars_df = bars_df.loc[symbol]
    bars_df = bars_df.reset_index()
    bars_df["date"] = pd.to_datetime(bars_df["timestamp"]).dt.date
    out = bars_df[["date", "open", "high", "low", "close", "volume"]].copy()
    out = out.sort_values("date").reset_index(drop=True)
    logger.info(f"{symbol}: fetched {len(out)} daily bars, {out['date'].min()}..{out['date'].max()}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    logger.info(f"{symbol}: cached to {cache_path}")
    return out


def report_missing_days(symbol: str, df: pd.DataFrame) -> int:
    """Count expected NYSE session gaps vs bars actually returned (crude check:
    business-day count over the span minus rows). Logged, not scored."""
    bdays = pd.bdate_range(START, END)
    missing = len(bdays) - len(df)
    logger.info(f"{symbol}: {len(df)} bars vs {len(bdays)} business days in span "
                f"({missing} 'missing' incl. legit market holidays)")
    if len(df) and df["date"].min() > START:
        logger.warning(f"{symbol}: data starts {df['date'].min()}, NOT the requested {START} "
                        f"(Alpaca account history limit) — TRAIN pre-{df['date'].min()} is EMPTY")
    return missing


def compute_nights(df: pd.DataFrame) -> pd.DataFrame:
    """Compute r_on (overnight, open_t/close_{t-1}-1) and r_id (intraday,
    close_t/open_t-1) per session. First row has no prior close -> dropped."""
    df = df.sort_values("date").reset_index(drop=True)
    df["prev_close"] = df["close"].shift(1)
    df["r_on"] = df["open"] / df["prev_close"] - 1.0
    df["r_id"] = df["close"] / df["open"] - 1.0
    df["r_full"] = df["close"] / df["prev_close"] - 1.0  # buy-and-hold daily return
    nights = df.dropna(subset=["prev_close"]).reset_index(drop=True)
    return nights[["date", "open", "close", "prev_close", "r_on", "r_id", "r_full"]]


def month_clustered_t(x: pd.Series, dates: pd.Series) -> float:
    """t-stat on monthly means (cluster by year-month) rather than raw obs."""
    tmp = pd.DataFrame({"x": x.values, "ym": pd.to_datetime(dates).dt.to_period("M")})
    monthly = tmp.groupby("ym")["x"].mean()
    n = len(monthly)
    if n < 2 or monthly.std(ddof=1) == 0:
        return float("nan")
    return float(monthly.mean() / (monthly.std(ddof=1) / np.sqrt(n)))


def iid_t(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(n)))


def max_drawdown(returns: pd.Series) -> float:
    """Max drawdown (fraction, negative) of the cumulative-product equity curve."""
    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def slice_split(nights: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    d = pd.to_datetime(nights["date"])
    mask = (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))
    return nights[mask].reset_index(drop=True)


def cell_stats(nights: pd.DataFrame, cost_bp_per_side: float = 0.0) -> dict:
    """Full stat block for one (symbol, split) cell. cost_bp_per_side > 0 applies
    the sensitivity line (round-trip cost subtracted from r_on each night)."""
    if len(nights) == 0:
        nan = float("nan")
        return {
            "n": 0, "mean_r_on_bp": nan, "mean_r_id_bp": nan, "sd_r_on_bp": nan,
            "iid_t": nan, "month_t": nan, "hit_rate": nan, "annualized_r_on": nan,
            "overnight_total_return": nan, "bh_total_return": nan,
            "overnight_mdd": nan, "bh_mdd": nan, "ex_top5_mean_bp": nan,
            "worst_night": {"date": "N/A", "r_on_bp": nan},
            "yearly_mean_bp": {}, "dow_mean_bp": {}, "cagr_annualized_pct": nan,
        }
    d = pd.to_datetime(nights["date"])
    r_on = nights["r_on"] - 2 * (cost_bp_per_side / 10000.0)  # buy+sell = 2 sides
    r_id = nights["r_id"]
    r_full = nights["r_full"]

    ex_top5_thresh = r_on.quantile(0.95)
    ex_top5_mean = r_on[r_on <= ex_top5_thresh].mean()

    overnight_total = float((1.0 + r_on).prod() - 1.0)
    bh_total = float((1.0 + r_full).prod() - 1.0)
    on_mdd = max_drawdown(r_on)
    bh_mdd = max_drawdown(r_full)

    years = d.dt.year
    yearly_mean = (r_on.groupby(years).mean() * 10000.0).round(2).to_dict()

    dow = d.dt.day_name()
    dow_mean = (r_on.groupby(dow).mean() * 10000.0).round(2).to_dict()

    worst_idx = r_on.idxmin()
    worst_night = {"date": str(nights.loc[worst_idx, "date"]), "r_on_bp": round(float(r_on.loc[worst_idx]) * 10000, 2)}

    n_years = max((pd.Timestamp(d.max()) - pd.Timestamp(d.min())).days / 365.25, 1e-9)

    return {
        "n": int(len(nights)),
        "mean_r_on_bp": float(r_on.mean() * 10000),
        "mean_r_id_bp": float(r_id.mean() * 10000),
        "sd_r_on_bp": float(r_on.std(ddof=1) * 10000),
        "iid_t": iid_t(r_on),
        "month_t": month_clustered_t(r_on, d),
        "hit_rate": float((r_on > 0).mean()),
        "annualized_r_on": float(r_on.mean() * 252),
        "overnight_total_return": overnight_total,
        "bh_total_return": bh_total,
        "overnight_mdd": on_mdd,
        "bh_mdd": bh_mdd,
        "ex_top5_mean_bp": float(ex_top5_mean * 10000),
        "worst_night": worst_night,
        "yearly_mean_bp": yearly_mean,
        "dow_mean_bp": dow_mean,
        "cagr_annualized_pct": float(((1 + overnight_total) ** (1 / n_years) - 1) * 100),
    }


def build_weekly_pnl_r(nights: pd.DataFrame) -> pd.DataFrame:
    """Weekly pnl_R series for scripts/cadence_bar.py.
    pnl_R per night = r_on * NOTIONAL / R_DOLLARS = r_on / 0.005 (base cost = 0).
    Grouped by ISO week, date = week's last trading day."""
    d = pd.to_datetime(nights["date"])
    pnl_r_night = nights["r_on"] * NOTIONAL / R_DOLLARS
    tmp = pd.DataFrame({"date": d, "pnl_R": pnl_r_night})
    tmp["iso_week"] = tmp["date"].dt.to_period("W-FRI")
    weekly = tmp.groupby("iso_week").agg(date=("date", "max"), pnl_R=("pnl_R", "sum")).reset_index(drop=True)
    weekly["date"] = weekly["date"].dt.strftime("%Y-%m-%d")
    return weekly[["date", "pnl_R"]]


def run_cadence_bar(weekly_csv: Path, split: str) -> str:
    """Shell out to scripts/cadence_bar.py --trades weekly_csv --split split.
    Returns captured stdout (or an error note)."""
    cmd = [sys.executable, str(CADENCE_SCRIPT), "--trades", str(weekly_csv), "--split", split]
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        out = proc.stdout.strip()
        if proc.returncode != 0:
            out += f"\n[cadence_bar exited {proc.returncode}]\n{proc.stderr.strip()}"
        return out
    except Exception as e:
        logger.error(f"cadence_bar.py failed for {weekly_csv} {split}: {e}")
        return f"[cadence_bar error: {e}]"


def verdict_for_cell(train_h1: dict, train_h2: dict, val: dict, val_1bp: dict,
                      cadence_c3: bool, cadence_c4: bool) -> str:
    """Apply PREREG pass bar 1-6 and return PASS/FAIL with reasons."""
    reasons = []
    ok = True

    c1 = val["mean_r_on_bp"] > 0 and (not np.isnan(val["month_t"])) and val["month_t"] >= 2
    if not c1:
        ok = False
        reasons.append(f"(1) VAL mean r_on={val['mean_r_on_bp']:.2f}bp month_t={val['month_t']:.2f} "
                        f"(need >0 and t>=2)")

    c2 = train_h1["mean_r_on_bp"] > 0 and train_h2["mean_r_on_bp"] > 0
    if not c2:
        ok = False
        reasons.append(f"(2) TRAIN halves mean r_on: H1={train_h1['mean_r_on_bp']:.2f}bp "
                        f"H2={train_h2['mean_r_on_bp']:.2f}bp (need both >0)")

    c3 = train_h1["mean_r_id_bp"] <= 0 and train_h2["mean_r_id_bp"] <= 0 and val["mean_r_id_bp"] <= 0
    if not c3:
        ok = False
        reasons.append(f"(3) r_id<=0 signature: TRAIN_H1={train_h1['mean_r_id_bp']:.2f} "
                        f"TRAIN_H2={train_h2['mean_r_id_bp']:.2f} VAL={val['mean_r_id_bp']:.2f} "
                        f"(need all <=0)")

    c4_mdd_ok = (val["overnight_mdd"] <= 0.60 * val["bh_mdd"]) if val["bh_mdd"] < 0 else True
    c4 = c4_mdd_ok and (val["overnight_total_return"] >= val["bh_total_return"])
    if not c4:
        ok = False
        reasons.append(f"(4) VAL overnight_mdd={val['overnight_mdd']:.2%} vs 60%*bh_mdd="
                        f"{0.6*val['bh_mdd']:.2%}; overnight_total={val['overnight_total_return']:.2%} "
                        f"vs bh_total={val['bh_total_return']:.2%}")

    c5 = cadence_c3 and cadence_c4
    if not c5:
        ok = False
        reasons.append("(5) cadence C3/C4 on VAL not both satisfied (see cadence_bar output)")

    c6_mdd_ok = (val_1bp["overnight_mdd"] <= 0.60 * val["bh_mdd"]) if val["bh_mdd"] < 0 else True
    c6 = (val_1bp["mean_r_on_bp"] > 0 and val_1bp["overnight_total_return"] >= val["bh_total_return"]
          and c6_mdd_ok)
    if not c6:
        ok = False
        reasons.append(f"(6) 1bp sensitivity fails criteria 1-4: mean_r_on={val_1bp['mean_r_on_bp']:.2f}bp")

    verdict = "PASS" if ok else "FAIL"
    return verdict, reasons


def fmt_dict_bp(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items()))


def main():
    logger.info(f"Overnight-effect walker starting: symbols={SYMBOLS} span={START}..{END}")
    report_lines = []
    report_lines.append("# REPORT — Overnight effect on index ETFs (cells 1,347-1,350)\n")
    report_lines.append(f"Generated by research/overnight/walker.py. Data: Alpaca daily bars, feed=sip, "
                         f"adjustment=raw, {START}..{END}. TEST (>=2025-01-01) not fetched.\n")
    report_lines.append(f"Notional ${NOTIONAL:,.0f}; R = 0.5% of notional = ${R_DOLLARS:,.0f}/night.\n")

    all_verdicts = {}

    for symbol in SYMBOLS:
        logger.info(f"=== {symbol} ===")
        bars = fetch_daily_bars(symbol)
        missing = report_missing_days(symbol, bars)
        nights = compute_nights(bars)

        nights_csv = BASE_DIR / f"nights_{symbol}.csv"
        nights[["date", "r_on", "r_id"]].to_csv(nights_csv, index=False)
        logger.info(f"{symbol}: wrote {nights_csv} ({len(nights)} nights)")

        train = slice_split(nights, TRAIN_START, TRAIN_END)
        train_h1 = slice_split(nights, TRAIN_H1_START, TRAIN_H1_END)
        train_h2 = slice_split(nights, TRAIN_H2_START, TRAIN_H2_END)
        val = slice_split(nights, VAL_START, VAL_END)

        st_train = cell_stats(train)
        st_train_h1 = cell_stats(train_h1)
        st_train_h2 = cell_stats(train_h2)
        st_val = cell_stats(val)
        st_val_1bp = cell_stats(val, cost_bp_per_side=COST_BP_SENSITIVITY)

        # Weekly pnl_R -> cadence_bar for TRAIN and VAL
        weekly_train = build_weekly_pnl_r(train)
        weekly_val = build_weekly_pnl_r(val)
        weekly_train_csv = BASE_DIR / f"nights_{symbol}_weeks_TRAIN.csv"
        weekly_val_csv = BASE_DIR / f"nights_{symbol}_weeks_VAL.csv"
        weekly_train.to_csv(weekly_train_csv, index=False)
        weekly_val.to_csv(weekly_val_csv, index=False)

        cadence_train_out = run_cadence_bar(weekly_train_csv, "TRAIN")
        cadence_val_out = run_cadence_bar(weekly_val_csv, "VAL")

        # crude parse of C3/C4 pass/fail from cadence_bar text output
        val_lower = cadence_val_out.lower()
        cadence_c3 = ("c3" in val_lower and "fail" not in val_lower.split("c3", 1)[1][:80].lower())
        cadence_c4 = ("c4" in val_lower and "fail" not in val_lower.split("c4", 1)[1][:80].lower())

        verdict, reasons = verdict_for_cell(st_train_h1, st_train_h2, st_val, st_val_1bp,
                                             cadence_c3, cadence_c4)
        all_verdicts[symbol] = verdict
        logger.info(f"{symbol}: verdict={verdict} reasons={reasons}")

        report_lines.append(f"\n## {symbol}\n")
        report_lines.append(f"Missing business-day bars (incl. holidays): {missing}. "
                             f"Nights computed: {len(nights)}.\n")
        report_lines.append("| Split | n | mean r_on (bp) | mean r_id (bp) | sd (bp) | iid t | month t | "
                             "hit rate | annualized r_on | overnight total ret | b&h total ret | "
                             "overnight MDD | b&h MDD | ex-top5% mean (bp) | worst night |")
        report_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for label, st in [("TRAIN (2010-2019)", st_train), ("TRAIN H1 (2010-2014)", st_train_h1),
                           ("TRAIN H2 (2015-2019)", st_train_h2), ("VAL (2020-2024)", st_val),
                           ("VAL +1bp/side", st_val_1bp)]:
            if st.get("n", 0) == 0:
                report_lines.append(f"| {label} | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - |")
                continue
            report_lines.append(
                f"| {label} | {st['n']} | {st['mean_r_on_bp']:.2f} | {st['mean_r_id_bp']:.2f} | "
                f"{st['sd_r_on_bp']:.1f} | {st['iid_t']:.2f} | {st['month_t']:.2f} | "
                f"{st['hit_rate']:.1%} | {st['annualized_r_on']:.2%} | {st['overnight_total_return']:.1%} | "
                f"{st['bh_total_return']:.1%} | {st['overnight_mdd']:.1%} | {st['bh_mdd']:.1%} | "
                f"{st['ex_top5_mean_bp']:.2f} | {st['worst_night']['date']} "
                f"({st['worst_night']['r_on_bp']:.1f}bp) |"
            )

        report_lines.append(f"\nVAL year-by-year mean r_on (bp): {fmt_dict_bp(st_val['yearly_mean_bp'])}\n")
        report_lines.append(f"VAL day-of-week mean r_on (bp, report-only): {fmt_dict_bp(st_val['dow_mean_bp'])}\n")

        report_lines.append(f"\n**cadence_bar.py TRAIN:**\n```\n{cadence_train_out}\n```\n")
        report_lines.append(f"**cadence_bar.py VAL:**\n```\n{cadence_val_out}\n```\n")

        report_lines.append(f"\n**Verdict ({symbol}): {verdict}**\n")
        for r in reasons:
            report_lines.append(f"- {r}")
        if not reasons:
            report_lines.append("- All pass-bar criteria (1-6) satisfied.")

    report_lines.append("\n## Summary\n")
    report_lines.append("| Cell | Symbol | Verdict |")
    report_lines.append("|---|---|---|")
    cell_ids = {"SPY": 1347, "QQQ": 1348, "TQQQ": 1349, "UPRO": 1350}
    for sym, v in all_verdicts.items():
        report_lines.append(f"| {cell_ids[sym]} | {sym} | {v} |")

    REPORT_PATH.write_text("\n".join(report_lines))
    logger.info(f"Wrote {REPORT_PATH}")
    logger.info(f"Verdicts: {all_verdicts}")


if __name__ == "__main__":
    main()
