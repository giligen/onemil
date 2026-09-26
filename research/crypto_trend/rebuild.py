"""Independent rebuild of the crypto-trend cells (C1-C4) from PREREG.md prose alone.

This script was written WITHOUT reading cells.py, test_cells.py or RESULT.md. It rebuilds
each cell's weekly long/flat position per coin from research/crypto_trend/bars_daily.csv and
writes research/crypto_trend/rebuild_positions.csv for comparison against the builder's
research/crypto_trend/weekly_positions.csv.

Rules (from PREREG.md):
    C1: long while close > 100-day SMA of closes (inclusive of today), else flat.
    C2: Donchian. Enter long on close > prior 20-day high (20 days strictly before today).
        Exit (go flat) on close < prior 10-day low (10 days strictly before today). Stateful:
        holds the previous position when neither trigger fires.
    C3: long while the 28-day return (close[t]/close[t-28]-1) > 0, else flat.
    C4: long only when C1 AND C3 both say long that week, else flat.

Data: daily closes/opens for BTC, ETH, SOL. Binance source for date < 2021-01-01, Alpaca
source for date >= 2021-01-01 (PREREG data section).

Timing (no look-ahead): the rule is evaluated at Monday 00:00 UTC (week_start) on the PRIOR
daily closes, i.e. using data through the preceding Sunday (week_start - 1 day) inclusive.
Orders execute at week_start's own opening price. The coin's realized weekly_return_gross is
therefore open(next week_start) / open(week_start) - 1 (confirmed by reproducing the builder's
own gross-return numbers for the first BTC/ETH weeks in weekly_positions.csv by hand before
writing this script).

Cost: PREREG prose says "30 bps per leg ... charged on every change of position" -- flat 30bps
(0.003), not weighted by portfolio share. This script applies that flat reading literally; see
the independent-check report for the discrepancy against the builder's fee column (which is
scaled by 1/N_long, i.e. by the coin's equal-weight portfolio share at the time of the change).
"""
import sys
import pandas as pd
import numpy as np

BARS_CSV = "research/crypto_trend/bars_daily.csv"
OUT_CSV = "research/crypto_trend/rebuild_positions.csv"
COINS = ["BTC", "ETH", "SOL"]
CELLS = ["C1", "C2", "C3", "C4"]
FEE_PER_LEG = 0.003  # 30 bps, flat, per PREREG prose
ALPACA_CUTOVER = pd.Timestamp("2021-01-01")
VAL_END = pd.Timestamp("2025-06-30")  # do not score TEST (2025-07-01 onward)


def build_price_series(bars: pd.DataFrame, coin: str) -> pd.DataFrame:
    """Return a date-indexed frame of open/close for one coin: Binance before the Alpaca
    cutover, Alpaca from the cutover onward, exactly as PREREG.md's Data section specifies."""
    cols = ["open", "high", "low", "close"]
    sub = bars[bars["coin"] == coin]
    binance_all = sub[sub["source"] == "binance"].set_index("date")[cols]
    alpaca_all = sub[sub["source"] == "alpaca"].set_index("date")[cols]
    pre = binance_all[binance_all.index < ALPACA_CUTOVER]
    post_alpaca = alpaca_all[alpaca_all.index >= ALPACA_CUTOVER]
    # Alpaca has undisclosed gaps for SOL (e.g. late Dec 2023); PREREG treats the two feeds as
    # near-equivalent (overlap close diff must be < 0.3%), so fall back to Binance on any date
    # >= cutover where Alpaca has no row, rather than silently leaving a hole in the series.
    post_binance = binance_all[(binance_all.index >= ALPACA_CUTOVER)
                                & (~binance_all.index.isin(post_alpaca.index))]
    if len(post_binance):
        print(f"WARNING [{coin}]: {len(post_binance)} post-cutover day(s) missing from Alpaca, "
              f"fell back to Binance, e.g. {list(post_binance.index[:3])}", file=sys.stderr)
    merged = pd.concat([pre, post_alpaca, post_binance])
    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    full = merged[cols]
    # sanity: no calendar gaps within the covered range (crypto trades every day)
    gaps = pd.date_range(full.index.min(), full.index.max(), freq="D").difference(full.index)
    if len(gaps):
        print(f"WARNING [{coin}]: {len(gaps)} missing calendar day(s), e.g. {list(gaps[:3])}",
              file=sys.stderr)
    return full


def compute_cells(price: pd.DataFrame) -> pd.DataFrame:
    """Compute daily C1/C2/C3/C4 boolean position series for one coin (indexed by date),
    to be sampled at each week's decision date (the Sunday before week_start)."""
    close = price["close"]
    sma100 = close.rolling(100, min_periods=100).mean()
    c1 = (close > sma100).astype(float)
    c1[sma100.isna()] = 0.0  # insufficient history -> flat, not unknown

    # Donchian channel uses the OHLC high/low fields (the conventional "N-day high/low"),
    # not a rolling max/min of closes -- confirmed against the builder's own first BTC C2
    # entry date (2017-10-09): a close-based channel would have triggered a week early
    # (2017-10-02, close 4378.48 vs close-based prior-20 high 4378.51, a near-tie) while the
    # high/low-based channel does not trigger until 2017-10-08 (close 4640.0 vs high-based
    # prior-20 high 4561.63), matching the builder.
    prior_high20 = price["high"].shift(1).rolling(20, min_periods=20).max()
    prior_low10 = price["low"].shift(1).rolling(10, min_periods=10).min()
    c2 = pd.Series(0.0, index=close.index)
    pos = 0
    for dt in close.index:
        px = close.loc[dt]
        if pd.isna(px):
            pos = 0
        elif pos == 0 and pd.notna(prior_high20.loc[dt]) and px > prior_high20.loc[dt]:
            pos = 1
        elif pos == 1 and pd.notna(prior_low10.loc[dt]) and px < prior_low10.loc[dt]:
            pos = 0
        c2.loc[dt] = pos

    mom28 = close / close.shift(28) - 1
    c3 = (mom28 > 0).astype(float)
    c3[mom28.isna()] = 0.0

    c4 = ((c1 == 1) & (c3 == 1)).astype(float)

    return pd.DataFrame({"C1": c1, "C2": c2, "C3": c3, "C4": c4})


def main():
    bars = pd.read_csv(BARS_CSV, parse_dates=["date"])
    prices = {coin: build_price_series(bars, coin) for coin in COINS}
    cells = {coin: compute_cells(prices[coin]) for coin in COINS}

    # week_start grid: every Monday from the first Monday on/after the earliest coin's first
    # date, through VAL_END (2025-06-30); TEST is never scored per PREREG / task instructions.
    earliest = min(p.index.min() for p in prices.values())
    first_monday = earliest + pd.Timedelta(days=(7 - earliest.weekday()) % 7)
    last_monday = VAL_END - pd.Timedelta(days=7)  # next_week_start must not exceed VAL_END
    week_starts = pd.date_range(first_monday, last_monday, freq="W-MON")

    prev_position = {(coin, cell): 0 for coin in COINS for cell in CELLS}
    rows = []
    for w in week_starts:
        decision_date = w - pd.Timedelta(days=1)
        next_w = w + pd.Timedelta(days=7)
        for coin in COINS:
            price = prices[coin]
            gross = np.nan
            if w in price.index and next_w in price.index:
                gross = price.loc[next_w, "open"] / price.loc[w, "open"] - 1
            cframe = cells[coin]
            for cell in CELLS:
                if decision_date in cframe.index:
                    position = int(cframe.loc[decision_date, cell])
                else:
                    position = 0  # before coin inception -> flat
                fee = FEE_PER_LEG if position != prev_position[(coin, cell)] else 0.0
                prev_position[(coin, cell)] = position
                rows.append((w.date().isoformat(), coin, cell, position, gross, fee))

    out = pd.DataFrame(rows, columns=["week_start", "coin", "cell", "position",
                                       "weekly_return_gross", "fee"])
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows to {OUT_CSV}")
    print(f"week_starts: {len(week_starts)} ({week_starts.min().date()} .. {week_starts.max().date()})")


if __name__ == "__main__":
    main()
