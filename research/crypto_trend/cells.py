"""Cells C1-C4 for the crypto-trend PREREG (research/crypto_trend/PREREG.md).

Weekly decision at Monday 00:00 UTC using daily closes STRICTLY BEFORE that instant
(i.e. through the prior Sunday's close; the Monday close itself is never visible to
the signal). Orders execute at the Monday 00:00 UTC price, which for a daily bar
that starts at 00:00 UTC is the Monday OPEN -- so opens are the transaction price
and closes are the signal price, never mixed.

Cells (long/flat only, no shorting on Alpaca crypto):
  C1  trend:     long while close > 100-day SMA of closes, else flat
  C2  Donchian:  enter long on a close above the prior 20-day high (of closes,
                 the 20 closes strictly before the trigger close); exit to flat on
                 a close below the prior 10-day low. Stateful (depends on last week's
                 position).
  C3  momentum:  long while the 28-day close-to-close return > 0, else flat
  C4  C1 AND C3, else flat

Portfolio: equal weight across coins long that week (weight = 1/n_long), cash when
n_long == 0. Fee: 30 bps charged on a coin's own leg (0->1 entry or 1->0 exit) only,
using the weight active on that leg (the week entered for an entry, the week just
vacated for an exit) -- NOT charged for a pure reweighting caused by another coin
joining/leaving while this coin's own position is unchanged (disclosed simplification,
matches the PREREG's "not modelled at weekly frequency" cost disclosure).
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cells")

COINS = ["BTC", "ETH", "SOL"]
FEE_RATE = 0.0030  # 30 bps per leg
NW_LAGS = 4

SPLITS = {
    "TRAIN": (pd.Timestamp("2017-08-17"), pd.Timestamp("2023-12-31")),
    "VAL": (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-06-30")),
    "TEST": (pd.Timestamp("2025-07-01"), pd.Timestamp("2026-09-26")),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_price_panel(bars_csv: str) -> dict:
    """Load bars_daily.csv into {coin: DataFrame(index=date, columns=[open, close])}.

    Prefers Alpaca (the executable venue) wherever it has a bar for that date;
    falls back to Binance (the only source before 2021-01-01, and for any date
    Alpaca is missing). Logs a warning on any calendar-day gap in the merged series.
    """
    df = pd.read_csv(bars_csv, parse_dates=["date"])
    panel = {}
    for coin in COINS:
        sub = df[df["coin"] == coin].copy()
        sub["source_rank"] = sub["source"].map({"alpaca": 0, "binance": 1})
        sub = sub.sort_values(["date", "source_rank"]).drop_duplicates("date", keep="first")
        sub = sub.set_index("date").sort_index()
        panel[coin] = sub[["open", "close"]]
        full_range = pd.date_range(sub.index.min(), sub.index.max(), freq="D")
        missing = full_range.difference(sub.index)
        if len(missing) > 0:
            log.warning(f"{coin}: {len(missing)} missing calendar day(s) in merged series, e.g. {list(missing[:5])}")
    return panel


def weekly_grid(panel: dict) -> list:
    """All Monday 00:00 UTC checkpoints spanning the earliest to latest data."""
    start = min(df.index.min() for df in panel.values())
    end = max(df.index.max() for df in panel.values())
    first_monday = start + pd.Timedelta(days=(7 - start.weekday()) % 7)
    mondays = list(pd.date_range(first_monday, end, freq="7D"))
    return mondays


# ---------------------------------------------------------------------------
# Signals (each returns a 0/1 pd.Series indexed by monday; decided using closes
# strictly before that monday, i.e. close.index < monday)
# ---------------------------------------------------------------------------
def sma_trend_signal(close: pd.Series, mondays: list, window: int = 100) -> pd.Series:
    out = []
    for m in mondays:
        prior = close.loc[close.index < m]
        if len(prior) < window:
            out.append(0)
            continue
        last_close = prior.iloc[-1]
        sma = prior.iloc[-window:].mean()
        out.append(1 if last_close > sma else 0)
    return pd.Series(out, index=mondays, dtype=int)


def momentum_signal(close: pd.Series, mondays: list, lookback: int = 28) -> pd.Series:
    out = []
    for m in mondays:
        prior = close.loc[close.index < m]
        if len(prior) < lookback + 1:
            out.append(0)
            continue
        last_close = prior.iloc[-1]
        past_close = prior.iloc[-(lookback + 1)]
        ret = last_close / past_close - 1
        out.append(1 if ret > 0 else 0)
    return pd.Series(out, index=mondays, dtype=int)


def donchian_signal(close: pd.Series, mondays: list, entry_window: int = 20, exit_window: int = 10) -> pd.Series:
    out = []
    state = 0
    req = max(entry_window, exit_window) + 1
    for m in mondays:
        prior = close.loc[close.index < m]
        if len(prior) < req:
            out.append(0)
            state = 0
            continue
        last_close = prior.iloc[-1]
        if state == 0:
            window_high = prior.iloc[-(entry_window + 1):-1].max()
            if last_close > window_high:
                state = 1
        else:
            window_low = prior.iloc[-(exit_window + 1):-1].min()
            if last_close < window_low:
                state = 0
        out.append(state)
    return pd.Series(out, index=mondays, dtype=int)


def c4_signal(close: pd.Series, mondays: list) -> pd.Series:
    c1 = sma_trend_signal(close, mondays)
    c3 = momentum_signal(close, mondays)
    return (c1 & c3).astype(int)


CELL_FUNCS = {
    "C1": sma_trend_signal,
    "C2": donchian_signal,
    "C3": momentum_signal,
    "C4": c4_signal,
}


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def cell_positions(cell: str, panel: dict, mondays: list) -> pd.DataFrame:
    """positions[coin] 0/1 per monday, for one cell."""
    fn = CELL_FUNCS[cell]
    cols = {coin: fn(panel[coin]["close"], mondays) for coin in COINS}
    return pd.DataFrame(cols, index=mondays)


def portfolio_weekly(cell: str, panel: dict, mondays: list) -> pd.DataFrame:
    """Per-week portfolio table: gross_return, fee, net_return, n_long, is_entry/-exit counts.

    Week [mondays[i], mondays[i+1]) return uses OPEN prices (the 00:00 UTC transacted
    price) for both entry and exit checkpoints; the last monday has no forward week
    and is dropped.
    """
    pos = cell_positions(cell, panel, mondays)
    rows = []
    coin_detail_rows = []  # per (week, coin): raw return, this-cell fee -- for weekly_positions.csv
    prev_weight = {c: 0.0 for c in COINS}
    for i in range(len(mondays) - 1):
        w_start, w_end = mondays[i], mondays[i + 1]
        n_long = int(pos.loc[w_start].sum())
        weight = {c: (1.0 / n_long if pos.loc[w_start, c] == 1 else 0.0) for c in COINS}

        gross = 0.0
        fee = 0.0
        entries = 0
        exits = 0
        for c in COINS:
            try:
                o0 = panel[c]["open"].loc[w_start]
                o1 = panel[c]["open"].loc[w_end]
                coin_ret = o1 / o0 - 1
            except KeyError:
                coin_ret = None  # coin not listed yet this week
            if weight[c] > 0 and coin_ret is not None:
                gross += weight[c] * coin_ret
            was_long = prev_weight[c] > 0
            is_long = weight[c] > 0
            coin_fee = 0.0
            if is_long and not was_long:
                coin_fee = FEE_RATE * weight[c]
                entries += 1
            elif was_long and not is_long:
                coin_fee = FEE_RATE * prev_weight[c]
                exits += 1
            fee += coin_fee
            coin_detail_rows.append({
                "week_start": w_start, "coin": c,
                "weekly_return_gross": coin_ret, "fee": coin_fee,
            })
        net = gross - fee
        rows.append({
            "week_start": w_start, "n_long": n_long, "gross_return": gross,
            "fee": fee, "net_return": net, "entries": entries, "exits": exits,
            "split": week_split(w_start, w_end),
        })
        prev_weight = weight
    df = pd.DataFrame(rows).set_index("week_start")
    detail = pd.DataFrame(coin_detail_rows).set_index(["week_start", "coin"])
    return df, pos, detail


def assign_split(week_start: pd.Timestamp) -> str:
    """Split containing a single date (used by tests and for the weekly_positions
    per-coin split filter, where only week_start is available)."""
    for name, (lo, hi) in SPLITS.items():
        if lo <= week_start <= hi:
            return name
    return None


def week_split(w_start: pd.Timestamp, w_end: pd.Timestamp) -> str:
    """A scored week must be FULLY contained in one split: both its decision-visible
    start and its return-realizing end. A week straddling a split boundary (its
    return would touch price data from the next, possibly-sealed, split) is dropped
    from scoring entirely rather than leaking data across the boundary -- this is why
    VAL is exactly 78 weeks (PREREG) rather than 79: the week starting Monday
    2025-06-30 realizes its return on 2025-07-07, inside the sealed TEST window, and
    is excluded from both VAL and TEST.
    """
    for name, (lo, hi) in SPLITS.items():
        if lo <= w_start and w_end <= hi:
            return name
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def newey_west_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """Newey-West HAC t-stat for H0: mean(x) == 0, Bartlett kernel, `lags` lags."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    xbar = x.mean()
    e = x - xbar
    gamma0 = np.mean(e * e)
    s = gamma0
    for lag in range(1, lags + 1):
        if lag >= n:
            break
        cov = np.mean(e[lag:] * e[:-lag])
        s += 2 * (1 - lag / (lags + 1)) * cov
    var_mean = s / n
    if var_mean <= 0:
        return float("nan")
    se = np.sqrt(var_mean)
    return xbar / se


def max_drawdown(returns: pd.Series) -> float:
    """Max drawdown of the compounded (1+r) equity curve, as a positive fraction."""
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(-dd.min()) if len(dd) else 0.0


def ex_top_k_pct_mean(returns: pd.Series, pct: float = 0.05) -> float:
    """Mean weekly return after dropping the top pct fraction of weeks by return."""
    n = len(returns)
    k = int(round(n * pct))
    if k == 0:
        return float(returns.mean())
    trimmed = returns.sort_values(ascending=False).iloc[k:]
    return float(trimmed.mean())


def count_matched_null_green_share(pos: pd.DataFrame, panel: dict, mondays: list, split_weeks: list,
                                    n_draws: int = 1000, seed: int = 1470) -> float:
    """Green-week share of `n_draws` random long/flat schedules, each coin keeping its
    own actual count of long weeks within the split (a random permutation of that
    coin's within-split position vector), portfolio built and fee-charged the same way.
    """
    rng = np.random.default_rng(seed)
    idx = {m: i for i, m in enumerate(mondays)}
    split_pos_idx = [idx[m] for m in split_weeks]
    n = len(split_weeks)

    opens = {c: panel[c]["open"] for c in COINS}
    actual_counts = {c: int(pos.loc[split_weeks, c].sum()) for c in COINS}

    green_shares = np.empty(n_draws)
    for draw in range(n_draws):
        rand_pos = {}
        for c in COINS:
            k = actual_counts[c]
            arr = np.zeros(n, dtype=int)
            if k > 0:
                on = rng.choice(n, size=k, replace=False)
                arr[on] = 1
            rand_pos[c] = arr
        prev_weight = {c: 0.0 for c in COINS}
        net_returns = np.empty(n)
        for j in range(n):
            m_i = split_pos_idx[j]
            w_start, w_end = mondays[m_i], mondays[m_i + 1]
            longs = [c for c in COINS if rand_pos[c][j] == 1]
            n_long = len(longs)
            weight = {c: (1.0 / n_long if c in longs else 0.0) for c in COINS}
            gross = 0.0
            fee = 0.0
            for c in COINS:
                if weight[c] > 0:
                    try:
                        o0, o1 = opens[c].loc[w_start], opens[c].loc[w_end]
                        gross += weight[c] * (o1 / o0 - 1)
                    except KeyError:
                        pass
                was_long, is_long = prev_weight[c] > 0, weight[c] > 0
                if is_long and not was_long:
                    fee += FEE_RATE * weight[c]
                elif was_long and not is_long:
                    fee += FEE_RATE * prev_weight[c]
            net_returns[j] = gross - fee
            prev_weight = weight
        green_shares[draw] = float(np.mean(net_returns > 0))
    return float(green_shares.mean())


def score_split(cell: str, split: str, wk: pd.DataFrame, pos: pd.DataFrame, panel: dict, mondays: list) -> dict:
    sub = wk[wk["split"] == split]
    if len(sub) == 0:
        return None
    net = sub["net_return"]
    gross = sub["gross_return"]
    t_nw = newey_west_t(net.values, NW_LAGS)
    green_share = float((net > 0).mean())
    dd = max_drawdown(net)
    ex5 = ex_top_k_pct_mean(net, 0.05)
    time_in_mkt = float((sub["n_long"] / 3.0).mean())
    round_trips = int(sub["entries"].sum())
    fees_pct_total = float(sub["fee"].sum() * 100)

    null_green = count_matched_null_green_share(pos, panel, mondays, list(sub.index))

    btc_open = panel["BTC"]["open"]
    bh_rows = []
    for w_start in sub.index:
        i = mondays.index(w_start)
        w_end = mondays[i + 1]
        bh_rows.append(btc_open.loc[w_end] / btc_open.loc[w_start] - 1)
    bh = pd.Series(bh_rows, index=sub.index)
    bh_mean = float(bh.mean())
    bh_dd = max_drawdown(bh)

    return {
        "cell": cell, "split": split, "weeks": int(len(sub)),
        "mean_weekly_pct": float(net.mean() * 100),
        "gross_mean_weekly_pct": float(gross.mean() * 100),
        "t_nw": float(t_nw),
        "green_share": green_share,
        "null_green_share": null_green,
        "max_dd_pct": float(dd * 100),
        "ex_top5_mean_weekly_pct": float(ex5 * 100),
        "time_in_market": time_in_mkt,
        "round_trips": round_trips,
        "fees_pct_total": fees_pct_total,
        "bh_btc_mean_weekly_pct": bh_mean * 100,
        "bh_btc_max_dd_pct": bh_dd * 100,
    }


def passes_val_bar(val_row: dict, train_row: dict) -> bool:
    """Frozen VAL pass bar (PREREG Pass-bar section), corroborated by TRAIN same-sign t>=2."""
    if val_row is None:
        return False
    checks = [
        val_row["mean_weekly_pct"] >= 0.35,
        val_row["t_nw"] >= 2.0,
        val_row["green_share"] >= 0.55,
        val_row["max_dd_pct"] <= 30.0,
        val_row["ex_top5_mean_weekly_pct"] >= 0.0,
    ]
    train_ok = (train_row is not None and train_row["t_nw"] >= 2.0
                and np.sign(train_row["mean_weekly_pct"]) == np.sign(val_row["mean_weekly_pct"]))
    return bool(all(checks) and train_ok)


def run_all(bars_csv: str = "research/crypto_trend/bars_daily.csv"):
    panel = load_price_panel(bars_csv)
    mondays = weekly_grid(panel)
    log.info(f"Weekly grid: {len(mondays)} mondays, {mondays[0].date()}..{mondays[-1].date()}")

    results = []
    weekly_pos_rows = []
    for cell in ["C1", "C2", "C3", "C4"]:
        wk, pos, detail = portfolio_weekly(cell, panel, mondays)
        log.info(f"{cell}: {len(wk)} scored weeks total")
        train_row = score_split(cell, "TRAIN", wk, pos, panel, mondays)
        val_row = score_split(cell, "VAL", wk, pos, panel, mondays)
        # TEST rows are fetched into bars_daily.csv but MUST NEVER be scored here.
        for row, split in ((train_row, "TRAIN"), (val_row, "VAL")):
            if row is None:
                continue
            row["passes_bar"] = (
                passes_val_bar(val_row, train_row) if split == "VAL"
                else bool(row["t_nw"] >= 2.0)  # TRAIN-only descriptive corroboration flag
            )
            results.append(row)

        for w_start in wk.index:
            split = wk.loc[w_start, "split"]
            if split != "TRAIN" and split != "VAL":
                continue  # TEST, and any boundary week dropped by week_split, never written
            for c in COINS:
                d = detail.loc[(w_start, c)]
                weekly_pos_rows.append({
                    "week_start": w_start.date().isoformat(), "coin": c, "cell": cell,
                    "position": int(pos.loc[w_start, c]),
                    "weekly_return_gross": d["weekly_return_gross"],
                    "fee": d["fee"],
                })
    return results, panel, mondays, weekly_pos_rows


if __name__ == "__main__":
    results, panel, mondays, weekly_pos_rows = run_all()
    import csv
    with open("research/crypto_trend/weekly_positions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["week_start", "coin", "cell", "position", "weekly_return_gross", "fee"])
        w.writeheader()
        w.writerows(weekly_pos_rows)
    log.info(f"Wrote {len(weekly_pos_rows)} rows to research/crypto_trend/weekly_positions.csv")
    for r in results:
        log.info(r)
