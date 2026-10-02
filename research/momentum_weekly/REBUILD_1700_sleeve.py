"""
REBUILD_1700_sleeve.py

Independent re-implementation of the "risk-adjusted top 20, weekly" momentum
sleeve, built ONLY from the prose spec handed to this agent and the two data
files named there (panel_2016_2026.parquet, 1700c_assets.csv). No file from
the research/momentum_weekly/1700*.py / RESULT_1700* lineage was opened.

Spec summary (see task prompt for the authoritative text):
  - Rebalance every Monday (or next trading day) 2017-01-02..2026-09-28.
  - Universe at each rebalance: close(t) >= $10, ADV20$(t) >= $200M,
    >=273 trading days of own history before the rebalance date, name
    exclusions (ETF/fund/trust/... regex, ^Z[A-Z]ZZT$, dotted/slashed
    tickers). t = the prior trading day (close-of/vol-through date).
  - Signal: (close[t-21]/close[t-252] - 1) / std(daily returns, 252d end t).
    Offsets are counted along EACH SYMBOL'S OWN trading-day sequence
    (not a reindexed master calendar) -- see RESOLUTIONS below.
  - Portfolio: top 20 by signal, equal weight on ENTRY only; names kept
    are not re-traded (weights drift); leavers sold at Monday's open,
    entrants bought at Monday's open at 1/20 of portfolio value.
  - Costs: on every traded dollar, 5bps commission + half the spread
    proxy (spread_proxy = (high-low)/close of the execution day * 0.1),
    total cost rate capped at 20bps.
  - No dividends either side. Weekly return = Monday-open to next-Monday-
    open, net of costs. Missing bar inside the week -> carry last price.
    Symbol with no bars at all after entry -> write off at last close
    (not -100%); count these.
  - SPY benchmark: open-to-open over the same weeks.

RESOLUTIONS (ambiguities this agent had to decide, stated here per protocol):
  R1. "close[t-21]" / "close[t-252]" offsets are counted along each
      symbol's own sorted bar sequence (standard momentum convention),
      not a master-calendar reindex. A symbol missing a bar on exactly
      t is simply excluded from that week's universe (liquid ADV20>=200M
      names rarely miss a session).
  R2. The 20bps cap applies to the TOTAL cost rate (commission + half
      spread), i.e. cost_rate = min(0.0005 + 0.5*spread_proxy, 0.0020).
  R3. Entry/exit execution price = that day's open; if a held symbol has
      no bar at all on the valuation Monday, its value is carried at the
      last available close (ffill) -- this is the literal "carry its
      last price" / write-off rule, implemented via a per-symbol
      forward-filled price path rather than a special-cased branch.
  R4. "Number of names traded" per week = entries + exits that week
      (first week: 20 entries, 0 exits).
  R5. Master trading calendar = SPY's own bar_date sequence in the panel.

Memory note: panel is ~22M rows; loaded as float32 for price/volume
columns to stay under ~1.5GB resident.
"""
import sys
import re
import time
import numpy as np
import pandas as pd

PANEL_PATH = "research/momentum_weekly/panel_2016_2026.parquet"
ASSETS_PATH = "research/momentum_weekly/1700c_assets.csv"
OUT_WEEKLY = "research/momentum_weekly/1700_rebuild_weekly.csv"
OUT_BY_YEAR = "research/momentum_weekly/1700_rebuild_by_year.csv"
OUT_MD = "research/momentum_weekly/REBUILD_1700_sleeve.md"

START_REBAL = pd.Timestamp("2017-01-02")
END_REBAL = pd.Timestamp("2026-09-28")

NAME_EXCL_RE = re.compile(
    r"ETF|ETN|Fund|Trust|Index|Warrant|Unit|Preferred|Depositary|Right|Notes|Bond|Portfolio",
    re.IGNORECASE,
)
SYM_ZZZT_RE = re.compile(r"^Z[A-Z]ZZT$")

MIN_PRICE = 10.0
MIN_ADV20 = 200e6
MIN_HISTORY_DAYS = 273
TOP_N = 20
COMMISSION = 0.0005
SPREAD_SCALE = 0.1
COST_CAP = 0.0020
STARTING_CAPITAL = 50000.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_panel():
    log("loading panel parquet (float32 cast)...")
    df = pd.read_parquet(
        PANEL_PATH,
        columns=["symbol", "bar_date", "open", "high", "low", "close", "volume"],
    )
    df["bar_date"] = pd.to_datetime(df["bar_date"])
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype("float32")
    df["volume"] = df["volume"].astype("float32")
    df = df.sort_values(["symbol", "bar_date"]).reset_index(drop=True)
    log(f"panel loaded: {len(df):,} rows, {df['symbol'].nunique():,} symbols, "
        f"{df['bar_date'].min().date()}..{df['bar_date'].max().date()}")
    return df


def load_exclusions():
    log("loading 1700c_assets.csv for name-based exclusions...")
    a = pd.read_csv(ASSETS_PATH, dtype=str)
    a.columns = [c.strip().lower() for c in a.columns]
    name_col = "name" if "name" in a.columns else None
    excl = set()
    if name_col:
        bad_name = a[name_col].fillna("").str.contains(NAME_EXCL_RE)
        excl |= set(a.loc[bad_name, "symbol"].dropna())
    log(f"name-based exclusion set: {len(excl):,} symbols")
    return excl


def compute_features(df):
    log("computing per-symbol features (shift/rolling)...")
    g = df.groupby("symbol", sort=False)
    df["hist_count"] = g.cumcount() + 1  # trading days up to & incl this row, own series
    df["close_lag21"] = g["close"].shift(21)
    df["close_lag252"] = g["close"].shift(252)
    df["mom"] = df["close_lag21"] / df["close_lag252"] - 1.0
    df["ret"] = g["close"].pct_change()
    return df


def compute_features_v2(df):
    """Correct within-group rolling std (the naive .rolling above leaks across
    symbol boundaries if not grouped first) -- compute explicitly grouped."""
    log("computing within-symbol rolling vol (252d) and ADV20...")
    df["vol252"] = df.groupby("symbol", sort=False)["ret"].transform(
        lambda s: s.rolling(252, min_periods=252).std()
    )
    df["dollar_vol"] = df["close"] * df["volume"]
    df["adv20"] = df.groupby("symbol", sort=False)["dollar_vol"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    df["signal"] = df["mom"] / df["vol252"]
    return df


def build_calendar(df):
    spy = df.loc[df["symbol"] == "SPY", "bar_date"].sort_values().unique()
    cal = pd.DatetimeIndex(spy)
    log(f"master calendar (SPY) has {len(cal):,} trading days, "
        f"{cal.min().date()}..{cal.max().date()}")
    return cal


def rebalance_dates(cal):
    mondays = pd.date_range(START_REBAL, END_REBAL, freq="W-MON")
    cal_sorted = cal.sort_values()
    out = []
    for m in mondays:
        pos = cal_sorted.searchsorted(m, side="left")
        if pos < len(cal_sorted) and cal_sorted[pos] == m:
            out.append(cal_sorted[pos])
        elif pos < len(cal_sorted):
            out.append(cal_sorted[pos])  # next trading day after the holiday Monday
        # else: past end of data, drop
    out = sorted(set(out))
    return [d for d in out if d <= cal_sorted.max()]


def prior_trading_day(cal_sorted, d):
    pos = cal_sorted.searchsorted(d, side="left")
    if pos == 0:
        return None
    return cal_sorted[pos - 1]


def main():
    df = load_panel()
    excl_names = load_exclusions()
    df = compute_features(df)
    df = compute_features_v2(df)

    cal = build_calendar(df)
    cal_sorted = cal.sort_values()
    rebals = rebalance_dates(cal)
    log(f"{len(rebals)} rebalance dates generated "
        f"({rebals[0].date()}..{rebals[-1].date()})")

    # exclude by symbol pattern up front (cheap, vectorized)
    syms = df["symbol"].astype(str)
    bad_sym_pattern = syms.str.match(SYM_ZZZT_RE) | syms.str.contains(r"[./]", regex=True)
    df = df.loc[~bad_sym_pattern].copy()
    df = df.loc[~df["symbol"].isin(excl_names)].copy()
    log(f"after name/pattern exclusions: {len(df):,} rows, "
        f"{df['symbol'].nunique():,} symbols remain")

    # index by (symbol, bar_date) for O(1) lookups at exact dates
    df = df.set_index(["symbol", "bar_date"]).sort_index()

    # per-symbol date/open/close arrays for weekly valuation (forward-fill lookups)
    log("building per-symbol price arrays for valuation...")
    sym_groups = {}
    for sym, sub in df.reset_index().groupby("symbol", sort=False):
        sym_groups[sym] = (
            sub["bar_date"].values,
            sub["open"].values.astype("float64"),
            sub["close"].values.astype("float64"),
        )
    log(f"{len(sym_groups):,} symbol price series built")

    def price_on_or_before(sym, date, use_open=False):
        """Forward-filled lookup: price at `date` if a bar exists, else the
        last close strictly before it (R3: carry-last-price / write-off)."""
        arr = sym_groups.get(sym)
        if arr is None:
            return None, False
        dates, opens, closes = arr
        pos = np.searchsorted(dates, np.datetime64(date), side="right") - 1
        if pos < 0:
            return None, False
        exact = dates[pos] == np.datetime64(date)
        if use_open and exact:
            return opens[pos], True
        return closes[pos], exact

    def has_future_bar(sym, after_date):
        arr = sym_groups.get(sym)
        if arr is None:
            return False
        dates = arr[0]
        return dates[-1] > np.datetime64(after_date)

    weekly_rows = []
    holdings = {}  # symbol -> dollar value (post last trade / drifted)
    cash_balance = STARTING_CAPITAL  # uninvested cash (earns 0), conserved across weeks
    writeoff_count = 0
    n_rebals_used = 0

    df_reset = df.reset_index()
    # fast lookup frame indexed by (symbol,bar_date) already built above (df multiindex)

    for i, reb in enumerate(rebals):
        t = prior_trading_day(cal_sorted, reb)
        if t is None:
            continue
        try:
            snap = df.xs(t, level="bar_date")
        except KeyError:
            snap = None
        if snap is None or len(snap) == 0:
            continue
        snap = snap.reset_index()  # columns: symbol, open, high, low, close, ...
        cand = snap[
            (snap["close"] >= MIN_PRICE)
            & (snap["adv20"] >= MIN_ADV20)
            & (snap["hist_count"] >= MIN_HISTORY_DAYS)
            & snap["signal"].notna()
            & np.isfinite(snap["signal"])
        ]
        if len(cand) < TOP_N:
            continue
        top20 = cand.nlargest(TOP_N, "signal")["symbol"].tolist()
        top20_set = set(top20)
        n_rebals_used += 1

        prev_syms = set(holdings.keys())
        leavers = prev_syms - top20_set
        entrants = top20_set - prev_syms
        kept = prev_syms & top20_set

        # portfolio value just before this Monday's trades = sum of drifted values
        pv_pre = sum(holdings.values()) + cash_balance

        # sell leavers at this Monday's open. `cash` starts as any undeployed
        # capital (== starting capital on week 0, when holdings is empty; 0
        # thereafter, since pv_pre already equals sum(holdings.values())).
        cash = cash_balance
        cost_total = 0.0
        for sym in leavers:
            px, exact = price_on_or_before(sym, reb, use_open=True)
            if px is None or px <= 0:
                px_val = holdings[sym]  # can't price, keep last value as proceeds
            else:
                px_val = holdings[sym]  # value already marked at entry; proceeds = current drifted value
            trade_val = holdings.pop(sym, 0.0)
            spread_proxy = None
            srow = snap.loc[snap["symbol"] == sym]
            if len(srow):
                h, l, c = float(srow["high"].iloc[0]), float(srow["low"].iloc[0]), float(srow["close"].iloc[0])
                spread_proxy = (h - l) / c * SPREAD_SCALE if c > 0 else 0.0
            cost_rate = min(COMMISSION + 0.5 * (spread_proxy or 0.0), COST_CAP)
            cost_total += trade_val * cost_rate
            cash += trade_val * (1 - cost_rate)

        pv_after_sells = cash + sum(holdings.values())
        per_entrant_target = pv_after_sells / TOP_N if (pv_after_sells > 0 and len(top20) > 0) else 0.0
        # equal-weight target for entrants only (kept names keep drifted value;
        # per spec only entrants are sized to 1/20 of portfolio value)
        for sym in entrants:
            trade_val = per_entrant_target
            srow = snap.loc[snap["symbol"] == sym]
            spread_proxy = 0.0
            if len(srow):
                h, l, c = float(srow["high"].iloc[0]), float(srow["low"].iloc[0]), float(srow["close"].iloc[0])
                spread_proxy = (h - l) / c * SPREAD_SCALE if c > 0 else 0.0
            cost_rate = min(COMMISSION + 0.5 * spread_proxy, COST_CAP)
            cost_total += trade_val * cost_rate
            cash -= trade_val
            holdings[sym] = trade_val * (1 - cost_rate)
        cash_balance = cash  # leftover/deficit from the 1/20 sizing rule is carried, never lost

        n_traded = len(leavers) + len(entrants)

        # determine next Monday (end of week valuation point)
        nxt = rebals[i + 1] if i + 1 < len(rebals) else None
        pv_start_week = sum(holdings.values())

        if nxt is not None:
            pv_end_week = 0.0
            for sym, val in list(holdings.items()):
                px_now, exact_now = price_on_or_before(sym, reb, use_open=True)
                px_next, exact_next = price_on_or_before(sym, nxt, use_open=True)
                if px_now is None or px_now <= 0 or px_next is None or px_next <= 0:
                    pv_end_week += val
                    continue
                if not has_future_bar(sym, reb):
                    writeoff_count += 1
                new_val = val * (px_next / px_now)
                holdings[sym] = new_val
                pv_end_week += new_val
            week_ret = (pv_end_week + cash_balance - pv_pre) / pv_pre if pv_pre > 0 else np.nan  # costs already embedded in holdings/cash

            # SPY open-to-open
            spy_now, _ = price_on_or_before("SPY", reb, use_open=True)
            spy_next, _ = price_on_or_before("SPY", nxt, use_open=True)
            spy_ret = (spy_next / spy_now - 1.0) if (spy_now and spy_next) else np.nan

            weekly_rows.append(
                dict(
                    week_start=reb.date().isoformat(),
                    book_return=week_ret,
                    spy_return=spy_ret,
                    n_names_traded=n_traded,
                    cost=cost_total,
                    pv_pre=pv_pre,
                )
            )

        if (i + 1) % 100 == 0:
            log(f"...{i+1}/{len(rebals)} rebalances processed")

    log(f"rebalances with a priced week: {len(weekly_rows)}; "
        f"rebalances with a valid 20-name universe: {n_rebals_used}; "
        f"write-off events: {writeoff_count}")

    wk = pd.DataFrame(weekly_rows)
    wk["week_start"] = pd.to_datetime(wk["week_start"])
    wk = wk.sort_values("week_start").reset_index(drop=True)
    wk["book_value"] = STARTING_CAPITAL * (1 + wk["book_return"]).cumprod()
    wk["spy_value"] = STARTING_CAPITAL * (1 + wk["spy_return"]).cumprod()
    wk.to_csv(OUT_WEEKLY, index=False)
    log(f"wrote {OUT_WEEKLY} ({len(wk)} rows)")

    # ---- by-year summary ----
    wk["year"] = wk["week_start"].dt.year
    rows = []
    for yr, sub in wk.groupby("year"):
        book_yr = (1 + sub["book_return"]).prod() - 1
        spy_yr = (1 + sub["spy_return"]).prod() - 1
        rows.append(
            dict(
                year=yr,
                book_pct=book_yr * 100,
                spy_pct=spy_yr * 100,
                n_weeks=len(sub),
                avg_names_traded=sub["n_names_traded"].mean(),
                cost_dollars=sub["cost"].sum(),
                beats_spy=book_yr > spy_yr,
            )
        )
    by_year = pd.DataFrame(rows)
    by_year.to_csv(OUT_BY_YEAR, index=False)
    log(f"wrote {OUT_BY_YEAR} ({len(by_year)} rows)")

    # ---- headline stats ----
    n_weeks = len(wk)
    years = n_weeks / 52.0
    book_cagr = (wk["book_value"].iloc[-1] / STARTING_CAPITAL) ** (1 / years) - 1 if years > 0 else np.nan
    spy_cagr = (wk["spy_value"].iloc[-1] / STARTING_CAPITAL) ** (1 / years) - 1 if years > 0 else np.nan
    book_sharpe = wk["book_return"].mean() / wk["book_return"].std() * np.sqrt(52)
    spy_sharpe = wk["spy_return"].mean() / wk["spy_return"].std() * np.sqrt(52)

    def max_dd(vals):
        peak = vals.cummax()
        dd = vals / peak - 1
        return dd.min()

    book_dd = max_dd(wk["book_value"])
    spy_dd = max_dd(wk["spy_value"])
    worst_year_book = by_year.loc[by_year["book_pct"].idxmin()]
    years_beat = int(by_year["beats_spy"].sum())
    total_years = len(by_year)
    avg_turnover = wk["n_names_traded"].mean()

    md = []
    md.append("# REBUILD_1700_sleeve — independent rebuild\n")
    md.append(f"Rebalances with valid universe: {n_rebals_used}; priced weeks: {n_weeks}; "
              f"write-off events: {writeoff_count}\n")
    md.append("| Year | Book % | SPY % | Book $ (from $50K) | SPY $ (from $50K) | Beats SPY |")
    md.append("|---|---|---|---|---|---|")
    cum_book = STARTING_CAPITAL
    cum_spy = STARTING_CAPITAL
    for _, r in by_year.iterrows():
        cum_book *= (1 + r["book_pct"] / 100)
        cum_spy *= (1 + r["spy_pct"] / 100)
        md.append(
            f"| {int(r['year'])} | {r['book_pct']:.1f}% | {r['spy_pct']:.1f}% | "
            f"${cum_book:,.0f} | ${cum_spy:,.0f} | {'Y' if r['beats_spy'] else 'N'} |"
        )
    md.append("")
    md.append(f"- Book CAGR: {book_cagr*100:.2f}%  |  SPY CAGR: {spy_cagr*100:.2f}%")
    md.append(f"- Book Sharpe (weekly, annualized): {book_sharpe:.2f}  |  SPY Sharpe: {spy_sharpe:.2f}")
    md.append(f"- Book max drawdown: {book_dd*100:.1f}%  |  SPY max drawdown: {spy_dd*100:.1f}%")
    md.append(f"- Worst book year: {int(worst_year_book['year'])} ({worst_year_book['book_pct']:.1f}%)")
    md.append(f"- Avg names traded/week: {avg_turnover:.2f} (of {TOP_N} held)")
    md.append(f"- Avg weekly cost $: {wk['cost'].mean():.2f}; total cost $ over window: {wk['cost'].sum():,.0f}")
    md.append(f"- Years beating SPY: {years_beat}/{total_years}")
    md.append(f"- Delisting write-off events: {writeoff_count}")
    md.append(f"- Exact rebalance count (valid-universe weeks): {n_rebals_used}")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    log(f"wrote {OUT_MD}")
    log("DONE")


if __name__ == "__main__":
    main()
