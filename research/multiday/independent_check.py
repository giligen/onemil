#!/usr/bin/env python3
"""Independent reimplementation of research cells F2-LO-60 and A1-b.

Written from a prose specification only (no reference to run_f2_a1.py /
build_panel.py / out_f2a1*), to allow a trade-by-trade cross-check.

Node rules honoured: single foreground process, column-pruned parquet reads,
no cache.db / config touched, outputs confined to out_indep/.

Usage:
    nice -n 10 python3 independent_check.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.environ.get("INDEP_OUT", os.path.join(BASE, "out_indep"))

# ---------------------------------------------------------------- constants
NOTIONAL = 3300.0            # $66,000 book / 20 slots
FIXED_COST_BPS = 0.4         # commission/half-spread floor, round trip
ADV_MIN = 1_000_000.0        # $1M ADV20$ gate
PRICE_MIN = 5.0              # raw close gate
F2_HOLD = 60                 # sessions held (entry S+1 -> exit S+1+60)
F2_MIN_REF = 200             # minimum size of the causal ranking reference set
F2_REF_DAYS = 250            # trailing calendar-day window for the reference
A1_LEAD = 5                  # buy 5 sessions before expected announcement
A1_CTRL_OFFSET = 31          # control window shift (sessions)
A1_YEAR_DAYS = 364           # calendar days to the expected next announcement
A1_REQUIRED_INTERVENING = 3  # quarters that must have landed before entry

# Years loaded.  Entries are always restricted to <= 2023-12-31 (2024+ sealed).
# The "sealed" variant additionally truncates the trading calendar at
# 2023-12-31 so that any trade needing a 2024 session is dropped.
# The "complete" variant lets an already-committed 2023 entry finish its hold
# using 2024 prices (no 2024 signal, no 2024 entry, no 2024 statistics).
YEARS = list(range(2016, 2025))
SEAL_DATE = np.datetime64("2023-12-31")

SPLITS = {
    "TRAIN": (np.datetime64("2016-01-01"), np.datetime64("2021-12-31")),
    "VAL": (np.datetime64("2022-01-01"), np.datetime64("2023-12-31")),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ loading
def load_universe() -> pd.DataFrame:
    return pq.read_table(os.path.join(DATA, "universe.parquet"),
                         columns=["symbol", "kind"]).to_pandas()


def load_events() -> pd.DataFrame:
    ev = pq.read_table(
        os.path.join(DATA, "earnings_events.parquet"),
        columns=["symbol", "event_session", "acceptance_utc", "acceptance_bucket"],
    ).to_pandas()
    ev = ev.dropna(subset=["symbol", "event_session"])
    return ev


def load_panel(symbols: set[str]):
    """Build (symbols, sessions) matrices of adjusted close, raw close, dollar vol.

    Reads only the columns needed, one year-file at a time.
    """
    def sym_codes(df, mapping):
        """Map a dictionary-encoded symbol column to panel row indices (-1 = drop)."""
        cat = df["symbol"]
        if str(cat.dtype) == "category":
            cats = np.array([mapping.get(c, -1) for c in cat.cat.categories], dtype=np.int64)
            codes = cat.cat.codes.to_numpy()
            out = np.where(codes >= 0, cats[np.maximum(codes, 0)], -1)
            return out
        return np.array([mapping.get(s, -1) for s in cat.to_numpy()], dtype=np.int64)

    # Pass 1: the trading-session calendar := distinct dates where SPY has a row
    # in the ADJUSTED panel.
    all_dates = []
    for y in YEARS:
        t = pq.read_table(os.path.join(DATA, f"prices_by_year/all/year={y}.parquet"),
                          columns=["symbol", "date"])
        df = t.to_pandas(date_as_object=False)
        del t
        si = sym_codes(df, {"SPY": 0})
        all_dates.append(df["date"].to_numpy()[si == 0])
        del df, si
    sessions = np.unique(np.concatenate(all_dates)).astype("datetime64[ns]")
    sess_index = {d: i for i, d in enumerate(sessions)}
    n_s = len(sessions)

    keep = set(symbols) | {"SPY"}
    sym_list = sorted(keep)
    sym_index = {s: i for i, s in enumerate(sym_list)}
    n_y = len(sym_list)
    log(f"panel: {n_y} symbols x {n_s} sessions")

    adj = np.full((n_y, n_s), np.nan, dtype=np.float32)
    raw = np.full((n_y, n_s), np.nan, dtype=np.float32)
    dv = np.full((n_y, n_s), np.nan, dtype=np.float64)

    def date_codes(dvals):
        pos = np.searchsorted(sessions, dvals.astype("datetime64[ns]"), side="left")
        pos_c = np.minimum(pos, n_s - 1)
        good = sessions[pos_c] == dvals.astype("datetime64[ns]")
        return np.where(good, pos_c, -1)

    for y in YEARS:
        t = pq.read_table(os.path.join(DATA, f"prices_by_year/all/year={y}.parquet"),
                          columns=["symbol", "date", "close", "vwap", "volume"])
        df = t.to_pandas(date_as_object=False)
        del t
        si = sym_codes(df, sym_index)
        ti = date_codes(df["date"].to_numpy())
        ok = (si >= 0) & (ti >= 0)
        adj[si[ok], ti[ok]] = df["close"].to_numpy(dtype=np.float32)[ok]
        dv[si[ok], ti[ok]] = (df["vwap"].to_numpy(dtype=np.float64)[ok]
                              * df["volume"].to_numpy(dtype=np.float64)[ok])
        del df, si, ti, ok

        t = pq.read_table(os.path.join(DATA, f"prices_by_year/raw/year={y}.parquet"),
                          columns=["symbol", "date", "close"])
        df = t.to_pandas(date_as_object=False)
        del t
        si = sym_codes(df, sym_index)
        ti = date_codes(df["date"].to_numpy())
        ok = (si >= 0) & (ti >= 0)
        raw[si[ok], ti[ok]] = df["close"].to_numpy(dtype=np.float32)[ok]
        del df, si, ti, ok
        log(f"  loaded {y}")

    return sym_list, sym_index, sessions, sess_index, adj, raw, dv


def rolling_adv(dv: np.ndarray, window: int = 20, min_obs: int = 10) -> np.ndarray:
    """Mean of dollar volume over the `window` sessions ending at and including t.

    Requires >= min_obs non-missing observations in the window, else NaN.
    """
    filled = np.nan_to_num(dv, nan=0.0)
    present = (~np.isnan(dv)).astype(np.float64)
    zeros = np.zeros((dv.shape[0], 1), dtype=np.float64)
    cs = np.concatenate([zeros, np.cumsum(filled, axis=1)], axis=1)
    cc = np.concatenate([zeros, np.cumsum(present, axis=1)], axis=1)
    n_s = dv.shape[1]
    idx = np.arange(n_s)
    lo = np.maximum(idx - window + 1, 0)
    s = cs[:, idx + 1] - cs[:, lo]
    c = cc[:, idx + 1] - cc[:, lo]
    adv = np.where(c >= min_obs, s / np.maximum(c, 1.0), np.nan)
    return adv


def daily_returns(adj: np.ndarray) -> np.ndarray:
    """r[t] = adj[t]/adj[t-1] - 1; missing on either side -> 0.0 (frozen)."""
    r = np.zeros_like(adj, dtype=np.float64)
    prev = adj[:, :-1].astype(np.float64)
    cur = adj[:, 1:].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = cur / prev - 1.0
    rr[~np.isfinite(rr)] = 0.0
    r[:, 1:] = rr
    return r


# -------------------------------------------------------------------- costs
def impact_bps(adv: float) -> float:
    return 10.0 * min(NOTIONAL / max(0.01 * adv, 1.0), 1.0)


def roundtrip_cost_bps(adv_in: float, adv_out: float) -> float:
    if not np.isfinite(adv_out):
        adv_out = adv_in
    return impact_bps(adv_in) + impact_bps(adv_out) + FIXED_COST_BPS


# --------------------------------------------------------------- portfolios
@dataclass
class Position:
    sym: int
    a: int          # entry session index (bought at this close)
    b: int          # exit session index (sold at this close)
    cost_bps: float


def daily_series(positions, r: np.ndarray, n_s: int) -> np.ndarray:
    """Overlapping equal-weighted daily portfolio return series.

    A position entered at a and exited at b contributes its daily return on
    sessions a+1..b.  Half the round-trip cost is charged as a return drag on
    the first contributing session and half on the last.
    """
    num = np.zeros(n_s, dtype=np.float64)
    cnt = np.zeros(n_s, dtype=np.float64)
    for p in positions:
        a, b = p.a, p.b
        if b <= a:
            continue
        num[a + 1:b + 1] += r[p.sym, a + 1:b + 1]
        cnt[a + 1:b + 1] += 1.0
        half = p.cost_bps / 2.0 / 1e4
        num[a + 1] -= half
        num[b] -= half
    out = np.zeros(n_s, dtype=np.float64)
    nz = cnt > 0
    out[nz] = num[nz] / cnt[nz]
    return out


def monthly(series: np.ndarray, sessions: np.ndarray) -> pd.Series:
    s = pd.Series(series, index=pd.DatetimeIndex(sessions))
    return s.groupby([s.index.year, s.index.month]).apply(lambda x: float(np.prod(1.0 + x.values) - 1.0))


def newey_west_t(x: np.ndarray, lag: int) -> float:
    """t-stat of the mean with a Newey-West (Bartlett) HAC variance.

    Overlapping holds make adjacent monthly returns correlated; the plain
    iid t-stat in the spec is inflated.  Reported alongside it.
    """
    n = len(x)
    if n < 3:
        return np.nan
    m = x.mean()
    e = x - m
    s = float(np.dot(e, e) / n)
    for l in range(1, min(lag, n - 1) + 1):
        g = float(np.dot(e[l:], e[:-l]) / n)
        s += 2.0 * (1.0 - l / (lag + 1.0)) * g
    if s <= 0:
        return np.nan
    return m / np.sqrt(s / n)


def excess_stats(m_book: pd.Series, m_bench: pd.Series, nw_lag: int = 3) -> dict:
    ex = (m_book - m_bench).dropna()
    ex = ex[ex != 0.0]
    n = len(ex)
    if n == 0:
        return dict(n_months=0, mean_bps=np.nan, t_stat=np.nan, pct_pos=np.nan,
                    t_nw=np.nan, series=ex)
    mean = ex.mean()
    sd = ex.std(ddof=1)
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    return dict(n_months=n, mean_bps=mean * 1e4, t_stat=t,
                pct_pos=100.0 * float((ex > 0).mean()),
                t_nw=newey_west_t(ex.to_numpy(), nw_lag), series=ex)


def gross_return(r: np.ndarray, sym: int, a: int, b: int) -> float:
    if b <= a:
        return 0.0
    return float(np.prod(1.0 + r[sym, a + 1:b + 1]) - 1.0)


# ---------------------------------------------------------------- cell F2
def build_f2(events, adj, raw, adv, r, sessions, sym_index, spy_idx, n_eff):
    """Return a DataFrame of ranked F2 candidate trades (all deciles)."""
    ev = events.sort_values(["s_idx", "symbol"], kind="mergesort").reset_index(drop=True)
    s_idx = ev["s_idx"].to_numpy()
    sym = ev["sym_idx"].to_numpy()
    dates = sessions

    # step 1+2: signal + eligibility at S
    ok = s_idx >= 2
    S = s_idx
    a_S = np.where(ok, adj[sym, np.clip(S, 0, None)], np.nan).astype(np.float64)
    a_S2 = np.where(ok, adj[sym, np.clip(S - 2, 0, None)], np.nan).astype(np.float64)
    spy_S = adj[spy_idx, np.clip(S, 0, None)].astype(np.float64)
    spy_S2 = adj[spy_idx, np.clip(S - 2, 0, None)].astype(np.float64)
    rawc = np.where(ok, raw[sym, np.clip(S, 0, None)], np.nan).astype(np.float64)
    advS = np.where(ok, adv[sym, np.clip(S, 0, None)], np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        r2_abn = (a_S / a_S2 - 1.0) - (spy_S / spy_S2 - 1.0)

    keep = (ok & np.isfinite(a_S) & np.isfinite(a_S2) & np.isfinite(rawc)
            & np.isfinite(advS) & np.isfinite(r2_abn)
            & (rawc >= PRICE_MIN) & (advS >= ADV_MIN))

    # AMBIGUITY: the spec ranks against "every already-processed event".  Default
    # reading = events that survived steps 1-2 (signal computable AND eligible).
    # F2_REF_POP=all ranks against every event with a computable signal, gated or not.
    ref_pop = os.environ.get("F2_REF_POP", "gated")
    signal_ok = (ok & np.isfinite(a_S) & np.isfinite(a_S2) & np.isfinite(r2_abn))
    if ref_pop == "all":
        ranking_mask = signal_ok
    else:
        ranking_mask = keep

    ev_all = ev.copy()
    ev_all["r2_abn"] = r2_abn
    ev_all["adv_S"] = advS
    ev_all["eligible"] = keep
    ev = ev_all.loc[ranking_mask].copy()
    ev = ev.sort_values(["s_idx", "symbol"], kind="mergesort").reset_index(drop=True)
    log(f"F2: ranking population={ref_pop}, {int(keep.sum())} eligible, "
        f"{len(ev)} in the ranking population")

    # step 3: causal decile assignment over a trailing 250-calendar-day window
    S = ev["s_idx"].to_numpy()
    vals = ev["r2_abn"].to_numpy()
    sdates = dates[S]
    deciles = np.full(len(ev), -1, dtype=np.int64)

    lo = 0
    grp_start = 0
    n = len(ev)
    while grp_start < n:
        grp_end = grp_start
        cur_s = S[grp_start]
        while grp_end < n and S[grp_end] == cur_s:
            grp_end += 1
        cutoff = sdates[grp_start] - np.timedelta64(F2_REF_DAYS, "D")
        while lo < grp_start and sdates[lo] < cutoff:
            lo += 1
        ref = vals[lo:grp_start]
        if len(ref) >= F2_MIN_REF:
            cuts = np.percentile(ref, np.arange(10, 100, 10))
            deciles[grp_start:grp_end] = np.searchsorted(cuts, vals[grp_start:grp_end], side="left")
        grp_start = grp_end

    ev["decile"] = deciles
    ev = ev[(ev["decile"] >= 0) & ev["eligible"]].copy().reset_index(drop=True)
    log(f"F2: {len(ev)} events ranked+eligible (reference >= {F2_MIN_REF})")

    # step 4: entry at close of S+1, exit at close of S+1+60
    S = ev["s_idx"].to_numpy()
    a = S + 1
    b = S + 1 + F2_HOLD
    alive = b <= (n_eff - 1)
    ev = ev.loc[alive].copy()
    a, b = a[alive], b[alive]
    sym = ev["sym_idx"].to_numpy()

    adv_in = adv[sym, a]
    adv_in = np.where(np.isfinite(adv_in), adv_in, ev["adv_S"].to_numpy())
    adv_out = adv[sym, b]
    cost = np.array([roundtrip_cost_bps(i, o) for i, o in zip(adv_in, adv_out)])
    gross = np.array([gross_return(r, s_, aa, bb) for s_, aa, bb in zip(sym, a, b)])

    out = pd.DataFrame({
        "symbol": ev["symbol"].to_numpy(),
        "sym_idx": sym,
        "event_session_index": S[alive],
        "entry_session_index": a,
        "exit_session_index": b,
        "entry_date": dates[a],
        "exit_date": dates[b],
        "decile": ev["decile"].to_numpy(),
        "gross": gross,
        "cost_bps": cost,
    })
    return out


# ---------------------------------------------------------------- cell A1
def build_a1(events, adj, raw, adv, r, sessions, n_eff):
    dates = sessions
    rows = []
    att = dict(candidates=0, drop_end=0, drop_entry_le_anchor=0,
               drop_intervening_lt3=0, drop_intervening_gt3=0,
               drop_missing=0, drop_gate=0, kept=0, ctrl_missing=0)
    for symbol, g in events.groupby("symbol", sort=True):
        g = g.sort_values("s_idx", kind="mergesort")
        S = g["s_idx"].to_numpy()
        sym = int(g["sym_idx"].iloc[0])
        if len(S) < 5:
            continue
        for k in range(4, len(S)):
            att["candidates"] += 1
            target = dates[S[k]] + np.timedelta64(A1_YEAR_DAYS, "D")
            ehat = int(np.searchsorted(dates, target, side="left"))
            if ehat > n_eff - 1 - 2:      # "within 2 sessions of the end"
                att["drop_end"] += 1
                continue
            entry = ehat - A1_LEAD
            if entry <= S[k]:
                att["drop_entry_le_anchor"] += 1
                continue
            # causality: exactly 3 of this symbol's events in (S[k], entry]
            nb = int(np.sum((S > S[k]) & (S <= entry)))
            if nb != A1_REQUIRED_INTERVENING:
                att["drop_intervening_lt3" if nb < 3 else "drop_intervening_gt3"] += 1
                continue
            rc = raw[sym, entry]
            av = adv[sym, entry]
            if not np.isfinite(rc) or not np.isfinite(av):
                att["drop_missing"] += 1
                continue
            if rc < PRICE_MIN or av < ADV_MIN:
                att["drop_gate"] += 1
                continue
            att["kept"] += 1
            adv_out = adv[sym, ehat]
            cost = roundtrip_cost_bps(float(av), float(adv_out))
            gross = gross_return(r, sym, entry, ehat)
            ca, cb = entry - A1_CTRL_OFFSET, ehat - A1_CTRL_OFFSET
            if ca < 0:
                att["ctrl_missing"] += 1
                cgross, ca_o, cb_o = np.nan, -1, -1
            else:
                cgross, ca_o, cb_o = gross_return(r, sym, ca, cb), ca, cb
            rows.append((symbol, sym, int(S[k]), entry, ehat, dates[entry], dates[ehat],
                         gross, cost, ca_o, cb_o, cgross))
    out = pd.DataFrame(rows, columns=[
        "symbol", "sym_idx", "anchor_session_index", "entry_session_index",
        "exit_session_index", "entry_date", "exit_date", "gross", "cost_bps",
        "ctrl_entry_index", "ctrl_exit_index", "ctrl"])
    log("A1 attrition: " + ", ".join(f"{k}={v}" for k, v in att.items()))
    return out


# ------------------------------------------------------------------- driver
def split_mask(dates: np.ndarray, split: str) -> np.ndarray:
    lo, hi = SPLITS[split]
    d = dates.astype("datetime64[D]")
    return (d >= lo) & (d <= hi)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    uni = load_universe()
    ev_raw = load_events()
    common = set(uni.loc[uni["kind"] == "common", "symbol"])
    scored = sorted(common & set(ev_raw["symbol"].unique()))
    log(f"scored symbols: {len(scored)}")

    sym_list, sym_index, sessions, sess_index, adj, raw, dv = load_panel(set(scored))
    spy_idx = sym_index["SPY"]
    adv = rolling_adv(dv)
    del dv
    r = daily_returns(adj)
    n_all = len(sessions)
    n_sealed = int(np.searchsorted(sessions, SEAL_DATE.astype("datetime64[ns]"), side="right"))
    log(f"sessions total {n_all}; sealed calendar length {n_sealed} "
        f"(last {sessions[n_sealed-1]})")

    # de-duplicate events: one per (symbol, event_session), earliest acceptance
    ev = ev_raw[ev_raw["symbol"].isin(scored)].copy()
    ev = ev.sort_values(["symbol", "event_session", "acceptance_utc"], kind="mergesort")
    ev = ev.drop_duplicates(subset=["symbol", "event_session"], keep="first")
    es = ev["event_session"].to_numpy().astype("datetime64[ns]")
    pos = np.minimum(np.searchsorted(sessions, es, side="left"), len(sessions) - 1)
    ev["s_idx"] = np.where(sessions[pos] == es, pos, -1)
    ev = ev[ev["s_idx"] >= 0].copy()
    ev["sym_idx"] = ev["symbol"].map(sym_index)
    log(f"events after dedup + calendar map: {len(ev)}")

    results = []
    monthly_dump = []
    tail_rows = []
    trades_f2_all, trades_a1_all = [], []

    for variant, n_eff in (("sealed", n_sealed), ("complete", n_all)):
        log(f"=== variant {variant} (n_eff={n_eff}) ===")

        f2 = build_f2(ev, adj, raw, adv, r, sessions, sym_index, spy_idx, n_eff)
        f2["variant"] = variant
        log(f"F2 {variant}: {len(f2)} tradable ranked events")

        a1 = build_a1(ev, adj, raw, adv, r, sessions, n_eff)
        a1["variant"] = variant
        log(f"A1 {variant}: {len(a1)} trades")

        for split in ("TRAIN", "VAL"):
            # ---- F2
            m = split_mask(f2["entry_date"].to_numpy(), split)
            sub = f2.loc[m]
            d9 = sub[sub["decile"] == 9]
            d0 = sub[sub["decile"] == 0]

            def pos(df):
                return [Position(int(s), int(a), int(b), float(c)) for s, a, b, c in
                        zip(df["sym_idx"], df["entry_session_index"],
                            df["exit_session_index"], df["cost_bps"])]

            # tail-dependence check: same book with the top 1% of winners removed
            if len(d9):
                cut = d9["gross"].quantile(0.99)
                d9_trim = d9[d9["gross"] < cut]
                st_t = excess_stats(monthly(daily_series(pos(d9_trim), r, n_all), sessions),
                                    monthly(daily_series(pos(sub), r, n_all), sessions), 3)
                tail_rows.append(dict(cell="F2-LO-60", split=split, variant=variant,
                                      test="drop_top_1pct_winners", n_trades=len(d9_trim),
                                      excess_mean_bps=st_t["mean_bps"], excess_t=st_t["t_stat"],
                                      pct_months_pos=st_t["pct_pos"]))

            m_d9 = monthly(daily_series(pos(d9), r, n_all), sessions)
            m_d0 = monthly(daily_series(pos(d0), r, n_all), sessions)
            m_all = monthly(daily_series(pos(sub), r, n_all), sessions)
            st = excess_stats(m_d9, m_all, nw_lag=3)
            st0 = excess_stats(m_d0, m_all, nw_lag=3)
            monthly_dump.append(st["series"].rename("excess").reset_index().assign(
                cell="F2-LO-60", split=split, variant=variant))
            results.append(dict(
                cell="F2-LO-60", split=split, variant=variant,
                n_trades=len(d9), n_trades_bench=len(sub),
                mean_gross_bps=d9["gross"].mean() * 1e4 if len(d9) else np.nan,
                mean_cost_bps=d9["cost_bps"].mean() if len(d9) else np.nan,
                mean_bench_gross_bps=sub["gross"].mean() * 1e4 if len(sub) else np.nan,
                n_months=st["n_months"], excess_mean_bps=st["mean_bps"],
                excess_t=st["t_stat"], excess_t_nw=st["t_nw"],
                pct_months_pos=st["pct_pos"],
                d0_n_months=st0["n_months"], d0_excess_mean_bps=st0["mean_bps"],
                d0_excess_t=st0["t_stat"], d0_pct_months_pos=st0["pct_pos"],
                ctrl_mean_bps=np.nan, gross_minus_ctrl_bps=np.nan,
            ))

            # ---- A1
            m = split_mask(a1["entry_date"].to_numpy(), split)
            sub = a1.loc[m]
            real_pos = [Position(int(s), int(a), int(b), float(c)) for s, a, b, c in
                        zip(sub["sym_idx"], sub["entry_session_index"],
                            sub["exit_session_index"], sub["cost_bps"])]
            cs = sub[sub["ctrl_entry_index"] >= 0]
            ctrl_pos = [Position(int(s), int(a), int(b), float(c)) for s, a, b, c in
                        zip(cs["sym_idx"], cs["ctrl_entry_index"],
                            cs["ctrl_exit_index"], cs["cost_bps"])]
            if len(sub):
                diff = (sub["gross"] - sub["ctrl"]).fillna(sub["gross"])
                keep_t = diff < diff.quantile(0.99)
                st_t = excess_stats(
                    monthly(daily_series([Position(int(s), int(a), int(b), float(c))
                                          for s, a, b, c in
                                          zip(sub.loc[keep_t, "sym_idx"],
                                              sub.loc[keep_t, "entry_session_index"],
                                              sub.loc[keep_t, "exit_session_index"],
                                              sub.loc[keep_t, "cost_bps"])], r, n_all), sessions),
                    monthly(daily_series([Position(int(s), int(a), int(b), float(c))
                                          for s, a, b, c in
                                          zip(sub.loc[keep_t, "sym_idx"],
                                              sub.loc[keep_t, "ctrl_entry_index"],
                                              sub.loc[keep_t, "ctrl_exit_index"],
                                              sub.loc[keep_t, "cost_bps"])
                                          if a >= 0], r, n_all), sessions), 1)
                tail_rows.append(dict(cell="A1-b", split=split, variant=variant,
                                      test="drop_top_1pct_winners", n_trades=int(keep_t.sum()),
                                      excess_mean_bps=st_t["mean_bps"], excess_t=st_t["t_stat"],
                                      pct_months_pos=st_t["pct_pos"]))

            m_real = monthly(daily_series(real_pos, r, n_all), sessions)
            m_ctrl = monthly(daily_series(ctrl_pos, r, n_all), sessions)
            st = excess_stats(m_real, m_ctrl, nw_lag=1)
            monthly_dump.append(st["series"].rename("excess").reset_index().assign(
                cell="A1-b", split=split, variant=variant))
            results.append(dict(
                cell="A1-b", split=split, variant=variant,
                n_trades=len(sub), n_trades_bench=len(cs),
                mean_gross_bps=sub["gross"].mean() * 1e4 if len(sub) else np.nan,
                mean_cost_bps=sub["cost_bps"].mean() if len(sub) else np.nan,
                mean_bench_gross_bps=np.nan,
                n_months=st["n_months"], excess_mean_bps=st["mean_bps"],
                excess_t=st["t_stat"], excess_t_nw=st["t_nw"],
                pct_months_pos=st["pct_pos"],
                d0_n_months=np.nan, d0_excess_mean_bps=np.nan,
                d0_excess_t=np.nan, d0_pct_months_pos=np.nan,
                ctrl_mean_bps=sub["ctrl"].mean() * 1e4 if len(sub) else np.nan,
                gross_minus_ctrl_bps=((sub["gross"] - sub["ctrl"]).mean() * 1e4
                                      if len(sub) else np.nan),
            ))

        trades_f2_all.append(f2)
        trades_a1_all.append(a1)

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OUT, "indep_cells.csv"), index=False)
    if tail_rows:
        pd.DataFrame(tail_rows).to_csv(os.path.join(OUT, "indep_tail_test.csv"), index=False)
    if monthly_dump:
        md = pd.concat(monthly_dump, ignore_index=True)
        md.columns = ["year", "month", "excess", "cell", "split", "variant"][:len(md.columns)]
        md.to_csv(os.path.join(OUT, "indep_monthly_excess.csv"), index=False)

    f2out = pd.concat(trades_f2_all, ignore_index=True)
    a1out = pd.concat(trades_a1_all, ignore_index=True)
    for df in (f2out, a1out):
        df["split"] = np.where(split_mask(df["entry_date"].to_numpy(), "TRAIN"), "TRAIN",
                               np.where(split_mask(df["entry_date"].to_numpy(), "VAL"),
                                        "VAL", "OUT"))
    f2out[f2out["split"] != "OUT"][
        ["symbol", "entry_session_index", "exit_session_index", "entry_date",
         "exit_date", "gross", "cost_bps", "decile", "split", "variant",
         "event_session_index"]
    ].to_csv(os.path.join(OUT, "indep_trades_F2LO60.csv"), index=False)
    a1out[a1out["split"] != "OUT"][
        ["symbol", "entry_session_index", "exit_session_index", "entry_date",
         "exit_date", "gross", "cost_bps", "ctrl", "split", "variant",
         "anchor_session_index", "ctrl_entry_index", "ctrl_exit_index"]
    ].to_csv(os.path.join(OUT, "indep_trades_A1b.csv"), index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    print(res.to_string(index=False))
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
