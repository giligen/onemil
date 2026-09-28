#!/usr/bin/env python3
"""Independent rebuild of cells 1,626-1,627 -- research/lev_decay/PREREG_1626.md (FROZEN).

Built from the PREREG prose ONLY (never opened cell_1626.py / cell_1626_days.csv / RESULT_1626.md).
Own pair-matching parser, own bars fetch, own P&L simulation.

Mechanism (Cheng & Madhavan 2009; Avellaneda & Zhang 2010): a daily-rebalanced Lx ETF loses
~0.5*(L^2-L)*sigma_daily^2 per day (volatility drag) relative to L x the underlying. Short both the
2x long and 2x short sides of the SAME single-stock underlying, dollar-balanced -> delta-neutral at
each rebalance, collects the drag of both legs minus borrow and trading cost.

Usage:
    python3 research/lev_decay/rebuild_1626.py pairs        # build + print pair list, hand-check sample
    python3 research/lev_decay/rebuild_1626.py fetch-bars   # Alpaca daily bars, resumable parquet cache
    python3 research/lev_decay/rebuild_1626.py fetch-flags  # Alpaca asset endpoint shortable/ETB flags
    python3 research/lev_decay/rebuild_1626.py score        # simulate cells 1626/1627, write outputs
    python3 research/lev_decay/rebuild_1626.py all          # run all stages in order
"""
import argparse
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ASSET_CSV = os.path.join(ROOT, "data/research/databento/alpaca_assets_all_20260905.csv")
BARS_PARQUET = os.path.join(HERE, "bars.parquet")
BARS_REBUILD_PARQUET = os.path.join(HERE, "bars_rebuild.parquet")
FLAGS_CSV = os.path.join(HERE, "asset_flags.csv")
PAIRS_CSV = os.path.join(HERE, "rebuild_1626_pairs.csv")
DAYS_CSV = os.path.join(HERE, "rebuild_1626_days.csv")
REPORT_MD = os.path.join(HERE, "REBUILD_1626.md")

sys.path.insert(0, os.path.join(ROOT, "research/hod_entry"))
from cell_1445 import day_clustered_t  # noqa: E402  (shared helper, reused per task instructions)

START_DATE = "2022-01-03"
END_DATE = "2026-09-26"
TRAIN_END = "2025-03-31"  # TRAIN 2022-01..2025-03
VAL_START = "2025-04-01"  # VAL 2025-04..2026-09
REBAL_SESSIONS = 5
LEG_COST_BPS = 5.0  # 5 bps per rebalance leg, on that leg's $1 target notional
BORROW_RAILS = {"5": 0.05, "15": 0.15, "30": 0.30}
GROSS_NOTIONAL = 2.0  # $1 short each leg
VOL_ENTER = 0.60  # annualised, 1627 gate
VOL_EXIT = 0.40
EXTREME_RET_CAP = 0.90  # |daily return| above this is treated as an uncorrected corporate-action
                         # artifact (split-adjusted bars still showed up to +2,589% single-day moves
                         # pre-fix, per CLAUDE.md's price-scale check) and neutralised to 0 for that
                         # symbol-day; every neutralisation is logged (see EXTREME_EVENTS).
EXTREME_EVENTS = []

# Company names spelled out in T-Rex-style names instead of the ticker.
NAME_TO_TICKER = {
    "APPLE": "AAPL", "MICROSOFT": "MSFT", "TESLA": "TSLA", "NVIDIA": "NVDA",
}
# Underlyings that are not single-stock equities even though the name pattern matches.
NON_EQUITY_UNDERLYINGS = {"BITCOIN", "ETHER", "ETHEREUM", "XRP", "SOLANA", "DOGECOIN"}

DIRECTION_WORDS = {"BULL": 1, "LONG": 1, "BEAR": -1, "SHORT": -1, "INVERSE": -1}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Stage 1: pair matching (own parser, independent of any prior implementation)
# --------------------------------------------------------------------------------------
def parse_leveraged_row(name):
    """Return (underlying_ticker, leverage_factor, direction) or None if `name` is not a
    recognisable single-stock leveraged/inverse daily ETF row. direction: +1 long, -1 short."""
    if not isinstance(name, str):
        return None
    upper = name.upper()
    lev_m = re.search(r"(\d+(?:\.\d+)?)\s*X\b", upper)
    if not lev_m:
        return None
    factor = float(lev_m.group(1))
    dir_words_present = [w for w in DIRECTION_WORDS if re.search(rf"\b{w}\b", upper)]
    if not dir_words_present:
        return None
    direction = DIRECTION_WORDS[dir_words_present[0]]

    candidate = None
    # Pattern A: "<TICKER> Bull|Bear <n>X" (Direxion style)
    m = re.search(r"\b([A-Z]{1,6})\s+(?:BULL|BEAR)\s+\d", upper)
    if m:
        candidate = m.group(1)
    # Pattern B: "Long|Short|Inverse <TICKER> Daily" (GraniteShares/Leverage Shares/Tradr/T-Rex)
    if candidate is None:
        m = re.search(r"\b(?:LONG|SHORT|INVERSE)\s+([A-Z]{1,6})\s+DAILY\b", upper)
        if m:
            candidate = m.group(1)
    # Pattern C: "Long|Short|Inverse <TICKER> ETF" (Defiance style, no "Daily" after ticker)
    if candidate is None:
        m = re.search(r"\b(?:LONG|SHORT|INVERSE)\s+([A-Z]{1,6})\s+ETF\b", upper)
        if m:
            candidate = m.group(1)
    # Pattern D: spelled-out company name instead of ticker
    if candidate is None:
        for word, tick in NAME_TO_TICKER.items():
            if re.search(rf"\b{word}\b", upper):
                candidate = tick
                break

    if candidate is None:
        return None
    if candidate in NON_EQUITY_UNDERLYINGS:
        return None
    STOP = {"DAILY", "TARGET", "ETF", "FUND", "TRUST", "SHARES", "LONG", "SHORT", "BULL",
            "BEAR", "INVERSE", "INDEX", "WEEKLY", "NEW", "UNIT", "SERIES", "DUE", "TIDAL",
            "II", "ETN", "ETNS"}
    if candidate in STOP:
        return None
    return candidate, factor, direction


def build_pairs():
    """Independent pair-matching: parse the asset CSV, validate each candidate underlying
    against its own common-stock row, group by (underlying, leverage), cross long x short."""
    assets = pd.read_csv(ASSET_CSV, dtype=str, keep_default_na=False)
    assets["common_b"] = assets["common"].astype(str).str.upper() == "TRUE"
    common_by_symbol = assets.loc[assets["common_b"]].set_index("symbol")["name"].to_dict()

    rows = []
    dropped_no_underlying = []
    for _, r in assets.iterrows():
        parsed = parse_leveraged_row(r["name"])
        if parsed is None:
            continue
        underlying, factor, direction = parsed
        if underlying == r["symbol"]:
            continue  # a common-stock row cannot also be its own leveraged product
        if underlying not in common_by_symbol:
            dropped_no_underlying.append((r["symbol"], r["name"], underlying))
            continue
        rows.append({
            "symbol": r["symbol"], "name": r["name"], "underlying": underlying,
            "leverage": factor, "direction": direction, "status": r["status"],
            "tradable": r["tradable"],
        })
    cand = pd.DataFrame(rows).drop_duplicates(subset=["symbol"])
    log(f"parsed {len(cand)} leveraged single-stock candidate rows "
        f"({len(dropped_no_underlying)} dropped: underlying ticker not a common-stock row)")

    pairs = []
    for (underlying, factor), g in cand.groupby(["underlying", "leverage"]):
        longs = g[g["direction"] == 1]
        shorts = g[g["direction"] == -1]
        if longs.empty or shorts.empty:
            continue

        def pick(df):
            act = df[(df["status"] == "active") & (df["tradable"].astype(str) == "True")]
            return act if not act.empty else df

        longs, shorts = pick(longs), pick(shorts)
        for _, lr in longs.iterrows():
            for _, sr in shorts.iterrows():
                pairs.append({
                    "underlying": underlying, "leverage": factor,
                    "long_symbol": lr["symbol"], "long_name": lr["name"],
                    "short_symbol": sr["symbol"], "short_name": sr["name"],
                    "both_active": (lr["status"] == "active") and (sr["status"] == "active"),
                })
    pairs_df = pd.DataFrame(pairs).drop_duplicates(subset=["underlying", "leverage", "long_symbol", "short_symbol"])
    pairs_df = pairs_df.sort_values(["leverage", "underlying"]).reset_index(drop=True)
    pairs_df.to_csv(PAIRS_CSV, index=False)
    log(f"built {len(pairs_df)} pairs across {pairs_df['underlying'].nunique()} underlyings "
        f"-> {PAIRS_CSV}")
    log(f"leverage factor counts:\n{pairs_df['leverage'].value_counts().to_string()}")

    sample = pairs_df.sample(n=min(20, len(pairs_df)), random_state=1626).sort_values("underlying")
    log("HAND-CHECK sample (20 pairs):")
    for _, p in sample.iterrows():
        log(f"  {p['underlying']:6s} {p['leverage']:.2f}x  "
            f"LONG {p['long_symbol']:6s} ({p['long_name'][:60]})  |  "
            f"SHORT {p['short_symbol']:6s} ({p['short_name'][:60]})")
    return pairs_df


# --------------------------------------------------------------------------------------
# Stage 2: Alpaca daily bars (resumable parquet cache)
# --------------------------------------------------------------------------------------
def fetch_bars(symbols):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET missing from .env -- cannot fetch bars")
    client = StockHistoricalDataClient(key, secret)

    existing = pd.DataFrame()
    if os.path.exists(BARS_REBUILD_PARQUET):
        existing = pd.read_parquet(BARS_REBUILD_PARQUET)
        log(f"resuming: {existing['symbol'].nunique()} symbols already cached in {BARS_REBUILD_PARQUET}")
    done = set(existing["symbol"].unique()) if len(existing) else set()
    todo = [s for s in symbols if s not in done]
    log(f"{len(done)} symbols cached, {len(todo)} to fetch, batches of 10")

    frames = [existing] if len(existing) else []
    lost = []
    for i in range(0, len(todo), 10):
        batch = todo[i:i + 10]
        try:
            req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                                    start=START_DATE, end=END_DATE, feed="iex", adjustment="split")
            resp = client.get_stock_bars(req)
            df = resp.df
            if df is None or df.empty:
                lost.extend(batch)
                log(f"batch {i // 10}: EMPTY for {batch}")
                continue
            df = df.reset_index()
            df = df.rename(columns={"timestamp": "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            got = set(df["symbol"].unique())
            batch_lost = [s for s in batch if s not in got]
            lost.extend(batch_lost)
            frames.append(df[["symbol", "date", "open", "high", "low", "close", "volume"]])
            log(f"batch {i // 10}: {len(got)}/{len(batch)} ok, {len(batch_lost)} LOST {batch_lost}")
        except Exception as e:
            lost.extend(batch)
            log(f"batch {i // 10}: ERROR {e} -- {batch} counted LOST")
        if (i // 10) % 5 == 0 and frames:
            pd.concat(frames, ignore_index=True).drop_duplicates(
                subset=["symbol", "date"]).to_parquet(BARS_REBUILD_PARQUET)

    all_bars = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["symbol", "date"]) if frames else existing
    all_bars.to_parquet(BARS_REBUILD_PARQUET)
    log(f"LOST total: {len(lost)}/{len(symbols)} symbols with zero bars returned: {sorted(set(lost))}")
    log(f"saved {len(all_bars)} rows, {all_bars['symbol'].nunique()} symbols -> {BARS_REBUILD_PARQUET}")
    return all_bars


# --------------------------------------------------------------------------------------
# Stage 3: Alpaca asset endpoint -- shortable / easy_to_borrow flags
# --------------------------------------------------------------------------------------
def fetch_flags(symbols):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass

    # Use the ORB paper-account credentials (ALPACA_ORB_API_KEY/SECRET, ALPACA_ORB_PAPER=true) for
    # this call, never the live-account keys: asset shortable/ETB metadata is exchange-level (not
    # account-specific), and this stays off the shared live account's endpoint entirely -- read-only
    # or not -- per the owner's manual-trades-untouchable rule and this task's "never place orders".
    key, secret = os.environ.get("ALPACA_ORB_API_KEY"), os.environ.get("ALPACA_ORB_API_SECRET")
    if not key or not secret:
        raise RuntimeError("ALPACA_ORB_API_KEY / ALPACA_ORB_API_SECRET missing from .env -- cannot fetch asset flags")
    client = TradingClient(key, secret, paper=True)
    req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
    all_assets = client.get_all_assets(req)
    log(f"asset endpoint returned {len(all_assets)} total us_equity assets (today's snapshot)")
    wanted = set(symbols)
    rows = [{"symbol": a.symbol, "status": str(a.status), "tradable": a.tradable,
             "shortable": a.shortable, "easy_to_borrow": a.easy_to_borrow,
             "marginable": a.marginable}
            for a in all_assets if a.symbol in wanted]
    df = pd.DataFrame(rows)
    df.to_csv(FLAGS_CSV, index=False)
    found = set(df["symbol"]) if len(df) else set()
    missing = wanted - found
    log(f"{len(df)}/{len(wanted)} pair-member symbols matched on the asset endpoint "
        f"({len(missing)} missing/delisted from today's snapshot: {sorted(missing)[:20]}...)")
    log(f"saved -> {FLAGS_CSV}")
    return df


# --------------------------------------------------------------------------------------
# Stage 4: simulation + scoring
# --------------------------------------------------------------------------------------
def split_of(date):
    if date <= pd.Timestamp(TRAIN_END):
        return "TRAIN"
    if date >= pd.Timestamp(VAL_START):
        return "VAL"
    return "GAP"  # none expected: TRAIN ends 2025-03-31, VAL starts 2025-04-01


def simulate_pair(underlying, leverage, long_sym, short_sym, bars):
    """Return a per-day DataFrame for this pair: gross/net bps (1626, all 3 rails) and the
    1627 vol-gated variant, plus the vol tercile / theory-drag inputs."""
    bl = bars[bars.symbol == long_sym][["date", "close"]].rename(columns={"close": "c_long"})
    bs = bars[bars.symbol == short_sym][["date", "close"]].rename(columns={"close": "c_short"})
    bu = bars[bars.symbol == underlying][["date", "close"]].rename(columns={"close": "c_under"})
    if bl.empty or bs.empty or bu.empty:
        return None
    df = bl.merge(bs, on="date").merge(bu, on="date").sort_values("date").reset_index(drop=True)
    df = df[(df.date >= pd.Timestamp(START_DATE)) & (df.date <= pd.Timestamp(END_DATE))]
    if len(df) < 30:
        return None
    df["r_long"] = df["c_long"].pct_change()
    df["r_short"] = df["c_short"].pct_change()
    df["r_under"] = df["c_under"].pct_change()
    for col, sym in (("r_long", long_sym), ("r_short", short_sym), ("r_under", underlying)):
        bad = df[col].abs() > EXTREME_RET_CAP
        for d, v in zip(df.loc[bad, "date"], df.loc[bad, col]):
            EXTREME_EVENTS.append((sym, str(d.date()), col, round(float(v), 3)))
        df.loc[bad, col] = 0.0
    df["sigma_daily"] = df["r_under"].rolling(20).std().shift(1)  # causal: known before day t's close
    df["vol_annual"] = df["sigma_daily"] * math.sqrt(252)
    df["theory_bps"] = 0.5 * (leverage ** 2 - leverage) * df["sigma_daily"] ** 2 * 10000
    df = df.iloc[21:].reset_index(drop=True)  # need 20d vol + 1 return warmup
    if df.empty:
        return None

    notional_l = notional_s = 1.0
    session_count = 0
    active_1627 = False
    session_count_1627 = 0
    notional_l27 = notional_s27 = 1.0
    out = []
    for _, row in df.iterrows():
        rl, rs = row["r_long"], row["r_short"]
        if pd.isna(rl) or pd.isna(rs):
            continue
        # ---- 1626: static, always on ----
        pnl = -(notional_l * rl) - (notional_s * rs)
        notional_l *= (1 + rl)
        notional_s *= (1 + rs)
        session_count += 1
        cost_dollar = 0.0
        if session_count >= REBAL_SESSIONS:
            cost_dollar = 2 * (LEG_COST_BPS / 10000.0) * 1.0
            notional_l, notional_s = 1.0, 1.0
            session_count = 0
        borrow_base = notional_l + notional_s
        gross_bps = pnl / GROSS_NOTIONAL * 10000
        net_bps = {}
        for rail_name, rail in BORROW_RAILS.items():
            borrow_dollar = borrow_base * rail / 252
            net_bps[rail_name] = (pnl - borrow_dollar - cost_dollar) / GROSS_NOTIONAL * 10000

        # ---- 1627: vol-gated (enter >=60% annualised, exit <40%), hysteresis ----
        gate_vol = row["vol_annual"]
        want_active = active_1627
        if not active_1627 and gate_vol >= VOL_ENTER:
            want_active = True
        elif active_1627 and gate_vol < VOL_EXIT:
            want_active = False
        entry_cost = exit_cost = 0.0
        if want_active and not active_1627:
            entry_cost = 2 * (LEG_COST_BPS / 10000.0) * 1.0
            notional_l27, notional_s27 = 1.0, 1.0
            session_count_1627 = 0
        if active_1627:
            pnl27 = -(notional_l27 * rl) - (notional_s27 * rs)
            notional_l27 *= (1 + rl)
            notional_s27 *= (1 + rs)
            session_count_1627 += 1
            rebal_cost = 0.0
            if session_count_1627 >= REBAL_SESSIONS:
                rebal_cost = 2 * (LEG_COST_BPS / 10000.0) * 1.0
                notional_l27, notional_s27 = 1.0, 1.0
                session_count_1627 = 0
            if not want_active:
                exit_cost = 2 * (LEG_COST_BPS / 10000.0) * 1.0
            borrow_base27 = notional_l27 + notional_s27
            gross_bps27 = pnl27 / GROSS_NOTIONAL * 10000
            net_bps27 = (pnl27 - borrow_base27 * BORROW_RAILS["15"] / 252 - rebal_cost - exit_cost) / GROSS_NOTIONAL * 10000
        else:
            gross_bps27 = 0.0
            net_bps27 = -(entry_cost) / GROSS_NOTIONAL * 10000 if entry_cost else 0.0
        active_1627 = want_active

        out.append({
            "date": row["date"], "underlying": underlying, "leverage": leverage,
            "long_symbol": long_sym, "short_symbol": short_sym,
            "split": split_of(row["date"]),
            "sigma_daily": row["sigma_daily"], "vol_annual": gate_vol, "theory_bps": row["theory_bps"],
            "gross_bps_1626": gross_bps,
            "net_bps_1626_rail5": net_bps["5"], "net_bps_1626_rail15": net_bps["15"], "net_bps_1626_rail30": net_bps["30"],
            "active_1627": active_1627,
            "gross_bps_1627": gross_bps27, "net_bps_1627_rail15": net_bps27,
        })
    return pd.DataFrame(out)


def score():
    pairs_df = pd.read_csv(PAIRS_CSV)
    bars_path = BARS_REBUILD_PARQUET if os.path.exists(BARS_REBUILD_PARQUET) else BARS_PARQUET
    bars = pd.read_parquet(bars_path)
    bars["date"] = pd.to_datetime(bars["date"]).dt.tz_localize(None).dt.normalize()
    log(f"scoring {len(pairs_df)} pairs against {bars_path} ({bars['symbol'].nunique()} symbols)")

    n_non2x = (pairs_df["leverage"] != 2.0).sum()
    pairs_2x = pairs_df[pairs_df["leverage"] == 2.0]
    log(f"cells 1,626/1,627 are the 2x book only (PREREG: 3x/1.5x report-only, cell 1,629 -- out of "
        f"this rebuild's scope): {len(pairs_2x)}/{len(pairs_df)} pairs used, {n_non2x} excluded")

    all_days = []
    n_skipped = 0
    for _, p in pairs_2x.iterrows():
        r = simulate_pair(p["underlying"], p["leverage"], p["long_symbol"], p["short_symbol"], bars)
        if r is None or r.empty:
            n_skipped += 1
            continue
        all_days.append(r)
    log(f"simulated {len(all_days)}/{len(pairs_2x)} 2x pairs with usable bars ({n_skipped} skipped: missing/short history)")
    days = pd.concat(all_days, ignore_index=True)
    days["vol_tercile"] = np.nan
    for split, g in days.groupby("split"):
        try:
            days.loc[g.index, "vol_tercile"] = pd.qcut(g["sigma_daily"], 3, labels=["Q1_low", "Q2_mid", "Q3_high"], duplicates="drop")
        except ValueError:
            pass
    days.to_csv(DAYS_CSV, index=False)
    log(f"wrote {len(days)} pair-days -> {DAYS_CSV}")

    lines = []
    lines.append("# REBUILD_1626 -- independent rebuild of cells 1,626-1,627\n")
    lines.append(f"Generated {pd.Timestamp.now(tz='UTC').isoformat()}. Built from PREREG_1626.md prose only; "
                 "cell_1626.py / cell_1626_days.csv / RESULT_1626.md were never opened.\n")
    lines.append(f"Pairs matched: {len(pairs_df)} (own parser, `build_pairs()` in this file). "
                 f"Pairs with usable bars and scored: {len(all_days)}. Skipped (missing/short bars): {n_skipped}.\n")

    n_dup_underlying = (pairs_df.groupby("underlying").size() > 1).sum()
    lines.append(f"Caveat: {n_dup_underlying}/{pairs_df['underlying'].nunique()} underlyings have >1 "
                 "matched pair (multiple issuers on the same underlying+factor, cross-joined long x "
                 "short) -- these pairs share a leg and are NOT independent draws; day-clustered t "
                 "(clustering on day, across all pairs) partially controls for this but 'share of pairs "
                 "positive' is inflated by near-duplicate pairs. Not deduplicated further given the step budget.\n")

    lines.append("\n## Hand-check sample (20 pairs, `random_state=1626` on the full pair list)\n")
    lines.append("| Underlying | Lev | Long | Long name | Short | Short name |")
    lines.append("|---|---|---|---|---|---|")
    hc = pairs_df.sample(n=min(20, len(pairs_df)), random_state=1626).sort_values("underlying")
    for _, p in hc.iterrows():
        lines.append(f"| {p['underlying']} | {p['leverage']:.2f}x | {p['long_symbol']} | "
                     f"{p['long_name'][:55]} | {p['short_symbol']} | {p['short_name'][:55]} |")

    def rail_block(df, col_prefix, rail_cols, label):
        out = [f"\n## {label}\n", "| Split | n pair-days | n pairs | mean bps/day | day-clustered t | share pairs positive |",
               "|---|---|---|---|---|---|"]
        for split in ["TRAIN", "VAL"]:
            g = df[df.split == split]
            for rail_name, col in rail_cols:
                if g.empty:
                    out.append(f"| {split} ({rail_name}) | 0 | 0 | n/a | n/a | n/a |")
                    continue
                mean_bps = g[col].mean()
                t = day_clustered_t(g[col], g["date"])
                per_pair = g.groupby(["long_symbol", "short_symbol"])[col].mean()
                share_pos = (per_pair > 0).mean() if len(per_pair) else float("nan")
                out.append(f"| {split} ({rail_name}) | {len(g)} | {g.groupby(['long_symbol','short_symbol']).ngroups} | "
                           f"{mean_bps:+.2f} | {t:.2f} | {share_pos:.1%} |")
        return out

    lines += rail_block(days, "1626", [("rail5", "net_bps_1626_rail5"), ("rail15", "net_bps_1626_rail15"),
                                        ("rail30", "net_bps_1626_rail30"), ("gross", "gross_bps_1626")],
                         "Cell 1,626 -- PAIR-SHORT static")
    lines += rail_block(days, "1627", [("rail15 (only while active)", "net_bps_1627_rail15")],
                         "Cell 1,627 -- PAIR-SHORT vol-gated (active days only)")
    active_share = days.groupby("split")["active_1627"].mean()
    lines.append("\nShare of pair-days active under the 1627 vol gate (>=60% enter / <40% exit, annualised): "
                f"TRAIN {active_share.get('TRAIN', float('nan')):.1%}, VAL {active_share.get('VAL', float('nan')):.1%}\n")

    lines.append("\n## Vol-tercile calibration (theory vs realised, gross bps/day, VAL split)\n")
    lines.append("| Tercile | n | mean sigma_daily | mean gross bps/day | mean theory bps/day | realised/theory |")
    lines.append("|---|---|---|---|---|---|")
    valg = days[days.split == "VAL"]
    for terc in ["Q1_low", "Q2_mid", "Q3_high"]:
        g = valg[valg.vol_tercile == terc]
        if g.empty:
            continue
        real_m, th_m = g["gross_bps_1626"].mean(), g["theory_bps"].mean()
        ratio = real_m / th_m if th_m else float("nan")
        lines.append(f"| {terc} | {len(g)} | {g['sigma_daily'].mean():.4f} | {real_m:+.2f} | {th_m:+.2f} | {ratio:.2f} |")

    lines.append("\n## Borrow-rail sensitivity, VAL, cell 1,626\n")
    lines.append("| Rail | mean bps/day | day-clustered t |")
    lines.append("|---|---|---|")
    for rail_name, col in [("5%/yr", "net_bps_1626_rail5"), ("15%/yr", "net_bps_1626_rail15"), ("30%/yr", "net_bps_1626_rail30")]:
        g = days[days.split == "VAL"]
        lines.append(f"| {rail_name} | {g[col].mean():+.2f} | {day_clustered_t(g[col], g['date']):.2f} |")

    dedup_events = sorted(set(EXTREME_EVENTS))
    lines.append(f"\n## Price-scale check (CLAUDE.md caveat 3)\n")
    lines.append(f"Split-adjusted bars (`adjustment=split`) still contained {len(dedup_events)} distinct "
                 f"symbol-day events with |daily return| > {EXTREME_RET_CAP:.0%} (pre-fix, raw bars had "
                 "50 events up to +2,589% in one day -- unadjusted reverse splits). Every such event is "
                 "treated as an uncorrected corporate-action artifact and NEUTRALISED to a 0% return for "
                 "that symbol-day (both the P&L and the notional carried flat through it) rather than "
                 "reported as real P&L. This is a disclosed limitation, not a resolution: a full "
                 "corporate-actions reconciliation was out of scope for this rebuild's step budget. "
                 "Sample (up to 15):\n")
    for sym, d, col, v in dedup_events[:15]:
        lines.append(f"- {sym} {d} {col}={v:+.2f}")
    if not dedup_events:
        lines.append("- none")

    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"wrote {REPORT_MD}")
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pairs", "fetch-bars", "fetch-flags", "score", "all"])
    args = ap.parse_args()

    if args.stage in ("pairs", "all"):
        pairs_df = build_pairs()
    if args.stage in ("fetch-bars", "all"):
        pairs_df = pd.read_csv(PAIRS_CSV)
        symbols = sorted(set(pairs_df["long_symbol"]) | set(pairs_df["short_symbol"]) | set(pairs_df["underlying"]))
        log(f"{len(symbols)} unique symbols to fetch (pair legs + underlyings)")
        fetch_bars(symbols)
    if args.stage in ("fetch-flags", "all"):
        pairs_df = pd.read_csv(PAIRS_CSV)
        symbols = sorted(set(pairs_df["long_symbol"]) | set(pairs_df["short_symbol"]))
        fetch_flags(symbols)
    if args.stage in ("score", "all"):
        score()


if __name__ == "__main__":
    main()
