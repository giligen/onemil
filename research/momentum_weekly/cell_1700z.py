"""
cell_1700z.py -- PREREG_1700z: extend the guarded momentum-sleeve backtest through October 2026 and reconcile it
with the paper sleeve.  Measurement only: no rule is touched (selection/cost come from REBUILD_1700tu.py).

Stages (run one at a time):
  python3 cell_1700z.py fetch   # Alpaca daily bars 2026-09-01..2026-10-30 for every panel symbol of 2026-09
                                # (resumable per batch; reuses scripts/momentum_sleeve.py fetch machinery)
  python3 cell_1700z.py check   # September overlap: share of (symbol,day) closes differing > 0.1 % vs the panel
  python3 cell_1700z.py score   # Z1..Z5 -> 1700z_weekly.csv, 1700z_names.csv, RESULT_1700z.md data
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/home/ec2-user/onemil"
D = ROOT + "/research/momentum_weekly/"
PANEL = D + "panel_2016_2026.parquet"
EXT = D + "panel_ext.parquet"
SCR = "/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/ext_cache/"
EXT_START, EXT_END = pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-31")
BATCH = 200
LOST_MAX = 0.05
for p in (ROOT, ROOT + "/scripts", D):
    if p not in sys.path:
        sys.path.insert(0, p)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sept_symbols():
    """Symbols present in the panel in 2026-09 (the fetch list)."""
    d = pd.read_parquet(PANEL, columns=["symbol", "bar_date"], filters=[("bar_date", ">=", EXT_START)])
    return sorted(d["symbol"].unique())


def make_client():
    """Alpaca client on the sleeve's dedicated PAPER key (data API only; no order call is ever made)."""
    from dotenv import load_dotenv
    load_dotenv(ROOT + "/.env")
    from data_sources.alpaca_client import AlpacaClient
    k, s = os.environ.get("ALPACA_MOM_API_KEY", ""), os.environ.get("ALPACA_MOM_API_SECRET", "")
    if not k or not s:
        raise RuntimeError("ALPACA_MOM_API_KEY/SECRET missing")
    return AlpacaClient(k, s, paper=True)


def stage_fetch():
    """Resumable per-batch fetch with adjustment=ALL feed=SIP (the sleeve's own convention)."""
    import momentum_sleeve as ms
    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    assert free >= 5e9, f"disk free {free/1e9:.1f} GB < 5 GB"
    os.makedirs(SCR, exist_ok=True)
    syms = sorted(set(sept_symbols()) | {"SPY"})
    cusip = [s for s in syms if re.match(r"^[0-9]", s)]
    syms = [s for s in syms if s not in set(cusip)]
    log(f"{len(syms):,} symbols, {len(cusip)} CUSIP-shaped pre-filtered (LOST)")
    client = make_client()
    batches = [syms[i:i + BATCH] for i in range(0, len(syms), BATCH)]
    lost = list(cusip)
    for i, b in enumerate(batches):
        f = SCR + f"all_{i:03d}.parquet"
        lf = SCR + f"all_{i:03d}.lost.json"
        if os.path.exists(f) and os.path.exists(lf):
            lost += json.load(open(lf)); continue
        try:
            bs, remaining, removed = ms.fetch_batch_resilient(client, b, EXT_START, EXT_END)
        except Exception as e:
            log(f"batch {i} FAILED {e!r}: {len(b)} LOST"); lost += b; continue
        l = list(removed)
        if bs is None:
            l += remaining; df = pd.DataFrame(columns=ms.COLS)
        else:
            df = ms.bars_to_df(bs)
            l += [s for s in remaining if s not in set(df["symbol"])]
        df.to_parquet(f + ".tmp", index=False); os.replace(f + ".tmp", f)
        json.dump(l, open(lf + ".tmp", "w")); os.replace(lf + ".tmp", lf)
        lost += l
        log(f"batch {i+1}/{len(batches)} rows {len(df):,} lost so far {len(lost)}")
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(SCR + "all_*.parquet"))], ignore_index=True)
    df = df.sort_values(["symbol", "bar_date"]).reset_index(drop=True)
    share = len(set(lost)) / len(syms)
    log(f"fetched rows {len(df):,} symbols {df['symbol'].nunique():,}; LOST {len(set(lost))} ({share:.2%}); "
        f"dates {df['bar_date'].min().date()}..{df['bar_date'].max().date()}")
    json.dump(sorted(set(lost)), open(SCR + "lost.json", "w"))
    if share > LOST_MAX:
        log(f"LOST share {share:.2%} > {LOST_MAX:.0%}: REFUSING to write the extension")
        return
    df.to_parquet(EXT + ".tmp", index=False); os.replace(EXT + ".tmp", EXT)
    log(f"wrote {EXT}")


def stage_check():
    """September overlap check: share of (symbol,day) closes differing > 0.1 % between panel and Alpaca."""
    p = pd.read_parquet(PANEL, filters=[("bar_date", ">=", EXT_START)])
    e = pd.read_parquet(EXT)
    e = e[e["bar_date"] <= pd.Timestamp("2026-09-30")]
    m = p.merge(e, on=["symbol", "bar_date"], suffixes=("_p", "_a"))
    m["rel"] = (m["close_a"].astype("float64") / m["close_p"].astype("float64") - 1).abs()
    big = m["rel"] > 0.001
    liquid = (m["close_p"] >= 10) & (m["close_p"] * m["volume_p"] >= 200e6)
    log(f"panel Sept rows {len(p):,}; matched {len(m):,}; unmatched panel rows {len(p)-len(m):,}")
    log(f"differ >0.1%: {big.mean():.3%} of all ({big.sum():,}); liquid subset {big[liquid].mean():.3%} "
        f"({big[liquid].sum()}/{liquid.sum()})")
    sy = m.groupby("symbol")["rel"].agg(["max", "median"])
    log(f"symbols with any >0.1% day: {(sy['max']>0.001).sum()} of {len(sy)}; median-diff>0.1%: {(sy['median']>0.001).sum()}")
    # is the difference a constant per-symbol ratio (dividend/split back-adjust)?
    m["ratio"] = m["close_a"].astype("float64") / m["close_p"].astype("float64")
    g = m.groupby("symbol")["ratio"].agg(["min", "max", "median"])
    bad = g[(g["median"] - 1).abs() > 0.001]
    const = bad[(bad["max"] / bad["min"] - 1) < 0.001]
    log(f"symbols with median ratio off by >0.1%: {len(bad)}; of which ratio constant over Sept (<0.1% spread): {len(const)}")
    log("largest ratio deviations:\n" + (g.assign(dev=(g['median']-1)).sort_values('dev', key=abs, ascending=False).head(15)).to_string())
    m[["symbol", "bar_date", "close_p", "close_a", "rel"]][big].to_csv(SCR + "overlap_diffs.csv", index=False)
    dev = (g["median"] - 1).abs()
    log(f"off-ratio symbols by size: <=1% {(bad.index.isin(g[(dev>0.001)&(dev<=0.01)].index)).sum()}, 1-10% "
        f"{(bad.index.isin(g[(dev>0.01)&(dev<=0.10)].index)).sum()}, >10% (split-sized) {(bad.index.isin(g[dev>0.10].index)).sum()}")
    lq = m[liquid].groupby("symbol")["ratio"].median()
    log(f"liquid-universe symbols (any liquid Sept day): {len(lq)}; off by >0.1%: {((lq-1).abs()>0.001).sum()}; "
        f"off by >10%: {((lq-1).abs()>0.10).sum()} {sorted(lq[(lq-1).abs()>0.10].index)[:10]}")


# ----------------------------------------------------------------------------------------------- scoring
OCT_MONDAYS = [pd.Timestamp(x) for x in ("2026-10-05", "2026-10-12", "2026-10-19", "2026-10-26")]
SENTINEL = pd.Timestamp("2026-11-02")          # "next Monday": makes the last week run to the 10/30 close
SEED = pd.Timestamp("2026-10-02")              # the paper sleeve's first (Friday) seed day
CAP = 20000.0


def extend_data(R, data):
    """Append the Alpaca extension rows (bar_date after each symbol's last panel date) to the loaded panel dict."""
    e = pd.read_parquet(EXT)
    n_add = 0
    for sym, g in e.groupby("symbol", sort=False):
        if sym not in data:
            continue
        d, o, h, l, c, v = data[sym]
        g = g[g["bar_date"] > pd.Timestamp(d[-1])].sort_values("bar_date")
        if g.empty:
            continue
        data[sym] = (np.concatenate([d, g["bar_date"].values.astype("datetime64[ns]")]),
                     *[np.concatenate([a, g[k].values.astype("float64")]) for a, k in
                       zip((o, h, l, c, v), ("open", "high", "low", "close", "volume"))])
        n_add += len(g)
    log(f"extension rows appended: {n_add:,}")
    return data


def splice_ratios():
    """Per-symbol median(Alpaca close / panel close) on the September overlap (sensitivity run only)."""
    p = pd.read_parquet(PANEL, columns=["symbol", "bar_date", "close"], filters=[("bar_date", ">=", EXT_START)])
    e = pd.read_parquet(EXT)
    e = e[e["bar_date"] <= pd.Timestamp("2026-09-30")]
    m = p.merge(e[["symbol", "bar_date", "close"]], on=["symbol", "bar_date"], suffixes=("_p", "_a"))
    m["r"] = m["close_a"].astype("float64") / m["close_p"].astype("float64")
    return m.groupby("symbol")["r"].median()


def picks_for(R, data, cal, rebs):
    """Rows of (sym, sig, bad, sp) per rebalance date and the guarded top-20 picks; exactly the REBUILD main() logic."""
    ts = np.array([cal[cal.searchsorted(r) - 1] for r in rebs], dtype="datetime64[ns]")
    rows = R.features(data, cal, ts, None)
    spmap, guarded, ranked = {}, [], []
    for i, rr in enumerate(rows):
        for s, sg, b, sp in rr:
            spmap[(i, s)] = sp
        elig = sorted([x for x in rr if not x[2]], key=lambda x: -x[1])
        ranked.append(elig)
        guarded.append([x[0] for x in elig[:R.TOPN]] if len(rr) >= R.TOPN else None)
    return ts, rows, ranked, guarded, spmap


def sim(R, data, cal, rebs, picks, spmap, skip=(), cap=CAP, halfflag=None):
    """Copy of REBUILD_1700tu.simulate (equal-weight reset every Monday, same cost) with ONE addition: weeks in
    ``skip`` do not trade (the held shares carry over).  Returns (week table, daily series, total cost $)."""
    cash, hold, qty = cap, {}, {}
    out, daily, tot_cost = [], [], 0.0
    for i, reb in enumerate(rebs[:-1]):
        nxt = rebs[i + 1]
        days = cal[(cal >= reb) & (cal < nxt)]
        if i in skip:
            entry = cash + sum(hold.values())
        else:
            tgt = picks[i]
            E = cash + sum(hold.values())
            w = 1.0 / (2 * R.TOPN if (halfflag and halfflag[i]) else R.TOPN)
            cost = 0.0
            for s in set(hold) | set(tgt):
                delta = abs((w * E if s in tgt else 0.0) - hold.get(s, 0.0))
                cost += delta * min(0.0005 + 0.5 * spmap.get((i, s), 0.0), 0.0020)
            cash = E - cost - w * E * len(tgt)
            tot_cost += cost
            entry = E - cost
            qty = {s: (w * E) / R.px(data, s, reb, True) for s in tgt}
        vals = [cash + sum(qty[s] * R.px(data, s, dd, False) for s in qty) for dd in days]
        daily += list(zip(days, vals))
        out.append((reb, entry, vals[-1] if vals else np.nan, "SKIP" if i in skip else "trade"))
        hold = {s: qty[s] * R.px(data, s, nxt, True) for s in qty}
    return out, daily, tot_cost


def paper_books():
    """Paper sleeve holdings after each rebalance row, replayed from names_in/names_out (checked vs state.json)."""
    w = pd.read_csv(ROOT + "/logs/momentum_sleeve_weekly.csv")
    held, books = set(), {}
    for _, r in w.iterrows():
        held |= set(str(r["names_in"]).split()) if isinstance(r["names_in"], str) else set()
        held -= set(str(r["names_out"]).split()) if isinstance(r["names_out"], str) else set()
        books[pd.Timestamp(r["date"])] = sorted(held)       # last row of a date wins
    st = json.load(open(ROOT + "/data/momentum_sleeve/state.json"))
    ok = sorted(st["positions"]) == books[pd.Timestamp("2026-10-19")]
    log(f"paper holdings replayed from names_in/out == state.json on 10/19: {ok}")
    return books, st


def diagnose(R, data, sym, ti, ts, rows, assets):
    """Why a paper-held name is not in the BT top 20 on week ti: its BT rank, or the universe/guard reason."""
    rr = rows[ti]
    elig = sorted([x for x in rr if not x[2]], key=lambda x: -x[1])
    pos = {x[0]: (k + 1, x[1]) for k, x in enumerate(elig)}
    if sym in pos:
        return pos[sym][0], pos[sym][1], "ranked below 20"
    if any(x[0] == sym and x[2] for x in rr):
        return None, None, "guard (corporate-action bad day / gap) in last 273 bars"
    if sym not in data:
        nm = assets.get(sym)
        if nm is None:
            return None, None, "not in the panel / assets file (listed after build or never in panel)"
        return None, None, "name-excluded or never passed price/ADV in panel"
    d, o, h, l, c, v = data[sym]
    p = np.searchsorted(d, ts[ti])
    if p >= len(d) or d[p] != ts[ti]:
        return None, None, "no bar on signal date"
    if c[p] < 10:
        return None, None, f"close {c[p]:.2f} < 10"
    adv = (c[p - 19:p + 1] * v[p - 19:p + 1]).mean() if p >= 19 else np.nan
    if adv < 200e6:
        return None, None, f"ADV20 ${adv/1e6:.0f}M < $200M"
    if p < 272:
        return None, None, "history < 273 bars"
    return None, None, "other (signal not finite)"


def stage_score():
    """Z1..Z5.  Needs cwd = ROOT (REBUILD_1700tu uses relative paths)."""
    os.chdir(ROOT)
    import REBUILD_1700tu as R
    import momentum_sleeve as ms
    R.CAP0 = CAP
    sens = "--splice" in sys.argv                       # sensitivity: rescale extension to the panel's basis
    data = R.load()
    if sens:
        ra = splice_ratios()
        n = 0
        e = pd.read_parquet(EXT)
        for sym, r in ra.items():
            if sym in data and abs(r - 1) > 1e-3:
                n += 1
        log(f"SPLICE sensitivity: {n} symbols rescaled by their overlap ratio")
        e2 = e.copy()
        f = e2["symbol"].map(ra).fillna(1.0).where((e2["symbol"].map(ra).fillna(1.0) - 1).abs() > 1e-3, 1.0)
        for k in ("open", "high", "low", "close"):
            e2[k] = (e2[k].astype("float64") / f).astype("float32")
        e2.to_parquet(SCR + "ext_spliced.parquet")
        globals()["EXT"] = SCR + "ext_spliced.parquet"
    data = extend_data(R, data)
    cal = pd.DatetimeIndex(data["SPY"][0])
    log(f"calendar {cal[0].date()}..{cal[-1].date()}")
    assets = {}
    a = pd.read_csv(R.ASSETS, dtype=str)
    a.columns = [c.strip().lower() for c in a.columns]
    assets = dict(zip(a["symbol"], a["name"].fillna("")))
    # --- rebalance sets: Monday-only (prereg) and paper calendar (extra Friday 10/2 seed)
    rebsM = OCT_MONDAYS + [SENTINEL]
    rebsP = [SEED] + OCT_MONDAYS + [SENTINEL]
    tsM, rowsM, rankM, gM, spM = picks_for(R, data, cal, rebsM)
    tsP, rowsP, rankP, gP, spP = picks_for(R, data, cal, rebsP)
    log("signal dates (Monday-only): " + ", ".join(str(pd.Timestamp(t).date()) for t in tsM[:-1]))
    log("universe sizes per signal date: " + str([len(r) for r in rowsM[:-1]]))
    # --- gate (half-size reference): CBOE VIX/VIX3M percentile < 20 %
    half = []
    for t in tsM[:-1]:
        info = ms.shadow_gate_info(pd.Timestamp(t))
        pct = info["percentile"] if info else np.nan
        half.append(bool(pct < 0.20) if np.isfinite(pct) else False)
        log(f"gate asof {pd.Timestamp(t).date()}: percentile {pct} half={half[-1]}")
    half.append(False)
    # --- Z1
    wk1, dl1, c1 = sim(R, data, cal, rebsM, gM, spM)
    wk1h, dl1h, c1h = sim(R, data, cal, rebsM, gM, spM, halfflag=half)
    # parity of my sim copy with the original function
    R.spmap = spM
    wkR, dlR = R.simulate(data, cal, rebsM, tsM, gM, [False] * len(rebsM), "z1")
    assert abs(dlR[-1][1] - dl1[-1][1]) < 1e-6, (dlR[-1], dl1[-1])
    log(f"sim copy == R.simulate: final {dl1[-1][1]:.2f} (ok)")
    def spy(d0, o_open=True):
        return R.px(data, "SPY", d0, o_open)
    spy_end = R.px(data, "SPY", cal[-1], False)
    z1 = dl1[-1][1] / CAP - 1
    z1h = dl1h[-1][1] / CAP - 1
    spy_M = spy_end / spy(OCT_MONDAYS[0]) - 1
    spy_P = spy_end / spy(SEED) - 1
    spy_m = spy_end / R.px(data, "SPY", pd.Timestamp("2026-09-30"), False) - 1
    log(f"Z1 Oct (10/5 open -> 10/30 close, $20K, BT cost): {z1*100:.2f}%  half-gate {z1h*100:.2f}%  | SPY 10/5 open: "
        f"{spy_M*100:.2f}%  10/2 open: {spy_P*100:.2f}%  9/30 close: {spy_m*100:.2f}%")
    # --- paper calendar variants
    wkP, dlP, cP = sim(R, data, cal, rebsP, gP, spP)
    skipP = {len(rebsP) - 2}                       # the 10/26 week (last real week)
    wkS, dlS, cS = sim(R, data, cal, rebsP, gP, spP, skip=skipP)
    z1p, z3 = dlP[-1][1] / CAP - 1, dlS[-1][1] / CAP - 1
    # --- Z3 on the Monday-only calendar as the prereg defines it
    wkM3, dlM3, cM3 = sim(R, data, cal, rebsM, gM, spM, skip={len(rebsM) - 2})
    z3M = dlM3[-1][1] / CAP - 1
    log(f"Z3 skip-10/26 (Monday-only calendar): {z3M*100:.2f}% vs Z1 {z1*100:.2f}% => TXSE bug cost "
        f"${(dlM3[-1][1]-dl1[-1][1]):+,.0f} ({'saved' if dlM3[-1][1] < dl1[-1][1] else 'cost'} if negative/positive resp.)")
    # --- paper books
    books, st = paper_books()
    pbooks = {SEED: books[SEED]}
    for m in OCT_MONDAYS[:3]:
        pbooks[m] = books[m]
    # Z2 table
    nrows = []
    for i, m in enumerate(OCT_MONDAYS[:3]):
        bt = gM[i]; pp = set(pbooks[m])
        ov = len(set(bt) & pp)
        log(f"Z2 {m.date()} (signal {pd.Timestamp(tsM[i]).date()}): overlap {ov}/20 | BT-only {sorted(set(bt)-pp)} | paper-only {sorted(pp-set(bt))}")
        rk = {x[0]: (k + 1, x[1]) for k, x in enumerate(rankM[i])}
        for k, x in enumerate(rankM[i][:20]):
            nrows.append((m.date(), "BT_top20", k + 1, x[0], round(float(x[1]), 4), x[0] in pp, ""))
        for s in sorted(pp - set(bt)):
            r_, sg_, why = diagnose(R, data, s, i, tsM, rowsM, assets)
            nrows.append((m.date(), "paper_only", r_, s, None if sg_ is None else round(float(sg_), 4), True, why))
        for s in sorted(set(bt) - pp):
            nrows.append((m.date(), "bt_only", rk[s][0], s, round(float(rk[s][1]), 4), False, ""))
    # the missed 10/26 rotation: what the BT would have bought vs what the paper still held
    pp26 = set(pbooks[OCT_MONDAYS[2]]); bt26 = gM[3]
    log(f"10/26 (signal {pd.Timestamp(tsM[3]).date()}): BT would hold {len(set(bt26)&pp26)}/20 of the 10/19 book | "
        f"BT would BUY {sorted(set(bt26)-pp26)} | would SELL {sorted(pp26-set(bt26))}")
    for k, x in enumerate(rankM[3][:20]):
        nrows.append((OCT_MONDAYS[3].date(), "bt_would_rotate_in" if x[0] not in pp26 else "bt_keeps", k + 1, x[0],
                      round(float(x[1]), 4), x[0] in pp26, "paper held through (missed rotation)"))
    # seed week 10/2 (signal 10/1)
    bt0 = gP[0]; pp0 = set(pbooks[SEED])
    log(f"Z2 10/2 seed (signal {pd.Timestamp(tsP[0]).date()}): overlap {len(set(bt0)&pp0)}/20 | BT-only {sorted(set(bt0)-pp0)} | paper-only {sorted(pp0-set(bt0))}")
    for s in sorted(pp0 - set(bt0)):
        r_, sg_, why = diagnose(R, data, s, 0, tsP, rowsP, assets)
        nrows.append((SEED.date(), "paper_only", r_, s, None if sg_ is None else round(float(sg_), 4), True, why))
    for s in sorted(set(bt0) - pp0):
        rk0 = {x[0]: (k + 1, x[1]) for k, x in enumerate(rankP[0])}
        nrows.append((SEED.date(), "bt_only", rk0[s][0], s, round(float(rk0[s][1]), 4), False, ""))
    names = pd.DataFrame(nrows, columns=["monday", "kind", "bt_rank", "symbol", "bt_score", "in_paper", "why_missing_from_bt"])
    names.to_csv(D + "1700z_names.csv", index=False)
    # --- S3: paper names (equal weight reset each Monday, BT cost), 10/26 skipped
    ppicks = [pbooks[SEED]] + [pbooks[m] for m in OCT_MONDAYS[:3]] + [None, None]
    wkQ, dlQ, cQ = sim(R, data, cal, rebsP, ppicks, spP, skip=skipP)
    z2q = dlQ[-1][1] / CAP - 1
    # --- paper actual
    close_end = {s: R.px(data, s, cal[-1], False) for s in st["positions"]}
    eq_end = st["cash"] + sum(q * close_end[s] for s, q in st["positions"].items())
    pret = eq_end / CAP - 1
    # --- Z4 ledger slippage
    L = pd.read_csv(ROOT + "/logs/momentum_sleeve_ledger.csv")
    L["slip$"] = L["notional"] * L["slip_bps_vs_open"] / 1e4
    def agg(x):
        return pd.Series({"n": len(x), "notional": x["notional"].sum(), "slip_$": x["slip$"].sum(),
                          "wmean_bps": x["slip$"].sum() / x["notional"].sum() * 1e4 if x["notional"].sum() else np.nan,
                          "n_blank": int(x["slip_bps_vs_open"].isna().sum())})
    z4tab = L.groupby(["date", "side"]).apply(agg)
    log("Z4 ledger slippage by day/side (positive = adverse):\n" + z4tab.round(2).to_string())
    clean = L[L["date"] >= "2026-10-05"]
    seed = L[L["date"] < "2026-10-05"]
    LL = pd.read_csv(ROOT + "/logs/momentum_sleeve_live_ledger.csv")
    LL["slip$"] = LL["notional"] * LL["slip_bps_vs_open"] / 1e4
    log(f"live ledger: n {len(LL)}, notional {LL['notional'].sum():.0f}, mean slip {LL['slip_bps_vs_open'].mean():.1f} bps, "
        f"notional-wtd {LL['slip$'].sum()/LL['notional'].sum()*1e4:.1f} bps, $ {LL['slip$'].sum():+.2f}; "
        f"buys {LL[LL.side=='buy']['slip_bps_vs_open'].describe()[['count','mean']].to_dict()}")
    act_slip = L["slip$"].sum()
    model_cost = cQ
    z4 = (model_cost - act_slip)                        # $ gained by actual execution vs the BT cost model
    z4_pp = z4 / CAP
    chain = {
        "Z1 rule (Mon-only, 10/5 open start)": z1,
        "seed: Fri 10/2 start (paper calendar, BT picks)": z1p - z1,
        "Z3 skip 10/26 (BT picks, paper calendar)": z3 - z1p,
        "Z2 paper names instead of BT names (skip 10/26)": z2q - z3,
        "Z4 execution (model cost - ledger slip $)": z4_pp,
    }
    pred = z2q + z4_pp
    resid = pret - pred
    log("Z5 chain (pp of $20K): " + "; ".join(f"{k} {v*100:+.2f}" for k, v in chain.items()))
    log(f"paper actual Oct: equity 10/30 ${eq_end:,.2f} => {pret*100:.2f}%  | chain end {pred*100:.2f}% | residual {resid*100:+.2f} pp")
    log(f"costs: BT model on S3 trades ${model_cost:.2f}; ledger slip $ all fills {act_slip:+.2f} "
        f"(seed days {seed['slip$'].sum():+.2f}, clean weeks {clean['slip$'].sum():+.2f})")
    # --- distribution of the rule's monthly returns (REBUILD_1700tu_daily_guarded.csv)
    dg = pd.read_csv(D + "REBUILD_1700tu_daily_guarded.csv", index_col=0, parse_dates=True).iloc[:, 0]
    me = dg.groupby([dg.index.year, dg.index.month]).last()
    mr = (me / me.shift(1) - 1).dropna() * 100
    log(f"monthly guarded returns 2017-02..2026-09: n {len(mr)}, mean {mr.mean():.2f}, sd {mr.std():.2f}, min {mr.min():.2f}, "
        f"p5 {mr.quantile(.05):.2f}, p10 {mr.quantile(.10):.2f}; months <= -10%: {(mr<=-10).sum()}, <= -15%: {(mr<=-15).sum()}")
    for nm, val in (("Z1", z1 * 100), ("-15.5", -15.5)):
        log(f"percentile of {nm} ({val:.2f}%) in the BT monthly distribution: {(mr < val).mean()*100:.1f}% of months are worse")
    log("worst 5 BT months: " + "; ".join(f"{a}-{b:02d} {v:.1f}" for (a, b), v in mr.nsmallest(5).items()))
    # --- weekly table
    rowsw = []
    def addwk(variant, wk, dl):
        d = pd.Series([v for _, v in dl], index=[d for d, _ in dl])
        for reb, entry, lastv, mode in wk:
            wend = d[(d.index >= reb)].index
            rowsw.append((variant, reb.date(), round(entry, 2), round(lastv, 2), round(lastv / entry - 1, 5), mode))
    addwk("Z1_guarded_MonOnly", wk1, dl1); addwk("Z1_halfgate_MonOnly", wk1h, dl1h)
    addwk("Z3_skip1026_MonOnly", wkM3, dlM3)
    addwk("S1_guarded_paperCalendar", wkP, dlP); addwk("S2_skip1026_paperCalendar", wkS, dlS)
    addwk("S3_paperNames_BTcost", wkQ, dlQ)
    pw = pd.read_csv(ROOT + "/logs/momentum_sleeve_weekly.csv").drop_duplicates("date", keep="last")
    for _, r in pw.iterrows():
        rowsw.append(("PAPER_ledger_equity_after_rebal", r["date"], r["equity"], np.nan, np.nan, "ledger"))
    rowsw.append(("PAPER_equity_1030_close", "2026-10-30", round(eq_end, 2), np.nan, round(pret, 5), "state qty x Alpaca close"))
    pd.DataFrame(rowsw, columns=["variant", "week_start", "entry_equity", "friday_close_equity", "week_ret", "mode"]
                 ).to_csv(D + ("1700z_weekly_splice.csv" if sens else "1700z_weekly.csv"), index=False)
    # --- affected names (corporate-action differences) among all picks
    ra = splice_ratios()
    allp = set(sum([g for g in gM[:-1]], []) + gP[0] + sum([pbooks[m] for m in pbooks], []))
    aff = {s: round(float(ra[s]), 4) for s in sorted(allp) if s in ra.index and abs(ra[s] - 1) > 1e-3}
    log(f"picked/held names with a Sept overlap ratio != 1 (corporate-action basis difference): {aff}")
    log("DONE")


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "score"
    {"fetch": stage_fetch, "check": stage_check, "score": stage_score}[st]()
