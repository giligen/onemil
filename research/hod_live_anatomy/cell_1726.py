"""Cell 1,726: (A) passive squeeze, (B) uncensored afternoon path + double-down of the runners (PREREG_1726.md, frozen).

Population: the 231-fill live HOD paper book (variants_1724.csv / fills_1721.csv / cache/bars_1724, no network).
Re-uses cell_1724 (loader, bars, stop-first walk, 12 bps SLIP, 14:00 handling: bars m < 840 are the last closed
before 14:00; a 14:00 action trades at the open of the 840 bar) and the R1/R2 hold flags of runners_1725.csv.

(A) Passive offers, paired against the live replay, Delta = 0 where the offer never fills:
    A1  R1 squeeze side (open at 14:00, not R1-hold): rest a sell at 13:59 close + 1 tick, bars 840..954, else live.
    A2  same, offer = 13:59 close + max(0.5 * (13:59 high - low), 1 tick), rounded to the cent.
    A3  every position still open at 15:30 (no stop/target before bar 930): rest at the 15:29 close + 1 tick,
        bars 930..954, else the live 15:55 flat.
    Fill rule: a bar fills the offer iff its high >= limit + 0.01 (touch is not enough); fill price = limit, cost 0.
    Within a bar the stop is checked first (as in cell 1,724), then the offer, then the live 2R target (the live
    target stays resting; the offer is below it unless the close is within a tick of it, then the lower price wins).
(B) Descriptive 14:00 open -> 15:55 close path, ignoring stop and target (uncensored), R1/R2 hold and squeeze sides;
    B1 double down: R1 hold side buys the ORIGINAL share count at the 14:00 open + 12 bps, add stop = 13:59 close - 1 R
    (original R per share), add leaves at its stop, the live target price, or the 15:55 close.  Add-leg R = pnl / rps.
Outputs: RESULT_1726.md, passive_1726.csv.  No orders, no DB writes.
"""
import math
import os

import numpy as np
import pandas as pd

import cell_1724 as c24

HERE = c24.HERE
SLIP, M_1400, M_1555 = c24.SLIP, c24.M_1400, c24.M_1555
TICK = 0.01
M_1530, M_1529, M_1554 = 930, 929, 954
GATE_NET_BPS, ROUND_TRIP_BPS = 30.0, 24.0


def log(msg):
    """Verbose, flushed progress line."""
    print(f"[1726] {msg}", flush=True)


# ---------------------------------------------------------------- engine
def walk_offer(b, m_from, stop, tp, limit):
    """Walk bars m_from..955 with stop-first, a resting sell `limit` (bars <= 954, needs high >= limit + tick) and the
    live target `tp`.  Returns (kind, price, minute); kind in stop / offer / target / eod.  limit=None -> no offer."""
    w = b[(b.m >= m_from) & (b.m <= M_1555)]
    for r in w.itertuples():
        if r.l <= stop:
            return "stop", min(r.o, stop) * (1 - SLIP), r.m
        off = limit is not None and r.m <= M_1554 and r.h >= limit + TICK - 1e-9
        tgt = r.h >= tp
        if off and tgt:
            return ("offer", limit, r.m) if limit <= tp else ("target", tp, r.m)
        if off:
            return "offer", limit, r.m
        if tgt:
            return "target", tp, r.m
    px, _ = c24.eod_price(b, np.nan)
    return "eod", px, np.nan


def r_of(res, fu, rps, charge_eod):
    """R of an exit tuple; optionally charge the 12 bps marketable cost on the EOD flat (sensitivity)."""
    kind, px = res[0], res[1]
    if charge_eod and kind == "eod":
        px = px * (1 - SLIP)
    return (px - fu) / rps


def last_close_before(b, m):
    """(close, high, low) of the last bar with start minute < m, or None."""
    w = b[b.m < m]
    return None if w.empty else (float(w.iloc[-1].c), float(w.iloc[-1].h), float(w.iloc[-1].l))


def position_row(r, v, rf):
    """All variants for one fill.  r = fills row (Series), v = variants row, rf = runners row or None."""
    b = c24.read_bars(v.trade_date, v.symbol)
    fu, rps, m0, stop0, tp = r.fill_used, r.rps, int(r.entry_min), r.stop_loss_price, r.take_profit_price
    row = dict(id=v.id, trade_date=v.trade_date, symbol=v.symbol, half=v.half, bars=len(b), live_R=v.live_R,
               live_kind=v.live_kind, open_1400=bool(v.open_1400),
               r1_hold=bool(rf.r1_hold) if rf is not None else False, r2_hold=bool(rf.r2_hold) if rf is not None else False)
    if b.empty:
        return row
    live = walk_offer(b, m0 + 1, stop0, tp, None)
    row["live_R_check"] = r_of(live, fu, rps, False)
    row["live_R_c"] = r_of(live, fu, rps, True)
    row["live_exit_min"] = live[2]
    # ---- A1 / A2 on the open-at-14:00 R1 squeeze side
    for name in ("a1", "a2", "a3"):
        row[f"{name}_elig"] = False
        row[f"{name}_offer"] = np.nan
        row[f"{name}_filled"] = False
        row[f"{name}_bar"] = np.nan
        row[f"{name}_R"] = v.live_R
        row[f"{name}_R_c"] = row["live_R_c"]
    lc = last_close_before(b, M_1400)
    if row["open_1400"] and rf is not None and not rf.r1_hold and lc is not None:
        c, h, l = lc
        for name, off in (("a1", c + TICK), ("a2", c + max(0.5 * (h - l), TICK))):
            off = round(off, 2)
            res = walk_offer(b, M_1400, stop0, tp, off)
            row[f"{name}_elig"], row[f"{name}_offer"] = True, off
            row[f"{name}_filled"] = res[0] == "offer"
            row[f"{name}_bar"] = res[2] if res[0] == "offer" else np.nan
            row[f"{name}_R"], row[f"{name}_R_c"] = r_of(res, fu, rps, False), r_of(res, fu, rps, True)
    # ---- A3: every position still open at 15:30
    lc3 = last_close_before(b, M_1530)
    still = (m0 < M_1529 and lc3 is not None and
             (c24.walk(b, m0 + 1, M_1530, stop0, lambda m: tp) is None))
    if still:
        off = round(lc3[0] + TICK, 2)
        res = walk_offer(b, M_1530, stop0, tp, off)
        row["a3_elig"], row["a3_offer"] = True, off
        row["a3_filled"] = res[0] == "offer"
        row["a3_bar"] = res[2] if res[0] == "offer" else np.nan
        row["a3_R"], row["a3_R_c"] = r_of(res, fu, rps, False), r_of(res, fu, rps, True)
    # ---- (B) uncensored path 14:00 open -> 15:55 close, only for positions open at 14:00
    post = b[(b.m >= M_1400) & (b.m <= M_1555)]
    row.update(path_bps=np.nan, path_R=np.nan, path_mfe_R=np.nan, path_mae_R=np.nan, path_up=np.nan,
               b1_add_R=np.nan, b1_kind=None)
    if row["open_1400"] and len(post):
        o14 = float(post.iloc[0].o)
        eod, _ = c24.eod_price(b, o14)
        row["path_bps"] = 1e4 * (eod / o14 - 1)
        row["path_R"] = (eod - o14) / rps
        row["path_mfe_R"] = (float(post.h.max()) - o14) / rps
        row["path_mae_R"] = (float(post.l.min()) - o14) / rps
        row["path_up"] = float(eod > o14)
        if rf is not None and rf.r1_hold and lc is not None:
            add_px = o14 * (1 + SLIP)
            stop_add = lc[0] - rps
            res = walk_offer(b, M_1400, stop_add, tp, None)
            row["b1_add_R"] = (res[1] - add_px) / rps
            row["b1_kind"] = res[0]
    return row


# ---------------------------------------------------------------- stats
def stats(d, days=None):
    """mean, iid t, day-clustered t, ex-top-5 % of a Delta vector (zeros included)."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 3:
        return dict(n=n, mean=d.mean() if n else np.nan, t=np.nan, tcl=np.nan, ex5=np.nan)
    m, sd = d.mean(), d.std(ddof=1)
    t = m / (sd / math.sqrt(n)) if sd > 0 else np.nan
    tcl = np.nan
    if days is not None and sd > 0:
        g = pd.Series(d - m).groupby(np.asarray(days)).sum()
        se = math.sqrt((g ** 2).sum()) / n
        tcl = m / se if se > 0 else np.nan
    return dict(n=n, mean=m, t=t, tcl=tcl, ex5=c24.ex_top5(d))


def f(x, k=3, sign=True):
    """Number formatter tolerant of NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{k}f}" if sign else f"{x:.{k}f}"


def variant_block(t, name, col):
    """Markdown for one A variant; returns (text, pass_flag, summary dict)."""
    out = []
    summ = {}
    for tag, rcol, lcol in (("primary (EOD flat uncosted, as in the live replay)", f"{col}_R", "live_R"),
                            ("sensitivity (live EOD flat charged 12 bps)", f"{col}_R_c", "live_R_c")):
        d = (t[rcol] - t[lcol]).fillna(0.0)
        el, fl = t[f"{col}_elig"], t[f"{col}_filled"]
        s, sa, sb = stats(d, t.trade_date), stats(d[t.half == "A"]), stats(d[t.half == "B"])
        sf = stats(d[fl], t.trade_date[fl])
        ok = s["mean"] >= 0.02 and sa["mean"] > 0 and sb["mean"] > 0 and s["ex5"] > 0
        out.append(f"| {tag} | {f(s['mean'], 4)} | {f(s['t'], 2)} / {f(s['tcl'], 2)} | {f(s['ex5'], 4)} | "
                   f"{f(sa['mean'], 4)} | {f(sb['mean'], 4)} | {f(sf['mean'], 3)} (n {int(fl.sum())}) | "
                   f"{'PASS' if ok else 'FAIL'} |")
        summ[tag.split()[0]] = (s, sa, sb, ok)
    head = [f"### {name}", "",
            f"Eligible positions {int(t[f'{col}_elig'].sum())}; filled via the offer **{int(t[f'{col}_filled'].sum())}** "
            f"= {100 * t[f'{col}_filled'].sum() / max(1, t[f'{col}_elig'].sum()):.0f}% of eligible, "
            f"{100 * t[f'{col}_filled'].sum() / len(t):.1f}% of all {len(t)} fills. Pooled over all fills "
            f"(Delta = 0 where the offer does not fill).", "",
            "| basis | pooled dR/trade | t iid / day-cl. | ex-top-5 % | half A | half B | dR on touched | verdict (bar: >= +0.02, both halves > 0, ex-top-5 % > 0) |",
            "|---|---|---|---|---|---|---|---|"]
    return "\n".join(head + out) + "\n", summ["primary"][3], summ


def path_table(t, flag, label):
    """Markdown rows of the uncensored 14:00 -> 15:55 path for hold / squeeze sides of one rule."""
    rows = []
    base = t[t.open_1400 & t.path_bps.notna()]
    for side, m in (("hold", base[flag]), ("squeeze", ~base[flag])):
        for hl in ("ALL", "A", "B"):
            x = base[m] if hl == "ALL" else base[m & (base.half == hl)]
            n = len(x)
            if n == 0:
                continue
            se = x.path_bps.std(ddof=1) / math.sqrt(n) if n > 2 else np.nan
            rows.append(f"| {label} {side} {hl} | {n} | {x.path_bps.mean():+.0f} | {x.path_bps.median():+.0f} | "
                        f"{f(x.path_bps.mean() / se if se else np.nan, 2)} | {x.path_R.mean():+.3f} | {x.path_R.median():+.3f} | "
                        f"{x.path_mfe_R.mean():+.2f} | {x.path_mae_R.mean():+.2f} | {100 * x.path_up.mean():.0f}% |")
    return rows


def main():
    """Build passive_1726.csv and RESULT_1726.md."""
    fills = c24.load_fills().set_index("id")
    var = pd.read_csv(os.path.join(HERE, "variants_1724.csv"))
    run = pd.read_csv(os.path.join(HERE, "runners_1725.csv")).set_index("id")
    rows = []
    for v in var.itertuples():
        r = fills.loc[v.id].copy()
        r.entry_min = int(r.entry_min)
        rows.append(position_row(r, v, run.loc[v.id] if v.id in run.index else None))
    t = pd.DataFrame(rows)
    t["live_R"] = t.live_R.astype(float)
    nb = int((t.bars == 0).sum())
    ok = t.live_R_check.notna()
    maxdiff = float((t.live_R_check[ok] - t.live_R[ok]).abs().max())
    log(f"positions {len(t)}, no bars {nb}, live replay max |diff| vs variants live_R = {maxdiff:.2e}")
    # a lost-bars fill (live_R NaN) enters every pooled mean as Delta 0, with live_R / variant R set to 0 there
    for c in [x for x in t.columns if x.endswith("_R") or x.endswith("_R_c")]:
        if c in ("path_R", "path_mfe_R", "path_mae_R", "b1_add_R", "live_R_check"):
            continue
        t[c] = t[c].fillna(0.0)
    t.to_csv(os.path.join(HERE, "passive_1726.csv"), index=False)
    log(f"wrote passive_1726.csv ({len(t)} rows)")

    md = ["# Cell 1,726: passive squeeze + uncensored afternoon path + double-down (PREREG_1726)", "",
          f"Live HOD paper book, {len(t)} fills ({nb} without cached bars: Delta 0), free Alpaca 1-min bars (cell 1,724 cache). "
          f"Live replay reproduces variants_1724 live_R to max |diff| {maxdiff:.1e}. Half = alternating trade dates (A/B).", ""]
    # ---------------- (A)
    md += ["## (A) Passive squeeze", ""]
    a_pass = {}
    for name, col in (("A1 offer = 13:59 close + 1 tick, R1 squeeze side", "a1"),
                      ("A2 offer = 13:59 close + 0.5 x bar range, R1 squeeze side", "a2"),
                      ("A3 passive EOD, every position open at 15:30 (15:29 close + 1 tick)", "a3")):
        txt, ok_, _ = variant_block(t, name, col)
        a_pass[col] = ok_
        md += [txt]
    md += [f"Verdict (primary basis): A1 {'PASS' if a_pass['a1'] else 'FAIL'}, A2 {'PASS' if a_pass['a2'] else 'FAIL'}, "
           f"A3 {'PASS' if a_pass['a3'] else 'FAIL'}. "
           + ("Both A1/A2 pass: ship the higher pooled dR." if a_pass['a1'] and a_pass['a2'] else ""), ""]
    # ---------------- (B)
    md += ["## (B) Uncensored 14:00 open -> 15:55 close path (stop and target ignored)", "",
           "| group | n | mean bps | median bps | t (bps) | mean R | median R | mean MFE R | mean MAE R | closes above 14:00 open |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    md += path_table(t, "r1_hold", "R1") + path_table(t, "r2_hold", "R2")
    md.append("")
    h1 = t[t.open_1400 & t.r1_hold & t.path_bps.notna()]
    gate = {}
    for hl in ("A", "B"):
        x = h1[h1.half == hl]
        gate[hl] = (x.path_bps.mean(), x.path_bps.median(), len(x))
    # ---------------- B1
    a = t[t.b1_add_R.notna()]
    s, sa, sb = stats(a.b1_add_R, a.trade_date), stats(a.b1_add_R[a.half == "A"]), stats(a.b1_add_R[a.half == "B"])
    pooled = stats(t.b1_add_R.fillna(0.0), t.trade_date)
    ok_b1 = s["mean"] >= 0.10 and sa["mean"] > 0 and sb["mean"] > 0 and s["ex5"] > 0
    kinds = a.b1_kind.value_counts().to_dict()
    md += ["### B1 double down (R1 hold side adds the original share count at the 14:00 open + 12 bps)", "",
           "| set | n | mean add R | t iid / day-cl. | ex-top-5 % | half A | half B |", "|---|---|---|---|---|---|---|",
           f"| add leg | {s['n']} | {f(s['mean'])} | {f(s['t'], 2)} / {f(s['tcl'], 2)} | {f(s['ex5'])} | "
           f"{f(sa['mean'])} (n {sa['n']}) | {f(sb['mean'])} (n {sb['n']}) |",
           f"| combined dR/trade over all {len(t)} fills (live + add; Delta = add R in original-R units) | {pooled['n']} | "
           f"{f(pooled['mean'], 4)} | {f(pooled['t'], 2)} / {f(pooled['tcl'], 2)} | {f(pooled['ex5'], 4)} | | |", "",
           f"Add exits by kind: {kinds}. Live R on the same 21 hold positions: mean {f(a.live_R.mean())}; combined "
           f"(live + add) mean on the 21 = {f(a.live_R.mean() + s['mean'])} R of original risk.", "",
           f"**B1 verdict (bar: add-leg mean >= +0.10, both halves > 0, ex-top-5 % > 0): {'PASS' if ok_b1 else 'FAIL'}**", ""]
    # ---------------- (C)
    need = GATE_NET_BPS + ROUND_TRIP_BPS
    g_ok = all(not math.isnan(gate[h][0]) and gate[h][0] - ROUND_TRIP_BPS >= GATE_NET_BPS for h in ("A", "B"))
    g_ok_med = all(not math.isnan(gate[h][1]) and gate[h][1] - ROUND_TRIP_BPS >= GATE_NET_BPS for h in ("A", "B"))
    md += ["## (C) Gate for a dedicated 14:00 runner scan (cell 1,727)", "",
           f"Rule: R1 hold side 14:00 -> 15:55 return >= +{GATE_NET_BPS:.0f} bps net of {ROUND_TRIP_BPS:.0f} bps round trip "
           f"(i.e. gross >= {need:.0f} bps) in BOTH halves. Mean basis (primary): "
           f"A {gate['A'][0]:+.0f} bps (n {gate['A'][2]}), B {gate['B'][0]:+.0f} bps (n {gate['B'][2]}) -> "
           f"net A {gate['A'][0] - ROUND_TRIP_BPS:+.0f}, net B {gate['B'][0] - ROUND_TRIP_BPS:+.0f}. "
           f"Median basis: A {gate['A'][1]:+.0f}, B {gate['B'][1]:+.0f}.",
           f"**Gate outcome: {'MET (mean) - PREREG 1,727 is triggered' if g_ok else 'NOT MET (mean basis)'}**"
           f" (median basis: {'MET' if g_ok_med else 'NOT MET'})"
           ". Cell 1,727 was NOT run here.", "",
           "## Implementation caveats (my own)", "",
           "- Offer placed at the 13:59 close + tick: bars carry no quotes; the real ask is unknown. A1's 'later bar' includes the "
           "14:00 bar itself (the order exists from 14:00:00); fill price = limit even when a bar opens above it (conservative).",
           "- Fill test is high >= limit + 0.01 on 1-min bars: a bar high can be a single print, no queue position is modelled, "
           "so fill shares are an upper bound on a real resting order.",
           "- Passive exits pay 0 bps, the live EOD flat in the replay also pays 0 (close of the 15:55 bar, cell 1,724 convention), "
           "so the primary basis does not credit the 12 bps cost lever on EOD exits; the sensitivity charges 12 bps to every EOD "
           "flat on both sides. Stops keep their measured slip in both.",
           "- The live 2R target stays resting beside the offer; stop-first inside a bar. R1 hold flags are the frozen 1,725 flags "
           "(features through the 13:59 bar; high-so-far = post-entry high since the cache starts at the fill).",
           "- B path uses the 14:00 bar open and the 15:55 close (last close <= 15:55 if absent); no cost in the descriptive table "
           "(the gate subtracts 24 bps). B1 add stop 1 R below the 13:59 close is always above the original stop (open P&L >= 0.5 R), "
           "so the add stops first; add R is divided by the ORIGINAL rps (actual risk is slightly above 1 R after the +12 bps "
           "entry). A gap above the live target at 14:00 sells the add at the target price (replay convention).",
           "- 21 hold positions: the add-leg and gate numbers are tiny-sample; cells counted: A1, A2, A3 (x2 bases), B1, path tables, gate = 7.",
           "- A lost-bars fill contributes Delta 0 (no cached bars)."]
    with open(os.path.join(HERE, "RESULT_1726.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    log("wrote RESULT_1726.md")


if __name__ == "__main__":
    main()
