"""Cell 1,725 (EXPLORATORY): can an afternoon runner be told from a fader at 14:00?  (PREREG_1725.md, frozen)

Population: positions open at 14:00 ET in the 231-fill live HOD paper book (variants_1724.csv, open_1400 == True).
Runner = live-replay R >= +1.0.  Two pre-stated hold rules (no feature search, no fitted model):
  R1 near the high : hold iff open P&L >= +0.5R AND decision price within 0.5R of the high so far.
  R2 momentum      : hold iff MFE so far >= +1.0R AND decision price >= 50 % of MFE.
Hold side = live rule (live_R).  Squeeze side = V2 of cell 1,724 (take the profit at 14:00 if open P&L >= 0, else live):
squeeze_R = v2_R from variants_1724.csv.  Paired delta = squeeze_R - live_R on the squeeze side, 0 on the hold side.

Timing convention (deviation, see RESULT caveats): V2 sells at the OPEN of the 14:00 bar, so a decision made from the
14:00 bar's CLOSE would be look-ahead.  Primary features therefore use bars with m < 840 (last bar = 13:59, closed
at 14:00:00), identical to cell 1,724's open_r_1400.  A sensitivity (features incl. the 14:00 bar, squeeze at the
14:01 bar open) is reported beside it.  Inputs: variants_1724.csv, fills_1721.csv, cache/bars_1724.  No network.
"""
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import cell_1724 as c24

HERE = c24.HERE
M_1400, M_1555 = c24.M_1400, c24.M_1555
RUNNER_R = 1.0


def log(msg):
    """Verbose progress line."""
    print(f"[1725] {msg}", flush=True)


def features(r, b, cutoff):
    """14:00 features for one open position from bars with fill_min < m < cutoff (m is the bar START minute).

    Returns dict: open_r (last close), mfe_r (max high since fill), dist_high_r (high-so-far minus last close, in R),
    vol_ratio (volume 13:30..cutoff / volume of the 30 minutes after the fill).  Day's high is the high since the fill:
    the cache holds bars only from the fill minute on."""
    w = b[(b.m > r.entry_min) & (b.m < cutoff)]
    if w.empty:
        return dict(open_r=np.nan, mfe_r=np.nan, dist_high_r=np.nan, vol_ratio=np.nan, px=np.nan)
    px = float(w.iloc[-1].c)
    hi = float(w.h.max())
    v_aft = b[(b.m > r.entry_min) & (b.m <= r.entry_min + 30)].v.sum()
    v_pm = b[(b.m >= cutoff - 30) & (b.m < cutoff)].v.sum()
    return dict(open_r=(px - r.fill_used) / r.rps, mfe_r=(hi - r.fill_used) / r.rps,
                dist_high_r=(hi - px) / r.rps, vol_ratio=v_pm / v_aft if v_aft > 0 else np.nan, px=px)


def hold_rules(f):
    """Apply R1 / R2 to a feature dict; returns (r1_hold, r2_hold)."""
    r1 = f["open_r"] >= 0.5 and f["dist_high_r"] <= 0.5
    r2 = f["mfe_r"] >= 1.0 and f["open_r"] >= 0.5 * f["mfe_r"]
    return bool(r1), bool(r2)


def fisher(rows, hold_col):
    """Hold-vs-squeeze runner table and Fisher exact p (two-sided) for rows (a DataFrame)."""
    h, s = rows[rows[hold_col]], rows[~rows[hold_col]]
    tab = [[int(h.runner.sum()), len(h) - int(h.runner.sum())], [int(s.runner.sum()), len(s) - int(s.runner.sum())]]
    p = fisher_exact(tab)[1] if len(h) and len(s) else np.nan
    return dict(hn=len(h), hr=int(h.runner.sum()), hrate=h.runner.mean() if len(h) else np.nan,
                sn=len(s), sr=int(s.runner.sum()), srate=s.runner.mean() if len(s) else np.nan, p=p)


def paired(rows, hold_col, sq_col="squeeze_R"):
    """Paired delta (squeeze the non-hold side) vs live: mean, t, ex-top-5 %, n touched."""
    d = np.where(rows[hold_col], 0.0, rows[sq_col] - rows.live_R)
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n) if n > 2 else np.nan
    return dict(n=n, touched=int((np.abs(d) > 1e-9).sum()), mean=d.mean() if n else np.nan,
                t=d.mean() / se if se and se > 0 else np.nan, extop5=c24.ex_top5(d) if n else np.nan)


def fmt_rate(x):
    """Format a rate as a percentage."""
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}%"


def table(rows, name, hold_col):
    """Markdown block: hold/squeeze runner table, Fisher p, paired delta overall and per half."""
    out = [f"### {name}", "", "| side | n | runners | rate |", "|---|---|---|---|"]
    f = fisher(rows, hold_col)
    out.append(f"| hold | {f['hn']} | {f['hr']} | {fmt_rate(f['hrate'])} |")
    out.append(f"| squeeze | {f['sn']} | {f['sr']} | {fmt_rate(f['srate'])} |")
    out += ["", f"Fisher exact p (two-sided) = {f['p']:.4f}", "",
            "| split | n open | n hold | hold runners/rate | squeeze runners/rate | Fisher p | paired mean dR | t | ex-top-5 % | touched |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, d in (("ALL", rows), ("A", rows[rows.half == "A"]), ("B", rows[rows.half == "B"])):
        f, p = fisher(d, hold_col), paired(d, hold_col)
        out.append(f"| {lab} | {len(d)} | {f['hn']} | {f['hr']}/{f['hn']} {fmt_rate(f['hrate'])} | "
                   f"{f['sr']}/{f['sn']} {fmt_rate(f['srate'])} | {f['p']:.4f} | {p['mean']:+.4f} | {p['t']:.2f} | "
                   f"{p['extop5']:+.4f} | {p['touched']} |")
    return "\n".join(out) + "\n"


def build():
    """Join variants + fills + bars; compute features (primary and sensitivity) for the 14:00-open population."""
    v = pd.read_csv(os.path.join(HERE, "variants_1724.csv"))
    v = v[v.open_1400 == True].copy()                                    # noqa: E712
    fills = c24.load_fills().set_index("id")
    rows = []
    for x in v.itertuples():
        r = fills.loc[x.id]
        r = pd.Series(r).rename("r")
        r.entry_min = int(r.entry_min)
        b = c24.read_bars(x.trade_date, x.symbol)
        f = features(r, b, M_1400)
        g = features(r, b, M_1400 + 1)                                   # sensitivity: incl. the 14:00 bar
        r1, r2 = hold_rules(f)
        s1, s2 = hold_rules(g)
        post = b[(b.m >= M_1400 + 1) & (b.m <= M_1555)]
        p1401 = float(post.iloc[0].o) * (1 - c24.SLIP) if len(post) else np.nan
        sq_alt = (p1401 - r.fill_used) / r.rps if g["open_r"] >= 0 and not math.isnan(p1401) else x.live_R
        rows.append(dict(id=x.id, trade_date=x.trade_date, symbol=x.symbol, half=x.half, open_r=f["open_r"],
                         mfe_r=f["mfe_r"], dist_high_r=f["dist_high_r"], vol_ratio=f["vol_ratio"],
                         runner=bool(x.live_R >= RUNNER_R), r1_hold=r1, r2_hold=r2, live_R=x.live_R,
                         squeeze_R=x.v2_R, sens_open_r=g["open_r"], sens_r1_hold=s1, sens_r2_hold=s2,
                         sens_squeeze_R=sq_alt))
    t = pd.DataFrame(rows)
    log(f"open-at-14:00 n={len(t)}, features missing={int(t.open_r.isna().sum())}")
    return t


def drift_section(t):
    """Markdown: afternoon drift = live-replay exit R - open P&L R at the 13:59 close, by rule side / half, plus bps of price."""
    fills = c24.load_fills().set_index("id")
    d = t.copy()
    d["drift_R"] = d.live_R - d.open_r
    fu, rps = fills.loc[d.id, "fill_used"].values, fills.loc[d.id, "rps"].values
    d["drift_bps"] = 1e4 * ((fu + d.live_R * rps) / (fu + d.open_r * rps) - 1)
    def row(lab, x):
        """One table row: n, mean dR, sd, t, mean/median bps."""
        n = len(x)
        if n < 2:
            return f"| {lab} | {n} | n/a | n/a | n/a | n/a | n/a |"
        sd = x.drift_R.std(ddof=1)
        return (f"| {lab} | {n} | {x.drift_R.mean():+.3f} | {sd:.3f} | {x.drift_R.mean() / (sd / math.sqrt(n)):.2f} | "
                f"{x.drift_bps.mean():+.0f} | {x.drift_bps.median():+.0f} |")
    out = ["", "## Afternoon drift from 14:00 to exit", "",
           "EXPLORATORY. drift = live-replay exit R - open P&L R at the 13:59 bar close (primary timing); bps = exit price vs "
           "13:59 close. Squeeze rows would, if drift ~ 0, have nothing to earn from an early exit.", "",
           "| group | n | mean dR | sd | t | mean bps | median bps |", "|---|---|---|---|---|---|---|"]
    for rule in ("r1_hold", "r2_hold"):
        for side, m in (("hold", d[rule]), ("squeeze", ~d[rule])):
            for lab, h in (("ALL", d), ("A", d[d.half == "A"]), ("B", d[d.half == "B"])):
                out.append(row(f"{rule[:2].upper()} {side} {lab}", h[m.loc[h.index]]))
        for lab, m in (("open>=0", d.open_r >= 0), ("open<0", d.open_r < 0)):
            x = d[~d[rule] & m]
            out.append(row(f"{rule[:2].upper()} squeeze {lab}", x))
            for hl in ("A", "B"):
                out.append(row(f"{rule[:2].upper()} squeeze {lab} {hl}", x[x.half == hl]))
    return "\n".join(out) + "\n"


def main():
    """Write runners_1725.csv and RESULT_1725.md."""
    t = build()
    t.to_csv(os.path.join(HERE, "runners_1725.csv"), index=False)
    s = t.copy()
    s["r1_hold"], s["r2_hold"], s["squeeze_R"] = s.sens_r1_hold, s.sens_r2_hold, s.sens_squeeze_R
    n, k = len(t), int(t.runner.sum())
    md = [f"# Cell 1,725 EXPLORATORY: afternoon runner vs fader at 14:00 (n = {n}, NOT a ship signal)", "",
          "PREREG_1725 frozen rules R1 / R2, live HOD paper book, 231 fills, free Alpaca 1-min bars (cell 1,724 cache).", "",
          f"- Positions open at 14:00 (live replay): **{n}** (A {int((t.half == 'A').sum())} / B {int((t.half == 'B').sum())})",
          f"- Runners (live-replay R >= +{RUNNER_R:.1f}): **{k}**, overall rate **{fmt_rate(k / n)}**",
          f"- Hold side = live_R; squeeze side = V2 (take the profit at the 14:00 bar open if open P&L >= 0, else live).", "",
          "## Primary (features from bars m < 840, i.e. through the 13:59 bar; squeeze at the 14:00 open as in V2)", "",
          table(t, "R1 near the high (open P&L >= +0.5R and within 0.5R of the high so far)", "r1_hold"),
          table(t, "R2 momentum intact (MFE >= +1R and price >= 50 % of MFE)", "r2_hold"),
          "## Sensitivity (features incl. the 14:00 bar; squeeze at the 14:01 open)", "",
          table(s, "R1 sensitivity", "r1_hold"), table(s, "R2 sensitivity", "r2_hold"),
          "## Pass bar (for the 500-fill run, not today)",
          "Hold-side runner rate >= 2x squeeze side with Fisher p <= 0.01 in BOTH halves AND paired dR >= +0.10 with "
          f"ex-top-5 % > 0. At n = {n} none of this can be met with confidence; nothing ships.", "",
          "## Implementation caveats",
          "- Timing: PREREG says features at the '14:00 bar close'; V2 sells at the 14:00 bar OPEN, so a bar-close feature "
          "would be look-ahead. Primary uses the last bar closed before 14:00 (as cell 1,724 open_r_1400); the literal "
          "reading is the sensitivity with a 14:01 squeeze exit.",
          "- 'Day's high so far' and MFE are the high since the FILL minute (cache bars start at the fill), not the true "
          "session high; R1's 'within 0.5R of the day high' is therefore 'of the post-entry high'. For positions entered "
          "near the HOD these coincide; others understate the distance.",
          "- Runner uses the REPLAYED live R (not the ledger R); replay matches the ledger exit reason on 95.7 %.",
          "- Volume ratio (13:30-14:00 / 30 min after fill) is computed and stored but used by no rule.",
          "- Fisher p is two-sided; halves are alternating trade dates (column `half`); rules were fixed before any number was seen; "
          "2 rules x 2 timings = 4 cells counted in the multiplicity ledger.",
          "- Squeeze side executes V2 (exit only if open P&L >= 0), so negative-open positions on the squeeze side still ride live."]
    with open(os.path.join(HERE, "RESULT_1725.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(HERE, "RESULT_1725.md"), "a") as fh:
        fh.write(drift_section(t))
    log("wrote RESULT_1725.md and runners_1725.csv")


if __name__ == "__main__":
    main()
