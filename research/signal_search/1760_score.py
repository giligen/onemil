"""Cell 1,760 scorer (PREREG_1760.md): ORB-13:30 (S1), HOD-24 (S2), weekend/weekday split of S2 (S3) on
Alpaca crypto hourly bars, long and short, taker/maker cost. TRAIN table is written BEFORE VAL is touched.

Spec choices where the PREREG is silent (all conservative, all causal):
* break = bar CLOSE beyond the level; entry at that bar's close (1 bar late vs the touch), first break per
  day per side per signal; entries with no later bar before the time-out are skipped (zero-hold = pure cost).
* S1 scans break bars 14..19 UTC (a 20:00 bar close = 21:00 = time-out). S2 scans bars 3..22.
* stop checked before target inside a bar; gap-through-stop fills at the open; 10 bp slip on every stop exit.
* cost = 2*fee + 5 bp round-trip spread (+10 bp if stopped), as % of price; net R = (gross% - cost%) / R%.
Usage: 1760_score.py [--train-only]
"""
import math, sys
import numpy as np
import pandas as pd

D = "/home/ec2-user/onemil/research/signal_search/"
BARS = D + "1760_cache/bars.parquet"
FEES = {"taker": 0.0025, "maker": 0.0015}
SPREAD, SLIP = 0.0005, 0.0010
SPLITS = {"TRAIN_2021": ("2021-01-01", "2022-01-01"), "TRAIN_2022": ("2022-01-01", "2023-01-01"),
          "TRAIN": ("2021-01-01", "2023-01-01"), "VAL": ("2023-01-01", "2024-01-01")}


def day_arrays(g):
    """Return (dates, o, h, l, c) with shape (ndays, 24); missing hours are NaN."""
    g = g.set_index("ts")
    idx = pd.date_range("2021-01-01", "2023-12-31 23:00", freq="h", tz="UTC")
    g = g.reindex(idx)
    n = len(idx) // 24
    f = lambda col: g[col].to_numpy().reshape(n, 24)
    return idx[::24].tz_localize(None), f("open"), f("high"), f("low"), f("close")


def exit_trade(side, entry, stop, o, h, l, c, t0, last):
    """Walk bars t0+1..last: stop first, then 2R target, else time-out at last present close.
    Returns (exit_px, stopped) or None when no later bar exists."""
    R = abs(entry - stop)
    tgt = entry + 2 * R if side == "long" else entry - 2 * R
    lastpx = None
    for j in range(t0 + 1, last + 1):
        if np.isnan(c[j]):
            continue
        lastpx = c[j]
        if side == "long":
            if o[j] <= stop: return o[j], True
            if l[j] <= stop: return stop, True
            if h[j] >= tgt: return tgt, False
        else:
            if o[j] >= stop: return o[j], True
            if h[j] >= stop: return stop, True
            if l[j] <= tgt: return tgt, False
    return None if lastpx is None else (lastpx, False)


def make_trade(sym, date, sig, side, t, entry, stop, o, h, l, c, last):
    """Simulate one trade; return a dict with gross pct, R pct and stopped flag or None."""
    if not (entry > stop if side == "long" else entry < stop) or np.isnan(entry) or np.isnan(stop):
        return None
    res = exit_trade(side, entry, stop, o, h, l, c, t, last)
    if res is None:
        return None
    px, stopped = res
    sgn = 1 if side == "long" else -1
    gross = sgn * (px - entry) / entry
    rpct = abs(entry - stop) / entry
    return dict(sym=sym, date=date, sig=sig, side=side, hour=t, gross_pct=gross, r_pct=rpct, stopped=stopped,
                weekend=date.dayofweek >= 5)


def signals_for_symbol(sym, g):
    """Generate all S1/S2 trades (both sides) for one symbol."""
    dates, O, H, L, C = day_arrays(g)
    out = []
    for d in range(len(dates)):
        o, h, l, c, date = O[d], H[d], L[d], C[d], dates[d]
        # S1: range = 13:00 bar
        if not np.isnan(h[13]):
            rh, rl = h[13], l[13]
            for side in ("long", "short"):
                for t in range(14, 20):
                    if np.isnan(c[t]): continue
                    if (side == "long" and c[t] > rh) or (side == "short" and c[t] < rl):
                        tr = make_trade(sym, date, "S1", side, t, c[t], rl if side == "long" else rh, o, h, l, c, 20)
                        if tr: out.append(tr)
                        break
        # S2: HOD/LOD of the UTC day set >= 3 h earlier
        for side in ("long", "short"):
            for t in range(3, 23):
                if np.isnan(c[t]): continue
                pre = h[:t] if side == "long" else l[:t]
                if np.all(np.isnan(pre)): continue
                lvl = np.nanmax(pre) if side == "long" else np.nanmin(pre)
                setj = int(np.where(pre == lvl)[0][0])
                if setj > t - 3: continue
                if (side == "long" and c[t] > lvl) or (side == "short" and c[t] < lvl):
                    win = (l if side == "long" else h)[t - 3:t]
                    if np.all(np.isnan(win)): continue
                    stop = np.nanmin(win) if side == "long" else np.nanmax(win)
                    tr = make_trade(sym, date, "S2", side, t, c[t], stop, o, h, l, c, 23)
                    if tr: out.append(tr)
                    break
    return pd.DataFrame(out)


def add_net(df):
    """Add net R columns for taker and maker cost."""
    for k, fee in FEES.items():
        cost = 2 * fee + SPREAD + np.where(df.stopped, SLIP, 0.0)
        df["cost_R_" + k] = cost / df.r_pct
        df["net_R_" + k] = (df.gross_pct - cost) / df.r_pct
    df["gross_R"] = df.gross_pct / df.r_pct
    return df


def tstat(x):
    """iid t-stat of the mean."""
    x = np.asarray(x, float)
    return np.nan if len(x) < 3 or x.std(ddof=1) == 0 else x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))


def tday(x, days):
    """Day-clustered t-stat of the mean (cluster-robust SE)."""
    x = np.asarray(x, float)
    if len(x) < 3: return np.nan
    m = x.mean()
    s = pd.Series(x - m).groupby(np.asarray(days)).sum()
    se = math.sqrt((s ** 2).sum()) / len(x)
    return np.nan if se == 0 else m / se


def stats(df, split, cost):
    """All PREREG numbers for one group of trades in one split."""
    a, b = (pd.Timestamp(x) for x in SPLITS[split])
    d = df[(df.date >= a) & (df.date < b)]
    wks = (b - a).days / 7
    col = "net_R_" + cost
    r = d[col].to_numpy()
    n = len(r)
    base = dict(split=split, cost=cost, n=n, n_per_wk=n / wks)
    if n < 3:
        return base
    srt = np.sort(r)[::-1]
    k = int(math.ceil(0.05 * n))
    wk = d.groupby(d.date.dt.to_period("W-SUN"))[col].sum()
    allw = pd.period_range(a, b - pd.Timedelta(days=1), freq="W-SUN")
    wk = wk.reindex(allw, fill_value=0.0)
    mo = d.groupby(d.date.dt.to_period("M"))[col].sum()
    base.update(gross_R=d.gross_R.mean(), net_R=r.mean(), t_iid=tstat(r), t_day=tday(r, d.date.values),
                ex_top5=srt[k:].mean() if n > k else np.nan, cap3=np.minimum(r, 3).mean(),
                wk_p10=wk.quantile(0.10), green_wk=(wk > 0).mean(), months_pos=int((mo > 0).sum()), months_n=len(mo),
                mde_R=2.8 * r.std(ddof=1) / math.sqrt(n), med_r_pct=100 * d.r_pct.median(),
                med_cost_R=d["cost_R_" + cost].median(), stopped_share=d.stopped.mean())
    return base


def table(trades, splits):
    """Build the summary rows over signal x side x daytype x scope x split x cost."""
    rows = []
    scopes = {"BTC": ["BTC"], "ETH": ["ETH"], "SOL": ["SOL"], "BTC+ETH": ["BTC", "ETH"], "ALL3": ["BTC", "ETH", "SOL"]}
    dts = {"all": None, "weekend": True, "weekday": False}
    for sig in ("S1", "S2"):
        for side in ("long", "short"):
            for dt, wk in dts.items():
                if sig == "S1" and dt != "all": continue
                name = {"S2": {"all": "S2", "weekend": "S3_weekend", "weekday": "S3_weekday"}, "S1": {"all": "S1"}}[sig][dt]
                for sc, syms in scopes.items():
                    d = trades[(trades.sig == sig) & (trades.side == side) & trades.sym.isin(syms)]
                    if wk is not None: d = d[d.weekend == wk]
                    for sp in splits:
                        for cost in FEES:
                            rows.append(dict(signal=name, side=side, scope=sc, **stats(d, sp, cost)))
    return pd.DataFrame(rows)


def coverage(bars):
    """Missing-hour share per symbol per split; returns (table, void set)."""
    rows, void = [], set()
    for sym, g in bars.groupby("symbol"):
        for sp, (a, b) in SPLITS.items():
            exp = int((pd.Timestamp(b) - pd.Timestamp(a)).days * 24)
            have = ((g.ts >= pd.Timestamp(a, tz="UTC")) & (g.ts < pd.Timestamp(b, tz="UTC"))).sum()
            miss = 1 - have / exp
            rows.append(dict(symbol=sym, split=sp, expected=exp, have=int(have), missing_pct=100 * miss))
            if sp != "TRAIN" and miss > 0.05: void.add(sym)
    return pd.DataFrame(rows), void


def kill_eval(tab):
    """PREREG kill rules on S2 long BTC+ETH taker."""
    g = lambda sp, cost="taker": tab[(tab.signal == "S2") & (tab.side == "long") & (tab.scope == "BTC+ETH")
                                      & (tab.split == sp) & (tab.cost == cost)].iloc[0]
    v, vm, h1, h2 = g("VAL"), g("VAL", "maker"), g("TRAIN_2021"), g("TRAIN_2022")
    t = min(v.t_iid, v.t_day)
    reasons = []
    if not v.net_R > 0: reasons.append(f"VAL net {v.net_R:.3f} <= 0")
    if not t >= 2: reasons.append(f"VAL t {t:.2f} < 2")
    if np.sign(h1.net_R) != np.sign(h2.net_R): reasons.append(f"TRAIN halves {h1.net_R:.3f}/{h2.net_R:.3f} opposite sign")
    if not v.ex_top5 > 0: reasons.append(f"VAL ex-top5 {v.ex_top5:.3f} <= 0")
    if v.n_per_wk < 3: reasons.append(f"VAL trades/wk {v.n_per_wk:.2f} < 3")
    if np.sign(vm.net_R) != np.sign(v.net_R): reasons.append(f"maker sign differs {vm.net_R:.3f}")
    return ("KILL" if reasons else "SURVIVE"), reasons


def main():
    train_only = "--train-only" in sys.argv
    bars = pd.read_parquet(BARS)
    cov, void = coverage(bars)
    print(cov.to_string(), "\nVOID:", void, flush=True)
    cov.to_csv(D + "1760_coverage.csv", index=False)
    parts = [signals_for_symbol(s, g) for s, g in bars.groupby("symbol") if s not in void]
    trades = add_net(pd.concat(parts, ignore_index=True))
    trades.to_parquet(D + "1760_cache/trades.parquet")
    tr_tab = table(trades, ["TRAIN_2021", "TRAIN_2022", "TRAIN"])
    tr_tab.to_csv(D + "1760_train.csv", index=False)  # TRAIN table written before VAL is scored
    print("TRAIN table written", flush=True)
    show = ["signal", "side", "scope", "split", "cost", "n", "n_per_wk", "gross_R", "net_R", "t_iid", "t_day", "ex_top5"]
    pd.set_option("display.width", 250)
    print(tr_tab[(tr_tab.cost == "taker") & tr_tab.scope.isin(["BTC+ETH"]) & tr_tab.signal.isin(["S1", "S2"])][show].round(3).to_string(), flush=True)
    if train_only:
        print("median R%, cost R:\n", trades.groupby(["sig"]).agg(r=("r_pct", "median"), c=("cost_R_taker", "median"), n=("r_pct", "size")))
        return
    full = pd.concat([tr_tab, table(trades, ["VAL"])], ignore_index=True)
    full.to_csv(D + "1760_summary.csv", index=False)
    pd.set_option("display.max_rows", 500)
    print(full[(full.split == "VAL") & (full.cost == "taker")][show + ["med_r_pct", "med_cost_R", "mde_R"]].round(3).to_string())
    print(kill_eval(full))


if __name__ == "__main__":
    main()
