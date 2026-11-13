"""Cell 1,759 scorer: F3b leveraged-ETF rebalance flow into the close, SPY/QQQ 1-min 2016-2023.

Follows PREREG_1759.md. TEST (2024 ->) is dropped at load and counted. The TRAIN grid is written first;
VAL is computed and appended only afterwards (read once). Cost 1 bp per leg (2 bp round trip).
"""
import sys
import numpy as np
import pandas as pd

CACHE = "/home/ec2-user/onemil/research/index_orb/cache/{}_1min.parquet"
OUT = "/home/ec2-user/onemil/research/hod_entry/1759_grid.csv"
COST_BP = 2.0
KS = [0.005, 0.010, 0.015, 0.020]
SYMS = ["SPY", "QQQ"]
WINDOWS = {"15:00": "c1500", "15:30": "c1530"}


def load_days(sym):
    """Return (per-day frame, n_test_rows_dropped, n_days_dropped_missing_bars) for one symbol."""
    df = pd.read_parquet(CACHE.format(sym), columns=["timestamp", "open", "close"])
    ts = pd.DatetimeIndex(df["timestamp"])
    if ts.tz is None:
        print("WARNING: naive timestamps, assuming US/Eastern", flush=True)
        ts = ts.tz_localize("US/Eastern")
    else:
        ts = ts.tz_convert("US/Eastern")
    df["date"] = ts.normalize().tz_localize(None)
    df["hm"] = ts.hour * 100 + ts.minute
    test = df["date"] >= pd.Timestamp("2024-01-01")
    n_test = int(test.sum())
    df = df[~test]

    def pick(hm, col):
        s = df[df.hm == hm].drop_duplicates("date").set_index("date")[col]
        return s
    d = pd.DataFrame({"open0930": pick(930, "open"), "c1500": pick(1500, "close"),
                      "c1530": pick(1530, "close"), "c1555": pick(1555, "close"),
                      "c1559": pick(1559, "close")})
    all_days = df["date"].drop_duplicates().sort_values()
    d = d.reindex(all_days)
    d["prev_close"] = d["c1559"].shift(1)
    n_all = len(d)
    d = d.drop(columns=["c1559"]).dropna()
    d = d[d.index > d.index[0]]
    n_drop = n_all - len(d)
    d["r_prime"] = d["c1500"] / d["prev_close"] - 1
    d["r"] = d["c1500"] / d["open0930"] - 1
    d["year"] = d.index.year
    return d, n_test, n_drop


def stats(pnl):
    """Summary stats for an array of per-day pnl in bp."""
    n = len(pnl)
    if n == 0:
        return dict(n=0, mean_bp=np.nan, t=np.nan, hit=np.nan, ex_top5=np.nan, worst=np.nan, mde_bp=np.nan)
    m = pnl.mean()
    sd = pnl.std(ddof=1) if n > 1 else np.nan
    t = m / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    k = int(np.ceil(0.05 * n))
    ex = np.sort(pnl)[:-k].mean() if n > k else np.nan
    return dict(n=n, mean_bp=m, t=t, hit=(pnl > 0).mean(), ex_top5=ex, worst=pnl.min(),
                mde_bp=2.8 * sd / np.sqrt(n) if n > 1 else np.nan)


def trade_pnl(d, retcol, wcol, k=None, comparator=False):
    """Per-day pnl (bp) trading WITH the sign of retcol from window close to 15:55 close."""
    x = d[retcol]
    if comparator:
        sel = x.abs() < 0.005
    else:
        sel = x.abs() >= k
    sel = sel & (x != 0)
    sub = d[sel]
    gross = np.sign(sub[retcol]) * (sub["c1555"] / sub[wcol] - 1) * 1e4
    return (gross - COST_BP), sub


def rows_for(dd, split_name, years):
    """Grid rows for one split over all symbols/defs/windows/k/sign, plus comparator and by-year."""
    out = []
    for sym, d in dd.items():
        d = d[d.year.isin(years)]
        for retcol in ["r_prime", "r"]:
            for wname, wcol in WINDOWS.items():
                cfgs = [(f"k{int(k*1000)/10}%", k, False) for k in KS] + [("cmp<0.5%", None, True)]
                for label, k, cmp_ in cfgs:
                    pnl, sub = trade_pnl(d, retcol, wcol, k, cmp_)
                    sgn = np.sign(sub[retcol])
                    for sg, mask in [("all", np.ones(len(sub), bool)), ("up", (sgn > 0).values), ("down", (sgn < 0).values)]:
                        out.append(dict(split=split_name, sym=sym, ret=retcol, window=wname, cfg=label, sign=sg,
                                        year="all", **stats(pnl.values[mask])))
                    if label in ("k1.0%", "cmp<0.5%"):
                        for y in sorted(set(sub.year)):
                            out.append(dict(split=split_name, sym=sym, ret=retcol, window=wname, cfg=label,
                                            sign="all", year=y, **stats(pnl.values[(sub.year == y).values])))
    return out


def main():
    dd, meta = {}, {}
    for s in SYMS:
        d, n_test, n_drop = load_days(s)
        dd[s] = d
        meta[s] = (n_test, n_drop, len(d))
        print(f"{s}: TEST rows dropped {n_test}; days dropped (missing bars) {n_drop}; days kept {len(d)} "
              f"({d.index.min().date()} -> {d.index.max().date()})", flush=True)
    train = rows_for(dd, "TRAIN_2016-18", [2016, 2017, 2018]) + rows_for(dd, "TRAIN_2019-20", [2019, 2020]) \
        + rows_for(dd, "TRAIN", [2016, 2017, 2018, 2019, 2020])
    pd.DataFrame(train).to_csv(OUT, index=False)
    print("TRAIN grid written; reading VAL once.", flush=True)
    val = rows_for(dd, "VAL", [2021, 2022, 2023])
    g = pd.concat([pd.DataFrame(train), pd.DataFrame(val)], ignore_index=True)
    g.to_csv(OUT, index=False)

    def pick(split, sym="SPY", ret="r_prime", w="15:00", cfg="k1.0%", sign="all", year="all"):
        q = g[(g.split == split) & (g.sym == sym) & (g.ret == ret) & (g.window == w) & (g.cfg == cfg)
              & (g.sign == sign) & (g.year.astype(str) == str(year))]
        return q.iloc[0]
    pd.set_option("display.width", 200)
    cols = ["n", "mean_bp", "t", "hit", "ex_top5", "worst", "mde_bp"]
    print("\n== TRAIN halves, SPY r' k1% 15:00 ==")
    for sp in ["TRAIN_2016-18", "TRAIN_2019-20", "TRAIN"]:
        print(sp, pick(sp)[cols].astype(float).round(3).to_dict())
    print("\n== VAL SPY r' 15:00 by k ==")
    for k in ["k0.5%", "k1.0%", "k1.5%", "k2.0%"]:
        print(k, pick("VAL", cfg=k)[cols].astype(float).round(3).to_dict())
    print("\n== VAL primary by sign ==")
    for sg in ["all", "up", "down"]:
        print(sg, pick("VAL", sign=sg)[cols].astype(float).round(3).to_dict())
    print("\n== VAL variants ==")
    print("15:30", pick("VAL", w="15:30")[cols].astype(float).round(3).to_dict())
    print("QQQ", pick("VAL", sym="QQQ")[cols].astype(float).round(3).to_dict())
    print("cmp<0.5%", pick("VAL", cfg="cmp<0.5%")[cols].astype(float).round(3).to_dict())
    print("open-based r", pick("VAL", ret="r")[cols].astype(float).round(3).to_dict())
    print("\n== year signs primary (SPY r' k1%) ==")
    for y in [2016, 2017, 2018, 2019, 2020]:
        print(y, pick("TRAIN", year=y)[["n", "mean_bp", "t"]].astype(float).round(2).to_dict())
    for y in [2021, 2022, 2023]:
        print(y, pick("VAL", year=y)[["n", "mean_bp", "t"]].astype(float).round(2).to_dict())
    v, c = pick("VAL"), pick("VAL", cfg="cmp<0.5%")
    h1, h2 = pick("TRAIN_2016-18"), pick("TRAIN_2019-20")
    kills = []
    if not (v.mean_bp > 5 and v.t >= 2):
        kills.append(f"VAL mean {v.mean_bp:.2f} <=5 or t {v.t:.2f} <2")
    if np.sign(h1.mean_bp) != np.sign(h2.mean_bp):
        kills.append("TRAIN halves opposite")
    if not (c.mean_bp < v.mean_bp):
        kills.append(f"comparator {c.mean_bp:.2f} not below trade {v.mean_bp:.2f}")
    if v.n < 30:
        kills.append(f"VAL n {v.n} < 30")
    print("\nVERDICT:", "KILL: " + "; ".join(kills) if kills else "SURVIVE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
