"""Cell 1,743 scorer: variance ratio / first-passage / drift diagnostics on stored HOD paths (PREREG_1743.md).

Reads only paths.parquet (day,symbol,m,c,vwap,minutes_since_entry), signals.parquet and b0_trades.csv.
Rows whose signal split is TEST are dropped at the inner join and never analysed.
"""
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LAB = "/home/ec2-user/onemil/research/hod_exit_lab/"
OUT = "/home/ec2-user/onemil/research/hod_entry/1743_"
QS = [2, 5, 10, 15, 30, 60]
NBOOT = 1000
SESSION_END_M = 955  # 15:55
rng = np.random.default_rng(1743)


def log(msg):
    """Verbose progress, flushed."""
    print(msg, flush=True)


def build_matrices(sig):
    """Return (sig_idx_df, dict price->(N x L) matrix, offset) of paths for TRAIN/VAL signals only."""
    p = pq.read_table(LAB + "paths.parquet",
                      columns=["day", "symbol", "m", "c", "vwap", "minutes_since_entry"]).to_pandas()
    log(f"paths rows {len(p)}; mse range {p.minutes_since_entry.min()}..{p.minutes_since_entry.max()}; m max {p.m.max()}")
    p = p[p.m <= SESSION_END_M]
    key = sig[["day", "symbol"]].copy()
    key["row"] = np.arange(len(sig))
    p = p.merge(key, on=["day", "symbol"], how="inner")  # drops TEST (not in sig)
    dup = p.duplicated(["row", "minutes_since_entry"]).sum()
    log(f"paths rows after join {len(p)}; dup (signal,mse) {dup}; signals with paths {p.row.nunique()} of {len(sig)}")
    off = -int(p.minutes_since_entry.min())
    L = int(p.minutes_since_entry.max()) + off + 1
    mats = {}
    for col in ["c", "vwap"]:
        M = np.full((len(sig), L), np.nan)
        M[p.row.values, p.minutes_since_entry.values + off] = p[col].values
        mats[col] = M
    return mats, off


def vr_suff(Mlog, off, lo, hi):
    """Per-signal sufficient stats for VR over minute window lo..hi (base lo-1). Returns array N x (3+3*len(QS))."""
    x = Mlog[:, lo - 1 + off: hi + off + 1]  # x_0..x_n
    r1 = x[:, 1:] - x[:, :-1]
    ok1 = ~np.isnan(r1)
    cols = [ok1.sum(1), np.nansum(r1, 1), np.nansum(r1 ** 2, 1)]
    for q in QS:
        rq = x[:, q:] - x[:, :-q]
        okq = ~np.isnan(rq)
        cols += [okq.sum(1), np.nansum(rq, 1), np.nansum(rq ** 2, 1)]
    return np.column_stack(cols).astype(float), r1


def vr_from_sums(T):
    """VR for each q from summed stats T (length 3+3*len(QS))."""
    n1, s1, s2 = T[0], T[1], T[2]
    mu = s1 / n1
    v1 = s2 / n1 - mu ** 2
    out = []
    for i, q in enumerate(QS):
        nq, sq, sq2 = T[3 + 3 * i: 6 + 3 * i]
        vq = (sq2 - 2 * q * mu * sq + q * q * mu * mu * nq) / nq
        out.append(vq / (q * v1))
    return np.array(out)


def lm_z(r1, vr):
    """Heteroskedasticity-robust Lo-MacKinlay z (within-signal lags only) for each q."""
    mu = np.nansum(r1) / np.sum(~np.isnan(r1))
    d = (r1 - mu) ** 2
    den = np.nansum(d) ** 2
    cache = {}
    z = []
    for q, v in zip(QS, vr):
        th = 0.0
        for j in range(1, q):
            if j not in cache:
                cache[j] = np.nansum(d[:, j:] * d[:, :-j])
            th += (2 * (q - j) / q) ** 2 * cache[j] / den
        z.append((v - 1) / np.sqrt(th))
    return np.array(z)


def boot_counts(n, nboot=NBOOT):
    """Bootstrap multiplicity matrix nboot x n (resample signals)."""
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(nboot)]).astype(float)


def vr_block(Mlog, off, idx, lo, hi, with_z=True):
    """VR point, bootstrap CI and z for subset idx. Returns list of dict per q."""
    S, r1 = vr_suff(Mlog[idx], off, lo, hi)
    pt = vr_from_sums(S.sum(0))
    W = boot_counts(len(idx))
    BT = W @ S
    bs = np.array([vr_from_sums(t) for t in BT])
    ci = np.percentile(bs, [2.5, 97.5], axis=0)
    z = lm_z(r1, pt) if with_z else np.full(len(QS), np.nan)
    return [dict(q=q, VR=pt[i], lo=ci[0, i], hi=ci[1, i], z_LM=z[i], n_ret1=int(S[:, 0].sum()),
                 n_signals=len(idx)) for i, q in enumerate(QS)]


def passage(c, entry, R, a, b, idx):
    """First passage outcomes (1=stop first, 2=target first, 0=neither) for closes at mse>=1 within the session."""
    off = PASS_OFF
    X = c[idx][:, off + 1:]
    e = entry[idx][:, None]
    r = R[idx][:, None]
    stop = X <= e - a * r
    tgt = X >= e + b * r
    ts = np.where(stop.any(1), stop.argmax(1), 10 ** 6)
    tt = np.where(tgt.any(1), tgt.argmax(1), 10 ** 6)
    out = np.zeros(len(idx), int)
    out[(ts < tt)] = 1
    out[(tt < ts)] = 2
    return out


def prop_ci(ind, W):
    """Point + bootstrap CI for mean of indicator vector using weights W."""
    pt = ind.mean()
    bs = (W @ ind) / W.sum(1)
    return pt, *np.percentile(bs, [2.5, 97.5])


def main():
    sig = pd.read_parquet(LAB + "signals.parquet")
    sig = sig[sig.split.isin(["TRAIN", "VAL"])].reset_index(drop=True)
    log(f"signals TRAIN/VAL {len(sig)}; dup day+symbol {sig.duplicated(['day', 'symbol']).sum()}")
    b0 = pd.read_csv(LAB + "b0_trades.csv", usecols=["day", "symbol", "net_R", "split"])
    b0 = b0[b0.split.isin(["TRAIN", "VAL"])]
    log(f"b0 rows TRAIN/VAL {len(b0)}; dup {b0.duplicated(['day', 'symbol']).sum()}")
    sig = sig.merge(b0[["day", "symbol", "net_R"]].drop_duplicates(["day", "symbol"]), on=["day", "symbol"], how="left")
    log(f"signals without b0 net_R: {sig.net_R.isna().sum()}")
    mats, off = build_matrices(sig)
    global PASS_OFF
    PASS_OFF = off
    with np.errstate(invalid="ignore", divide="ignore"):
        Lc = np.log(mats["c"])
        Lv = np.log(mats["vwap"])
    sets = {"TRAIN": np.where(sig.split == "TRAIN")[0], "VAL": np.where(sig.split == "VAL")[0]}
    sets["TRAIN_H1"] = np.where((sig.split == "TRAIN") & (sig.half == "H1"))[0]
    sets["TRAIN_H2"] = np.where((sig.split == "TRAIN") & (sig.half == "H2"))[0]
    log({k: len(v) for k, v in sets.items()})
    windows = {"1-120": (1, 120), "1-15": (1, 15), "16-120": (16, 120), "pre-60": (-59, 0)}

    # ---- 1. variance ratios
    rows = []
    for sname, idx in sets.items():
        for price, M in [("c", Lc), ("vwap", Lv)]:
            for wname, (lo, hi) in windows.items():
                if price == "vwap" and wname != "1-120":
                    continue
                for d in vr_block(M, off, idx, lo, hi, with_z=(price == "c")):
                    d.update(split=sname, price=price, window=wname)
                    rows.append(d)
        log(f"VR done {sname}")
    vr = pd.DataFrame(rows)
    vr.to_csv(OUT + "vr.csv", index=False)

    # ---- 2. first passage
    cmat, entry, R = mats["c"], sig.entry.values, sig.R.values
    prow = []
    pas = {}
    for sname in ["TRAIN", "VAL", "TRAIN_H1", "TRAIN_H2"]:
        idx = sets[sname]
        W = boot_counts(len(idx))
        for a, b in [(1, 2), (1, 1), (0.5, 1)]:
            o = passage(cmat, entry, R, a, b, idx)
            pas[(sname, a, b)] = o
            bound = a / (a + b)
            row = dict(split=sname, stop_R=-a, target_R=b, n=len(idx), driftless_stop_bound=bound)
            for lab, code in [("stop_first", 1), ("target_first", 2), ("neither", 0)]:
                pt, lo, hi = prop_ci((o == code).astype(float), W)
                row.update({f"P_{lab}": pt, f"P_{lab}_lo": lo, f"P_{lab}_hi": hi})
            prow.append(row)
    pd.DataFrame(prow).to_csv(OUT + "passage.csv", index=False)
    log("passage done")

    # ---- 3. drift
    ff = pd.DataFrame(Lc).ffill(axis=1).values  # last known log close
    drow = []
    for sname in ["TRAIN", "VAL"]:
        idx = sets[sname]
        for wname, (lo, hi) in [("1-15", (1, 15)), ("16-60", (16, 60)), ("61-120", (61, 120))]:
            ret = (ff[idx, hi + off] - ff[idx, lo - 1 + off]) / (hi - lo + 1) * 1e4
            ret = ret[~np.isnan(ret)]
            drow.append(dict(split=sname, window=wname, mean_bps_per_min=ret.mean(),
                             se_signal=ret.std(ddof=1) / np.sqrt(len(ret)), n=len(ret)))
        lastc = np.array([np.exp(ff[i, -1]) for i in idx])
        end_bps = (np.log(lastc) - np.log(entry[idx])) * 1e4
        end_R = (lastc - entry[idx]) / R[idx]
        days = sig.day.values[idx]
        dser = pd.Series(end_R).groupby(days).sum()
        se_day = np.sqrt((dser ** 2).sum() - len(dser) * (dser.mean() ** 2)) / len(idx)
        drow.append(dict(split=sname, window="entry-to-1555 bps", mean_bps_per_min=end_bps.mean(),
                         se_signal=end_bps.std(ddof=1) / np.sqrt(len(idx)), n=len(idx)))
        drow.append(dict(split=sname, window="entry-to-1555 R", mean_bps_per_min=end_R.mean(),
                         se_signal=end_R.std(ddof=1) / np.sqrt(len(idx)), n=len(idx), se_day=se_day))
    pd.DataFrame(drow).to_csv(OUT + "drift.csv", index=False)
    log("drift done")

    # ---- 4. subgroups
    srows = []
    for sname in ["TRAIN", "VAL"]:
        idx_all = sets[sname]
        sub = sig.iloc[idx_all]
        for var in ["half_spread", "r_pct", "entry_m"]:
            terc = pd.qcut(sub[var].rank(method="first"), 3, labels=["T1", "T2", "T3"])
            for t in ["T1", "T2", "T3"]:
                idx = idx_all[(terc == t).values]
                v = vr_block(Lc, off, idx, 1, 120, with_z=False)[QS.index(10)]
                o = passage(cmat, entry, R, 1, 2, idx)
                W = boot_counts(len(idx))
                pt, plo, phi = prop_ci((o == 2).astype(float), W)
                nr = sig.net_R.values[idx]
                nr = nr[~np.isnan(nr)]
                srows.append(dict(split=sname, var=var, tercile=t, n=len(idx),
                                  var_lo=sub.loc[terc == t, var].min(), var_hi=sub.loc[terc == t, var].max(),
                                  VR10=v["VR"], VR10_lo=v["lo"], VR10_hi=v["hi"],
                                  P_target=pt, P_target_lo=plo, P_target_hi=phi,
                                  B0_netR=nr.mean(), B0_netR_se=nr.std(ddof=1) / np.sqrt(len(nr))))
        log(f"subgroups done {sname}")
    pd.DataFrame(srows).to_csv(OUT + "subgroups.csv", index=False)
    log("ALL DONE")


if __name__ == "__main__":
    main()
