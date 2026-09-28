"""Independent rebuild of cells 1,640-1,642 (VIX basis carry) from PREREG_1640.md prose alone.

Do not import or reference cell_1640.py. This script recomputes F30, basis, signal, and
P&L from raw data under research/vix_carry/data/ only.
"""
import glob
import os
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = "/home/ec2-user/onemil/research/vix_carry/data"
OUT_CSV = "/home/ec2-user/onemil/research/vix_carry/daily_1640_rebuild.csv"

ENTRY_THRESH = 0.03
COST_BPS = 0.0005  # 5 bps per leg


def load_vx_contracts():
    """Parse each vx_raw/VX_<expiry>.csv into a long (date, expiry, settle) frame."""
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, "vx_raw", "VX_*.csv"))):
        m = re.search(r"VX_(\d{4}-\d{2}-\d{2})\.csv$", path)
        expiry = pd.Timestamp(m.group(1))
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["Trade Date"], format="mixed")
        # Data-quality fallback: 2013-2014-vintage CBOE files leave Settle==0 for many
        # rows that DID trade (nonzero Close); use Close as the settlement proxy there.
        # Irrelevant to the reportable window (Alpaca ETP data starts 2016-01-04) but
        # kept for correctness of the full contracts table.
        df["settle_eff"] = np.where(df["Settle"] > 0, df["Settle"], df["Close"])
        df = df[df["settle_eff"] > 0][["date", "settle_eff"]].copy()
        df["expiry"] = expiry
        rows.append(df.rename(columns={"settle_eff": "settle"}))
    return pd.concat(rows, ignore_index=True)


def compute_f30(contracts):
    """F30_t by linear interpolation of the two nearest (by calendar days) live expiries."""
    out = []
    for date, grp in contracts.groupby("date"):
        g = grp[grp["expiry"] >= date].sort_values("expiry")
        if len(g) < 2:
            continue
        front, nxt = g.iloc[0], g.iloc[1]
        dA = (front["expiry"] - date).days
        dB = (nxt["expiry"] - date).days
        if dB == dA:
            continue
        f30 = front["settle"] + (nxt["settle"] - front["settle"]) * (30 - dA) / (dB - dA)
        out.append((date, f30, front["expiry"], nxt["expiry"], dA, dB,
                     front["settle"], nxt["settle"]))
    f = pd.DataFrame(out, columns=["date", "F30", "front_expiry", "next_expiry",
                                    "dA", "dB", "front_settle", "next_settle"])
    return f.sort_values("date").reset_index(drop=True)


def load_vix_spot():
    df = pd.read_csv(os.path.join(DATA, "vix_spot.csv"))
    df["date"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    df = df[["date", "CLOSE"]].rename(columns={"CLOSE": "vix"})
    return df.sort_values("date").reset_index(drop=True)


def load_etp(symbol):
    df = pd.read_csv(os.path.join(DATA, f"{symbol}_daily.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "open", "close"]].sort_values("date").reset_index(drop=True)


def build_basis(f30, vix):
    m = pd.merge(f30, vix, on="date", how="inner")
    m["basis"] = m["F30"] / m["vix"] - 1.0
    m["vix_mean20"] = m["vix"].rolling(20, min_periods=20).mean()
    return m.sort_values("date").reset_index(drop=True)


def run_backtest(basis_df, etp_df, two_close=True):
    """State machine: decisions made on day D-1's close data, executed at day D's open."""
    m = pd.merge(basis_df, etp_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    m["basis_lag1"] = m["basis"].shift(1)
    m["basis_lag2"] = m["basis"].shift(2)
    m["vix_lag1"] = m["vix"].shift(1)
    m["vix_mean20_lag1"] = m["vix_mean20"].shift(1)
    m["close_lag1"] = m["close"].shift(1)

    state = "FLAT"
    pnl = np.full(len(m), np.nan)
    in_mkt = np.zeros(len(m), dtype=bool)
    action = [""] * len(m)

    for i in range(len(m)):
        row = m.iloc[i]
        if pd.isna(row["basis_lag1"]) or pd.isna(row["vix_mean20_lag1"]) or pd.isna(row["close_lag1"]):
            pnl[i] = np.nan
            continue
        entry_ok = row["basis_lag1"] >= ENTRY_THRESH and (
            (row["basis_lag2"] >= ENTRY_THRESH) if two_close else True
        )
        exit_ok = (row["basis_lag1"] <= 0) or (row["vix_lag1"] > 1.25 * row["vix_mean20_lag1"])

        if state == "FLAT":
            if not pd.isna(row["basis_lag2"]) and entry_ok:
                pnl[i] = (row["close"] / row["open"] - 1.0) - COST_BPS
                in_mkt[i] = True
                action[i] = "ENTER"
                state = "LONG"
            else:
                pnl[i] = 0.0
                in_mkt[i] = False
        else:  # LONG
            if exit_ok:
                pnl[i] = (row["open"] / row["close_lag1"] - 1.0) - COST_BPS
                in_mkt[i] = True
                action[i] = "EXIT"
                state = "FLAT"
            else:
                pnl[i] = (row["close"] / row["close_lag1"] - 1.0)
                in_mkt[i] = True
                action[i] = "HOLD"

    m["pnl"] = pnl
    m["in_market"] = in_mkt
    m["action"] = action
    return m


def nw_tstat(x, lags=5):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < lags + 2:
        return np.nan, len(x)
    X = np.ones(len(x))
    model = sm.OLS(x, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return model.tvalues[0], len(x)


def max_drawdown(pnl_series):
    eq = (1 + pnl_series.fillna(0)).cumprod()
    run_max = eq.cummax()
    dd = eq / run_max - 1.0
    return dd.min()


def split_metrics(df, start, end, label):
    sub = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if len(sub) == 0:
        return {"split": label, "n_days": 0}
    in_mkt = sub[sub["in_market"]]
    t, n = nw_tstat(in_mkt["pnl"])
    ann_ret = (1 + sub["pnl"].fillna(0)).prod() ** (252.0 / len(sub)) - 1.0 if len(sub) > 0 else np.nan
    worst_day = sub["pnl"].min()
    monthly = sub.set_index("date")["pnl"].fillna(0).groupby(pd.Grouper(freq="ME")).apply(lambda x: (1 + x).prod() - 1)
    worst_month = monthly.min() if len(monthly) else np.nan
    mdd = max_drawdown(sub.set_index("date")["pnl"])
    return {
        "split": label,
        "n_days": len(sub),
        "days_in_market": int(in_mkt.shape[0]),
        "share_in_market": in_mkt.shape[0] / len(sub),
        "mean_bps_day_in_market": in_mkt["pnl"].mean() * 1e4 if len(in_mkt) else np.nan,
        "nw_t": t,
        "n_in_market": n,
        "annualised_return": ann_ret,
        "worst_day_pct": worst_day * 100 if pd.notna(worst_day) else np.nan,
        "worst_month_pct": worst_month * 100 if pd.notna(worst_month) else np.nan,
        "max_drawdown_pct": mdd * 100,
        "pnl_750_total": (sub["pnl"].fillna(0) * 750).sum(),
        "pnl_5000_total": (sub["pnl"].fillna(0) * 5000).sum(),
        "worst_day_dollar_750": worst_day * 750 if pd.notna(worst_day) else np.nan,
        "worst_day_dollar_5000": worst_day * 5000 if pd.notna(worst_day) else np.nan,
    }


def decile_table(basis_df, etp_df, start, end, drop_worst_n=0):
    m = pd.merge(basis_df, etp_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    m["next_ret"] = m["close"].shift(-1) / m["close"] - 1.0
    sub = m[(m["date"] >= start) & (m["date"] <= end)].dropna(subset=["basis", "next_ret"]).copy()
    if drop_worst_n > 0:
        sub = sub.sort_values("next_ret").iloc[drop_worst_n:]
    sub["decile"] = pd.qcut(sub["basis"], 10, labels=False, duplicates="drop")
    tbl = sub.groupby("decile").agg(mean_next_ret=("next_ret", "mean"), n=("next_ret", "size"),
                                     mean_basis=("basis", "mean"))
    corr = sub["decile"].astype(float).corr(sub["next_ret"])
    return tbl, corr


def price_scale_check(etp_df, symbol, thresh=0.40):
    df = etp_df.copy().sort_values("date")
    df["ret"] = df["close"].pct_change()
    flagged = df[df["ret"].abs() > thresh][["date", "ret"]]
    flagged["symbol"] = symbol
    return flagged


def main():
    print("Loading VX contract settlements...")
    contracts = load_vx_contracts()
    print(f"  {contracts['expiry'].nunique()} contracts, {len(contracts)} settle rows")

    print("Computing F30 (linear interpolation, two nearest expiries)...")
    f30 = compute_f30(contracts)
    print(f"  F30 computed for {len(f30)} dates, {f30['date'].min().date()} - {f30['date'].max().date()}")

    vix = load_vix_spot()
    basis = build_basis(f30, vix)
    print(f"  basis computed for {len(basis)} dates")

    svxy = load_etp("SVXY")
    svix = load_etp("SVIX")

    # TEST (2024-01-01..2026-09) is SEALED per PREREG/task instructions: compute
    # NOTHING on it. Truncate all inputs to TRAIN+VAL (<=2023-12-31) before any
    # backtest, decile, or price-scale computation.
    SEAL = pd.Timestamp("2023-12-31")
    basis = basis[basis["date"] <= SEAL].reset_index(drop=True)
    svxy_full = svxy.copy()  # kept only for the out-of-seal row count printed below
    svxy = svxy[svxy["date"] <= SEAL].reset_index(drop=True)
    svix = svix[svix["date"] <= SEAL].reset_index(drop=True)
    print(f"  SVXY (<=TEST seal) {svxy['date'].min().date()}..{svxy['date'].max().date()} ({len(svxy)} rows;"
          f" {len(svxy_full) - len(svxy)} rows after 2023-12-31 excluded as sealed TEST)")
    print(f"  SVIX (<=TEST seal) {svix['date'].min().date()}..{svix['date'].max().date()} ({len(svix)} rows)")

    print("Running backtests...")
    bt_1640 = run_backtest(basis, svxy, two_close=True)
    bt_1640_one_close = run_backtest(basis, svxy, two_close=False)
    bt_1641 = run_backtest(basis, svix, two_close=True)

    # 1,642 always-in SVXY: merge basis+svxy for date alignment, in_market always True after warmup
    always_in = pd.merge(basis, svxy, on="date", how="inner").sort_values("date").reset_index(drop=True)
    always_in["close_lag1"] = always_in["close"].shift(1)
    always_in["pnl"] = always_in["close"] / always_in["close_lag1"] - 1.0
    always_in.loc[always_in.index[0], "pnl"] = np.nan
    always_in["in_market"] = always_in["pnl"].notna()
    always_in["action"] = np.where(always_in["in_market"], "HOLD", "")

    # ---- write combined daily CSV ----
    def tag(df, cell):
        d = df[["date", "basis", "vix", "in_market", "pnl", "action"]].copy()
        d["cell"] = cell
        d["pnl_bps"] = d["pnl"] * 1e4
        return d

    combined = pd.concat([
        tag(bt_1640, "1640_SVXY_gated"),
        tag(bt_1641, "1641_SVIX_gated"),
        tag(always_in, "1642_SVXY_always_in"),
    ], ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(combined)} rows)")

    # ---- splits ----
    splits_1640 = [
        ("TRAIN_-1x_2016-01..2018-02-27", "2016-01-04", "2018-02-27"),
        ("TRAIN_-0.5x_2018-02-28..2019-12-31", "2018-02-28", "2019-12-31"),
        ("VAL_2020-01..2023-12", "2020-01-01", "2023-12-31"),
    ]
    print("\n=== 1,640 SVXY basis-gated (two-close, frozen) ===")
    results_1640 = [split_metrics(bt_1640, pd.Timestamp(a), pd.Timestamp(b), lbl) for lbl, a, b in splits_1640]
    for r in results_1640:
        print(r)

    print("\n=== 1,640 one-close variant (report-only) ===")
    results_1640_1c = [split_metrics(bt_1640_one_close, pd.Timestamp(a), pd.Timestamp(b), lbl) for lbl, a, b in splits_1640]
    for r in results_1640_1c:
        print(r)

    print("\n=== 1,642 always-in SVXY (report-only) ===")
    results_1642 = [split_metrics(always_in, pd.Timestamp(a), pd.Timestamp(b), lbl) for lbl, a, b in splits_1640]
    for r in results_1642:
        print(r)

    print("\n=== 1,641 SVIX basis-gated (short sample, report-only) ===")
    r1641 = split_metrics(bt_1641, pd.Timestamp("2022-03-30"), pd.Timestamp("2023-12-31"), "VAL_partial_2022-03..2023-12")
    print(r1641)

    # ---- decile tables (VAL) ----
    print("\n=== VAL decile table (all days) ===")
    tbl_all, corr_all = decile_table(basis, svxy, pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31"), drop_worst_n=0)
    print(tbl_all)
    print("corr(decile, next_ret) =", corr_all)

    print("\n=== VAL decile table (ex-worst-5-days) ===")
    tbl_ex5, corr_ex5 = decile_table(basis, svxy, pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31"), drop_worst_n=5)
    print(tbl_ex5)
    print("corr(decile, next_ret) ex5 =", corr_ex5)

    # ---- price scale check ----
    print("\n=== Price scale check (|daily return| > 40%) ===")
    ps_svxy = price_scale_check(svxy, "SVXY")
    ps_svix = price_scale_check(svix, "SVIX")
    ps = pd.concat([ps_svxy, ps_svix], ignore_index=True)
    print(ps.to_string())
    ps.to_csv("/home/ec2-user/onemil/research/vix_carry/price_scale_check_rebuild.csv", index=False)

    # ---- roll-day discontinuity check ----
    print("\n=== F30 discontinuity at expiry (roll) days vs other days ===")
    basis_sorted = basis.sort_values("date").reset_index(drop=True)
    basis_sorted["db"] = basis_sorted["basis"].diff().abs()
    f30_expiries = set(f30["front_expiry"].unique())
    basis_sorted["is_roll_day"] = basis_sorted["date"].isin(f30_expiries)
    roll_stats = basis_sorted.groupby("is_roll_day")["db"].agg(["max", "mean", "count"])
    print(roll_stats)

    # save summary dict for report generation
    import json
    summary = {
        "1640_two_close": results_1640,
        "1640_one_close": results_1640_1c,
        "1642_always_in": results_1642,
        "1641": r1641,
        "corr_all": corr_all,
        "corr_ex5": corr_ex5,
        "roll_stats": roll_stats.reset_index().to_dict(orient="records"),
        "n_flagged_scale": len(ps),
    }
    with open("/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/summary_1640.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print("\nDone.")


if __name__ == "__main__":
    main()
