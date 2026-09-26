"""Independent rebuild of PREREG_1552 -- the EDGAR event desk (cells 1,552-1,561).

Written from research/edgar_desk/PREREG_1552.md prose ONLY. This session did not open
cell_1552.py, test_cell_1552.py, cell_1552_events.csv, RESULT_1552.md or events_raw.csv.

Scope: TRAIN (2019-2022) and VAL (2023-2024H1) only -- TEST is sealed per the PREREG and is
not read here. Universe is a seed-1552 deterministic SAMPLE of the primary-CIK-mapped liquid
population (see rebuild_fetch.py and REBUILD_1552.md Scope section) -- not the full 8,488-name
population -- because a full live SEC fetch does not fit this task's time/step budget. All
statistics below are computed for real on the fetched sample; n / coverage are reported, not
assumed.
"""
import gzip
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

REPO = "/home/ec2-user/onemil"
SUB_DIR = os.path.join(REPO, "research/edgar_desk/submissions")
SCRATCH = "/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/edgar"
EVENTS_CSV = os.path.join(REPO, "research/edgar_desk/rebuild_1552_events.csv")
LOG_PATH = os.path.join(REPO, "research/edgar_desk/rebuild_1552.log")
FORM4_CACHE_DIR = os.path.join(SCRATCH, "form4_docs")
os.makedirs(FORM4_CACHE_DIR, exist_ok=True)

UA = "onemil research giligen@gmail.com"
RATE_S = 1.05

COST_LEG_BPS = 5.0          # per auction leg, 5bps; E1 and E5 both use exactly one entry + one exit leg
BORROW_APY = 0.03            # short borrow, 3%/yr, pro-rata on calendar days held
WINNER_CAP = 0.20             # +-20% winner cap
SSR_PRIOR_RET = -0.10         # proxy: prior session's own day-over-day return <= -10%
SSR_MIN_PRICE = 5.0
MIN_PRICE = 1.0
MIN_DVOL20 = 1_000_000.0
N_NULL_DRAWS = 1000
NULL_SEED = 1552

TRAIN_START, TRAIN_END = "2019-01-01", "2022-12-31"
VAL_START, VAL_END = "2023-01-01", "2024-06-30"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                     handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("rebuild_1552")

CLASS_DIRECTION = {
    "OFFERING": -1, "SHELF": -1, "REVERSE_SPLIT": -1, "AUDITOR": -1, "NON_RELIANCE": -1,
    "LATE_FILING": -1, "OFFICER_EXIT": -1,
    "CONTRACT": 1, "ACTIVIST": 1, "BUYBACK_OR_INSIDER": 1,
}
CELL_ID = {
    "OFFERING": 1552, "SHELF": 1553, "REVERSE_SPLIT": 1554, "AUDITOR": 1555, "NON_RELIANCE": 1556,
    "LATE_FILING": 1557, "OFFICER_EXIT": 1558, "CONTRACT": 1559, "ACTIVIST": 1560, "BUYBACK_OR_INSIDER": 1561,
}
SHORT_CLASSES = {c for c, d in CLASS_DIRECTION.items() if d == -1}


# ---------------------------------------------------------------------------
# 1. Load submissions cache -> per-filing rows with a class label
# ---------------------------------------------------------------------------

def _items_set(items_str):
    if not items_str:
        return set()
    return {x.strip() for x in items_str.split(",") if x.strip()}


def classify(form, items_str):
    """Structured-code-only class assignment (no filing text is read), per the PREREG table.

    Judgment calls made explicit here (documented again in REBUILD_1552.md):
      * OFFERING takes 424B1-5 and 8-K/3.02 only. An 8-K/1.01 is NEVER routed to OFFERING in
        this structured-only pass (the PREREG's "8-K 1.01 whose text is skipped (structured
        only)" clause is read as an explicit exclusion note, not a third OFFERING trigger --
        deciding whether a 1.01 is a financing agreement needs the text, which this pass may
        not read).
      * CONTRACT (1.01) excludes any filing that ALSO carries 3.02 or 2.03 on the same
        accession (dilutive/debt-raising 1.01s), per the row's own "WITHOUT" clause.
      * SHELF matches only the bare forms S-3 / S-3ASR / S-1 (not '/A' amendments) -- "initial
        ... registration" is read as excluding amendments.
      * OFFICER_EXIT keys on item 5.02 alone; that item code does not structurally separate a
        departure from an appointment, so this pass -- deliberately, per "Not allowed: text
        classification" -- includes BOTH under the same class and flags it as a purity caveat.
      * ACTIVIST is the bare 'SC 13D' form only, not 'SC 13D/A'.
      * BUYBACK_OR_INSIDER (8.01) is a CANDIDATE here; the Form-4-officer-purchase join happens
        in build_events() because it needs a second, targeted fetch.
    """
    items = _items_set(items_str)
    if form in ("424B1", "424B2", "424B3", "424B4", "424B5"):
        return "OFFERING"
    if form == "8-K" and "3.02" in items:
        return "OFFERING"
    if form in ("S-3", "S-3ASR", "S-1"):
        return "SHELF"
    if form == "8-K" and "5.03" in items:
        return "REVERSE_SPLIT"
    if form == "8-K" and "4.01" in items:
        return "AUDITOR"
    if form == "8-K" and "4.02" in items:
        return "NON_RELIANCE"
    if form in ("NT 10-K", "NT 10-Q"):
        return "LATE_FILING"
    if form == "8-K" and "5.02" in items:
        return "OFFICER_EXIT"
    if form == "8-K" and "1.01" in items and not ({"3.02", "2.03"} & items):
        return "CONTRACT"
    if form == "SC 13D":
        return "ACTIVIST"
    if form == "8-K" and "8.01" in items:
        return "BUYBACK_OR_INSIDER_CANDIDATE"
    return None


def load_filings():
    """One row per (symbol, cik, class) filing event from the cached submissions JSONs."""
    rows = []
    n_files = 0
    for fn in sorted(os.listdir(SUB_DIR)):
        if not fn.endswith(".json.gz"):
            continue
        n_files += 1
        with gzip.open(os.path.join(SUB_DIR, fn), "rt") as f:
            doc = json.load(f)
        if doc.get("_missing"):
            continue
        symbol = doc.get("_symbol", "")
        cik = doc.get("cik", "")
        pages = [doc.get("filings", {}).get("recent", {})] + doc.get("_extra_pages", [])
        for page in pages:
            forms = page.get("form", [])
            for i, form in enumerate(forms):
                items_str = (page.get("items", [None] * len(forms)) or [None] * len(forms))[i]
                accept = (page.get("acceptanceDateTime", [None] * len(forms)) or [None] * len(forms))[i]
                fdate = (page.get("filingDate", [None] * len(forms)) or [None] * len(forms))[i]
                accession = (page.get("accessionNumber", [None] * len(forms)) or [None] * len(forms))[i]
                primary_doc = (page.get("primaryDocument", [None] * len(forms)) or [None] * len(forms))[i]
                cls = classify(form, items_str)
                if cls is None:
                    continue
                rows.append(dict(symbol=symbol, cik=cik, form=form, items=items_str,
                                  accept=accept, filingDate=fdate, accession=accession,
                                  primary_doc=primary_doc, cls=cls))
    log.info("loaded %d submissions files, %d classified filing rows", n_files, len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Form-4 officer-purchase join for BUYBACK_OR_INSIDER candidates
# ---------------------------------------------------------------------------

def _get(url, retries=2):
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": UA}, timeout=20)
            time.sleep(RATE_S)
            return resp
        except requests.exceptions.RequestException as e:
            log.warning("Form4 GET failed (attempt %d/%d) %s: %s", attempt, retries, url, e)
    log.error("Form4 GET permanently failed url=%s -- treated as no-purchase-found", url)
    return None


def form4_has_officer_purchase(cik, accession, primary_doc):
    """Fetch one Form-4 XML and return True iff it shows an officer's open-market PURCHASE
    (transactionCode 'P' on a non-derivative table row, isOfficer true). Cached to disk.
    """
    cache_path = os.path.join(FORM4_CACHE_DIR, f"{cik}_{accession}.json")
    if os.path.exists(cache_path):
        return json.load(open(cache_path))["hit"]
    acc_nodash = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{primary_doc}"
    resp = _get(url)
    hit = False
    if resp is not None and resp.status_code == 200:
        text = resp.text
        is_officer = bool(re.search(r"<isOfficer>\s*1\s*</isOfficer>|<isOfficer>true</isOfficer>", text, re.I))
        has_purchase = bool(re.search(r"<transactionCode>\s*P\s*</transactionCode>", text))
        hit = is_officer and has_purchase
    else:
        log.warning("Form4 fetch failed/absent for CIK%s accession=%s -- no purchase credited", cik, accession)
    json.dump({"hit": hit}, open(cache_path, "w"))
    return hit


def resolve_buyback_insider(filings_df, form4_df):
    """8.01 candidate -> BUYBACK_OR_INSIDER iff a same-issuer Form 4 with an officer purchase
    was filed within 2 trading sessions of the 8.01's acceptance (structured join per the row).
    """
    cand = filings_df[filings_df["cls"] == "BUYBACK_OR_INSIDER_CANDIDATE"].copy()
    if cand.empty or form4_df.empty:
        return pd.DataFrame(columns=filings_df.columns)
    cand["accept_dt"] = pd.to_datetime(cand["accept"])
    form4_df = form4_df.copy()
    form4_df["accept_dt"] = pd.to_datetime(form4_df["accept"])
    hits = []
    for _, ev in cand.iterrows():
        window = form4_df[(form4_df["cik"] == ev["cik"]) &
                           (form4_df["accept_dt"] >= ev["accept_dt"]) &
                           (form4_df["accept_dt"] <= ev["accept_dt"] + pd.Timedelta(days=4))]
        found = False
        for _, f4 in window.iterrows():
            if form4_has_officer_purchase(ev["cik"], f4["accession"], f4["primary_doc"]):
                found = True
                break
        if found:
            row = ev.drop(labels=["accept_dt"]).to_dict()
            row["cls"] = "BUYBACK_OR_INSIDER"
            hits.append(row)
    log.info("BUYBACK_OR_INSIDER: %d of %d 8.01 candidates joined to an officer Form-4 purchase",
              len(hits), len(cand))
    return pd.DataFrame(hits)


# ---------------------------------------------------------------------------
# 3. Price panel, population filter, entry/exit auction logic
# ---------------------------------------------------------------------------

def load_prices():
    p = pd.read_parquet(os.path.join(REPO, "research/overnight_high/alpaca_daily_2019_2024H1.parquet"))
    p["bar_date"] = pd.to_datetime(p["bar_date"])
    p = p.sort_values(["symbol", "bar_date"]).reset_index(drop=True)
    p["dvol"] = p["close"] * p["volume"]
    p["dvol20"] = p.groupby("symbol")["dvol"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    g = p.groupby("symbol")
    p["dvol20_prior"] = g["dvol20"].shift(1)
    p["close_prior"] = g["close"].shift(1)
    p["close_prior2"] = g["close"].shift(2)
    p["prior_day_ret"] = p["close_prior"] / p["close_prior2"] - 1
    p["session_idx"] = g.cumcount()
    return p


def entry_and_exits(sym_bars, accept_dt):
    """Return (entry_row, exit_e1_row, exit_e5_row, prior_row) or None if unresolvable within
    this symbol's cached bar range.
    """
    accept_date = pd.Timestamp(accept_dt.date())
    before_9am = accept_dt.time() < pd.Timestamp("09:00:00").time()
    if before_9am:
        cand = sym_bars[sym_bars["bar_date"] >= accept_date]
    else:
        cand = sym_bars[sym_bars["bar_date"] > accept_date]
    if cand.empty:
        return None
    entry_row = cand.iloc[0]
    pos = sym_bars.index.get_loc(entry_row.name)
    if pos == 0:
        return None  # no prior session available for the population/SSR gate
    prior_row = sym_bars.iloc[pos - 1]
    e1_row = entry_row
    if pos + 5 >= len(sym_bars):
        e5_row = None
    else:
        e5_row = sym_bars.iloc[pos + 5]
    return entry_row, e1_row, e5_row, prior_row


def build_events(filings_df, prices):
    """One row per eligible filing event with entry/exit prices, net returns, and split label."""
    by_sym = {s: df.set_index(df.index) for s, df in prices.groupby("symbol")}
    out = []
    lost_no_price = lost_no_prior = lost_illiquid = 0
    for _, r in filings_df.iterrows():
        sym = r["symbol"]
        if sym not in by_sym or not r["accept"]:
            lost_no_price += 1
            continue
        try:
            accept_dt = pd.Timestamp(r["accept"])
        except Exception:
            lost_no_price += 1
            continue
        res = entry_and_exits(by_sym[sym], accept_dt)
        if res is None:
            lost_no_price += 1
            continue
        entry_row, e1_row, e5_row, prior_row = res
        if pd.isna(prior_row["close_prior"]) or pd.isna(prior_row["dvol20_prior"]):
            lost_no_prior += 1
            continue
        if prior_row["close"] < MIN_PRICE or prior_row["dvol20"] < MIN_DVOL20:
            lost_illiquid += 1
            continue
        cls = r["cls"]
        direction = CLASS_DIRECTION[cls]
        if direction == -1:
            if prior_row["close"] < SSR_MIN_PRICE or (not pd.isna(prior_row["prior_day_ret"]) and
                                                       prior_row["prior_day_ret"] <= SSR_PRIOR_RET):
                continue  # SHORT-only SSR/locate exclusion
        open_e = entry_row["open"]
        close_e1 = e1_row["close"]
        raw_e1 = close_e1 / open_e - 1
        cost_e1 = 2 * COST_LEG_BPS / 1e4
        borrow_e1 = BORROW_APY * (0 / 365.0) if direction == -1 else 0.0  # same-session: ~0 days held
        net_e1 = direction * raw_e1 - cost_e1 - borrow_e1
        net_e5 = np.nan
        if e5_row is not None:
            raw_e5 = e5_row["close"] / open_e - 1
            days_held = (e5_row["bar_date"] - entry_row["bar_date"]).days
            borrow_e5 = BORROW_APY * (days_held / 365.0) if direction == -1 else 0.0
            net_e5 = direction * raw_e5 - cost_e1 - borrow_e5
        out.append(dict(symbol=sym, cik=r["cik"], cls=cls, cell=CELL_ID[cls], form=r["form"],
                         items=r["items"], accept=r["accept"], entry_date=entry_row["bar_date"],
                         session=entry_row["bar_date"].strftime("%Y-%m-%d"), direction=direction,
                         net_e1_bps=net_e1 * 1e4, net_e5_bps=net_e5 * 1e4 if not pd.isna(net_e5) else np.nan))
    log.info("events: %d built; lost_no_price=%d lost_no_prior=%d lost_illiquid(pop-filter)=%d",
              len(out), lost_no_price, lost_no_prior, lost_illiquid)
    ev = pd.DataFrame(out)
    if ev.empty:
        return ev
    ev["split"] = np.select(
        [ev["entry_date"].between(TRAIN_START, TRAIN_END), ev["entry_date"].between(VAL_START, VAL_END)],
        ["TRAIN", "VAL"], default="OUT_OF_SCOPE")
    ev = ev[ev["split"] != "OUT_OF_SCOPE"].reset_index(drop=True)
    return ev


# ---------------------------------------------------------------------------
# 4. Universe placebo + count-matched null (per cell, per session)
# ---------------------------------------------------------------------------

def build_universe_returns(prices, leg):
    """Per-session mean return of ALL prior-session-eligible names, for one leg (e1 or e5)."""
    p = prices.copy()
    elig = p[(p["close_prior"] >= MIN_PRICE) & (p["dvol20_prior"] >= MIN_DVOL20)].copy()
    if leg == "e1":
        elig["ret"] = elig["close"] / elig["open"] - 1
    else:
        g = elig.sort_values(["symbol", "bar_date"]).groupby("symbol")
        fwd_close = g["close"].shift(-5)
        fwd_open = elig["open"]
        elig["ret"] = fwd_close / fwd_open - 1
    per_session = elig.groupby(elig["bar_date"].dt.strftime("%Y-%m-%d"))["ret"].agg(["mean", list])
    return per_session


def day_clustered_t(y, day):
    y = pd.Series(y).dropna()
    if len(y) < 2:
        return np.nan
    d = pd.Series(day).loc[y.index]
    if d.nunique() < 2:
        return np.nan
    X = np.ones((len(y), 1))
    model = sm.OLS(y.to_numpy(), X).fit(cov_type="cluster", cov_kwds={"groups": d.to_numpy()})
    return float(model.tvalues[0])


def ex_top_mean(y, pct):
    y = pd.Series(y).dropna().sort_values(ascending=False)
    n = len(y)
    if n == 0:
        return np.nan
    k = int(round(pct * n))
    return float(y.iloc[k:].mean()) if k < n else float(y.mean())


def winner_capped_mean(y, direction, cap=WINNER_CAP):
    y = pd.Series(y).dropna() / 1e4  # bps -> fraction
    if not len(y):
        return np.nan
    return float(np.minimum(y, cap).mean()) * 1e4


def cell_stats(ev_cell, split, leg_col, universe_by_session):
    sub = ev_cell[(ev_cell["split"] == split)].dropna(subset=[leg_col])
    n = len(sub)
    if n == 0:
        return dict(n=0)
    mean_bps = float(sub[leg_col].mean())
    t = day_clustered_t(sub[leg_col], sub["session"])
    ex5 = ex_top_mean(sub[leg_col], 0.05)
    ex1 = ex_top_mean(sub[leg_col], 0.01)
    wc = winner_capped_mean(sub[leg_col], sub["direction"].iloc[0])
    med = float(sub[leg_col].median())
    share_dir = float((sub[leg_col] > 0).mean())
    n_weeks = max(1, (pd.to_datetime(sub["entry_date"]).max() - pd.to_datetime(sub["entry_date"]).min()).days / 7.0)
    events_per_week = n / n_weeks if n_weeks > 0 else np.nan
    uni_sessions = [universe_by_session.at[s, "mean"] * 1e4 for s in sub["session"] if s in universe_by_session.index]
    uni_mean = float(np.mean(uni_sessions)) if uni_sessions else np.nan
    direction = sub["direction"].iloc[0]
    placebo_margin = mean_bps - direction * uni_mean if not np.isnan(uni_mean) else np.nan
    return dict(n=n, events_per_week=events_per_week, mean_bps=mean_bps, t=t, ex_top5_bps=ex5,
                ex_top1_bps=ex1, winner_capped_bps=wc, median_bps=med, share_trade_dir=share_dir,
                universe_placebo_bps=uni_mean, placebo_margin_bps=placebo_margin)


def count_matched_null(sub, universe_by_session, n_draws=N_NULL_DRAWS, seed=NULL_SEED):
    """1,000 draws of the same #events/session at random from that session's eligible universe,
    same direction sign and same costs; returns the percentile rank of the observed mean.
    """
    rng = np.random.default_rng(seed)
    direction = sub["direction"].iloc[0] if len(sub) else 1
    per_session_n = sub.groupby("session").size()
    pools = {}
    for s in per_session_n.index:
        if s in universe_by_session.index:
            pools[s] = universe_by_session.at[s, "list"]
    draw_means = []
    for _ in range(n_draws):
        vals = []
        for s, n in per_session_n.items():
            pool = pools.get(s)
            if not pool:
                continue
            k = min(n, len(pool))
            picks = rng.choice(pool, size=k, replace=len(pool) < k)
            vals.extend(direction * np.asarray(picks) * 1e4 - 2 * COST_LEG_BPS)
        if vals:
            draw_means.append(np.mean(vals))
    if not draw_means or sub.empty:
        return np.nan
    obs = sub["net_e1_bps"].mean() if "net_e1_bps" in sub else np.nan
    return float(100 * np.mean(np.array(draw_means) < obs))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    filings = load_filings()
    if filings.empty:
        log.error("NO filings loaded -- submissions cache empty or incomplete; aborting")
        return
    form4 = filings[filings["form"] == "4"].copy()
    buyback = resolve_buyback_insider(filings, form4)
    filings = filings[filings["cls"] != "BUYBACK_OR_INSIDER_CANDIDATE"]
    filings = pd.concat([filings, buyback], ignore_index=True)

    prices = load_prices()
    ev = build_events(filings, prices)
    if ev.empty:
        log.error("NO events survived the population/price filters -- aborting")
        return
    ev.to_csv(EVENTS_CSV, index=False)
    log.info("wrote %s (%d rows)", EVENTS_CSV, len(ev))

    uni_e1 = build_universe_returns(prices, "e1")
    uni_e5 = build_universe_returns(prices, "e5")

    report = {}
    for cls, cell in CELL_ID.items():
        ev_cell = ev[ev["cls"] == cls]
        train_e1 = cell_stats(ev_cell, "TRAIN", "net_e1_bps", uni_e1)
        train_e5 = cell_stats(ev_cell, "TRAIN", "net_e5_bps", uni_e5)
        named_leg = "E5" if (train_e5.get("n", 0) > 0 and
                              train_e5.get("mean_bps", -1e9) > train_e1.get("mean_bps", -1e9)) else "E1"
        val_e1 = cell_stats(ev_cell, "VAL", "net_e1_bps", uni_e1)
        val_e5 = cell_stats(ev_cell, "VAL", "net_e5_bps", uni_e5)
        null_pct = count_matched_null(ev_cell[ev_cell["split"] == "VAL"], uni_e1)
        report[cls] = dict(cell=cell, named_leg=named_leg, train_e1=train_e1, train_e5=train_e5,
                            val_e1=val_e1, val_e5=val_e5, val_null_pct=null_pct,
                            n_total=len(ev_cell))
    with open(os.path.join(REPO, "research/edgar_desk/rebuild_1552_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("=== rebuild_1552 complete ===")
    for cls, r in report.items():
        log.info("%s (cell %d): n_total=%d named_leg=%s VAL n=%s mean_bps=%s t=%s",
                  cls, r["cell"], r["n_total"], r["named_leg"],
                  r["val_e1"].get("n"), r["val_e1"].get("mean_bps"), r["val_e1"].get("t"))


if __name__ == "__main__":
    main()
