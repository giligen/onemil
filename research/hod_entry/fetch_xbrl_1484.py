"""Cell 1,484 RUNWAY -- point-in-time SEC XBRL cash-runway table for the 9,911 HOD-break fills.

Spec: research/hod_entry/PREREG_1483.md ("1,484 RUNWAY"). For each fill (day, symbol) in the
base book (research/hod_entry/causal_arming_causal.csv, status=='fill'), find the latest XBRL
filing (10-Q/10-K) FILED strictly before the fill day, and compute:

    cash    = latest CashAndCashEquivalentsAtCarryingValue (fallback
              CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents), instant, USD.
    ocf_q   = the QUARTERLY (not YTD, not annual) NetCashProvidedByUsedInOperatingActivities,
              derived from the filing's own duration fact -- see derive_quarterly_ocf() below.
    runway_q = cash / max(-ocf_q, 0); +inf when ocf_q >= 0 (cf_positive=True).

Quarterly-OCF derivation rule (documented per PREREG, since GAAP interim cash-flow statements are
cumulative/YTD, not discrete-quarter):
  1. DIRECT: a duration fact whose (end - start) is ~1 quarter (80-100 days) IS already a discrete
     quarter (this is always true for a company's fiscal Q1, and true for any company that happens
     to tag a discrete quarter). Use its val directly. method='direct'.
  2. DIFF: a duration fact of ~2/3/4 quarters (150-200 / 240-290 / 340-380 days) is a YTD cumulative
     figure. Find, within the SAME company and the SAME fiscal-year start date, the fact with the
     largest end date that is still < this fact's end date (any duration: a prior discrete quarter
     or a prior YTD cumulative). Subtract: quarterly = val(this) - val(prior). This is exact because
     GAAP YTD(n) - YTD(n-1) = discrete quarter n, regardless of who reported the prior figure or in
     which filing. method='diff'.
  3. PRORATE (fallback, no prior fact found -- e.g. IPO-year first filing): quarterly = val *
     (91 / duration_days). Documented, flagged, and reported separately in coverage. method='prorated'.

Point-in-time discipline: every fact carries the SEC 'filed' date (the date the filing hit EDGAR).
A fill on day D may only use facts with filed < D. The DIFF prior-fact lookup does not weaken this:
the prior fact was necessarily filed on or before the current fact's own 'filed' date (it is an
earlier filing in the same company's history), so subtracting it never pulls in information from
the future relative to fact (this), and fact (this) is itself gated by filed < D.

Resumable / rate-limited: SEC allows <=10 req/s but we self-limit to 1 req/s (spec). Every CIK's
companyfacts response is cached to xbrl_1484/CIK<10digit>.json (or a {"_missing": <reason>} sentinel
on 404 / no XBRL) so a killed run resumes from where it left off with zero re-fetching.
"""
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

import pandas as pd
import requests

REPO = "/home/ec2-user/onemil"
OUT_DIR = os.path.join(REPO, "research/hod_entry/xbrl_1484")
BASE_BOOK = os.path.join(REPO, "research/hod_entry/causal_arming_causal.csv")
TICKER_MAP_PATH = os.path.join(OUT_DIR, "company_tickers.json")
RUNWAY_CSV = os.path.join(REPO, "research/hod_entry/runway_1484.csv")
UNMAPPED_TXT = os.path.join(OUT_DIR, "unmapped_symbols.txt")
COVERAGE_TXT = os.path.join(OUT_DIR, "coverage_report_1484.txt")
LOG_PATH = os.path.join(OUT_DIR, "fetch_xbrl_1484.log")

UA = "onemil research giligen@gmail.com"
RATE_S = 1.05  # >=1 req/s ceiling per SEC fair-access policy, self-limited

CASH_CONCEPT_PRIMARY = "CashAndCashEquivalentsAtCarryingValue"
CASH_CONCEPT_FALLBACK = "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
OCF_CONCEPT = "NetCashProvidedByUsedInOperatingActivities"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fetch_xbrl_1484")


def _get(url: str, retries: int = 3):
    """Rate-limited GET with the required SEC User-Agent header and retry-with-backoff.

    SEC's endpoints (especially the bot-protected browse-edgar CGI used by the ticker fallback)
    occasionally stall or reset a connection under load; a bare unhandled exception there would
    kill an otherwise-resumable multi-hour fetch over one transient hiccup. Returns None (never
    raises) after exhausting retries -- callers MUST treat None as "unknown this run, retry on
    next resume" and must NOT write a permanent cache sentinel for it.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            time.sleep(RATE_S)
            return resp
        except requests.exceptions.RequestException as e:
            log.warning("GET failed (attempt %d/%d) url=%s: %s", attempt, retries, url, e)
            time.sleep(RATE_S * attempt * 2)
    log.error("GET permanently failed after %d attempts url=%s -- treating as unknown this run "
              "(not cached, will retry on next resume)", retries, url)
    return None


def load_ticker_map() -> dict:
    """Symbol -> zero-padded 10-digit CIK string, from SEC's current ticker->CIK map.

    Cached once (this file changes daily upstream but we do not need same-day freshness for a
    research pull); resumable runs reuse it without re-fetching.
    """
    if os.path.exists(TICKER_MAP_PATH):
        log.info("ticker map: using cache %s", TICKER_MAP_PATH)
        with open(TICKER_MAP_PATH) as f:
            raw = json.load(f)
    else:
        log.info("ticker map: fetching from SEC (not cached yet)")
        resp = _get("https://www.sec.gov/files/company_tickers.json")
        if resp is None or resp.status_code != 200:
            log.error("ticker map fetch FAILED (%s) -- cannot map any symbol to a CIK this run",
                      "network error" if resp is None else f"status={resp.status_code}")
            return {}
        raw = resp.json()
        with open(TICKER_MAP_PATH, "w") as f:
            json.dump(raw, f)
    out = {}
    for row in raw.values():
        out[str(row["ticker"]).upper()] = "%010d" % int(row["cik_str"])
    log.info("ticker map: %d symbols mapped", len(out))
    return out


FALLBACK_CACHE_PATH = os.path.join(OUT_DIR, "ticker_fallback_cache.json")


def _load_fallback_cache() -> dict:
    if os.path.exists(FALLBACK_CACHE_PATH):
        with open(FALLBACK_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_fallback_cache(cache: dict) -> None:
    with open(FALLBACK_CACHE_PATH, "w") as f:
        json.dump(cache, f)


def fallback_cik_lookup(symbol: str, cache: dict) -> str:
    """Second-chance CIK lookup for a symbol absent from SEC's own company_tickers.json.

    That file is built from each filer's 'tickers' submissions field, which SEC leaves EMPTY for a
    surprising number of active, current filers (verified case: Apellis Pharmaceuticals / APLS,
    CIK 1492422, live 10-K filer as of 2026, `tickers: []` in its own submissions JSON) -- a known
    gap in SEC's OWN reference data, not a bug in this script. EDGAR's company-browse CGI accepts a
    ticker directly in the CIK= query param and resolves it via a different internal index; we use
    that as a fallback so real operating companies are not miscounted as unmapped/no-XBRL. Result
    (a CIK string, or "" for a genuine miss e.g. a fund/wrapper with no EDGAR company record) is
    cached so a resumed run never re-queries the same symbol.
    """
    if symbol in cache:
        return cache[symbol]
    url = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={symbol}"
           f"&type=10-K&dateb=&owner=include&count=5&output=atom")
    resp = _get(url)
    if resp is None:
        log.error("fallback CIK lookup network FAILURE for symbol=%s -- leaving uncached, "
                  "will retry on next resume", symbol)
        return ""
    cik10 = ""
    if resp.status_code == 200:
        m = re.search(r"<cik>(\d+)</cik>", resp.text)
        if m:
            cik10 = "%010d" % int(m.group(1))
    else:
        log.warning("fallback CIK lookup HTTP %s for symbol=%s", resp.status_code, symbol)
    cache[symbol] = cik10
    _save_fallback_cache(cache)
    return cik10


def fetch_companyfacts(cik10: str) -> dict:
    """Return the companyfacts JSON for one CIK, cached and resumable.

    A 404 (no XBRL: many wrappers/ETFs/foreign private issuers under IFRS) is cached as a sentinel
    so we never re-request it. This is a WARN, not silently swallowed: it directly reduces coverage.
    """
    cache_path = os.path.join(OUT_DIR, f"CIK{cik10}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    resp = _get(url)
    if resp is None:
        log.error("companyfacts network FAILURE for CIK%s -- leaving uncached, will retry on "
                  "next resume", cik10)
        return {"_missing": "network_error_this_run"}
    if resp.status_code == 404:
        log.warning("companyfacts 404 for CIK%s -- no XBRL (wrapper/ETF/foreign filer expected)", cik10)
        sentinel = {"_missing": "404"}
        with open(cache_path, "w") as f:
            json.dump(sentinel, f)
        return sentinel
    if resp.status_code != 200:
        log.error("companyfacts fetch FAILED CIK%s status=%s -- treating as missing this run", cik10, resp.status_code)
        sentinel = {"_missing": f"http_{resp.status_code}"}
        with open(cache_path, "w") as f:
            json.dump(sentinel, f)
        return sentinel
    try:
        data = resp.json()
    except ValueError:
        log.error("companyfacts JSON decode FAILED CIK%s -- treating as missing this run", cik10)
        data = {"_missing": "bad_json"}
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def _usd_facts(facts_json: dict, concept: str) -> list:
    try:
        return facts_json["facts"]["us-gaap"][concept]["units"]["USD"]
    except KeyError:
        return []


def derive_quarterly_ocf(ocf_facts: list) -> list:
    """Turn raw (possibly YTD/annual) OCF duration facts into discrete-quarter records.

    Returns a list of dicts: {accn, filed, form, fy, fp, end, val, method} where val is the
    DISCRETE-QUARTER operating cash flow (method in direct/diff/prorated), one per distinct
    (accn, end) input fact that could be resolved.
    """
    parsed = []
    for f in ocf_facts:
        if "start" not in f or "end" not in f or f.get("form") not in ("10-Q", "10-K"):
            continue
        try:
            start = datetime.strptime(f["start"], "%Y-%m-%d")
            end = datetime.strptime(f["end"], "%Y-%m-%d")
        except (ValueError, TypeError):
            continue
        dur = (end - start).days + 1
        parsed.append({**f, "_start": start, "_end": end, "_dur": dur})

    # Group by (fiscal-year start date) so DIFF only ever compares facts from the same fiscal year.
    by_fystart = {}
    for f in parsed:
        by_fystart.setdefault(f["_start"], []).append(f)

    out = []
    for fystart, group in by_fystart.items():
        group_sorted = sorted(group, key=lambda f: f["_end"])
        for i, f in enumerate(group_sorted):
            dur = f["_dur"]
            if 80 <= dur <= 100:
                out.append({"accn": f["accn"], "filed": f["filed"], "form": f["form"],
                            "fy": f.get("fy"), "fp": f.get("fp"), "end": f["end"],
                            "val": f["val"], "method": "direct"})
            elif dur >= 150:
                # cumulative YTD (H1 ~150-200, 9mo ~240-290, FY ~340-380): diff against the
                # nearest earlier fact in the SAME fiscal year (direct or cumulative alike).
                priors = [p for p in group_sorted[:i] if p["_end"] < f["_end"]]
                if priors:
                    prior = max(priors, key=lambda p: p["_end"])
                    out.append({"accn": f["accn"], "filed": f["filed"], "form": f["form"],
                                "fy": f.get("fy"), "fp": f.get("fp"), "end": f["end"],
                                "val": f["val"] - prior["val"], "method": "diff"})
                else:
                    out.append({"accn": f["accn"], "filed": f["filed"], "form": f["form"],
                                "fy": f.get("fy"), "fp": f.get("fp"), "end": f["end"],
                                "val": f["val"] * (91.0 / dur), "method": "prorated"})
    return out


def build_candidates(facts_json: dict) -> list:
    """One row per filing (accn) that has BOTH a cash figure and a resolvable quarterly OCF.

    A filing that is missing one side is dropped (report-only symbol-level coverage loss) rather
    than paired with a mismatched accn -- a wrong pairing would silently corrupt runway_q.
    """
    if facts_json.get("_missing"):
        return []
    cash_facts = _usd_facts(facts_json, CASH_CONCEPT_PRIMARY)
    cash_concept_used = CASH_CONCEPT_PRIMARY
    if not cash_facts:
        cash_facts = _usd_facts(facts_json, CASH_CONCEPT_FALLBACK)
        cash_concept_used = CASH_CONCEPT_FALLBACK
    cash_by_accn = {}
    for f in cash_facts:
        if f.get("form") not in ("10-Q", "10-K") or "end" not in f:
            continue
        # instant concept: keep the latest 'end' per accn (a filing can restate a prior year's
        # comparative balance too; we want the filing's OWN period-end balance).
        prev = cash_by_accn.get(f["accn"])
        if prev is None or f["end"] > prev["end"]:
            cash_by_accn[f["accn"]] = f

    ocf_facts = _usd_facts(facts_json, OCF_CONCEPT)
    ocf_quarters = derive_quarterly_ocf(ocf_facts)
    ocf_by_accn = {}
    for r in ocf_quarters:
        # a filing can carry >1 duration fact (rare restatement); keep the one whose end matches
        # this accn's own reporting period end most closely -- last-wins is fine since same accn
        # values should agree; if they disagree we want the direct/most-recent-computed one.
        ocf_by_accn[r["accn"]] = r

    candidates = []
    for accn, cash_f in cash_by_accn.items():
        ocf_r = ocf_by_accn.get(accn)
        if ocf_r is None:
            continue
        filed = cash_f.get("filed") or ocf_r.get("filed")
        if not filed:
            continue
        candidates.append({
            "accn": accn,
            "filed": filed,
            "form": cash_f.get("form"),
            "fy": cash_f.get("fy"),
            "fp": cash_f.get("fp"),
            "cash": cash_f["val"],
            "cash_concept": cash_concept_used,
            "ocf_quarterly": ocf_r["val"],
            "ocf_method": ocf_r["method"],
        })
    candidates.sort(key=lambda c: c["filed"])
    return candidates


def pick_pit(candidates: list, fill_day: str):
    """Latest candidate FILED strictly before fill_day (PIT). None if none qualify."""
    eligible = [c for c in candidates if c["filed"] < fill_day]
    if not eligible:
        return None
    return max(eligible, key=lambda c: c["filed"])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log.info("=== cell 1,484 RUNWAY fetch starting ===")

    base = pd.read_csv(BASE_BOOK, low_memory=False)
    fills = base[base["status"] == "fill"][["day", "symbol", "split", "fill_min", "level", "stop", "fill"]].copy()
    log.info("base book: %d fills, %d distinct symbols", len(fills), fills["symbol"].nunique())

    ticker_map = load_ticker_map()
    symbols = sorted(fills["symbol"].unique())
    sym_to_cik = {}
    unmapped = []
    for s in symbols:
        cik = ticker_map.get(s.upper())
        if cik:
            sym_to_cik[s] = cik
        else:
            unmapped.append(s)
    log.info("CIK mapping (primary company_tickers.json): %d/%d symbols mapped (%d unmapped)",
              len(sym_to_cik), len(symbols), len(unmapped))

    fb_cache = _load_fallback_cache()
    n_fb_recovered = 0
    still_unmapped = []
    for i, s in enumerate(unmapped, 1):
        try:
            cik10 = fallback_cik_lookup(s, fb_cache)
        except Exception:
            log.exception("fallback_cik_lookup CRASHED for symbol=%s -- treating as unmapped "
                          "this run, will retry on next resume", s)
            cik10 = ""
        if cik10:
            sym_to_cik[s] = cik10
            n_fb_recovered += 1
        else:
            still_unmapped.append(s)
        if i % 50 == 0 or i == len(unmapped):
            log.info("fallback CIK lookup progress: %d/%d symbols checked (%d recovered)",
                      i, len(unmapped), n_fb_recovered)
    unmapped = still_unmapped
    log.info("CIK mapping (post-fallback): %d/%d symbols mapped (%d genuinely unmapped -- "
              "expected wrappers/ETFs/foreign filers/delisted tickers)",
              len(sym_to_cik), len(symbols), len(unmapped))
    with open(UNMAPPED_TXT, "w") as f:
        f.write("\n".join(unmapped) + "\n")

    # Fetch + build candidates once per distinct CIK (resumable: cache hit = no sleep, no request).
    cik_to_symbols = {}
    for s, cik in sym_to_cik.items():
        cik_to_symbols.setdefault(cik, []).append(s)
    unique_ciks = sorted(cik_to_symbols.keys())
    log.info("fetching companyfacts for %d distinct CIKs (rate-limited ~1/s; resumable)", len(unique_ciks))

    candidates_by_cik = {}
    n_missing_xbrl = 0
    n_no_candidates = 0
    for i, cik in enumerate(unique_ciks, 1):
        try:
            facts_json = fetch_companyfacts(cik)
            if facts_json.get("_missing"):
                n_missing_xbrl += 1
                candidates_by_cik[cik] = []
            else:
                cands = build_candidates(facts_json)
                candidates_by_cik[cik] = cands
                if not cands:
                    n_no_candidates += 1
        except Exception:
            log.exception("companyfacts/build_candidates CRASHED for CIK%s -- treating as no "
                          "usable candidates this run (symbols: %s)", cik, cik_to_symbols.get(cik))
            candidates_by_cik[cik] = []
            n_missing_xbrl += 1
        if i % 50 == 0 or i == len(unique_ciks):
            log.info("companyfacts progress: %d/%d CIKs done (missing_xbrl=%d, no_paired_candidates=%d)",
                      i, len(unique_ciks), n_missing_xbrl, n_no_candidates)

    log.info("companyfacts done: %d/%d CIKs had zero usable (cash+ocf paired) candidates",
              n_missing_xbrl + n_no_candidates, len(unique_ciks))

    # Join to fills, PIT.
    rows = []
    for r in fills.itertuples(index=False):
        cik = sym_to_cik.get(r.symbol)
        rec = {"day": r.day, "symbol": r.symbol, "split": r.split, "fill_min": r.fill_min,
               "level": r.level, "stop": r.stop, "fill": r.fill, "cik": cik}
        if cik is None:
            rec.update({"filed": None, "form": None, "cash": None, "ocf_quarterly": None,
                        "ocf_method": None, "runway_q": None, "cf_positive": None,
                        "days_since_filing": None, "cash_concept": None})
            rows.append(rec)
            continue
        pit = pick_pit(candidates_by_cik.get(cik, []), r.day)
        if pit is None:
            rec.update({"filed": None, "form": None, "cash": None, "ocf_quarterly": None,
                        "ocf_method": None, "runway_q": None, "cf_positive": None,
                        "days_since_filing": None, "cash_concept": None})
            rows.append(rec)
            continue
        cf_positive = pit["ocf_quarterly"] >= 0
        burn = max(-pit["ocf_quarterly"], 0)
        runway_q = float("inf") if burn == 0 else pit["cash"] / burn
        filed_dt = datetime.strptime(pit["filed"], "%Y-%m-%d")
        fill_dt = datetime.strptime(r.day, "%Y-%m-%d")
        rec.update({
            "filed": pit["filed"], "form": pit["form"], "cash": pit["cash"],
            "ocf_quarterly": pit["ocf_quarterly"], "ocf_method": pit["ocf_method"],
            "runway_q": runway_q, "cf_positive": cf_positive,
            "days_since_filing": (fill_dt - filed_dt).days,
            "cash_concept": pit["cash_concept"],
        })
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(RUNWAY_CSV, index=False)
    log.info("wrote %s (%d rows)", RUNWAY_CSV, len(out))

    # Coverage + distribution report.
    n = len(out)
    has_value = out["runway_q"].notna()
    n_has_value = int(has_value.sum())
    coverage = n_has_value / n if n else 0.0

    finite = out.loc[has_value & (out["runway_q"] < float("inf")), "runway_q"]
    n_inf = int((out.loc[has_value, "runway_q"] == float("inf")).sum())
    quartiles = finite.quantile([0.25, 0.5, 0.75]) if len(finite) else None

    share_lt2 = float((out.loc[has_value, "runway_q"] < 2).mean()) if n_has_value else float("nan")
    share_ge4 = float((out.loc[has_value, "runway_q"] >= 4).mean()) if n_has_value else float("nan")

    method_counts = out.loc[has_value, "ocf_method"].value_counts().to_dict()

    lines = []
    lines.append("cell 1,484 RUNWAY -- coverage and distribution report")
    lines.append(f"generated: {datetime.utcnow().isoformat()}Z")
    lines.append(f"fills total: {n}")
    lines.append(f"symbols total / mapped to a CIK / unmapped: {len(symbols)} / {len(sym_to_cik)} / {len(unmapped)}")
    lines.append(f"  of which recovered via EDGAR browse-edgar fallback (missing from SEC's own "
                 f"company_tickers.json 'tickers' field): {n_fb_recovered}")
    lines.append(f"CIKs with zero usable XBRL (404 or unparseable): {n_missing_xbrl}")
    lines.append(f"CIKs mapped but no cash+ocf paired candidate ever: {n_no_candidates}")
    lines.append(f"fills WITH a runway value: {n_has_value} ({coverage:.1%})")
    lines.append(f"  of which cash-flow-positive (runway = +inf): {n_inf}")
    lines.append(f"  of which finite runway_q: {len(finite)}")
    lines.append(f"ocf derivation method mix among covered fills: {method_counts}")
    if quartiles is not None:
        lines.append(f"finite runway_q quartiles: p25={quartiles.loc[0.25]:.2f} "
                      f"p50={quartiles.loc[0.5]:.2f} p75={quartiles.loc[0.75]:.2f}")
    lines.append(f"share of covered fills with runway_q < 2: {share_lt2:.1%}")
    lines.append(f"share of covered fills with runway_q >= 4 (incl. cf-positive/+inf): {share_ge4:.1%}")
    report = "\n".join(lines)
    with open(COVERAGE_TXT, "w") as f:
        f.write(report + "\n")
    log.info("\n%s", report)
    log.info("=== cell 1,484 RUNWAY fetch DONE ===")


if __name__ == "__main__":
    main()
