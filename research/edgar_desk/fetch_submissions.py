"""FETCH stage for PREREG_1552 -- the EDGAR event desk (cells 1,552-1,561).

Spec: research/edgar_desk/PREREG_1552.md. This script does ONLY the fetch + raw event-table
build; no scoring, no pass-bar evaluation (that is a separate PREREG'd scoring pass on this
output). It:

  1. Builds the universe: common-stock symbols from `data/research/databento/
     alpaca_assets_all_20260905.csv` (column `common`) UNION common-stock symbols from every
     month of the Databento point-in-time listing feed (`research/scripts/pit_listings.py`),
     test tickers (^Z[A-Z]ZZT$ and ^ZZ) excluded.
  2. Maps every universe symbol to a CIK: SEC's own `company_tickers.json` first (reusing the
     existing read-only cache at research/hod_entry/xbrl_1484/company_tickers.json, 10,428
     entries, per the task -- this script never writes to that path), then the EDGAR
     browse-edgar CIK fallback for symbols SEC's own ticker file omits (same code pattern as
     research/hod_entry/fetch_xbrl_1484.py:fallback_cik_lookup). Symbols still unmapped after
     both steps are LOST and counted.
  3. Pulls each mapped CIK's submissions JSON (https://data.sec.gov/submissions/CIK##########
     .json) plus every additional filing-history page listed in `filings.files` (older filings
     for that CIK, needed for 2019 coverage), merges them into one `filings.recent`-shaped
     dict, and caches the merged result gzipped under `submissions/CIK##########.json.gz` --
     resumable: a killed run re-reads every cached CIK for free and only fetches what is
     missing.
  4. Writes `events_raw.csv`: one row per filing 2019-01-01..2026-09-25, per symbol mapped to
     that CIK (a CIK can map to >1 symbol across a ticker history; each gets its own row),
     with the PREREG class(es) assigned by the structured item-code / form-type rules below.

Rate limit: SEC fair-access is <=10 req/s; this script self-limits to <=1 req/s (RATE_S) exactly
as fetch_xbrl_1484.py does, including on the browse-edgar fallback and on the additional
filing-history pages.

Completeness gate (fallback-logging rule, CLAUDE.md): after the submissions pull, if more than
3% of *mapped* CIKs have no usable submissions record (permanent network failure after retries,
or a 404 on a CIK that SEC's own ticker file said was a live filer), the script logs ERROR and
exits 1 rather than silently building events_raw.csv on a population with an undisclosed hole --
per the fetch-completeness-gate lesson (5,000 tickers were silently dropped once before, see
CLAUDE.md / memory). Re-running resumes from the gzip cache; fix the network issue and rerun to
close the gate.

CONTRACT / OFFERING interpretation (flagged, not resolved by code): the PREREG's OFFERING row
reads "form 424B1-424B5 or 8-K item 3.02 (unregistered sale of equity) or 8-K 1.01 whose text is
skipped (structured only)". Taken as a literal third OR-clause, EVERY 8-K 1.01 filing would land
in OFFERING, which directly contradicts cell 1,559's CONTRACT definition ("8-K item 1.01 WITHOUT
items 3.02/2.03") -- if 1.01 always triggered OFFERING, CONTRACT could never fire. Per the
PREREG's own "Not allowed: text classification in this pass" rule, this script reads the clause
as a CAVEAT (1.01 is not further split by text in this pass, so it is left to the CONTRACT rule
below) rather than a third trigger: OFFERING = form in the 424B set OR 8-K item 3.02 present.
This is a genuine spec ambiguity -- flag it to the owner / the independent-check pass before any
number from this desk is trusted; the raw items field is preserved unmodified per filing so a
re-classification needs no re-fetch.

BUYBACK_OR_INSIDER (cell 1,561) is marked NOT COMPUTABLE from this fetch. Transaction code ('P'
for an open-market purchase) and officer/insider role are fields inside each Form 4's own XML
document, not in the submissions JSON (which only records that a Form "4" was filed, by whom,
and when) -- computing this class would mean fetching and parsing every individual Form 4
document for every mapped CIK, an order of magnitude more requests than this fetch, and it is
out of scope for the FETCH stage per the task. events_raw.csv still records every Form "4"
filing's form/date/CIK/symbol (so item 8.01 co-occurrence can be checked later) but assigns it
no BUYBACK_OR_INSIDER class; the scoring pass must either implement the Form 4 XML parse as its
own dated addition or report this cell as not computable.
"""
import glob
import gzip
import json
import logging
import os
import re
import sys
import time

import pandas as pd
import requests

REPO = "/home/ec2-user/onemil"
OUT_DIR = os.path.join(REPO, "research/edgar_desk")
SUB_DIR = os.path.join(OUT_DIR, "submissions")
LOG_PATH = os.path.join(OUT_DIR, "fetch_submissions.log")
EVENTS_CSV = os.path.join(OUT_DIR, "events_raw.csv")
CIK_MAP_CSV = os.path.join(OUT_DIR, "symbol_cik_map.csv")
SUMMARY_TXT = os.path.join(OUT_DIR, "fetch_summary.txt")
DONE_SENTINEL = os.path.join(OUT_DIR, "FETCH_DONE")

# Reuse SEC's own ticker->CIK file already fetched for cell 1,484 -- READ ONLY, never written here.
EXTERNAL_TICKER_MAP = os.path.join(REPO, "research/hod_entry/xbrl_1484/company_tickers.json")
LOCAL_TICKER_MAP = os.path.join(OUT_DIR, "company_tickers.json")  # only used if the external one is absent
FALLBACK_CACHE_PATH = os.path.join(OUT_DIR, "ticker_fallback_cache.json")

UA = "onemil research giligen@gmail.com"
RATE_S = 1.05  # >=1 req/s ceiling per SEC fair-access policy, self-limited (same as fetch_xbrl_1484.py)

START_DATE = pd.Timestamp("2019-01-01")
END_DATE = pd.Timestamp("2026-09-25")

TEST_TICKER_RE = re.compile(r'^Z[A-Z]ZZT$|^ZZ')
# Real-ticker shape only -- excludes CUSIP-shaped escrow/CVR/when-issued identifiers
# (e.g. 004CVR049, 18506U302, 382ESC010) that the Alpaca asset list marks `common=True` but
# can never map to a CIK. Verified 2026-09-26: only ~394 of 30,058 common==True symbols are
# non-ticker-shaped by this regex -- most of the 23,213 primary-unmapped symbols ARE
# ticker-shaped (delisted/foreign/OTC names not in SEC's company_tickers.json), so this rule
# alone does not fix the runtime; see FALLBACK_ELIGIBLE gating below.
TICKER_SHAPE_RE = re.compile(r'^[A-Z]{1,5}([.-][A-Z]{1,2})?$')

COMPLETENESS_GATE_PCT = 3.0  # fail if > this % of mapped CIKs have no usable submissions record

FORM_424B = {"424B1", "424B2", "424B3", "424B4", "424B5"}
FORM_SHELF = {"S-3", "S-3ASR", "S-1"}
FORM_LATE = {"NT 10-K", "NT 10-Q"}

os.makedirs(SUB_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fetch_submissions")


def _get(url: str, retries: int = 3):
    """Rate-limited GET with the required SEC User-Agent header and retry-with-backoff.

    Never raises; returns None after exhausting retries. Callers MUST treat None as "unknown
    this run, retry on next resume" and must not write a permanent cache sentinel for it (same
    contract as research/hod_entry/fetch_xbrl_1484.py:_get).
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


def build_universe() -> tuple[list[str], set[str]]:
    """Union of common-stock symbols from the Alpaca asset list and the PIT listing feed.

    Test tickers and CUSIP-shaped identifiers excluded. Every drop/inclusion count is logged
    (fallback-logging rule): a silent universe shrink here would be exactly the kind of defect
    CLAUDE.md's fetch-completeness-gate memory warns about.

    Returns (symbols, fallback_eligible): fallback_eligible is the subset of `symbols` allowed
    to go to the slow browse-edgar CIK fallback when SEC's own ticker file misses them -- a
    symbol only qualifies if it is CURRENTLY ACTIVE+TRADABLE (per the Alpaca asset list's own
    `status`/`tradable` columns) or appears in the point-in-time listing feed (so a historically
    real, in-regime name is never silently dropped). Purely-delisted/non-tradable names absent
    from the PIT feed are still counted as universe members (for LOST accounting) but are not
    sent to the fallback -- they are the bulk of what made the previous run 11+ hours.
    """
    assets = pd.read_csv(os.path.join(REPO, "data/research/databento/alpaca_assets_all_20260905.csv"))
    common_mask = assets["common"].astype(str).str.strip().str.lower() == "true"
    common_assets = assets.loc[common_mask].copy()
    common_assets["symbol"] = common_assets["symbol"].astype(str).str.upper()
    alpaca_syms = set(common_assets["symbol"])
    log.info("universe: %d common-stock symbols from alpaca_assets_all (of %d total rows, "
              "`common` column)", len(alpaca_syms), len(assets))

    sys.path.insert(0, REPO)
    from research.scripts.pit_listings import PitListings
    pit = PitListings()
    first, last = pit.coverage
    pit_syms: set[str] = set()
    for key in pit._months:
        date = f"{key[:4]}-{key[4:6]}-15"
        pit_syms |= set(pit.common_stock_symbols(date))
    log.info("universe: %d distinct common-stock symbols across PIT listing months %s..%s",
              len(pit_syms), first, last)

    union = alpaca_syms | pit_syms
    n_union = len(union)

    ticker_shaped = {s for s in union if s and TICKER_SHAPE_RE.match(s)}
    n_cusip_dropped = n_union - len(ticker_shaped)
    log.info("universe rule 1 (ticker shape): %d union -> %d ticker-shaped, %d CUSIP/escrow-"
              "shaped dropped", n_union, len(ticker_shaped), n_cusip_dropped)

    final = {s for s in ticker_shaped if not TEST_TICKER_RE.match(s)}
    n_test_dropped = len(ticker_shaped) - len(final)
    log.info("universe rule 2 (test tickers): %d ticker-shaped -> %d final, %d test tickers "
              "excluded", len(ticker_shaped), len(final), n_test_dropped)

    active_tradable = set(
        common_assets.loc[(common_assets["status"] == "active") & (common_assets["tradable"] == True),
                           "symbol"]
    )
    fallback_eligible = {s for s in final if s in pit_syms or s in active_tradable}
    log.info("fallback eligibility: %d/%d final symbols eligible for the browse-edgar fallback "
              "(in PIT feed or active+tradable); %d ineligible (delisted/non-tradable, not in "
              "PIT -- go straight to LOST if the primary map misses them, no fallback request)",
              len(fallback_eligible), len(final), len(final) - len(fallback_eligible))

    return sorted(final), fallback_eligible


def load_primary_ticker_map() -> dict:
    """Symbol -> zero-padded 10-digit CIK, from SEC's own company_tickers.json."""
    path = EXTERNAL_TICKER_MAP if os.path.exists(EXTERNAL_TICKER_MAP) else LOCAL_TICKER_MAP
    if os.path.exists(path):
        log.info("ticker map: reusing existing cache %s (read-only)", path)
        with open(path) as f:
            raw = json.load(f)
    else:
        log.info("ticker map: fetching fresh from SEC (no cache found at %s or %s)",
                  EXTERNAL_TICKER_MAP, LOCAL_TICKER_MAP)
        resp = _get("https://www.sec.gov/files/company_tickers.json")
        if resp is None or resp.status_code != 200:
            log.error("ticker map fetch FAILED (%s) -- cannot map any symbol via the primary path this run",
                      "network error" if resp is None else f"status={resp.status_code}")
            return {}
        raw = resp.json()
        with open(LOCAL_TICKER_MAP, "w") as f:
            json.dump(raw, f)
    out = {}
    for row in raw.values():
        out[str(row["ticker"]).upper()] = "%010d" % int(row["cik_str"])
    log.info("ticker map: %d symbols mapped (primary)", len(out))
    return out


def _load_json_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json_cache(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)


def fallback_cik_lookup(symbol: str, cache: dict) -> str:
    """Second-chance CIK lookup via EDGAR's browse-edgar CGI for symbols SEC's own ticker file
    omits (verified gap: e.g. APLS/Apellis, see research/hod_entry/fetch_xbrl_1484.py). Cached
    so a resumed run never re-queries the same symbol. Returns "" for a genuine miss.
    """
    if symbol in cache:
        return cache[symbol]
    url = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={symbol}"
           f"&type=10-K&dateb=&owner=include&count=5&output=atom")
    resp = _get(url)
    if resp is None:
        log.error("fallback CIK lookup network FAILURE for symbol=%s -- leaving uncached, will retry", symbol)
        return ""
    cik10 = ""
    if resp.status_code == 200:
        m = re.search(r"<cik>(\d+)</cik>", resp.text)
        if m:
            cik10 = "%010d" % int(m.group(1))
    else:
        log.warning("fallback CIK lookup HTTP %s for symbol=%s", resp.status_code, symbol)
    cache[symbol] = cik10
    _save_json_cache(FALLBACK_CACHE_PATH, cache)
    return cik10


def map_symbols_to_ciks(symbols: list[str], fallback_eligible: set[str]) -> tuple[dict[str, str], int]:
    """Symbol -> CIK for every universe symbol, primary map then browse-edgar fallback.

    The fallback (one request/second) only runs for primary-unmapped symbols in
    `fallback_eligible` (active+tradable or PIT-listed, per build_universe) -- symbols that are
    both unmapped and ineligible are counted straight into LOST with zero network cost, which is
    the fix for the 11+ hour runtime (23,213 candidates -> ~5,000).

    Returns (mapping, lost_count). Writes symbol_cik_map.csv with a `source` column
    (primary/fallback/lost/lost_ineligible) so the mapping is auditable without re-running the fetch.
    """
    primary = load_primary_ticker_map()
    mapping: dict[str, str] = {}
    sources: dict[str, str] = {}
    unmapped = []
    for s in symbols:
        cik = primary.get(s)
        if cik:
            mapping[s] = cik
            sources[s] = "primary"
        else:
            unmapped.append(s)
    log.info("CIK mapping (primary): %d/%d symbols mapped, %d unmapped", len(mapping), len(symbols), len(unmapped))

    to_fallback = [s for s in unmapped if s in fallback_eligible]
    lost_ineligible = [s for s in unmapped if s not in fallback_eligible]
    projected_s = len(to_fallback) * RATE_S
    log.info("fallback routing: %d/%d unmapped symbols eligible -> browse-edgar fallback "
              "(projected %.1f min at 1 req/s); %d unmapped symbols ineligible -> LOST with no "
              "request (delisted/non-tradable, not in PIT feed)",
              len(to_fallback), len(unmapped), projected_s / 60.0, len(lost_ineligible))
    for s in lost_ineligible:
        sources[s] = "lost_ineligible"

    fb_cache = _load_json_cache(FALLBACK_CACHE_PATH)
    recovered = 0
    for i, s in enumerate(to_fallback, 1):
        cik = fallback_cik_lookup(s, fb_cache)
        if cik:
            mapping[s] = cik
            sources[s] = "fallback"
            recovered += 1
        else:
            sources[s] = "lost"
        if i % 200 == 0:
            log.info("fallback CIK lookup progress: %d/%d checked (%d recovered)", i, len(to_fallback), recovered)

    lost = len(unmapped) - recovered
    log.info("CIK mapping (post-fallback): %d/%d symbols mapped (%d recovered via fallback), %d LOST",
              len(mapping), len(symbols), recovered, lost)

    with open(CIK_MAP_CSV, "w") as f:
        f.write("symbol,cik,source\n")
        for s in symbols:
            f.write(f"{s},{mapping.get(s, '')},{sources[s]}\n")
    return mapping, lost


def fetch_submissions_json(cik10: str) -> dict | None:
    """Fetch + merge one CIK's primary submissions JSON with all its additional filing-history
    pages (older filings, needed for 2019 coverage). Cached gzipped, resumable. Returns None on
    a permanent (post-retry) failure -- caller must NOT count that as a genuine 404 miss.
    """
    cache_path = os.path.join(SUB_DIR, f"CIK{cik10}.json.gz")
    if os.path.exists(cache_path):
        with gzip.open(cache_path, "rt") as f:
            return json.load(f)

    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    resp = _get(url)
    if resp is None:
        log.error("submissions network FAILURE for CIK%s -- leaving uncached, will retry on next resume", cik10)
        return None
    if resp.status_code == 404:
        log.warning("submissions 404 for CIK%s -- no filer record at SEC", cik10)
        data = {"_missing": "404"}
        with gzip.open(cache_path, "wt") as f:
            json.dump(data, f)
        return data
    if resp.status_code != 200:
        log.error("submissions fetch FAILED CIK%s status=%s -- treating as missing this run", cik10, resp.status_code)
        return None
    try:
        data = resp.json()
    except Exception:
        log.error("submissions JSON decode FAILED CIK%s -- treating as missing this run", cik10)
        return None

    files = data.get("filings", {}).get("files", [])
    if files:
        recent = data["filings"]["recent"]
        merged = {k: list(v) for k, v in recent.items()}
        for finfo in files:
            fname = finfo.get("name")
            if not fname:
                continue
            page_url = f"https://data.sec.gov/submissions/{fname}"
            presp = _get(page_url)
            if presp is None or presp.status_code != 200:
                log.error("additional submissions page FAILED CIK%s file=%s -- filings on that page missing "
                          "(older-history coverage gap for this CIK)", cik10, fname)
                continue
            try:
                page = presp.json()
            except Exception:
                log.error("additional submissions page JSON decode FAILED CIK%s file=%s", cik10, fname)
                continue
            for k in merged:
                if k in page:
                    merged[k] = merged[k] + list(page[k])
        data["filings"]["recent"] = merged

    with gzip.open(cache_path, "wt") as f:
        json.dump(data, f)
    return data


def classify(form: str, items_set: set) -> list[str]:
    """PREREG cells 1,552-1,560 class assignment from structured form/item codes only.

    A filing can carry several items and so several classes (except CONTRACT, which is gated
    OFF by 3.02/2.03 presence). See the module docstring for the OFFERING/CONTRACT interpretation
    flag and for why BUYBACK_OR_INSIDER (1,561) is never assigned here.
    """
    classes = []
    if form in FORM_424B or "3.02" in items_set:
        classes.append("OFFERING")
    if form in FORM_SHELF:
        classes.append("SHELF")
    if "5.03" in items_set:
        classes.append("REVERSE_SPLIT")
    if "4.01" in items_set:
        classes.append("AUDITOR")
    if "4.02" in items_set:
        classes.append("NON_RELIANCE")
    if form in FORM_LATE:
        classes.append("LATE_FILING")
    if "5.02" in items_set:
        classes.append("OFFICER_EXIT")
    if "1.01" in items_set and "3.02" not in items_set and "2.03" not in items_set:
        classes.append("CONTRACT")
    if form == "SC 13D":
        classes.append("ACTIVIST")
    return classes


def main():
    log.info("=== FETCH stage, PREREG_1552 -- start ===")

    symbols, fallback_eligible = build_universe()
    mapping, lost = map_symbols_to_ciks(symbols, fallback_eligible)

    cik_to_symbols: dict[str, list[str]] = {}
    for s, cik in mapping.items():
        cik_to_symbols.setdefault(cik, []).append(s)
    unique_ciks = sorted(cik_to_symbols)
    log.info("fetching submissions for %d distinct CIKs (rate-limited ~1/s incl. additional pages; resumable)",
              len(unique_ciks))

    n_missing = 0
    n_done = 0
    rows = []
    for i, cik in enumerate(unique_ciks, 1):
        data = fetch_submissions_json(cik)
        if data is None or data.get("_missing"):
            n_missing += 1
        else:
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            filing_dates = recent.get("filingDate", [])
            accept_dts = recent.get("acceptanceDateTime", [])
            items_list = recent.get("items", [])
            primary_docs = recent.get("primaryDocument", [])
            n = len(forms)
            for j in range(n):
                fdate = pd.Timestamp(filing_dates[j]) if j < len(filing_dates) and filing_dates[j] else None
                if fdate is None or fdate < START_DATE or fdate > END_DATE:
                    continue
                form = forms[j]
                items_raw = items_list[j] if j < len(items_list) else ""
                items_set = {it.strip() for it in items_raw.split(",") if it.strip()}
                classes = classify(form, items_set)
                items_out = items_raw.replace(",", ";")
                accept_dt = accept_dts[j] if j < len(accept_dts) else ""
                pdoc = primary_docs[j] if j < len(primary_docs) else ""
                for sym in cik_to_symbols[cik]:
                    rows.append({
                        "cik": cik,
                        "symbol": sym,
                        "form": form,
                        "filing_date": filing_dates[j],
                        "acceptance_datetime": accept_dt,
                        "items": items_out,
                        "primary_document": pdoc,
                        "classes": ";".join(classes),
                    })
        n_done += 1
        if n_done % 200 == 0:
            log.info("submissions progress: %d/%d CIKs done (missing=%d, events so far=%d)",
                      n_done, len(unique_ciks), n_missing, len(rows))

    pct_missing = 100.0 * n_missing / max(len(unique_ciks), 1)
    log.info("submissions done: %d/%d CIKs missing (%.2f%%)", n_missing, len(unique_ciks), pct_missing)
    if pct_missing > COMPLETENESS_GATE_PCT:
        log.error("COMPLETENESS GATE FAILED: %.2f%% of mapped CIKs missing a usable submissions record "
                  "(gate = %.1f%%) -- refusing to write events_raw.csv on an undisclosed hole. Re-run to "
                  "resume from the gzip cache after the network issue clears.", pct_missing, COMPLETENESS_GATE_PCT)
        with open(SUMMARY_TXT, "w") as f:
            f.write(f"COMPLETENESS GATE FAILED: {n_missing}/{len(unique_ciks)} CIKs missing "
                    f"({pct_missing:.2f}% > {COMPLETENESS_GATE_PCT}%)\n")
        sys.exit(1)

    events = pd.DataFrame(rows, columns=[
        "cik", "symbol", "form", "filing_date", "acceptance_datetime", "items", "primary_document", "classes",
    ])
    events.sort_values(["filing_date", "cik", "symbol"], inplace=True)
    events.to_csv(EVENTS_CSV, index=False)
    log.info("wrote %s: %d rows", EVENTS_CSV, len(events))

    class_names = ["OFFERING", "SHELF", "REVERSE_SPLIT", "AUDITOR", "NON_RELIANCE",
                   "LATE_FILING", "OFFICER_EXIT", "CONTRACT", "ACTIVIST"]
    counts = {c: int(events["classes"].str.contains(rf"\b{c}\b", regex=True).sum()) for c in class_names}
    counts["BUYBACK_OR_INSIDER"] = -1  # sentinel: not computable from this fetch, see module docstring

    with open(SUMMARY_TXT, "w") as f:
        f.write(f"universe symbols: {len(symbols)}\n")
        f.write(f"mapped to CIK: {len(mapping)}  LOST: {lost}\n")
        f.write(f"distinct CIKs fetched: {len(unique_ciks)}  missing: {n_missing} ({pct_missing:.2f}%)\n")
        f.write(f"filings 2019-01-01..2026-09-25: {len(events)}\n")
        f.write("class counts (a filing may carry several classes):\n")
        for c, n in counts.items():
            f.write(f"  {c}: {n if n >= 0 else 'NOT_COMPUTABLE (needs Form 4 XML parse, see docstring)'}\n")

    with open(DONE_SENTINEL, "w") as f:
        f.write("done\n")
    log.info("=== FETCH stage complete ===")


if __name__ == "__main__":
    main()
