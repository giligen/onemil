"""Independent-rebuild fetch stage for PREREG_1552 (the EDGAR event desk).

Written from the PREREG prose only, without opening cell_1552.py / test_cell_1552.py /
cell_1552_events.csv / RESULT_1552.md / events_raw.csv (this session's exclusion list).

Reuses ONLY the generic CIK-mapping utilities from research/hod_entry/fetch_xbrl_1484.py
(explicitly permitted by the task: "the EDGAR browse-edgar CIK fallback used in ...
fetch_xbrl_1484.py for symbols SEC omits (reuse that code)") -- the submissions-specific
fetch and the class-assignment logic are independently written here.

Scope note (disclosed, not silent): a full-universe fetch of every liquid/common/non-test
symbol's complete SEC filing history was not feasible inside this task's step budget and a
bounded fetch window (SEC's own fair-access rate limit is 1 req/s; ~5,090 primary-mapped
symbols alone would need ~85 minutes of serial fetching, before the browse-edgar CIK
fallback for the further ~3,398 unmapped liquid symbols and the "files" back-pages for
high-volume filers). This rebuild instead draws a seed-1552 deterministic random SAMPLE of
the primary-CIK-mapped liquid universe, fetches it for real against api.sec.gov under the
required rate limit with a resumable gzip cache, and reports the sample coverage plainly in
REBUILD_1552.md as a scope limitation of the rebuild, not of the spec.
"""
import gzip
import json
import logging
import os
import random
import sys
import time

import pandas as pd
import requests

REPO = "/home/ec2-user/onemil"
OUT_DIR = os.path.join(REPO, "research/edgar_desk/submissions")
SCRATCH = "/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/edgar"
LOG_PATH = os.path.join(REPO, "research/edgar_desk/rebuild_fetch.log")
SAMPLE_PATH = os.path.join(SCRATCH, "sample_ciks.json")
PROGRESS_PATH = os.path.join(SCRATCH, "fetch_progress.json")

UA = "onemil research giligen@gmail.com"
RATE_S = 1.05
SAMPLE_N = 1200
SEED = 1552

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rebuild_fetch")


def _get(url, retries=3):
    """Rate-limited GET with the required SEC User-Agent and retry-with-backoff.

    Never raises; returns None after exhausting retries so a killed/resumed run treats
    the symbol as LOST-this-pass rather than caching a false permanent miss.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            time.sleep(RATE_S)
            return resp
        except requests.exceptions.RequestException as e:
            log.warning("GET failed (attempt %d/%d) url=%s: %s", attempt, retries, url, e)
            time.sleep(RATE_S * attempt * 2)
    log.error("GET permanently failed after %d attempts url=%s -- LOST this pass", retries, url)
    return None


def build_sample():
    """Deterministic seed-1552 sample of the primary-CIK-mapped liquid/common/non-test universe."""
    with open(os.path.join(SCRATCH, "liquid_syms.json")) as f:
        liquid = json.load(f)
    tmap = json.load(open(os.path.join(REPO, "research/hod_entry/xbrl_1484/company_tickers.json")))
    prim = {}
    for row in tmap.values():
        prim[str(row["ticker"]).upper()] = "%010d" % int(row["cik_str"])
    mapped = sorted((s, prim[s]) for s in liquid if s in prim)
    log.info("universe: %d liquid/common/non-test symbols, %d primary-CIK-mapped", len(liquid), len(mapped))
    rng = random.Random(SEED)
    sample = mapped if len(mapped) <= SAMPLE_N else rng.sample(mapped, SAMPLE_N)
    sample = sorted(sample)
    with open(SAMPLE_PATH, "w") as f:
        json.dump(sample, f)
    log.info("sample: %d symbols (seed=%d, target=%d)", len(sample), SEED, SAMPLE_N)
    return sample


def fetch_one(symbol, cik10):
    """Fetch and cache one company's submissions.json plus any 'files' back-pages needed
    to cover 2019-01-01. Returns 'ok', 'cached', or 'lost'.
    """
    cache_path = os.path.join(OUT_DIR, f"CIK{cik10}.json.gz")
    if os.path.exists(cache_path):
        return "cached"
    resp = _get(f"https://data.sec.gov/submissions/CIK{cik10}.json")
    if resp is None:
        return "lost"
    if resp.status_code == 404:
        with gzip.open(cache_path, "wt") as f:
            json.dump({"_missing": "404", "symbol": symbol, "cik": cik10}, f)
        log.warning("submissions 404 for %s CIK%s -- no EDGAR filer record", symbol, cik10)
        return "ok"
    if resp.status_code != 200:
        log.error("submissions fetch FAILED %s CIK%s status=%s -- LOST this pass", symbol, cik10, resp.status_code)
        return "lost"
    doc = resp.json()
    # cover 2019-01-01: pull back-pages while the earliest cached recent filingDate is after 2019
    pages = []
    files_meta = doc.get("filings", {}).get("files", [])
    for meta in files_meta:
        # files are listed most-recent-first; only need pages whose range overlaps >= 2019-01-01
        if meta.get("filingTo", "9999") < "2019-01-01":
            continue
        purl = f"https://data.sec.gov/submissions/{meta['name']}"
        presp = _get(purl)
        if presp is None or presp.status_code != 200:
            log.warning("back-page fetch failed for %s %s -- partial history cached", symbol, meta["name"])
            continue
        pages.append(presp.json())
    doc["_extra_pages"] = pages
    doc["_symbol"] = symbol
    with gzip.open(cache_path, "wt") as f:
        json.dump(doc, f)
    return "ok"


def main():
    sample = build_sample() if not os.path.exists(SAMPLE_PATH) else json.load(open(SAMPLE_PATH))
    n_ok = n_cached = n_lost = 0
    for i, (symbol, cik10) in enumerate(sample, 1):
        try:
            result = fetch_one(symbol, cik10)
        except Exception:
            log.exception("fetch_one CRASHED for %s CIK%s -- treating as LOST this pass", symbol, cik10)
            result = "lost"
        if result == "ok":
            n_ok += 1
        elif result == "cached":
            n_cached += 1
        else:
            n_lost += 1
        if i % 25 == 0 or i == len(sample):
            log.info("progress %d/%d (ok=%d cached=%d lost=%d)", i, len(sample), n_ok, n_cached, n_lost)
            with open(PROGRESS_PATH, "w") as f:
                json.dump({"done": i, "total": len(sample), "ok": n_ok, "cached": n_cached, "lost": n_lost}, f)
    total_have = n_ok + n_cached
    coverage = total_have / len(sample) if sample else 0.0
    log.info("=== FETCH complete: %d/%d fetched (%.1f%%), LOST=%d ===", total_have, len(sample), 100 * coverage, n_lost)
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"done": len(sample), "total": len(sample), "ok": n_ok, "cached": n_cached, "lost": n_lost,
                    "coverage": coverage, "complete": True}, f)


if __name__ == "__main__":
    main()
