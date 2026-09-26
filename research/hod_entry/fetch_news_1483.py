#!/usr/bin/env python3
"""Build masked news windows for cells 1,483-1,485 (research/hod_entry/PREREG_1483.md).

For every base HOD-break fill (causal_arming_causal.csv, status=='fill', 9,911 rows):
pull Benzinga articles (Alpaca News API) for that symbol in the point-in-time window
    [16:00 ET the previous NYSE session  ->  arm instant = ET minute (fill_min - 1) on the fill day]
keep own-name articles (<= 3 symbols listed), MASK the fill symbol and company name
with "the company" and any other listed ticker with "another company", then split
into JSON batches for a later Haiku classification pass (not done by this script).

Resumable: one JSON cache file per (symbol, day) under news_1483/cache/ -- a rerun
skips any fill whose cache file already exists, so a killed/restarted run continues
where it left off. Rate limit <= 5 req/s (Alpaca news endpoint).

Outputs (all under research/hod_entry/news_1483/, gitignored):
    cache/<symbol>__<day>.json   one file per fetched fill (own-name articles only)
    raw.parquet                  every kept raw (unmasked) article, one row each
    masked.parquet                same rows with entities masked, text truncated
    batches/batch_NNN.json       <= 200 items each: {item_id, fill_key, created_at, text}
    manifest.json                {n_batches, n_items}
    FETCH_REPORT.md              coverage, items/fill distribution, n_batches

Usage:
    nohup python3 -u research/hod_entry/fetch_news_1483.py \
        > research/hod_entry/news_1483/fetch.log 2>&1 &

    # Rebuild raw/masked/batches from whatever is already cached, no new API calls:
    python3 research/hod_entry/fetch_news_1483.py --aggregate-only

    # Smoke-test on the first N base fills:
    python3 research/hod_entry/fetch_news_1483.py --max-rows 20
"""
import argparse
import html
import json
import logging
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "research/hod_entry/news_1483"
CACHE_DIR = OUT_DIR / "cache"
BATCH_DIR = OUT_DIR / "batches"
BASE_CSV = ROOT / "research/hod_entry/causal_arming_causal.csv"
ASSETS_CSV = ROOT / "data/research/databento/alpaca_assets_all_20260905.csv"

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

RATE_LIMIT_S = 0.22          # ~4.5 req/s, under the 5 req/s Alpaca news cap
MAX_RETRIES = 5
LIMIT_PER_WINDOW = 50        # NewsRequest.limit; the SDK auto-paginates to this total
OWN_NAME_MAX_SYMBOLS = 3     # PREREG_1483: "own-name only (articles listing <= 3 symbols)"
SUMMARY_TRUNCATE = 600
BATCH_SIZE = 200
PROGRESS_EVERY = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fetch_news_1483")

# NYSE full-market-closure holidays, 2025-2026 (base book spans 2025-07-01..2026-05-29).
# Early-close days (e.g. Jul 3, day after Thanksgiving, Dec 24) are NOT full closures
# and are deliberately excluded: the market is open for the previous-session 16:00 bar.
NYSE_HOLIDAYS = {
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
}


def previous_session(d: date) -> date:
    """Previous NYSE trading day strictly before calendar date d."""
    prev = d - timedelta(days=1)
    while prev.weekday() >= 5 or prev in NYSE_HOLIDAYS:
        prev -= timedelta(days=1)
    return prev


def build_window(day_str: str, fill_min: float):
    """Point-in-time news window for one fill, per PREREG_1483.

    window_start = 16:00:00 ET on the previous NYSE session.
    window_end   = the arm instant = the ET minute (fill_min - 1) on the fill day
                   (fill_min is minutes-since-ET-midnight of the fill bar; the arm
                   instant is one minute before it, floored to the minute).
    Returns (start_utc, end_utc) as tz-aware datetimes.
    """
    fill_day = datetime.strptime(day_str, "%Y-%m-%d").date()
    prev_day = previous_session(fill_day)
    window_start = datetime(prev_day.year, prev_day.month, prev_day.day, 16, 0, 0, tzinfo=ET)

    arm_min = max(int(fill_min) - 1, 0)
    arm_hour, arm_minute = divmod(arm_min, 60)
    window_end = (
        datetime(fill_day.year, fill_day.month, fill_day.day, 0, 0, 0, tzinfo=ET)
        + timedelta(hours=arm_hour, minutes=arm_minute)
    )
    return window_start.astimezone(UTC), window_end.astimezone(UTC)


# --------------------------------------------------------------------------- #
# Entity masking
# --------------------------------------------------------------------------- #

_STOPWORDS = {
    "the", "a", "an", "and", "of", "co", "inc", "corp", "corporation", "company",
    "limited", "ltd", "llc", "plc", "trust", "fund", "group", "holding",
    "holdings", "one", "first", "new", "class", "common", "stock", "shares",
    "ordinary", "depositary", "american", "global", "national", "international",
    "partners", "partnership", "series", "preferred", "warrant", "warrants",
    "rights", "units", "reit", "bank", "general", "united", "royal", "western",
    "eastern", "southern", "northern", "capital", "financial",
}
_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
_SUFFIX_RE = re.compile(
    r"\b("
    r"common stock|ordinary shares|depositary shares|class [a-z]|"
    r"incorporated|inc\.?|corporation|corp\.?|company|co\.?|"
    r"limited|ltd\.?|l\.l\.c\.?|llc|plc|group|holdings?|holding|"
    r"s\.a\.?|n\.v\.?|trust|l\.p\.?|lp|reit|warrants?|units?|rights?|"
    r"preferred|series [a-z0-9]+"
    r")\s*\.?\s*$",
    re.IGNORECASE,
)


def name_variants(raw_name: str) -> set:
    """All usable masking variants of a company name: the raw name plus every
    intermediate state of iteratively stripping trailing parentheticals and
    corporate/security suffixes (e.g. 'Apple Inc. Common Stock' -> 'Apple Inc.'
    -> 'Apple'). A candidate is kept only if it is >= 3 chars and not, on its
    own, a generic stopword -- guards against over-stripping down to a common
    word ('Group', 'A', ...) that would mask unrelated text everywhere it appears.

    News prose almost always refers to a multi-word name by its first word
    alone ('Ultragenyx' for 'Ultragenyx Pharmaceutical'), so the first word of
    every kept multi-word variant is also added as its own candidate, subject
    to the same stopword/length guard.
    """
    variants = set()
    cur = raw_name.strip()
    prev = None
    steps = 0
    while cur != prev and steps < 6:
        prev = cur
        if len(cur) >= 3 and cur.lower() not in _STOPWORDS:
            variants.add(cur)
        cur = _PAREN_RE.sub("", cur).strip()
        cur = _SUFFIX_RE.sub("", cur).strip().rstrip(",").strip()
        steps += 1
    if len(cur) >= 3 and cur.lower() not in _STOPWORDS:
        variants.add(cur)

    for v in list(variants):
        words = v.split()
        if len(words) < 2:
            continue
        first = words[0].strip(",.")
        if len(first) >= 3 and first.lower() not in _STOPWORDS:
            variants.add(first)
    return variants


def load_asset_names(assets_csv: Path) -> dict:
    """symbol -> set of masking name-variants, from the Alpaca asset list."""
    df = pd.read_csv(assets_csv, dtype=str).dropna(subset=["symbol", "name"])
    if "status" in df.columns:
        df = df.sort_values("status", ascending=False)  # 'active' before 'inactive'
    out = {}
    for sym, grp in df.groupby("symbol"):
        variants = set()
        for nm in grp["name"].unique():
            variants |= name_variants(nm)
        out[sym] = variants
    return out


def _mask_pattern(token: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9])\$?" + re.escape(token) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def mask_text(text: str, symbol: str, own_names: set, article_symbols: list) -> str:
    """Replace the fill symbol / company name with 'the company' and any other
    listed ticker with 'another company'. Longest patterns first so a longer
    name is not left partially masked by a shorter substring match."""
    if not text:
        return text
    masked = text
    own_tokens = sorted({symbol, *own_names}, key=len, reverse=True)
    for tok in own_tokens:
        masked = _mask_pattern(tok).sub("the company", masked)
    for sym in article_symbols:
        if sym == symbol:
            continue
        masked = _mask_pattern(sym).sub("another company", masked)
    return masked


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #

def make_news_client():
    """Alpaca NewsClient from .env credentials (raw_data=True: plain dicts, no
    pydantic model guessing needed)."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    from config import Config

    c = Config()
    if not c.alpaca_api_key or not c.alpaca_api_secret:
        log.error("ALPACA_API_KEY / ALPACA_API_SECRET missing or empty in .env -- cannot fetch news")
        sys.exit(1)
    from alpaca.data.historical.news import NewsClient

    return NewsClient(c.alpaca_api_key, c.alpaca_api_secret, raw_data=True)


def fetch_symbol_day(client, symbol: str, day_str: str, window_start, window_end):
    """One news pull for (symbol, day). Returns a list of raw article dicts on
    success (possibly empty), or None on failure after retries (distinct from
    an empty list, so a retry on resume is not mistaken for 'fetched, zero
    articles'). The Alpaca SDK auto-paginates internally up to `limit` total."""
    from alpaca.data.requests import NewsRequest

    req = NewsRequest(
        symbols=symbol, start=window_start, end=window_end,
        limit=LIMIT_PER_WINDOW, include_content=False,
    )
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.get_news(req)
            return resp.get("news", []) if isinstance(resp, dict) else []
        except Exception as e:  # noqa: BLE001 -- any transient API/network error
            last_err = e
            wait = min(2 ** attempt, 30)
            log.warning(
                f"fetch_symbol_day: {symbol} {day_str} attempt {attempt}/{MAX_RETRIES} "
                f"failed ({e!r}); retrying in {wait}s"
            )
            time.sleep(wait)
    log.error(f"fetch_symbol_day: {symbol} {day_str} FAILED after {MAX_RETRIES} attempts: {last_err!r}")
    return None


def cache_path(symbol: str, day_str: str) -> Path:
    return CACHE_DIR / f"{symbol}__{day_str}.json"


def run_fetch(fills: pd.DataFrame, max_rows=None):
    """Resumable fetch loop: one API call per (symbol, day) not already cached."""
    client = make_news_client()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    rows = fills if max_rows is None else fills.head(max_rows)
    n = len(rows)
    n_cached, n_fetched, n_failed, n_kept_articles = 0, 0, 0, 0
    t0 = time.time()

    for i, row in enumerate(rows.itertuples(index=False), start=1):
        symbol, day_str, fill_min = row.symbol, row.day, row.fill_min
        cp = cache_path(symbol, day_str)
        if cp.exists():
            n_cached += 1
        else:
            window_start, window_end = build_window(day_str, fill_min)
            articles = fetch_symbol_day(client, symbol, day_str, window_start, window_end)
            time.sleep(RATE_LIMIT_S)
            if articles is None:
                n_failed += 1
                continue  # leave no cache file -> retried on next resume
            own_name = [a for a in articles if len(a.get("symbols", [])) <= OWN_NAME_MAX_SYMBOLS]
            cp.write_text(json.dumps(own_name))
            n_fetched += 1
            n_kept_articles += len(own_name)

        if i % PROGRESS_EVERY == 0 or i == n:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta_min = (n - i) / rate / 60 if rate > 0 else float("nan")
            log.info(
                f"progress {i}/{n} | cached={n_cached} fetched={n_fetched} failed={n_failed} "
                f"kept_articles(new)={n_kept_articles} | {rate:.2f} rows/s | ETA {eta_min:.1f} min"
            )

    log.info(
        f"fetch done: {n} rows, {n_cached} pre-cached, {n_fetched} newly fetched, "
        f"{n_failed} failed (will retry on next run)"
    )
    if n_failed:
        log.warning(f"{n_failed} symbol-days failed after retries -- rerun the script to retry them")


# --------------------------------------------------------------------------- #
# Aggregate + mask + batch
# --------------------------------------------------------------------------- #

def aggregate_raw(fills: pd.DataFrame) -> pd.DataFrame:
    """Join every cached (symbol, day) file's own-name articles onto the base
    fills; one output row per kept article. Fills with no cache file yet (not
    fetched) or zero own-name articles are absent from the article rows but are
    still counted in the coverage report via `fills`."""
    records = []
    for row in fills.itertuples(index=False):
        symbol, day_str = row.symbol, row.day
        cp = cache_path(symbol, day_str)
        if not cp.exists():
            continue
        articles = json.loads(cp.read_text())
        fill_key = f"{symbol}__{day_str}"
        for a in articles:
            records.append({
                "fill_key": fill_key,
                "symbol": symbol,
                "day": day_str,
                "article_id": a.get("id"),
                "created_at": a.get("created_at"),
                "headline": a.get("headline", ""),
                "summary": a.get("summary", ""),
                "symbols": a.get("symbols", []),
                "source": a.get("source", ""),
                "url": a.get("url", ""),
            })
    raw = pd.DataFrame.from_records(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(raw):
        raw_for_parquet = raw.copy()
        raw_for_parquet["symbols"] = raw_for_parquet["symbols"].apply(json.dumps)
        raw_for_parquet.to_parquet(OUT_DIR / "raw.parquet", index=False)
    else:
        pd.DataFrame(columns=[
            "fill_key", "symbol", "day", "article_id", "created_at", "headline",
            "summary", "symbols", "source", "url",
        ]).to_parquet(OUT_DIR / "raw.parquet", index=False)
    log.info(f"raw.parquet: {len(raw)} own-name article rows")
    return raw


def mask_and_batch(raw: pd.DataFrame, asset_names: dict):
    """Mask entities, truncate, write masked.parquet + JSON batches + manifest."""
    if len(raw) == 0:
        log.warning("mask_and_batch: zero raw articles -- writing empty masked.parquet / no batches")
        pd.DataFrame(columns=["item_id", "fill_key", "created_at", "text"]).to_parquet(
            OUT_DIR / "masked.parquet", index=False
        )
        BATCH_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "manifest.json").write_text(json.dumps({"n_batches": 0, "n_items": 0}))
        return pd.DataFrame(columns=["item_id", "fill_key", "created_at", "text"])

    items = []
    for row in raw.itertuples(index=False):
        own_names = asset_names.get(row.symbol, set())
        masked_headline = html.unescape(mask_text(row.headline, row.symbol, own_names, row.symbols))
        masked_summary = html.unescape(mask_text(row.summary, row.symbol, own_names, row.symbols))[:SUMMARY_TRUNCATE]
        text = (masked_headline or "").strip()
        if masked_summary.strip():
            text = f"{text}\n\n{masked_summary.strip()}"
        item_id = f"{row.fill_key}::{row.article_id}"
        items.append({"item_id": item_id, "fill_key": row.fill_key, "created_at": row.created_at, "text": text})

    masked = pd.DataFrame.from_records(items)
    masked.to_parquet(OUT_DIR / "masked.parquet", index=False)
    log.info(f"masked.parquet: {len(masked)} items")

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in BATCH_DIR.glob("batch_*.json"):
        old.unlink()
    n_batches = 0
    for start in range(0, len(items), BATCH_SIZE):
        chunk = items[start:start + BATCH_SIZE]
        n_batches += 1
        (BATCH_DIR / f"batch_{n_batches:03d}.json").write_text(json.dumps(chunk))
    manifest = {"n_batches": n_batches, "n_items": len(items)}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest))
    log.info(f"manifest: {manifest}")
    return masked


def write_report(fills: pd.DataFrame, raw: pd.DataFrame, manifest_path: Path):
    """Coverage + items/fill distribution + n_batches, per the task's report spec."""
    n_fills = len(fills)
    per_fill = raw.groupby("fill_key").size() if len(raw) else pd.Series(dtype=int)
    fill_keys_all = fills["symbol"] + "__" + fills["day"]
    counts = fill_keys_all.map(per_fill).fillna(0).astype(int)
    n_with_article = int((counts >= 1).sum())
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"n_batches": 0, "n_items": 0}

    dist = counts.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])
    lines = [
        "# fetch_news_1483 report",
        "",
        f"base fills: {n_fills}",
        f"fills with >= 1 own-name article: {n_with_article} ({100 * n_with_article / n_fills:.1f}%)",
        f"fills with 0 own-name articles: {n_fills - n_with_article}",
        "",
        "## items per fill (own-name articles kept, incl. zeros)",
        f"mean={dist['mean']:.2f} std={dist['std']:.2f} "
        f"p25={dist['25%']:.0f} p50={dist['50%']:.0f} p75={dist['75%']:.0f} "
        f"p90={dist['90%']:.0f} p99={dist['99%']:.0f} max={dist['max']:.0f}",
        "",
        f"## batches\nn_batches={manifest['n_batches']} n_items={manifest['n_items']}",
    ]
    report = "\n".join(lines)
    (OUT_DIR / "FETCH_REPORT.md").write_text(report + "\n")
    log.info("\n" + report)


# --------------------------------------------------------------------------- #

def load_base_fills() -> pd.DataFrame:
    df = pd.read_csv(BASE_CSV, low_memory=False)
    fills = df[df.status == "fill"][["day", "symbol", "split", "fill_min"]].reset_index(drop=True)
    log.info(f"base fills loaded: {len(fills)} rows (status=='fill') from {BASE_CSV.name}")
    return fills


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-rows", type=int, default=None, help="smoke-test on the first N base fills")
    ap.add_argument("--aggregate-only", action="store_true", help="skip fetching, rebuild outputs from cache/")
    args = ap.parse_args()

    fills = load_base_fills()
    if args.max_rows:
        fills = fills.head(args.max_rows)

    if not args.aggregate_only:
        run_fetch(fills, max_rows=args.max_rows)

    asset_names = load_asset_names(ASSETS_CSV)
    log.info(f"asset name map loaded: {len(asset_names)} symbols")
    raw = aggregate_raw(fills)
    mask_and_batch(raw, asset_names)
    write_report(fills, raw, OUT_DIR / "manifest.json")


if __name__ == "__main__":
    main()
