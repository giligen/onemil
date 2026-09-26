#!/usr/bin/env python3
"""Cells 1,483-1,485 -- research/hod_entry/PREREG_1483.md (FROZEN 2026-09-26).

WHY is a name moving (LLM catalyst attribution of the day's own-name news, entities masked,
Haiku classification) and CAN it afford to (SEC XBRL cash runway, point-in-time by filing date)
as structural filters on the 9,911 cell-1,438 fills (correct levels), 1,478-standard outcome_R.

Pipeline (each stage resumable -- re-running skips work already on disk under research/hod_entry/
news_1483/ and research/hod_entry/xbrl_1484/, both gitignored):
  1. load_base()            -- model_1478_predictions.csv (day,symbol,fill_min,split,outcome_R,
                                 L1,L2,store_served_1438) + exit_m from causal_arming_causal.csv.
  2. symbol_master()         -- SEC company_tickers.json once: symbol -> (cik, title).
  3. fetch_news()            -- Alpaca NewsClient per unique symbol, full date span, paginated,
                                 <=4 req/s; raw articles cached to news_1483/raw/{symbol}.json.
  4. build_candidates()      -- per fill, own-name articles (<=3 symbols listed, symbol present)
                                 created_at in [prior session 16:00 ET, fill_min ET) -- CAUSAL:
                                 fill_min is a looser (later) proxy for the PREREG's "arm instant"
                                 since arm_min is not a column on the base book; documented in
                                 RESULT as a caveat, not silently assumed away. Entities masked
                                 (own ticker/title -> "the company"; other listed tickers ->
                                 "another company") BEFORE the item is shown to any classifier.
                                 Writes news_1483/masked.parquet (one row per unique item).
  5. classify_batches()      -- Haiku (claude-haiku-4-5), batches of 20 masked items, JSON out;
                                 labels_A/batch_NNNN.json. Second, independently-worded pass on a
                                 500-item seed-1483 sample -> robustness_500.json.
  6. fetch_xbrl()            -- SEC companyfacts per CIK, 1 req/s, raw cached to xbrl_1484/raw/.
  7. compute_runway()        -- cash (latest instant fact filed < fill day) and a duration-
                                 normalised quarterly operating cash flow (val * 91.25/duration_days,
                                 so a 10-K's annual figure and a 10-Q's YTD figure both reduce to a
                                 quarterly-equivalent burn -- an explicit, documented approximation,
                                 not a silent one) filed < fill day. runway_1484.csv.
  8. build_flags()           -- per fill: class, materiality, hard_catalyst, runway_q, runway_ok/
                                 bad. cell_1483_fills.csv.
  9. score_cells()           -- 1,483 CATALYST / 1,484 RUNWAY / 1,485 JOINT vs the frozen pass bar
                                 (PREREG "Pass bar" section) on TRAIN-H2 and VAL, plus the two
                                 report-only tables. RESULT_1483.md.

Usage:
    nice -n 19 python3 research/hod_entry/cell_1483.py --stage all
    nice -n 19 python3 research/hod_entry/cell_1483.py --stage news       # resumable, one stage
    nice -n 19 python3 research/hod_entry/cell_1483.py --stage score      # re-score only

Not allowed (PREREG): re-labelling classes after a number exists; changing runway thresholds;
sentiment scores of any kind; unmasked entities in the classifier prompt; reading TEST.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

REPO = '/home/ec2-user/onemil'
sys.path.insert(0, REPO)

from research.hod_consol import run_consol as consol          # noqa: E402  (simulate_slots)
from research.hod_entry import cell_1445 as c1445             # noqa: E402  (t/ex5/null/fills_wk)

OUT_DIR = os.path.join(REPO, 'research/hod_entry')
NEWS_DIR = os.path.join(OUT_DIR, 'news_1483')
XBRL_DIR = os.path.join(OUT_DIR, 'xbrl_1484')
NEWS_RAW = os.path.join(NEWS_DIR, 'raw')
LABELS_A = os.path.join(NEWS_DIR, 'labels_A')
XBRL_RAW = os.path.join(XBRL_DIR, 'raw')
for d in (NEWS_DIR, NEWS_RAW, LABELS_A, XBRL_DIR, XBRL_RAW):
    os.makedirs(d, exist_ok=True)

BASE_FILLS = os.path.join(OUT_DIR, 'model_1478_predictions.csv')
CAUSAL_CSV = os.path.join(OUT_DIR, 'causal_arming_causal.csv')
COMPANY_TICKERS_URL = 'https://www.sec.gov/files/company_tickers.json'
SEC_UA = 'onemil research giligen@gmail.com'
XBRL_URL = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json'

CLASSES = ['earnings_guidance', 'fda_clinical', 'ma_strategic', 'contract_product',
           'financing_dilution', 'analyst_action', 'legal_regulatory', 'sector_sympathy',
           'no_news']
HARD_CLASSES = {'earnings_guidance', 'fda_clinical', 'ma_strategic', 'contract_product'}
HAIKU_MODEL = 'claude-haiku-4-5'
NULL_SEED = 1483
NULL_DRAWS = 1000
BATCH_SIZE = 20
ROBUST_SAMPLE_N = 500
ROBUST_AGREE_MIN = 0.85
QUARTER_DAYS = 91.25

PASS_KEPT_MEAN = 0.15
PASS_T = 2.5
PASS_FILLS_WK = 3.0
PASS_TRAINH2_T = 1.0
PASS_NULL_PCTILE = 99.0
PASS_COVERAGE = 0.70
PASS_CACHEONLY_TOL = 0.05
CACHEONLY_REF = 0.195


def log(msg):
    """Verbose progress line, timestamped, flushed (print() inside -c is buffered; this is a
    script file run directly, but flush=True keeps nohup logs live regardless)."""
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)


# ================================================================================================
# Step 1: base book
# ================================================================================================

def load_base():
    """9,911 fills: day, symbol, split, fill_min, outcome_R, L1, L2, store_served_1438, exit_m
    (exit_m merged from causal_arming_causal.csv on day+symbol, both unique keys -- verified)."""
    base = pd.read_csv(BASE_FILLS)
    assert len(base) == 9911, f'expected 9,911 fills, got {len(base)}'
    ca = pd.read_csv(CAUSAL_CSV, low_memory=False)
    ca = ca[ca.status == 'fill'][['day', 'symbol', 'exit_m']]
    assert ca[['day', 'symbol']].duplicated().sum() == 0, 'causal_arming_causal.csv day+symbol not unique'
    merged = base.merge(ca, on=['day', 'symbol'], how='left', validate='one_to_one')
    n_missing_exit = merged['exit_m'].isna().sum()
    if n_missing_exit:
        log(f'WARNING load_base: {n_missing_exit} fills missing exit_m after merge -- '
            f'fills/wk slotting will drop them')
    log(f'load_base: {len(merged)} fills, splits {merged.split.value_counts().to_dict()}')
    return merged


# ================================================================================================
# Step 2: symbol -> (cik, title) via SEC's free ticker map (single fetch, cached)
# ================================================================================================

def symbol_master():
    path = os.path.join(XBRL_DIR, 'company_tickers.json')
    if not os.path.exists(path):
        log('symbol_master: fetching SEC company_tickers.json (single request)')
        req = urllib.request.Request(COMPANY_TICKERS_URL, headers={'User-Agent': SEC_UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        with open(path, 'w') as f:
            json.dump(data, f)
    else:
        with open(path) as f:
            data = json.load(f)
    out = {}
    for row in data.values():
        out[row['ticker'].upper()] = dict(cik=int(row['cik_str']), title=row['title'])
    log(f'symbol_master: {len(out)} SEC ticker->CIK/title entries loaded')
    return out


# ================================================================================================
# Step 3: news fetch -- Alpaca NewsClient, one request-stream per unique symbol, paginated
# ================================================================================================

def fetch_news(symbols, day_min, day_max):
    """Per unique symbol: full [day_min-7, day_max+1] span in one paginated stream (day+symbol is
    a unique key on the base book -- one fetch per symbol covers every fill for that symbol).
    Resumable: a symbol whose raw file already exists is skipped. Rate limit <=4 req/s (Alpaca
    caps at 5); NewsClient reads ALPACA_API_KEY/SECRET from Config, same as production."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, '.env'), override=True)
    from config import Config
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    cfg = Config()
    nc = NewsClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    start = (pd.Timestamp(day_min) - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    end = (pd.Timestamp(day_max) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    todo = [s for s in symbols if not os.path.exists(os.path.join(NEWS_RAW, f'{s}.json'))]
    log(f'fetch_news: {len(symbols)} symbols total, {len(todo)} to fetch '
        f'({len(symbols) - len(todo)} already cached), window {start}..{end}')
    n_err = 0
    for i, sym in enumerate(todo):
        articles, token = [], None
        try:
            while True:
                req = NewsRequest(symbols=sym, start=start, end=end, limit=50,
                                   include_content=False, page_token=token)
                res = nc.get_news(req)
                items = res.data.get('news', [])
                for it in items:
                    articles.append(dict(
                        id=it.id, headline=it.headline, summary=it.summary or '',
                        created_at=it.created_at.astimezone(timezone.utc).isoformat(),
                        symbols=list(it.symbols), source=it.source))
                token = res.next_page_token
                time.sleep(0.25)
                if not token:
                    break
        except Exception as e:
            n_err += 1
            log(f'WARNING fetch_news: {sym} failed ({e!r}); writing empty article list so the '
                f'symbol is not silently missing from coverage accounting')
            articles = []
        with open(os.path.join(NEWS_RAW, f'{sym}.json'), 'w') as f:
            json.dump(articles, f)
        if (i + 1) % 100 == 0 or i == len(todo) - 1:
            log(f'fetch_news: {i + 1}/{len(todo)} symbols done ({n_err} errors so far)')
    log(f'fetch_news: complete, {n_err} symbol-level fetch errors')


# ================================================================================================
# Step 4: causal windowing + own-name filter + entity masking
# ================================================================================================

def mask_text(text, own_symbol, own_title, other_symbols):
    """Replace the fill's own ticker/company-title tokens with 'the company', and every OTHER
    ticker present in the article's own symbols list with 'another company'. Bounded to
    ticker/title-token replacement (documented limitation, not a full NER masker -- see
    RESULT_1483.md caveats)."""
    if not text:
        return text
    out = text
    import re
    out = re.sub(rf'\b{re.escape(own_symbol)}\b', 'the company', out, flags=re.IGNORECASE)
    if own_title:
        first_word = own_title.split()[0]
        if len(first_word) >= 4:
            out = re.sub(rf'\b{re.escape(first_word)}\b', 'the company', out, flags=re.IGNORECASE)
    for other in other_symbols:
        if other == own_symbol:
            continue
        out = re.sub(rf'\b{re.escape(other)}\b', 'another company', out, flags=re.IGNORECASE)
    return out


def prev_session(day, calendar):
    """Latest trading day strictly before `day` in the sorted `calendar` (array of 'YYYY-MM-DD'
    strings, which sort identically to their date order). idx==0 (day predates the whole
    calendar) is a data problem, not a valid prior session -- returns `day` itself with a
    WARNING rather than silently producing a zero-length or wrong-day window."""
    day = str(day)
    idx = np.searchsorted(calendar, day)
    if idx == 0:
        log(f'WARNING prev_session: {day} at or before the start of the loaded calendar; '
            f'using {day} itself as the "previous session" (news window will be empty/short)')
        return day
    return str(calendar[idx - 1])


def build_candidates(base, master):
    """One row per fill: the masked own-name news items in its causal window. Also writes
    masked.parquet (unique items, deduped by id) for the classifier stage."""
    cal_path = os.path.join(REPO, 'data/research/databento/equs_daily_2025_2026.parquet')
    cal = np.sort(pd.read_parquet(cal_path, columns=['bar_date'])['bar_date'].unique())
    log(f'build_candidates: trading calendar {cal[0]}..{cal[-1]} ({len(cal)} sessions)')

    items_by_id = {}
    fill_item_ids = []   # list of (day, symbol, item_id, created_at)
    n_no_cache = 0
    for sym, g in base.groupby('symbol'):
        path = os.path.join(NEWS_RAW, f'{sym}.json')
        if not os.path.exists(path):
            n_no_cache += 1
            continue
        with open(path) as f:
            arts = json.load(f)
        title = master.get(sym, {}).get('title')
        for _, row in g.iterrows():
            day = row['day']
            prev = prev_session(day, cal)
            lo = pd.Timestamp(f'{prev} 16:00:00', tz='America/New_York').tz_convert('UTC')
            fill_ts = pd.Timestamp(day) + pd.to_timedelta(row['fill_min'], unit='m')
            hi = fill_ts.tz_localize('America/New_York').tz_convert('UTC')
            for a in arts:
                ca_ts = pd.Timestamp(a['created_at'])
                if not (lo <= ca_ts < hi):
                    continue
                if len(a['symbols']) > 3 or sym not in a['symbols']:
                    continue
                iid = a['id']
                if iid not in items_by_id:
                    items_by_id[iid] = dict(
                        item_id=iid, symbol=sym, created_at=a['created_at'],
                        headline_masked=mask_text(a['headline'], sym, title, a['symbols']),
                        summary_masked=mask_text(a['summary'], sym, title, a['symbols']))
                fill_item_ids.append((day, sym, iid, a['created_at']))
    if n_no_cache:
        log(f'WARNING build_candidates: {n_no_cache} symbols had no cached news file (fetch_news '
            f'not yet run for them) -- their fills get class=no_news_uncovered, excluded from '
            f'catalyst coverage')

    masked = pd.DataFrame(items_by_id.values())
    masked.to_parquet(os.path.join(NEWS_DIR, 'masked.parquet'), index=False)
    fmap = pd.DataFrame(fill_item_ids, columns=['day', 'symbol', 'item_id', 'created_at'])
    fmap.to_parquet(os.path.join(NEWS_DIR, 'fill_item_map.parquet'), index=False)
    log(f'build_candidates: {len(masked)} unique own-name items, {len(fmap)} fill-item edges, '
        f'{fmap[["day", "symbol"]].drop_duplicates().shape[0]} fills with >=1 candidate item')
    return masked, fmap


# ================================================================================================
# Step 5: Haiku classification
# ================================================================================================

PROMPT_PASS1 = """You are classifying short news items about a publicly traded company. The
company's own name and ticker have been masked and replaced with "the company"; other companies
mentioned are masked as "another company". Classify EACH item below into exactly one of these
classes, and rate its materiality to THE COMPANY on a 1-3 scale (3 = major, 1 = minor/routine):

- earnings_guidance: earnings results, guidance, or outlook changes
- fda_clinical: FDA action or clinical trial result
- ma_strategic: M&A, acquisition, merger, strategic partnership/investment
- contract_product: new contract, product launch, or major partnership (non-M&A)
- financing_dilution: securities offering, ATM program, convertible note, S-3 shelf filing
- analyst_action: analyst upgrade/downgrade/price-target change
- legal_regulatory: lawsuit, investigation, regulatory action (non-FDA)
- sector_sympathy: the item is about a peer or the sector, not directly about the company
- no_news: no substantive own-name news content (listicles, generic mentions, whale/options
  activity noise)

Items (JSON array, each with item_id, headline_masked, summary_masked):
{items_json}

Respond with ONLY a JSON array, one object per item, each exactly:
{{"item_id": <same id>, "class": "<one class>", "materiality": <1|2|3>}}
No prose, no markdown fences."""

PROMPT_PASS2 = """Read each masked news snippet below (the subject company's name/ticker has been
replaced with "the company"; other firms are "another company"). For each one, pick the single
best-fitting category and a materiality score:

Categories: earnings_guidance (results/guidance), fda_clinical (FDA/clinical trial), ma_strategic
(M&A/merger/strategic deal), contract_product (new deal/product, not M&A), financing_dilution
(share offering/ATM/convertible/shelf), analyst_action (analyst rating/target change),
legal_regulatory (lawsuit/investigation/non-FDA regulatory), sector_sympathy (about a peer/sector,
not the company itself), no_news (no real news content).

Materiality: 3=major event, 2=moderate, 1=minor/routine.

Snippets:
{items_json}

Output a bare JSON array only (no other text), each element exactly:
{{"item_id": <id>, "class": "<category>", "materiality": <1|2|3>}}"""


def call_haiku(client, prompt_template, batch):
    """One Messages API call classifying `batch` (list of item dicts). Returns {item_id: (class,
    materiality)} or {} on unrecoverable failure (logged as ERROR -- those items surface as
    class=classify_error, dropped from catalyst coverage, never silently folded into no_news)."""
    items_json = json.dumps([dict(item_id=b['item_id'], headline_masked=b['headline_masked'],
                                   summary_masked=b['summary_masked']) for b in batch])
    prompt = prompt_template.format(items_json=items_json)
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=HAIKU_MODEL, max_tokens=4096,
                messages=[{'role': 'user', 'content': prompt}])
            text = ''.join(b.text for b in resp.content if b.type == 'text').strip()
            text = text.strip('`')
            if text.lower().startswith('json'):
                text = text[4:]
            parsed = json.loads(text)
            out = {}
            for row in parsed:
                cls = row.get('class')
                if cls not in CLASSES:
                    cls = 'no_news'
                out[row['item_id']] = (cls, int(row.get('materiality', 1)))
            return out
        except Exception as e:
            wait = 2 ** attempt
            log(f'WARNING call_haiku: attempt {attempt + 1} failed ({e!r}); retrying in {wait}s')
            time.sleep(wait)
    log(f'ERROR call_haiku: batch of {len(batch)} items unrecoverable after 4 attempts')
    return {}


def _anthropic_client():
    """The ambient shell may already export an ANTHROPIC_API_KEY (the harness's own, not scoped
    for direct Messages API use) -- override=True forces the repo's own .env key to win, same
    fix as the fetch_news/Alpaca path."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, '.env'), override=True)
    import anthropic
    return anthropic.Anthropic()


def classify_batches(masked):
    client = _anthropic_client()
    n_batches = (len(masked) + BATCH_SIZE - 1) // BATCH_SIZE
    log(f'classify_batches: {len(masked)} items, {n_batches} batches of {BATCH_SIZE}')
    for bi in range(n_batches):
        out_path = os.path.join(LABELS_A, f'labelsA_batch_{bi:05d}.json')
        if os.path.exists(out_path):
            continue
        batch = masked.iloc[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE].to_dict('records')
        result = call_haiku(client, PROMPT_PASS1, batch)
        with open(out_path, 'w') as f:
            json.dump({str(k): v for k, v in result.items()}, f)
        if (bi + 1) % 25 == 0 or bi == n_batches - 1:
            log(f'classify_batches: {bi + 1}/{n_batches} batches done')
    log('classify_batches: complete')


def load_labels_a():
    """Reads only this pipeline's own labelsA_batch_*.json files -- a pre-existing, differently-
    shaped batch_000.json was found in this directory on a prior/parallel attempt (item_id format
    'SYM__day::id', key 'cls' not 'class'); it is not touched (not ours to delete) but is
    deliberately excluded by the distinct 'labelsA_batch_' prefix, and any unexpected shape is
    skipped with a WARNING rather than crashing the whole scoring run."""
    labels = {}
    n_skipped = 0
    for fn in sorted(os.listdir(LABELS_A)):
        if not fn.startswith('labelsA_batch_'):
            continue
        with open(os.path.join(LABELS_A, fn)) as f:
            d = json.load(f)
        if not isinstance(d, dict):
            n_skipped += 1
            log(f'WARNING load_labels_a: {fn} is not a {{item_id: [class, materiality]}} dict '
                f'(got {type(d).__name__}); skipped')
            continue
        for k, v in d.items():
            try:
                labels[int(k)] = tuple(v)
            except (TypeError, ValueError):
                n_skipped += 1
    if n_skipped:
        log(f'WARNING load_labels_a: {n_skipped} malformed entries/files skipped')
    return labels


def robustness_check(masked):
    """Second, independently-worded Haiku pass on a seed-1483 500-item sample; agreement rate on
    class label vs pass 1 is the PREREG's classification-robustness gate (>=85% or catalyst cells
    VOID)."""
    out_path = os.path.join(NEWS_DIR, 'robustness_500.json')
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)['agreement']
    labels1 = load_labels_a()
    rng = random.Random(NULL_SEED)
    pool = [i for i in masked['item_id'] if i in labels1]
    sample_ids = rng.sample(pool, min(ROBUST_SAMPLE_N, len(pool)))
    sample = masked[masked.item_id.isin(sample_ids)]
    client = _anthropic_client()
    pass2 = {}
    for i in range(0, len(sample), BATCH_SIZE):
        batch = sample.iloc[i:i + BATCH_SIZE].to_dict('records')
        pass2.update(call_haiku(client, PROMPT_PASS2, batch))
    n_compared = sum(1 for i in sample_ids if i in pass2)
    n_agree = sum(1 for i in sample_ids if i in pass2 and pass2[i][0] == labels1[i][0])
    agreement = n_agree / n_compared if n_compared else float('nan')
    log(f'robustness_check: {n_compared} items compared, agreement={agreement:.3f} '
        f'(pass bar {ROBUST_AGREE_MIN})')
    with open(out_path, 'w') as f:
        json.dump(dict(n_compared=n_compared, n_agree=n_agree, agreement=agreement), f)
    return agreement


# ================================================================================================
# Step 6-7: SEC XBRL runway
# ================================================================================================

CASH_TAGS = ['CashAndCashEquivalentsAtCarryingValue',
             'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',
             'CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations']
OCF_TAGS = ['NetCashProvidedByUsedInOperatingActivities',
            'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations']


def fetch_xbrl(symbols, master):
    mapped = [s for s in symbols if s in master]
    log(f'fetch_xbrl: {len(symbols)} symbols, {len(mapped)} mapped to a CIK via SEC '
        f'({len(symbols) - len(mapped)} unmapped -- wrappers/foreign/OTC, runway=NaN by design)')
    todo = [s for s in mapped if not os.path.exists(os.path.join(XBRL_RAW, f'{s}.json'))]
    log(f'fetch_xbrl: {len(todo)} to fetch, {len(mapped) - len(todo)} cached, 1 req/s')
    n_404 = n_err = 0
    for i, sym in enumerate(todo):
        cik = master[sym]['cik']
        url = XBRL_URL.format(cik=cik)
        req = urllib.request.Request(url, headers={'User-Agent': SEC_UA})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                n_404 += 1
                data = None
            else:
                n_err += 1
                log(f'WARNING fetch_xbrl: {sym} HTTP {e.code}; treated as no data')
                data = None
        except Exception as e:
            n_err += 1
            log(f'WARNING fetch_xbrl: {sym} failed ({e!r}); treated as no data')
            data = None
        facts = None
        if data:
            gaap = data.get('facts', {}).get('us-gaap', {})
            facts = {tag: gaap[tag] for tag in CASH_TAGS + OCF_TAGS if tag in gaap}
        with open(os.path.join(XBRL_RAW, f'{sym}.json'), 'w') as f:
            json.dump(facts, f)
        time.sleep(1.0)
        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            log(f'fetch_xbrl: {i + 1}/{len(todo)} done ({n_404} 404s, {n_err} other errors)')
    log(f'fetch_xbrl: complete, {n_404} 404 (no XBRL filer), {n_err} other errors')


def _latest_before(units_list, cutoff_date, want_duration=False):
    """Latest fact (by 'filed') with filed < cutoff_date, form in 10-Q/10-K. `want_duration`
    picks facts with both start/end (duration); else instant facts (end only)."""
    best = None
    for u in units_list:
        if u.get('form') not in ('10-Q', '10-K'):
            continue
        filed = u.get('filed')
        if not filed or filed >= cutoff_date:
            continue
        has_start = 'start' in u
        if want_duration != has_start:
            continue
        if best is None or filed > best['filed']:
            best = u
    return best


def compute_runway(base, master):
    rows = []
    n_no_facts = 0
    for sym, g in base.groupby('symbol'):
        path = os.path.join(XBRL_RAW, f'{sym}.json')
        facts = None
        if os.path.exists(path):
            with open(path) as f:
                facts = json.load(f)
        if not facts:
            n_no_facts += 1
            for _, row in g.iterrows():
                rows.append(dict(day=row['day'], symbol=sym, cash=np.nan, quarterly_ocf=np.nan,
                                  runway_q=np.nan, cf_positive=False, coverage=False))
            continue
        cash_units, ocf_units = [], []
        for tag in CASH_TAGS:
            if tag in facts:
                cash_units += facts[tag].get('units', {}).get('USD', [])
        for tag in OCF_TAGS:
            if tag in facts:
                ocf_units += facts[tag].get('units', {}).get('USD', [])
        for _, row in g.iterrows():
            day = row['day']
            cash_f = _latest_before(cash_units, day, want_duration=False)
            ocf_f = _latest_before(ocf_units, day, want_duration=True)
            cash = cash_f['val'] if cash_f else np.nan
            q_ocf = np.nan
            if ocf_f:
                dur = (pd.Timestamp(ocf_f['end']) - pd.Timestamp(ocf_f['start'])).days
                if 25 <= dur <= 400:
                    q_ocf = ocf_f['val'] * QUARTER_DAYS / dur
            cf_positive = bool(q_ocf >= 0) if not np.isnan(q_ocf) else False
            if np.isnan(cash) or np.isnan(q_ocf):
                runway_q, covered = np.nan, False
            else:
                covered = True
                runway_q = float('inf') if cf_positive else cash / max(-q_ocf, 1e-6)
            rows.append(dict(day=day, symbol=sym, cash=cash, quarterly_ocf=q_ocf,
                              runway_q=runway_q, cf_positive=cf_positive, coverage=covered))
    if n_no_facts:
        log(f'WARNING compute_runway: {n_no_facts} symbols had no usable XBRL cash/OCF facts')
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'runway_1484.csv'), index=False)
    log(f'compute_runway: {out.coverage.mean():.3f} coverage across {len(out)} fill-rows, '
        f'{out.cf_positive.mean():.3f} cf_positive share')
    return out


# ================================================================================================
# Step 8: flags
# ================================================================================================

def build_flags(base, masked, fmap, runway):
    labels = load_labels_a()
    masked = masked.copy()
    masked['class'] = masked['item_id'].map(lambda i: labels.get(i, (np.nan, np.nan))[0])
    masked['materiality'] = masked['item_id'].map(lambda i: labels.get(i, (np.nan, np.nan))[1])

    fmap = fmap.merge(masked[['item_id', 'class', 'materiality']], on='item_id', how='left')
    fmap = fmap.dropna(subset=['class'])
    fmap['rank_key'] = list(zip(-fmap['materiality'], fmap['created_at']))
    best = (fmap.sort_values(['day', 'symbol', 'materiality', 'created_at'],
                              ascending=[True, True, False, False])
                .drop_duplicates(subset=['day', 'symbol'], keep='first')
                [['day', 'symbol', 'class', 'materiality']])

    out = base.merge(best, on=['day', 'symbol'], how='left')
    out['class'] = out['class'].fillna('no_news')
    out['hard_catalyst'] = out['class'].isin(HARD_CLASSES) & (out['materiality'] >= 2)
    out = out.merge(runway[['day', 'symbol', 'runway_q', 'cf_positive', 'coverage']],
                     on=['day', 'symbol'], how='left', suffixes=('', '_rw'))
    out = out.rename(columns={'coverage': 'runway_coverage'})
    out['runway_ok'] = out['cf_positive'].fillna(False) | (out['runway_q'] >= 4)
    out['runway_bad'] = (~out['cf_positive'].fillna(False)) & (out['runway_q'] < 2)

    cols = ['day', 'symbol', 'split', 'fill_min', 'exit_m', 'outcome_R', 'L1', 'L2',
            'store_served_1438', 'class', 'materiality', 'hard_catalyst', 'runway_q',
            'cf_positive', 'runway_coverage', 'runway_ok', 'runway_bad']
    out[cols].to_csv(os.path.join(OUT_DIR, 'cell_1483_fills.csv'), index=False)
    log(f'build_flags: class coverage (non no_news, non-error) = '
        f'{(out["class"] != "no_news").mean():.3f}; hard_catalyst rate = {out.hard_catalyst.mean():.3f}; '
        f'runway coverage = {out.runway_coverage.mean():.3f}')
    return out


# ================================================================================================
# Step 9: scoring
# ================================================================================================

def score_mask(name, holdout_df, keep_mask, weeks, coverage_mask):
    """One cell x holdout row: kept = keep_mask & coverage_mask (uncovered rows excluded from
    BOTH kept and dropped, tracked separately in `coverage`)."""
    covered = holdout_df[coverage_mask]
    kept = covered[keep_mask[coverage_mask.index][coverage_mask]]
    dropped = covered[~keep_mask[coverage_mask.index][coverage_mask]]
    kept_R, dropped_R = kept['outcome_R'], dropped['outcome_R']
    t = c1445.day_clustered_t(kept_R, kept['day']) if len(kept_R) else float('nan')
    ex5 = c1445.ex_top5_mean(kept_R) if len(kept_R) else float('nan')
    slot_in = kept.rename(columns={'fill_min': 'entry_m'})[['day', 'entry_m', 'exit_m']].dropna()
    fwk = c1445.fills_per_week(slot_in, weeks) if len(slot_in) else 0.0
    return dict(
        name=name, n_covered=len(covered), n_total=len(holdout_df),
        coverage=len(covered) / len(holdout_df) if len(holdout_df) else float('nan'),
        n_kept=int(len(kept)), n_dropped=int(len(dropped)),
        kept_mean=float(kept_R.mean()) if len(kept_R) else float('nan'),
        dropped_mean=float(dropped_R.mean()) if len(dropped_R) else float('nan'),
        t_kept=float(t) if t is not None else float('nan'),
        ex_top5=float(ex5) if ex5 is not None else float('nan'),
        fills_wk=float(fwk),
        cacheonly_share=float(kept['store_served_1438'].mean()) if len(kept) else float('nan'),
    )


def null_pctile(holdout_df, sc):
    if sc['n_kept'] == 0 or np.isnan(sc['kept_mean']):
        return float('nan')
    return c1445.null_percentile_of(holdout_df['outcome_R'], sc['n_kept'], sc['kept_mean'],
                                     seed=NULL_SEED, n_draws=NULL_DRAWS)


def score_cells(flags, robust_agreement):
    weeks = dict(TRAIN=c1445.weeks_spanned(flags.loc[flags.split == 'TRAIN', 'day']),
                 VAL=c1445.weeks_spanned(flags.loc[flags.split == 'VAL', 'day']))
    catalyst_void = robust_agreement < ROBUST_AGREE_MIN if not np.isnan(robust_agreement) else True
    if catalyst_void:
        log(f'WARNING score_cells: classification robustness {robust_agreement} < '
            f'{ROBUST_AGREE_MIN} -- cells 1,483 and 1,485 are VOID per PREREG')

    cell_defs = [
        ('1483_CATALYST', lambda d: d['hard_catalyst'], lambda d: d['class'] != 'classify_error',
         catalyst_void),
        ('1484_RUNWAY', lambda d: d['runway_ok'], lambda d: d['runway_coverage'].fillna(False),
         False),
        ('1485_JOINT', lambda d: d['hard_catalyst'] & d['runway_ok'],
         lambda d: (d['class'] != 'classify_error') & d['runway_coverage'].fillna(False),
         catalyst_void),
    ]
    rows = []
    for cell_id, keep_fn, cov_fn, void_flag in cell_defs:
        for holdout in ('TRAIN-H2', 'VAL'):
            split_name = 'TRAIN' if holdout == 'TRAIN-H2' else 'VAL'
            d = flags[flags.split == split_name].reset_index(drop=True)
            keep_mask = keep_fn(d)
            cov_mask = cov_fn(d)
            sc = score_mask(cell_id, d, keep_mask, weeks[split_name], cov_mask)
            sc['holdout'] = holdout
            sc['null_pctile'] = null_pctile(d, sc)
            rows.append(sc)

    by_id = {}
    for r in rows:
        by_id.setdefault(r['name'], {})[r['holdout']] = r
    for cell_id, keep_fn, cov_fn, void_flag in cell_defs:
        val = by_id[cell_id]['VAL']
        trh2 = by_id[cell_id]['TRAIN-H2']
        same_sign = (np.sign(trh2['kept_mean']) == np.sign(val['kept_mean'])) if not (
            np.isnan(trh2['kept_mean']) or np.isnan(val['kept_mean'])) else False
        passes = (not void_flag and
                  val['coverage'] >= PASS_COVERAGE and
                  val['kept_mean'] >= PASS_KEPT_MEAN and val['t_kept'] >= PASS_T and
                  val['ex_top5'] > 0 and val['fills_wk'] >= PASS_FILLS_WK and
                  val['dropped_mean'] < val['kept_mean'] and
                  trh2['dropped_mean'] < trh2['kept_mean'] and
                  same_sign and abs(trh2['t_kept']) >= PASS_TRAINH2_T and
                  val['null_pctile'] >= PASS_NULL_PCTILE and
                  abs(val['cacheonly_share'] - CACHEONLY_REF) <= PASS_CACHEONLY_TOL)
        for r in rows:
            if r['name'] == cell_id:
                r['passes_bar'] = bool(passes) if r['holdout'] == 'VAL' else None
                r['void'] = void_flag
    return rows


# ================================================================================================
# Report-only tables
# ================================================================================================

def report_tables(flags):
    class_rows = []
    for (cls, holdout), g in flags.assign(
            holdout=flags.split.map({'TRAIN': 'TRAIN-H2', 'VAL': 'VAL'})).groupby(['class', 'holdout']):
        class_rows.append(dict(cls=cls, holdout=holdout, n=len(g), mean_net_R=float(g.outcome_R.mean()),
                                big_day_L1=float(g.L1.mean()), big_day_L2=float(g.L2.mean())))
    class_tbl = pd.DataFrame(class_rows).sort_values(['cls', 'holdout'])

    def bucket(row):
        if row['cf_positive']:
            return 'cf_positive'
        rq = row['runway_q']
        if pd.isna(rq):
            return 'NaN'
        if rq < 2:
            return '<2'
        if rq < 4:
            return '2-4'
        return '>=4'
    flags = flags.copy()
    flags['runway_bucket'] = flags.apply(bucket, axis=1)
    rw_rows = []
    for (buck, holdout), g in flags.assign(
            holdout=flags.split.map({'TRAIN': 'TRAIN-H2', 'VAL': 'VAL'})).groupby(['runway_bucket', 'holdout']):
        rw_rows.append(dict(bucket=buck, holdout=holdout, n=len(g), mean_net_R=float(g.outcome_R.mean()),
                             big_day_L1=float(g.L1.mean())))
    rw_tbl = pd.DataFrame(rw_rows).sort_values(['bucket', 'holdout'])

    # overnight close(day) -> next-session open return, by runway bucket
    cal_path = os.path.join(REPO, 'data/research/databento/equs_daily_2025_2026.parquet')
    daily = pd.read_parquet(cal_path, columns=['bar_date', 'symbol', 'instrument_id', 'open', 'close'])
    daily = daily.dropna(subset=['symbol']).sort_values(['symbol', 'bar_date'])
    daily['next_open'] = daily.groupby('symbol')['open'].shift(-1)
    daily['overnight_ret'] = daily['next_open'] / daily['close'] - 1
    onret = daily[['bar_date', 'symbol', 'overnight_ret']].rename(columns={'bar_date': 'day'})
    onret['day'] = onret['day'].astype(str)
    joined = flags.merge(onret, on=['day', 'symbol'], how='left')
    on_rows = []
    for buck, g in joined.groupby('runway_bucket'):
        on_rows.append(dict(bucket=buck, n=int(g['overnight_ret'].notna().sum()),
                             mean_overnight_ret=float(g['overnight_ret'].mean())))
    on_tbl = pd.DataFrame(on_rows).sort_values('bucket')
    return class_tbl, rw_tbl, on_tbl


# ================================================================================================
# main
# ================================================================================================

def fmt_row(r):
    return (f"| {r['name']} | {r['holdout']} | {r['n_kept']} | {r['kept_mean']:.3f} | "
            f"{r['t_kept']:.2f} | {r['ex_top5']:.3f} | {r['fills_wk']:.1f} | "
            f"{r['dropped_mean']:.3f} | {r['null_pctile']:.1f} | {r['coverage']*100:.1f}% | "
            f"{r['cacheonly_share']:.3f} | {r.get('passes_bar')} |")


def write_result(rows, class_tbl, rw_tbl, on_tbl, robust_agreement, coverage_news, coverage_rw):
    lines = ['# RESULT 1,483-1,485 -- catalyst attribution and cash runway',
             '',
             f'Classification robustness (500-item seed-1483 sample, second independent prompt): '
             f'agreement={robust_agreement:.3f} (bar {ROBUST_AGREE_MIN}) -> catalyst cells VOID = '
             f'{robust_agreement < ROBUST_AGREE_MIN}',
             f'News coverage (fills with a cached article stream): {coverage_news:.3f}; '
             f'XBRL runway coverage: {coverage_rw:.3f}', '',
             '## Pass-bar table (cell x holdout)',
             '| cell | holdout | n_kept | kept_R | t | ex_top5 | fills/wk | dropped_R | '
             'null_pctile | coverage | cache-only | PASS |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(fmt_row(r))
    lines += ['', '## Report-only: mean net R and big-day rate by news class x holdout',
              class_tbl.to_markdown(index=False), '',
              '## Report-only: mean net R and big-day rate by runway bucket x holdout',
              rw_tbl.to_markdown(index=False), '',
              '## Report-only: overnight close->next-open return by runway bucket',
              on_tbl.to_markdown(index=False), '']
    with open(os.path.join(OUT_DIR, 'RESULT_1483.md'), 'w') as f:
        f.write('\n'.join(lines))
    log('write_result: RESULT_1483.md written')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                     choices=['all', 'news', 'classify', 'xbrl', 'runway', 'flags', 'score'])
    args = ap.parse_args()

    base = load_base()
    master = symbol_master()
    symbols = sorted(base['symbol'].unique())
    day_min, day_max = base['day'].min(), base['day'].max()

    if args.stage in ('all', 'news'):
        fetch_news(symbols, day_min, day_max)
    if args.stage == 'news':
        return

    masked, fmap = build_candidates(base, master)
    if args.stage in ('all', 'classify'):
        classify_batches(masked)
        robust_agreement = robustness_check(masked)
    if args.stage == 'classify':
        return

    if args.stage in ('all', 'xbrl'):
        fetch_xbrl(symbols, master)
    if args.stage == 'xbrl':
        return

    if args.stage in ('all', 'runway'):
        runway = compute_runway(base, master)
    else:
        runway = pd.read_csv(os.path.join(OUT_DIR, 'runway_1484.csv'))
    if args.stage == 'runway':
        return

    flags = build_flags(base, masked, fmap, runway)
    if args.stage == 'flags':
        return

    robust_path = os.path.join(NEWS_DIR, 'robustness_500.json')
    if os.path.exists(robust_path):
        with open(robust_path) as f:
            robust_agreement = json.load(f)['agreement']
    else:
        robust_agreement = robustness_check(masked)

    rows = score_cells(flags, robust_agreement)
    class_tbl, rw_tbl, on_tbl = report_tables(flags)
    coverage_news = (flags['class'] != 'no_news').mean()
    coverage_rw = flags['runway_coverage'].fillna(False).mean()
    write_result(rows, class_tbl, rw_tbl, on_tbl, robust_agreement, coverage_news, coverage_rw)
    log('main: DONE')


if __name__ == '__main__':
    main()
