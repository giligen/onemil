"""Cell 1,428 — the E1/1,438 causal-arming rule on the GAPPER (ORB candidate) universe. LADDER.md row 1,428.

Population change vs cell 1,438 ONLY (arming/fill mechanism is identical, reused unmodified from causal_arming.py):
  * population = point-in-time ORB-candidate symbol-days: gap >= 5 % vs prior close at the 09:30 open, open in
    [$3, $30], prior-day (single day, not adv20) volume >= 500,000, 2025-07-01..2026-05-31 (TRAIN-H2/VAL split
    dates = causal_arming.SPLITS). Built from Databento EQUS.SUMMARY consolidated daily bars
    (data/research/databento/equs_daily_2025_2026.parquet, survivorship-free, delisted included) — never from the
    live daily_bars table. Test tickers ^Z[A-Z]ZZT$ excluded. adv20 (20-day causal rolling mean volume, shifted so
    day D never sees day D's own volume) is still computed and fed to rv_profile exactly as the live rule does —
    only the ADMISSION gate uses prior-day volume, per LADDER.md/PREREG.
  * floor = 3.0 (the pool's price floor, live_params() gives 20 for the $20+ book) and min_adv = 500,000
    (prior-day volume gate) override live_params(); every other HodBreakParams constant is untouched.
  * minute bars: fetched fresh SIP via the SAME fetcher research/bf_zero/backfill_bars_sip.py.fetch_day into a NEW
    sqlite db research/hod_entry/bars_sip_gapper.db (same `bars` table schema: symbol, day, t, o, h, l, c, v),
    resumable per day (skips symbol-days already in the db). A LOST count (requested vs received symbol-days) is
    logged; the run refuses to score if LOST > 5 % of the population.
  * arming, tick fetch (fetch_window), fill and B0 exit physics: causal_arming.{arm_state, armed_crossing_bars,
    resolve_day, process_symbol_day, fetch_window} reused UNCHANGED. Tick cache: research/hod_entry/sip_cache_gapper/
    c1428_{day}.pkl.gz (cell_1439.py's per-day cache pattern). No 1,427 cache exists for this population (c1427={}).
  * cost: measured half-spread at the fill + B0 exit leg, exactly as causal_arming (nbbo_half falls back to the
    fill-instant half-spread for symbol-days absent from research/bf_zero/causal_filter/nbbo.csv, which is most of
    this population — a different pool). 30 bps stop-slip variant via causal_arming.stop_slip_net.

Usage: nice -n 19 python3 research/hod_entry/cell_1428.py --stage population|fetch|run|report|all --workers 2
       --resume continues a partial fetch or run.
"""
import argparse
import gzip
import json
import logging
import os
import pickle
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'research/bf_zero'))
import sip_rebuild as sr           # noqa: E402
import causal_arming as ca         # noqa: E402
import backfill_bars_sip as bbs    # noqa: E402

log = logging.getLogger('cell_1428')

PARQUET = os.path.join(ROOT, 'data/research/databento/equs_daily_2025_2026.parquet')
POP_CSV = os.path.join(HERE, 'cell_1428_population.csv')
BARS_DB = os.path.join(HERE, 'bars_sip_gapper.db')
CACHE_DIR = os.path.join(HERE, 'sip_cache_gapper')
OUT_CSV = os.path.join(HERE, 'cell_1428_{variant}.csv')
RESULT_MD = os.path.join(HERE, 'RESULT_1428.md')
WEEKEND_RESULTS = os.path.join(HERE, 'WEEKEND_RESULTS.md')

TEST_TICKER = re.compile(r'^Z[A-Z]ZZT$')
SYMBOL_RE = re.compile(r'^[A-Z]{1,5}(\.[A-Z])?$')
SPLITS = ca.SPLITS                          # TRAIN 2025-07-01..2025-12-31, VAL 2026-01-01..2026-05-31
GAP_MIN, PRICE_LO, PRICE_HI, ADV_MIN = 0.05, 3.0, 30.0, 500_000
FLOOR = 3.0
FETCH_BATCH = 100
LOST_GATE = 0.05                            # refuse to score if > 5 % of requested symbol-days are unfetchable


# --------------------------------------------------------------------------------------------- population
def build_population():
    """Point-in-time gapper universe from the Databento EQUS.SUMMARY daily parquet: gap >= 5 %, open $3-30,
    prior-day volume >= 500K, 2025-07-01..2026-05-31, test tickers excluded. adv20 = causal 20-day rolling mean
    volume (shift(1) so day D excludes D's own volume) for the rv_profile input (NOT the admission gate)."""
    d = pd.read_parquet(PARQUET)
    d = d[d.symbol.notna() & d.symbol.str.match(SYMBOL_RE) & (d.volume > 0)].sort_values(['symbol', 'bar_date'])
    g = d.groupby('symbol')
    d['prev_close'] = g['close'].shift(1)
    d['prev_vol'] = g['volume'].shift(1)
    d['adv20'] = g['volume'].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    gap = (d.open - d.prev_close) / d.prev_close
    u = d[(gap >= GAP_MIN) & (d.open >= PRICE_LO) & (d.open <= PRICE_HI) & (d.prev_vol >= ADV_MIN)
          & (d.bar_date >= SPLITS['TRAIN'][0]) & (d.bar_date <= SPLITS['VAL'][1])
          & (d.high >= FLOOR + sr.TICK) & d.adv20.notna()]
    u = u[~u.symbol.str.match(TEST_TICKER)].drop_duplicates(['symbol', 'bar_date']).rename(columns={'bar_date': 'day'})
    u['split'] = np.where(u.day <= SPLITS['TRAIN'][1], 'TRAIN', 'VAL')
    u['wk'] = pd.to_datetime(u.day).dt.to_period('W-FRI').astype(str)
    u = u[['day', 'symbol', 'adv20', 'split', 'wk']].reset_index(drop=True)
    log.info('[pop] %d symbol-days (TRAIN-H2 %d, VAL %d), %d symbols, %s..%s', len(u),
             int((u.split == 'TRAIN').sum()), int((u.split == 'VAL').sum()), u.symbol.nunique(), u.day.min(), u.day.max())
    u.to_csv(POP_CSV, index=False)
    return u


# --------------------------------------------------------------------------------------------- fetch
def fetch_bars(pop, limit_days=0):
    """SIP 1-min bars for `pop`'s symbol-days into BARS_DB (bbs.fetch_day, same schema as bars_sip.db). Resumable:
    skips (symbol, day) pairs already present. Logs and returns (requested, lost) symbol-day counts."""
    con = sqlite3.connect(BARS_DB)
    con.execute('CREATE TABLE IF NOT EXISTS bars (symbol TEXT, day TEXT, t TEXT, o REAL, h REAL, l REAL, c REAL, v REAL)')
    con.execute('CREATE INDEX IF NOT EXISTS ix_bars_day_sym ON bars(day, symbol)')
    con.commit()
    have = pd.read_sql('select distinct symbol, day from bars', con)
    have_by_day = have.groupby('day').symbol.apply(set).to_dict()
    days = sorted(pop.day.unique())
    if limit_days:
        days = days[:limit_days]
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, '.env'))
    from config import Config
    from alpaca.data.historical import StockHistoricalDataClient
    cfg = Config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_api_secret)
    requested = 0
    for di, day in enumerate(days):
        syms = sorted(set(pop[pop.day == day].symbol) - have_by_day.get(day, set()))
        requested += len(pop[pop.day == day].symbol.unique())
        if not syms:
            continue
        for j in range(0, len(syms), FETCH_BATCH):
            chunk = syms[j:j + FETCH_BATCH]
            for attempt in range(3):
                try:
                    df = bbs.fetch_day(client, chunk, day)
                    if len(df):
                        df.to_sql('bars', con, if_exists='append', index=False)
                    break
                except Exception as e:  # noqa: BLE001
                    log.warning('[fetch] day %s chunk %d attempt %d: %s', day, j // FETCH_BATCH, attempt + 1, e)
                    time.sleep(2 * (attempt + 1))
        if (di + 1) % 20 == 0 or di == len(days) - 1:
            con.commit()
            log.info('[fetch] day %d/%d (%s)', di + 1, len(days), day)
    con.commit()
    have2 = pd.read_sql('select distinct symbol, day from bars', con)
    con.close()
    have_set = set(map(tuple, have2.values))
    pop_pairs = set(zip(pop.symbol, pop.day))
    lost = len(pop_pairs - have_set)
    log.info('[fetch] requested %d symbol-days, %d LOST (%.1f %%)', len(pop_pairs), lost, 100 * lost / len(pop_pairs))
    return len(pop_pairs), lost


# --------------------------------------------------------------------------------------------- run
def run(pop, workers):
    """Scan + arm + fill, reusing causal_arming's arming/fill/exit code unchanged with FLOOR=3.0. Resumable
    per-day tick cache under sip_cache_gapper/c1428_{day}.pkl.gz."""
    p, _, _ = ca.live_params()
    log.info('[params] %s | floor %.2f | min_adv %d (pool overrides)', p, FLOOR, ADV_MIN)
    nb = pd.read_csv(ca.NBBO_CSV, dtype={'symbol': str, 'day': str}, keep_default_na=False, na_values=[''])
    nbbo_half = (0.5 * nb.drop_duplicates(['day', 'symbol']).set_index(['day', 'symbol']).spread_mean).to_dict()
    con = sqlite3.connect(sr.CACHE_DB_URI, uri=True, timeout=120)
    sipcon = sqlite3.connect('file:' + BARS_DB + '?mode=ro', uri=True, timeout=120)
    rows, n_nobars, src = [], 0, {}
    days = sorted(pop.day.unique())
    for di, day in enumerate(days):
        sub = pop[pop.day == day]
        bars = ca.load_day_bars(con, day, sub.symbol.tolist(), sipcon, src)
        path = os.path.join(CACHE_DIR, f'c1428_{day}.pkl.gz')
        cache = {}
        if os.path.exists(path):
            with gzip.open(path, 'rb') as f:
                cache = pickle.load(f)
        jobs = [(r, bars[r.symbol]) for r in sub.itertuples()
                if r.symbol in bars and len(bars[r.symbol]) >= p.consol_bars + 2]
        n_nobars += len(sub) - len(jobs)
        new_all = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(ca.process_symbol_day, r, b, p, FLOOR, nbbo_half, cache, {}) for r, b in jobs]
            for fu in futs:
                try:
                    rr, new = fu.result()
                    rows += rr
                    new_all.update(new)
                except Exception as e:  # noqa: BLE001
                    log.error('[run] ERROR %s: %s: %s — symbol-day LOST (re-run to resume)', day, type(e).__name__, e)
        if new_all:
            cache.update(new_all)
            tmp = path + '.tmp'
            with gzip.open(tmp, 'wb') as f:
                pickle.dump(cache, f)
            os.replace(tmp, path)
        if (di + 1) % 10 == 0 or di == len(days) - 1:
            d = pd.DataFrame(rows)
            nf = int(((d.status == 'fill') & (d.variant == 'causal')).sum()) if len(d) else 0
            log.info('[run] day %d/%d (%s) | causal fills so far %d | new windows %d', di + 1, len(days), day, nf,
                      len(new_all))
    con.close()
    sipcon.close()
    log.info('[run] minute-bar source: %s; %d superset symbol-days unsimulable (< K+2 bars)', src, n_nobars)
    res = pd.DataFrame(rows)
    for v in ('causal', 'tick_rv'):
        res[res.variant == v].to_csv(OUT_CSV.format(variant=v), index=False)
    return res, n_nobars


# --------------------------------------------------------------------------------------------- score / report
def _legs(v):
    """The 1,428 pass-bar legs (LADDER.md row 1,428 + the frozen VAL bar): mean >= +0.15 (both holdouts), VAL
    t >= 2, ex-top-5 % > 0, coverage >= 80 %, gap <= 5 pp, >= 3 fills/week."""
    return [('mean >= +0.15', v['mean_R'] >= 0.15, f'{v["mean_R"]:+.3f}'), ('t >= 2', v['t'] >= 2, f'{v["t"]:.2f}'),
            ('ex-top-5 % > 0', v['extop5'] > 0, f'{v["extop5"]:+.3f}'),
            ('coverage >= 80 %', v['coverage'] >= 0.8, f'{v["coverage"]*100:.1f} %'),
            ('gap <= 5 pp', v['miss_gap'] <= 0.05, f'{v["miss_gap"]*100:.1f} pp'),
            ('>= 3 fills/wk', v['fills_wk'] >= 3, f'{v["fills_wk"]:.1f}')]


def report(res, n_nobars, pop_n, lost):
    """RESULT_1428.md: one table (both holdouts) + <= 5 line verdict, and appends the same under a
    '## 1,428 — gapper universe' heading to WEEKEND_RESULTS.md."""
    sc, extra = {}, {}
    for sp in ('TRAIN', 'VAL'):
        allv = res[(res.variant == 'causal') & (res.split == sp)]
        f = allv[allv.status == 'fill']
        sc[sp] = sr.score(allv[allv.status != 'not_armed'].copy(), allv)
        spread_bps = (f.cost_R * f.R - sr.SLIP_BP * f.exit_price) / f.fill * 10000  # approx full round-trip spread
        extra[sp] = dict(stop_slip=ca.stop_slip_net(f).mean() if len(f) else np.nan,
                          r_pct=(f.R / f.fill * 100).median() if len(f) else np.nan,
                          spread_bps=spread_bps.median() if len(f) else np.nan)
    lines = ['# RESULT — cell 1,428: gapper-universe causal arming (LADDER.md row 1,428)', '',
             f'Population: {pop_n} symbol-days requested, {lost} LOST ({100*lost/pop_n:.1f} %), '
             f'{n_nobars} unsimulable (< K+2 minute bars).', '',
             '| holdout | fills (rate) | mean net R | stop-slip R | day-clust t | ex-top-5 % | fills/wk | '
             'coverage / gap | median spread bps | R % of price | verdict |', '|' + '---|' * 11]
    for sp, cn in (('TRAIN', 'TRAIN-H2'), ('VAL', 'VAL')):
        v, e = sc[sp], extra[sp]
        legs = _legs(v)
        verdict = ('PASS' if all(x[1] for x in legs) else 'FAIL: ' + ', '.join(n for n, g, _ in legs if not g)) \
            if sp == 'VAL' else 'report-only (TRAIN-H2)'
        lines.append(f'| {cn} | {v["fills"]} ({v["fill_rate"]*100:.1f} %) | {v["mean_R"]:+.3f} | {e["stop_slip"]:+.3f} '
                     f'| {v["t"]:.2f} | {v["extop5"]:+.3f} | {v["fills_wk"]:.1f} | {v["coverage"]*100:.1f} % / '
                     f'{v["miss_gap"]*100:.1f} pp | {e["spread_bps"]:.0f} | {e["r_pct"]:.2f} % | {verdict} |')
    val_legs = _legs(sc['VAL'])
    verdict_line = 'PASS' if all(x[1] for x in val_legs) else 'FAIL: ' + ', '.join(n for n, g, _ in val_legs if not g)
    caveats = [
        '', f'* VAL verdict: {verdict_line}.',
        '* Cost: measured half-spread at fill + B0 exit leg (causal_arming); nbbo.csv is the $20+ book\'s spread '
        'table and rarely covers this pool, so exit cost falls back to the fill-instant half-spread for most fills.',
        '* Spread bps is derived from cost_R * R (approx, assumes entry and exit half-spreads are close) — not an '
        'independent NBBO measurement for this pool; read as indicative.',
        '* Small caps: R must exceed the spread — see the R %-of-price and spread-bps columns above per CLAUDE.md.',
        f'* Population: point-in-time Databento EQUS.SUMMARY daily bars, gap >= 5 %, open $3-30, prior-day volume '
        f'>= 500K, {SPLITS["TRAIN"][0]}..{SPLITS["VAL"][1]}, test tickers excluded; TEST not read.']
    with open(RESULT_MD, 'w') as fh:
        fh.write('\n'.join(lines + caveats))
    with open(WEEKEND_RESULTS, 'a') as fh:
        fh.write('\n\n## 1,428 — gapper universe\n\n' + '\n'.join(lines) + '\n' + '\n'.join(caveats[1:]) + '\n')
    log.info('[report] VAL %s', '; '.join(f'{n}={x}' for n, _, x in val_legs))
    return sc


# --------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    """CLI: --stage population|fetch|run|report|all (default all), --workers N (<=2 on this node), --resume,
    --limit-days N (fetch stage debug)."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stage', default='all', choices=['population', 'fetch', 'run', 'report', 'all'])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--limit-days', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    os.makedirs(CACHE_DIR, exist_ok=True)
    pop = pd.read_csv(POP_CSV, dtype={'symbol': str, 'day': str}) if (a.resume and os.path.exists(POP_CSV)) \
        else build_population()
    if a.stage == 'population':
        return 0
    if a.stage in ('fetch', 'all'):
        requested, lost = fetch_bars(pop, a.limit_days)
        if lost / requested > LOST_GATE:
            log.error('[fetch] ERROR: LOST %d/%d (%.1f %%) exceeds the %.0f %% gate — refusing to score', lost,
                       requested, 100 * lost / requested, 100 * LOST_GATE)
            return 1
        state = dict(requested=requested, lost=lost)
        with open(os.path.join(HERE, 'cell_1428_fetch_state.json'), 'w') as fh:
            json.dump(state, fh)
    if a.stage == 'fetch':
        return 0
    with open(os.path.join(HERE, 'cell_1428_fetch_state.json')) as fh:
        state = json.load(fh)
    if a.stage in ('run', 'all'):
        res, n_nobars = run(pop, a.workers)
        res.to_pickle(os.path.join(HERE, 'cell_1428_res.pkl'))
        with open(os.path.join(HERE, 'cell_1428_run_state.json'), 'w') as fh:
            json.dump({'n_nobars': n_nobars}, fh)
    if a.stage == 'run':
        return 0
    res = pd.read_pickle(os.path.join(HERE, 'cell_1428_res.pkl'))
    with open(os.path.join(HERE, 'cell_1428_run_state.json')) as fh:
        n_nobars = json.load(fh)['n_nobars']
    report(res, n_nobars, state['requested'], state['lost'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
