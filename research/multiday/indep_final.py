#!/usr/bin/env python3
"""INDEPENDENT reimplementation of cells F5-LO, A4-LO and the ex-date regime
reconciliation, written from the prose specification only.

Outputs -> research/multiday/out_indep_final/
"""
import os, sys, json, gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = '/home/ec2-user/onemil/research/multiday/data'
OUT = '/home/ec2-user/onemil/research/multiday/out_indep_final'
os.makedirs(OUT, exist_ok=True)

BUDGET = 66_000.0
SLOTS = 20
TAF_BPS = 0.4
IMPACT_COEF = 10.0          # bps per 1.0 of participation ratio, one way
LOAD_YEARS = range(2016, 2024)   # TEST (2024+) is SEALED -> never loaded
SEAL = pd.Timestamp('2024-01-01')

SPLITS = {
    'TRAIN': (pd.Timestamp('2016-01-01'), pd.Timestamp('2021-12-31')),
    'VAL':   (pd.Timestamp('2022-01-01'), pd.Timestamp('2023-12-31')),
}

def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- panel build
def build_panel():
    uni = pd.read_parquet(os.path.join(BASE, 'universe.parquet'),
                          columns=['symbol', 'kind'])
    common = set(uni.loc[uni['kind'] == 'common', 'symbol'])
    log(f'universe common symbols: {len(common)}')

    frames_adj, frames_raw = [], []
    for y in LOAD_YEARS:
        for kind, store in (('all', frames_adj), ('raw', frames_raw)):
            cols = ['symbol', 'date', 'close'] if kind == 'all' else ['symbol', 'date', 'close', 'volume', 'vwap']
            t = pq.read_table(os.path.join(BASE, f'prices_by_year/{kind}/year={y}.parquet'),
                              columns=cols)
            df = t.to_pandas()
            del t
            df['symbol'] = df['symbol'].astype(str)
            df = df[df['symbol'].isin(common) | (df['symbol'] == 'SPY')]
            store.append(df)
            log(f'  loaded {kind} {y}: {len(df):,} rows')
            gc.collect()
    adj = pd.concat(frames_adj, ignore_index=True); del frames_adj
    raw = pd.concat(frames_raw, ignore_index=True); del frames_raw
    gc.collect()
    adj['date'] = pd.to_datetime(adj['date'])
    raw['date'] = pd.to_datetime(raw['date'])

    # SESSION CALENDAR = distinct dates on SPY in the ADJUSTED files
    sessions = np.sort(adj.loc[adj['symbol'] == 'SPY', 'date'].unique())
    sessions = pd.DatetimeIndex(sessions)
    log(f'sessions {len(sessions)}: {sessions[0].date()} .. {sessions[-1].date()}')

    symbols = np.array(sorted(common & set(adj['symbol'].unique())))
    log(f'panel symbols (common & priced): {len(symbols)}')

    sidx = pd.Series(np.arange(len(symbols)), index=symbols)
    didx = pd.Series(np.arange(len(sessions)), index=sessions)
    S, T = len(symbols), len(sessions)

    def to_mat(df, col, dtype=np.float32):
        m = np.full((S, T), np.nan, dtype=dtype)
        si = df['symbol'].map(sidx)
        di = df['date'].map(didx)
        ok = si.notna() & di.notna()
        m[si[ok].to_numpy(np.int32), di[ok].to_numpy(np.int32)] = df.loc[ok, col].to_numpy()
        return m

    # duplicate (symbol,date) check
    dup = adj.duplicated(subset=['symbol', 'date']).sum()
    dup_r = raw.duplicated(subset=['symbol', 'date']).sum()
    log(f'duplicate (symbol,date) rows: adjusted={dup} raw={dup_r}')

    adj_close = to_mat(adj, 'close')
    raw_close = to_mat(raw, 'close')
    raw['dvol'] = raw['vwap'].astype(np.float64) * raw['volume'].astype(np.float64)
    dvol = to_mat(raw, 'dvol', np.float64)
    del adj, raw
    gc.collect()
    return dict(symbols=symbols, sessions=sessions, adj=adj_close, rawc=raw_close,
                dvol=dvol, dup=(int(dup), int(dup_r)))


def adv20_at(dvol, t):
    """mean of raw dollar volume over the 20 sessions ending at and incl. t,
    >=10 non-missing observations required."""
    lo = max(0, t - 19)
    w = dvol[:, lo:t + 1]
    cnt = np.isfinite(w).sum(axis=1)
    s = np.nansum(w, axis=1)
    out = np.where(cnt >= 10, s / np.maximum(cnt, 1), np.nan)
    return out


# ------------------------------------------------------------------- plumbing
def month_end_anchors(sessions):
    s = pd.Series(np.arange(len(sessions)), index=sessions)
    return s.groupby([sessions.year, sessions.month]).max().to_numpy()


def decile_ranks(values):
    """decile 1 = lowest .. 10 = highest, equal-count, stable sort."""
    n = len(values)
    order = np.argsort(values, kind='stable')
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    return (pos * 10) // n + 1


class Trades:
    def __init__(self):
        self.sym, self.ent, self.exi, self.rank = [], [], [], []

    def add(self, sym, ent, exi, rank=0.0):
        self.sym.append(sym); self.ent.append(ent); self.exi.append(exi)
        self.rank.append(rank)

    def arrays(self):
        return (np.asarray(self.sym, np.int32), np.asarray(self.ent, np.int32),
                np.asarray(self.exi, np.int32), np.asarray(self.rank, np.float64))


def positions_held(ent, exi, T):
    """count of trades with entry_day <= d <= exit_day, for every session d."""
    c = np.zeros(T + 2, dtype=np.int64)
    np.add.at(c, ent, 1)
    np.add.at(c, np.minimum(exi + 1, T + 1), -1)
    return np.cumsum(c)[:T]


def trade_costs(ent, adv_at_entry, held_on_entry_day):
    order_usd = BUDGET / np.maximum(held_on_entry_day, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        part = order_usd / (0.01 * adv_at_entry)
    part = np.where(np.isfinite(part), part, 1.0)
    return 2.0 * IMPACT_COEF * np.minimum(part, 1.0) + TAF_BPS


def daily_series(ret, sym, ent, exi, cost_bps, T):
    """equal-weighted overlapping daily series; cost split half on entry_day+1,
    half on exit_day. Positions with a non-finite return that day are skipped."""
    num = np.zeros(T); den = np.zeros(T)
    for i in range(len(sym)):
        a, b = ent[i] + 1, exi[i]
        if b < a:
            continue
        r = ret[sym[i], a:b + 1].astype(np.float64).copy()
        ok = np.isfinite(r)
        if cost_bps is not None and cost_bps[i] != 0.0:
            c = cost_bps[i] / 1e4
            r[0] -= c / 2.0
            r[-1] -= c / 2.0
        r[~ok] = 0.0
        num[a:b + 1] += r
        den[a:b + 1] += ok
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return out


def gross_returns(ret, sym, ent, exi):
    out = np.empty(len(sym))
    for i in range(len(sym)):
        a, b = ent[i] + 1, exi[i]
        if b < a:
            out[i] = 0.0; continue
        r = ret[sym[i], a:b + 1].astype(np.float64)
        r = r[np.isfinite(r)]
        out[i] = np.prod(1.0 + r) - 1.0
    return out


def monthly(daily, sessions, lo, hi):
    m = (sessions >= lo) & (sessions <= hi)
    s = pd.Series(daily[m], index=sessions[m])
    mo = (1.0 + s).resample('ME').prod() - 1.0
    return mo


def stats(excess):
    e = excess.to_numpy()
    n = len(e)
    if n < 2:
        return dict(n_months=n, bps=np.nan, t=np.nan, pct_pos=np.nan, mde=np.nan)
    mu, sd = e.mean(), e.std(ddof=1)
    se = sd / np.sqrt(n)
    jan = excess.index.month == 1
    return dict(n_months=n, bps=mu * 1e4, t=(mu / se if se > 0 else np.nan),
                pct_pos=float((e > 0).mean() * 100),
                ex_jan_bps=(e[~jan].mean() * 1e4 if (~jan).any() else np.nan),
                jan_bps=(e[jan].mean() * 1e4 if jan.any() else np.nan),
                mde_bps=2 * se * 1e4)


# ------------------------------------------------------------- cell machinery
def run_cell(name, panel, book_t, bench_t, hold_desc, notes):
    ret = panel['ret']; sessions = panel['sessions']; T = len(sessions)
    bs, be, bx, brank = book_t.arrays()
    ns, ne, nx, _ = bench_t.arrays()
    log(f'[{name}] book trades={len(bs):,} bench trades={len(ns):,}')

    held_book = positions_held(be, bx, T)
    adv_e = panel['adv_lookup'](bs, be)
    cost = trade_costs(be, adv_e, held_book[be])

    gross = gross_returns(ret, bs, be, bx)

    # ---- executable 20-slot book
    order = np.lexsort((-brank, be))          # first-come by entry session, best signal first
    occupied_until = np.full(SLOTS, -1)
    take = []
    for i in order:
        for s in range(SLOTS):
            if occupied_until[s] < be[i]:      # reusable on exit_day
                occupied_until[s] = bx[i]
                take.append(i)
                break
    take = np.array(sorted(take), dtype=np.int64)
    xs, xe, xx = bs[take], be[take], bx[take]
    held_x = positions_held(xe, xx, T)
    cost_x = trade_costs(xe, panel['adv_lookup'](xs, xe), held_x[xe])
    exec_daily = daily_series(ret, xs, xe, xx, cost_x, T)

    rows, mo_out = [], {}
    for split, (lo, hi) in SPLITS.items():
        in_split = (sessions[be] >= lo) & (sessions[bx] <= hi) & (sessions[be] >= lo) & (sessions[be] <= hi)
        nb = (sessions[ne] >= lo) & (sessions[nx] <= hi) & (sessions[ne] <= hi)
        xsel = (sessions[xe] >= lo) & (sessions[xx] <= hi) & (sessions[xe] <= hi)

        bd = daily_series(ret, bs[in_split], be[in_split], bx[in_split], cost[in_split], T)
        nd = daily_series(ret, ns[nb], ne[nb], nx[nb], None, T)
        mb, mn = monthly(bd, sessions, lo, hi), monthly(nd, sessions, lo, hi)
        keep = (mb != 0.0) | (mn != 0.0)
        exc = (mb - mn)[keep]

        r = dict(cell=name, split=split, n_trades=int(in_split.sum()),
                 n_bench_trades=int(nb.sum()), mean_cost_bps=float(cost[in_split].mean()) if in_split.any() else np.nan)
        r.update(stats(exc))

        # tail drops (book only; benchmark unchanged)
        g = gross[in_split]
        idx = np.where(in_split)[0]
        for pct, tag in ((1, 'ex1'), (5, 'ex5')):
            if len(g) == 0:
                r[f'{tag}_bps'] = np.nan; r[f'{tag}_t'] = np.nan; continue
            k = int(np.ceil(len(g) * pct / 100.0))
            drop = idx[np.argsort(-g, kind='stable')[:k]]
            sel = in_split.copy(); sel[drop] = False
            bd2 = daily_series(ret, bs[sel], be[sel], bx[sel], cost[sel], T)
            mb2 = monthly(bd2, sessions, lo, hi)
            e2 = (mb2 - mn)[keep]
            st2 = stats(e2)
            r[f'{tag}_bps'] = st2['bps']; r[f'{tag}_t'] = st2['t']; r[f'{tag}_ndrop'] = k

        # executable book
        xd = daily_series(ret, xs[xsel], xe[xsel], xx[xsel], cost_x[xsel], T)
        mx = monthly(xd, sessions, lo, hi)
        dollars = (mx - mn)[keep] * BUDGET
        n = len(dollars)
        weeks = max((hi - lo).days / 7.0, 1)
        r['exec_trades'] = int(xsel.sum())
        r['exec_trades_per_week'] = xsel.sum() / weeks
        r['exec_dollar_alpha'] = float(dollars.mean()) if n else np.nan
        r['exec_t'] = float(dollars.mean() / (dollars.std(ddof=1) / np.sqrt(n))) if n > 1 and dollars.std(ddof=1) > 0 else np.nan
        r['exec_mde_dollars'] = float(2 * dollars.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        # ---- diagnostics for the critique
        r['mean_impact_bps'] = float((cost[in_split] - TAF_BPS).mean()) if in_split.any() else np.nan
        r['exec_mean_cost_bps'] = float(cost_x[xsel].mean()) if xsel.any() else np.nan
        r['exec_mean_impact_bps'] = float((cost_x[xsel] - TAF_BPS).mean()) if xsel.any() else np.nan
        adjm = panel['adj']
        r['dark_trades'] = int(np.sum(~np.isfinite(adjm[bs[in_split], bx[in_split]])))
        # alternative excess: daily difference then compounded (vs monthly-minus-monthly)
        alt = monthly(bd - nd, sessions, lo, hi)[keep]
        r['excess_alt_bps'] = float(alt.mean() * 1e4)
        r['excess_alt_t'] = float(alt.mean() / (alt.std(ddof=1) / np.sqrt(len(alt)))) if len(alt) > 1 else np.nan
        r['mean_open_positions'] = float(held_book[be[in_split]].mean()) if in_split.any() else np.nan
        rows.append(r)
        mo_out[split] = pd.DataFrame({'book_net': mb[keep], 'bench_gross': mn[keep],
                                      'excess': exc, 'exec_net': mx[keep],
                                      'exec_dollar': dollars})
    for split, df in mo_out.items():
        df.to_csv(os.path.join(OUT, f'monthly_{name}_{split}.csv'))
    notes[name] = dict(hold=hold_desc, book_trades=int(len(bs)), bench_trades=int(len(ns)),
                       exec_trades_total=int(len(take)))
    return rows


# -------------------------------------------------------------------- cell 1
def cell_f5(panel):
    sessions = panel['sessions']; adj = panel['adj']; rawc = panel['rawc']
    dvol = panel['dvol']; T = len(sessions)
    anchors = month_end_anchors(sessions)
    book, bench = Trades(), Trades()
    elig_counts = []
    adv_cache = {}
    used = 0
    for k, t in enumerate(anchors):
        if t < 252 or k + 6 >= len(anchors):
            continue
        ent, exi = t + 1, anchors[k + 6] + 1
        if exi >= T:
            continue
        if sessions[ent] >= SEAL:            # TEST sealed
            continue
        w = adj[:, t - 251:t + 1]
        cnt = np.isfinite(w).sum(axis=1)
        with np.errstate(invalid='ignore'):
            mx = np.nanmax(np.where(np.isfinite(w), w, -np.inf), axis=1)
        mx = np.where(cnt >= 200, mx, np.nan)
        near = adj[:, t] / mx
        adv = adv20_at(dvol, t)
        elig = (np.isfinite(adj[:, t]) & (rawc[:, t] >= 5.0) & (adv >= 1e6)
                & np.isfinite(near) & (mx > 0))
        ids = np.where(elig)[0]
        if len(ids) < 10:
            continue
        used += 1
        elig_counts.append(len(ids))
        d = decile_ranks(near[ids])
        for i, sym in enumerate(ids):
            if d[i] == 10:
                book.add(sym, ent, exi, near[sym])
            else:
                bench.add(sym, ent, exi, near[sym])
        adv_cache[ent] = adv20_at(dvol, ent)
    panel['_advc'] = adv_cache
    return book, bench, used, float(np.mean(elig_counts)) if elig_counts else np.nan


# -------------------------------------------------------------------- cell 2
def cell_a4(panel):
    sessions = panel['sessions']; adj = panel['adj']; rawc = panel['rawc']
    dvol = panel['dvol']; symbols = panel['symbols']; T = len(sessions)
    anchors = month_end_anchors(sessions)

    d = pd.read_parquet(os.path.join(BASE, 'dividends.parquet'),
                        columns=['symbol', 'ex_date'])
    d = d[d['ex_date'].notna() & (d['ex_date'] <= pd.Timestamp('2023-12-31'))]
    sidx = pd.Series(np.arange(len(symbols)), index=symbols)
    d['si'] = d['symbol'].map(sidx)
    d = d[d['si'].notna()]
    d['si'] = d['si'].astype(int)
    d['mi'] = d['ex_date'].dt.year * 12 + (d['ex_date'].dt.month - 1)
    first_ex = d.groupby(['si', 'mi'])['ex_date'].min()
    M0 = int(d['mi'].min()); M1 = int(d['mi'].max())
    S = len(symbols); NM = M1 - M0 + 1
    exmat = np.full((S, NM), np.datetime64('NaT'), dtype='datetime64[ns]')
    ii = first_ex.index.get_level_values(0).to_numpy()
    jj = first_ex.index.get_level_values(1).to_numpy() - M0
    exmat[ii, jj] = first_ex.to_numpy()

    book, bench = Trades(), Trades()
    elig_counts, pred_counts, bench_counts = [], [], []
    adv_cache = {}
    used = 0
    skipped = 0
    for k, t in enumerate(anchors):
        if k + 1 >= len(anchors):
            continue
        ent, exi = t + 1, anchors[k + 1] + 1
        if exi >= T or sessions[ent] >= SEAL:
            continue
        a_date = np.datetime64(sessions[t])
        m_anchor = sessions[t].year * 12 + (sessions[t].month - 1)
        M = m_anchor + 1                      # month being predicted
        lo, hi = M - 24, M - 1                # 24 calendar months ending with M-1
        cols = [m - M0 for m in range(lo, hi + 1) if 0 <= m - M0 < NM]
        if not cols:
            continue
        sub = exmat[:, cols]
        valid = (~np.isnat(sub)) & (sub < a_date)
        nmonths = valid.sum(axis=1)
        history = nmonths >= 2
        c12 = M - 12 - M0
        if not (0 <= c12 < NM):
            continue
        col = exmat[:, c12]
        predicted = history & (~np.isnat(col)) & (col < a_date)

        adv = adv20_at(dvol, t)
        elig = (np.isfinite(adj[:, t]) & (rawc[:, t] >= 5.0) & (adv >= 1e6))
        bids = np.where(elig & predicted)[0]
        nids = np.where(elig & history & ~predicted)[0]
        if len(bids) < 10 or len(nids) < 50:
            skipped += 1
            continue
        used += 1
        elig_counts.append(int(elig.sum()))
        pred_counts.append(len(bids)); bench_counts.append(len(nids))
        adv_e = adv20_at(dvol, ent)
        adv_cache[ent] = adv_e
        for sym in bids:
            book.add(sym, ent, exi, adv_e[sym] if np.isfinite(adv_e[sym]) else 0.0)
        for sym in nids:
            bench.add(sym, ent, exi, 0.0)
    panel['_advc'] = adv_cache
    return (book, bench, used, skipped,
            float(np.mean(elig_counts)) if elig_counts else np.nan,
            float(np.mean(pred_counts)) if pred_counts else np.nan,
            float(np.mean(bench_counts)) if bench_counts else np.nan)


# -------------------------------------------------------------------- cell 3
def cell3():
    d = pd.read_parquet(os.path.join(BASE, 'dividends.parquet'),
                        columns=['symbol', 'ex_date', 'record_date'])
    d = d[d['ex_date'].notna() & d['record_date'].notna()].copy()
    cut = pd.Timestamp('2024-05-28')
    d['era'] = np.where(d['ex_date'] < cut, 'pre', 'post')
    bd = pd.offsets.BDay()
    rec = d['record_date'].dt.normalize()
    d['f_minus1'] = rec - bd
    d['f_same'] = rec
    ex = d['ex_date'].dt.normalize()
    d['w_minus1'] = d['f_minus1'] != ex
    d['w_same'] = d['f_same'] != ex
    d['wm_minus1'] = (d['f_minus1'].dt.to_period('M') != ex.dt.to_period('M'))
    d['wm_same'] = (d['f_same'].dt.to_period('M') != ex.dt.to_period('M'))
    d['two'] = np.where(d['era'] == 'pre', d['f_minus1'], d['f_same'])
    d['w_two'] = d['two'] != ex
    d['wm_two'] = (pd.DatetimeIndex(d['two']).to_period('M') != ex.dt.to_period('M'))

    res = {}
    for era in ('pre', 'post'):
        s = d[d['era'] == era]
        res[era] = dict(rows=int(len(s)),
                        wrong_minus1=int(s['w_minus1'].sum()),
                        wrong_same=int(s['w_same'].sum()),
                        wrong_month_minus1=int(s['wm_minus1'].sum()),
                        wrong_month_same=int(s['wm_same'].sum()),
                        residual_two_regime=int(s['w_two'].sum()),
                        residual_two_regime_wrong_month=int(s['wm_two'].sum()))
    # (e) month-by-month
    buckets = {}
    def add(tag, mask):
        s = d[mask]
        buckets[tag] = dict(rows=int(len(s)),
                            wrong_minus1=int(s['w_minus1'].sum()),
                            wrong_same=int(s['w_same'].sum()),
                            wrong_month_minus1=int(s['wm_minus1'].sum()),
                            wrong_month_same=int(s['wm_same'].sum()),
                            residual_two_regime=int(s['w_two'].sum()),
                            residual_two_regime_wrong_month=int(s['wm_two'].sum()))
    ex = d['ex_date']
    add('2024-04', (ex >= '2024-04-01') & (ex < '2024-05-01'))
    add('2024-05 (1st-27th)', (ex >= '2024-05-01') & (ex < '2024-05-28'))
    add('2024-05 (28th-31st)', (ex >= '2024-05-28') & (ex < '2024-06-01'))
    add('2024-06', (ex >= '2024-06-01') & (ex < '2024-07-01'))
    add('2024-07', (ex >= '2024-07-01') & (ex < '2024-08-01'))
    return res, buckets


# ------------------------------------------------------------------------ main
def main():
    log('=== building panel ===')
    panel = build_panel()
    adj = panel['adj']
    ret = np.full_like(adj, np.nan)
    ret[:, 1:] = adj[:, 1:] / adj[:, :-1] - 1.0
    panel['ret'] = ret
    sessions = panel['sessions']
    log(f'panel shape {adj.shape}')

    # sanity scan of the data
    with np.errstate(invalid='ignore'):
        wild = np.isfinite(ret) & (np.abs(ret) > 2.0)
    data_notes = dict(
        duplicate_rows=panel['dup'],
        daily_returns_gt_200pct=int(wild.sum()),
        daily_returns_lt_neg90pct=int((np.isfinite(ret) & (ret < -0.9)).sum()),
        nonpositive_adj_close=int((np.isfinite(adj) & (adj <= 0)).sum()),
        nonpositive_raw_close=int((np.isfinite(panel['rawc']) & (panel['rawc'] <= 0)).sum()),
        symbols_with_dot_in_name=int(sum(1 for s in panel['symbols'] if '.' in s)),
        test_tickers=[s for s in panel['symbols'] if s.upper().startswith(('ZVZZT', 'ZXZZT', 'ZJZZT', 'ZWZZT'))],
    )
    log(f'data notes: {data_notes}')

    notes = {}
    all_rows = []

    log('=== cell F5-LO ===')
    bk, bn, n_anch, mean_elig = cell_f5(panel)
    advc = panel['_advc']
    panel['adv_lookup'] = lambda s, e: np.array([advc[e[i]][s[i]] for i in range(len(s))])
    all_rows += run_cell('F5-LO', panel, bk, bn, '6-month overlapping, month-end anchors', notes)
    notes['F5-LO']['anchors_used'] = n_anch
    notes['F5-LO']['mean_eligible_per_rebalance'] = mean_elig
    del bk, bn; gc.collect()

    log('=== cell A4-LO ===')
    bk, bn, n_anch, skipped, mean_elig, mean_pred, mean_bench = cell_a4(panel)
    advc = panel['_advc']
    panel['adv_lookup'] = lambda s, e: np.array([advc[e[i]][s[i]] for i in range(len(s))])
    all_rows += run_cell('A4-LO', panel, bk, bn, '1-month, month-end anchors', notes)
    notes['A4-LO'].update(anchors_used=n_anch, anchors_skipped=skipped,
                          mean_eligible_per_rebalance=mean_elig,
                          mean_predicted=mean_pred, mean_benchmark=mean_bench)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT, 'cells.csv'), index=False)
    log('\n' + df.to_string())

    log('=== cell 3 ===')
    era, buckets = cell3()
    with open(os.path.join(OUT, 'cell3.json'), 'w') as f:
        json.dump({'era': era, 'months': buckets}, f, indent=2)
    log(json.dumps({'era': era, 'months': buckets}, indent=2))

    with open(os.path.join(OUT, 'notes.json'), 'w') as f:
        json.dump({'cells': notes, 'data': data_notes}, f, indent=2, default=str)
    log('DONE')


def diag3():
    """Supplementary: is the pd.offsets.BDay residual just exchange holidays?
    Recompute 'one business day before' on the real SPY session calendar."""
    ses = []
    for y in range(2016, 2027):
        t = pq.read_table(os.path.join(BASE, f'prices_by_year/all/year={y}.parquet'),
                          columns=['symbol', 'date'])
        df = t.to_pandas(); df['symbol'] = df['symbol'].astype(str)
        ses.append(pd.to_datetime(df.loc[df['symbol'] == 'SPY', 'date']))
    ses = pd.DatetimeIndex(np.sort(pd.concat(ses).unique()))
    log(f'SPY sessions 2016-2026: {len(ses)} {ses[0].date()}..{ses[-1].date()}')
    d = pd.read_parquet(os.path.join(BASE, 'dividends.parquet'),
                        columns=['symbol', 'ex_date', 'record_date', 'foreign'])
    d = d[d['ex_date'].notna() & d['record_date'].notna()].copy()
    ex = d['ex_date'].dt.normalize(); rec = d['record_date'].dt.normalize()
    pos = ses.searchsorted(rec.to_numpy(), side='left')
    prev = np.where(pos > 0, ses.to_numpy()[np.maximum(pos - 1, 0)], np.datetime64('NaT'))
    cut = pd.Timestamp('2024-05-28')
    era = np.where(ex < cut, 'pre', 'post')
    two = np.where(era == 'pre', prev, rec.to_numpy())
    wrong = two != ex.to_numpy()
    out = {}
    for e in ('pre', 'post'):
        m = era == e
        out[e] = dict(rows=int(m.sum()),
                      residual_two_regime_session_calendar=int((wrong & m).sum()),
                      residual_foreign_share=float(d.loc[m & wrong, 'foreign'].mean()) if (m & wrong).any() else 0.0)
    out['foreign_share_all'] = float(d['foreign'].mean())
    log(json.dumps(out, indent=2))
    with open(os.path.join(OUT, 'cell3_session_calendar.json'), 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'diag3':
        diag3()
    else:
        main()
