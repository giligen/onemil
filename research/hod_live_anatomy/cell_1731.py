#!/usr/bin/env python3
"""Cell 1,731 — thin breaks: a CAUSAL break-volume filter on the HOD BT population (PREREG_1731.md, frozen 10/31).

Book (primary) = population B1 of cell 1,730 (the FIRST armed crossing per symbol-day, re-derived with
causal_arming.armed_crossing_bars under the live gates) filled at the live +15 bps cap on the trigger-bar open
(d_bps <= 15), exits walked by the live rule (cache/opens_1730.parquet). Population A (the 5,417 recovered 1,663 BT
fills) is scored the same way, report-only.

Measures (all causal — computed from information available AT the crossing print):
  M1 pre-cross rate = (SIP volume inside minute T strictly before the first print >= trigger) / (seconds elapsed in
     T) / (v_cons10 / 60), v_cons10 = mean volume of the <= 10 bars before T (bars j-9..j, as 1,730). Tape = the 1,438
     SIP caches (research/hod_entry/sip_cache/c1438_{day}.pkl.gz keyed sym|m_lo|m_hi, then 1,427's {day}.pkl.gz keyed
     sym|break_m). Crossing in the first 5 s of T -> undefined, counted.
  M2 last-bar ratio = v_j / v_cons10 (bars only).
  M3 cumulative participation = sum(v, 09:30..end of bar j) / adv20 (research/bf_zero/universe.csv).
  Report-only: VT = v_T / v_cons10 (the 1,730 leaking read) and M2L = v_j / mean(v[0..j]) (the engine's
     arm_bar_vol_x definition, trading/hod_break.py l.215/238 — the live-ledger twin used for V3).

Cells: V1 quintiles (cuts fixed on TRAIN-H2), V2 drop-Q1 filter paired vs the unfiltered book (day-cluster
bootstrap), V3 live kept vs dropped on the ledger twin. Outputs vol_1731.csv, filter_1731.csv, cell_1731_out.txt.
Resumable: cache/vol_1731_bars.parquet (bar-level fields) and cache/vol_1731_tape.parquet (tape fields).
Read-only on bars_sip.db and the sip caches.
"""
import glob
import gzip
import os
import pickle
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'research/hod_entry'))
sys.path.insert(0, ROOT)
import causal_arming as ca  # noqa: E402
import cell_1730 as c30  # noqa: E402  (walk/statistics helpers; its module-level log file is append-only)

sr = ca.sr
OPENS = os.path.join(HERE, 'cache', 'opens_1730.parquet')
BARS_CACHE = os.path.join(HERE, 'cache', 'vol_1731_bars.parquet')
TAPE_CACHE = os.path.join(HERE, 'cache', 'vol_1731_tape.parquet')
OUT_TXT = os.path.join(HERE, 'cell_1731_out.txt')
VOL_CSV = os.path.join(HERE, 'vol_1731.csv')
FILT_CSV = os.path.join(HERE, 'filter_1731.csv')
BARS_URI = 'file:' + os.path.join(ROOT, 'research/bf_zero/bars_sip.db') + '?mode=ro'
TAPE_DIR = os.path.join(ROOT, 'research/hod_entry/sip_cache')
LEDGER = os.path.join(ROOT, 'logs/hod_dry_entry_ledger.csv')
CAP = 15
EARLY_S = 5.0
N_BOOT = 2000
MEASURES = ['M1', 'M2', 'M3', 'VT', 'M2L']
FILTER_MEASURES = ['M1', 'M2', 'M3']
SPLITS = ('TRAIN-H2', 'VAL')
_out = open(OUT_TXT, 'a')


def log(*a):
    """Print flushed and append to the cell's output file."""
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    _out.write(s + '\n')
    _out.flush()


# ------------------------------------------------------------------------------------------------ inputs
def load_population() -> pd.DataFrame:
    """A and B1 rows of the 1,730 cache with adv20 joined (universe.csv, the arming's adv20)."""
    x = pd.read_parquet(OPENS)
    x = x[(x['pop'] == 'A') | ((x['pop'] == 'B') & (x.k == 0))].copy()
    x['pop'] = np.where(x['pop'] == 'A', 'A', 'B1')
    u = pd.read_csv(ca.UNIVERSE_CSV, dtype={'symbol': str}, keep_default_na=False)
    u = u.rename(columns={'bar_date': 'date'})[['date', 'symbol', 'adv20']].drop_duplicates(['date', 'symbol'])
    u['adv20'] = pd.to_numeric(u.adv20, errors='coerce')
    x = x.merge(u, on=['date', 'symbol'], how='left')
    log(f'[pop] A {int((x["pop"] == "A").sum())}, B1 {int((x["pop"] == "B1").sum())} '
        f'(B1 at cap 15: {int(((x["pop"] == "B1") & (x.d_bps <= CAP)).sum())}); adv20 missing '
        f'{int(x.adv20.isna().sum())}')
    return x


def bar_fields(x: pd.DataFrame) -> pd.DataFrame:
    """Per (date, symbol, m_T): m_lo (bar j minute), v_j, cumv_j, mean_v_day_j, n_bars_cons. Resumable cache."""
    keys = x[['date', 'symbol', 'm_T']].drop_duplicates()
    old = pd.read_parquet(BARS_CACHE) if os.path.exists(BARS_CACHE) else pd.DataFrame(
        columns=['date', 'symbol', 'm_T'])
    done = set(zip(old.date, old.symbol))
    log(f'[bars] {len(keys)} crossings, symbol-days cached {len(done)}')
    todo = keys[[k not in done for k in zip(keys.date, keys.symbol)]]
    rows = old.to_dict('records')
    con = sqlite3.connect(BARS_URI, uri=True, timeout=60)
    t0 = time.time()
    groups = list(todo.groupby(['date', 'symbol']))
    for n, ((day, sym), g) in enumerate(groups, 1):
        b = c30.rth_bars(con, sym, day)
        for mT in g.m_T.astype(int):
            r = dict(date=day, symbol=sym, m_T=mT, m_lo=np.nan, v_j=np.nan, cumv_j=np.nan, mean_v_day_j=np.nan)
            if len(b):
                hit = np.nonzero(b.m.values == mT)[0]
                if len(hit) and hit[0] > 0:
                    j = int(hit[0]) - 1
                    v = b.v.values.astype(float)
                    r.update(m_lo=int(b.m.iat[j]), v_j=v[j], cumv_j=float(v[: j + 1].sum()),
                             mean_v_day_j=float(v[: j + 1].mean()))
            rows.append(r)
        if n % 1000 == 0 or n == len(groups):
            tmp = BARS_CACHE + '.tmp'
            pd.DataFrame(rows).to_parquet(tmp, index=False)
            os.replace(tmp, BARS_CACHE)
            el = time.time() - t0
            log(f'[bars] {n}/{len(groups)} symbol-days, {el:.0f}s, eta {el / n * (len(groups) - n):.0f}s')
    con.close()
    return pd.DataFrame(rows).drop_duplicates(['date', 'symbol', 'm_T'])


def load_tapes(day: str) -> tuple:
    """(c1438 dict sym|m_lo|m_hi -> (trades, quotes), 1,427 dict sym|break_m -> (trades, quotes)) for a day."""
    out = []
    for name in (f'c1438_{day}.pkl.gz', f'{day}.pkl.gz'):
        p = os.path.join(TAPE_DIR, name)
        if os.path.exists(p):
            with gzip.open(p, 'rb') as f:
                out.append(pickle.load(f))
        else:
            out.append({})
    return out[0], out[1]


def tape_fields(x: pd.DataFrame, bf: pd.DataFrame) -> pd.DataFrame:
    """Per crossing: tape source, status (no_tape / no_cross / early / ok), seconds to the cross, pre-cross volume,
    whole-minute tape volume (sanity vs v_T). Resumable per day."""
    k = x[['date', 'symbol', 'm_T', 'level', 'v_T']].drop_duplicates(['date', 'symbol', 'm_T']).merge(
        bf[['date', 'symbol', 'm_T', 'm_lo']], on=['date', 'symbol', 'm_T'], how='left')
    old = pd.read_parquet(TAPE_CACHE) if os.path.exists(TAPE_CACHE) else pd.DataFrame(columns=['date'])
    done_days = set(old.date)
    rows = old.to_dict('records')
    days = sorted(set(k.date) - done_days)
    log(f'[tape] {len(k)} crossings on {k.date.nunique()} days; days cached {len(done_days)}, to do {len(days)}')
    t0 = time.time()
    for n, day in enumerate(days, 1):
        c38, c27 = load_tapes(day)
        by_sym_end = {}
        for key in c38:
            s, _lo, hi = key.split('|')
            by_sym_end.setdefault((s, int(hi)), key)
        for r in k[k.date == day].itertuples():
            mT = int(r.m_T)
            tape, src = None, 'none'
            if pd.notna(r.m_lo) and f'{r.symbol}|{int(r.m_lo)}|{mT}' in c38:
                tape, src = c38[f'{r.symbol}|{int(r.m_lo)}|{mT}'], 'c1438'
            elif (r.symbol, mT) in by_sym_end:
                tape, src = c38[by_sym_end[(r.symbol, mT)]], 'c1438_end'
            elif f'{r.symbol}|{mT}' in c27:
                tape, src = c27[f'{r.symbol}|{mT}'], 'c1427'
            row = dict(date=day, symbol=r.symbol, m_T=mT, tape_src=src, status='no_tape', sec_to_cross=np.nan,
                       vol_pre=np.nan, vol_min_tape=np.nan)
            if tape is not None:
                tr = tape[0]
                s_ns, e_ns = sr.et_ns(day, mT * 60), sr.et_ns(day, (mT + 1) * 60)
                w = tr[(tr.ts >= s_ns) & (tr.ts < e_ns)].sort_values('ts', kind='stable')
                trig = round(float(r.level) + sr.TICK, 6)
                hits = w[w.price >= trig - 1e-9]
                row['vol_min_tape'] = float(w['size'].sum())
                if not len(hits):
                    row['status'] = 'no_cross'
                else:
                    tc = int(hits.ts.iloc[0])
                    sec = (tc - s_ns) / 1e9
                    row.update(sec_to_cross=sec, vol_pre=float(w[w.ts < tc]['size'].sum()),
                               status='early' if sec < EARLY_S else 'ok')
            rows.append(row)
        if n % 20 == 0 or n == len(days):
            tmp = TAPE_CACHE + '.tmp'
            pd.DataFrame(rows).to_parquet(tmp, index=False)
            os.replace(tmp, TAPE_CACHE)
            el = time.time() - t0
            log(f'[tape] {n}/{len(days)} days, {el:.0f}s, eta {el / n * (len(days) - n):.0f}s')
    return pd.DataFrame(rows)


def add_measures(x: pd.DataFrame, bf: pd.DataFrame, tf: pd.DataFrame) -> pd.DataFrame:
    """Join bar/tape fields and compute M1, M2, M3, VT, M2L (NaN when undefined)."""
    x = x.merge(bf, on=['date', 'symbol', 'm_T'], how='left').merge(
        tf[['date', 'symbol', 'm_T', 'tape_src', 'status', 'sec_to_cross', 'vol_pre', 'vol_min_tape']],
        on=['date', 'symbol', 'm_T'], how='left')
    vc = x.v_cons10.replace(0, np.nan)
    x['M1'] = np.where(x.status == 'ok', (x.vol_pre / x.sec_to_cross) / (vc / 60.0), np.nan)
    x['M2'] = x.v_j / vc
    x['M3'] = x.cumv_j / x.adv20.where(x.adv20 > 0)
    x['VT'] = x.v_T / vc
    x['M2L'] = x.v_j / x.mean_v_day_j.replace(0, np.nan)
    x['wkn'] = pd.to_datetime(x.date).dt.isocalendar().week.astype(int)
    return x


# ------------------------------------------------------------------------------------------------ statistics
def boot_mask_delta(xs: pd.DataFrame, keep: np.ndarray, rng) -> tuple:
    """Paired delta mean(net | keep) - mean(net | all) with a day-cluster bootstrap SE. Returns (delta, t, se)."""
    d0 = xs.net_R[keep].mean() - xs.net_R.mean()
    kk = keep.astype(float)
    pdf = pd.DataFrame({'Sk': kk * xs.net_R.values, 'Nk': kk, 'Sa': xs.net_R.values,
                        'Na': np.ones(len(xs))}).groupby(xs.date.values).sum()
    M = pdf.values
    W = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=N_BOOT)
    T = W @ M
    with np.errstate(invalid='ignore', divide='ignore'):
        ds = T[:, 0] / T[:, 1] - T[:, 2] / T[:, 3]
    se = float(np.nanstd(ds))
    return d0, (d0 / se if se > 0 else np.nan), se


def v1_quintiles(x: pd.DataFrame, pop: str) -> pd.DataFrame:
    """V1: per measure, quintile cuts on TRAIN-H2 (defined rows), applied to VAL; n, net, t, ex-top-5 % per bucket."""
    out = []
    for mcol in MEASURES:
        tr = x[(x.split == 'TRAIN-H2') & x[mcol].notna()][mcol]
        cuts = np.quantile(tr, [0.2, 0.4, 0.6, 0.8])
        for s in SPLITS:
            xs = x[(x.split == s) & x[mcol].notna()]
            q = np.searchsorted(cuts, xs[mcol].values, side='right')
            for i in range(5):
                v = xs[q == i]
                out.append(dict(pop=pop, measure=mcol, split=s, q=i + 1,
                                lo=v[mcol].min() if len(v) else np.nan, hi=v[mcol].max() if len(v) else np.nan,
                                cut_hi=cuts[i] if i < 4 else np.inf, n=len(v), net=v.net_R.mean(),
                                gross=v.gross_R.mean(), t=c30.dclust_t(v.net_R, v.date), ex5=c30.ex_top5(v.net_R)))
    return pd.DataFrame(out)


def v2_filter(x: pd.DataFrame, pop: str, weeks: dict, rng) -> pd.DataFrame:
    """V2: drop measure < TRAIN-H2 Q1 cut (undefined rows kept); paired stats per split and per alternating-week
    half; plus the same on the measure-defined subset (cov)."""
    out = []
    for mcol in FILTER_MEASURES:
        cut = float(np.quantile(x[(x.split == 'TRAIN-H2') & x[mcol].notna()][mcol], 0.2))
        for s in SPLITS:
            for scope in ('all', 'cov'):
                xs0 = x[x.split == s] if scope == 'all' else x[(x.split == s) & x[mcol].notna()]
                for half in ('both', 'A', 'B'):
                    xs = xs0 if half == 'both' else xs0[(xs0.wkn % 2 == 0) == (half == 'A')]
                    xs = xs.reset_index(drop=True)
                    keep = ~(xs[mcol] < cut).values
                    d, t, se = boot_mask_delta(xs, keep, rng)
                    k = xs[keep]
                    wk = weeks[s] / (1 if half == 'both' else 2)
                    out.append(dict(pop=pop, measure=mcol, split=s, scope=scope, half=half, cut=cut, n_all=len(xs),
                                    n_kept=int(keep.sum()), drop_share=1 - keep.mean(), fills_wk_kept=len(k) / wk,
                                    net_all=xs.net_R.mean(), net_kept=k.net_R.mean(), net_dropped=xs.net_R[~keep].mean(),
                                    delta=d, delta_t=t, delta_se=se, mde80=2.8 * se,
                                    R_wk_all=xs.net_R.sum() / wk, R_wk_kept=k.net_R.sum() / wk,
                                    ex5_delta=c30.ex_top5(k.net_R) - c30.ex_top5(xs.net_R),
                                    capped_delta=k.net_R.clip(upper=2).mean() - xs.net_R.clip(upper=2).mean()))
    return pd.DataFrame(out)


def pass_bar(f: pd.DataFrame, q: pd.DataFrame) -> dict:
    """The PREREG pass bar per filter measure on the primary book (scope all), excluding the live sign check."""
    res = {}
    for mcol in FILTER_MEASURES:
        ok, why = True, []
        for s in SPLITS:
            r = f[(f.measure == mcol) & (f.split == s) & (f.scope == 'all') & (f.half == 'both')].iloc[0]
            ha = f[(f.measure == mcol) & (f.split == s) & (f.scope == 'all') & (f.half == 'A')].iloc[0].delta
            hb = f[(f.measure == mcol) & (f.split == s) & (f.scope == 'all') & (f.half == 'B')].iloc[0].delta
            qq = q[(q.measure == mcol) & (q.split == s)]
            checks = {'delta>=0.03': r.delta >= 0.03, 't>=2': r.delta_t >= 2, 'Rwk>=all': r.R_wk_kept >= r.R_wk_all,
                      'ex5>=0': r.ex5_delta >= 0, 'halves same sign': np.sign(ha) == np.sign(hb) == np.sign(r.delta),
                      'fills/wk>=3': r.fills_wk_kept >= 3, 'Q1 worst': qq.net.idxmin() == qq[qq.q == 1].index[0]}
            for kname, v in checks.items():
                if not v:
                    ok = False
                    why.append(f'{s}:{kname}')
        res[mcol] = (ok, why)
    return res


# ------------------------------------------------------------------------------------------------ live OOS
def live_join() -> pd.DataFrame:
    """Live R-valid fills (fills_1721.csv) joined to the dry/paper entry ledger on (date, symbol, level)."""
    L = c30.load_live()
    D = pd.read_csv(LEDGER, on_bad_lines='warn')
    D = D[D.filled == 1].rename(columns={'account': 'ledger_account'}).copy()
    D['lv'] = D.level.round(2)
    L['lv'] = L.level.round(2)
    D = D.sort_values('ledger_account').drop_duplicates(['date', 'symbol', 'lv'], keep='last')
    j = L.merge(D[['date', 'symbol', 'lv', 'ledger_account', 'arm_bar_vol_x', 'cumvol_to_arm', 'cumvol_over_adv20']],
                on=['date', 'symbol', 'lv'], how='left')
    log(f'[live] {len(L)} R-valid fills; ledger matches {int(j.arm_bar_vol_x.notna().sum())} '
        f'(ledger accounts {j.ledger_account.value_counts().to_dict()})')
    return j


def v3_live(j: pd.DataFrame, x: pd.DataFrame, rng) -> list:
    """Live kept vs dropped at the BT TRAIN-H2 Q1 cut of the ledger twin: M3 <-> cumvol_over_adv20 (same definition),
    M2L <-> arm_bar_vol_x (engine definition; M2 itself has no ledger twin). Bootstrap 95 % CI of kept - dropped."""
    out = []
    for bt_col, lv_col in (('M3', 'cumvol_over_adv20'), ('M2L', 'arm_bar_vol_x')):
        cut = float(np.quantile(x[(x.split == 'TRAIN-H2') & x[bt_col].notna()][bt_col], 0.2))
        z = j[j[lv_col].notna()]
        kept, drop = z[z[lv_col] >= cut].R.values, z[z[lv_col] < cut].R.values
        if len(drop) < 3 or len(kept) < 3:
            log(f'  V3 {bt_col}<->{lv_col}: cut {cut:.4g}, kept {len(kept)}, dropped {len(drop)} — too few')
            continue
        bs = [rng.choice(kept, len(kept)).mean() - rng.choice(drop, len(drop)).mean() for _ in range(N_BOOT)]
        r = dict(measure=bt_col, ledger=lv_col, cut=cut, n_kept=len(kept), n_drop=len(drop), net_kept=kept.mean(),
                 net_drop=drop.mean(), diff=kept.mean() - drop.mean(), ci_lo=np.percentile(bs, 2.5),
                 ci_hi=np.percentile(bs, 97.5), live_all=z.R.mean(), live_q1_share=len(drop) / len(z))
        log(f'  V3 {bt_col}<->{lv_col}: cut {cut:.4g}; kept n {len(kept)} {kept.mean():+.3f} R, dropped n {len(drop)} '
            f'{drop.mean():+.3f} R, kept - dropped {r["diff"]:+.3f} CI [{r["ci_lo"]:+.3f}, {r["ci_hi"]:+.3f}]; '
            f'live share below the cut {r["live_q1_share"]:.1%}')
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------ main
def coverage(x: pd.DataFrame, label: str) -> None:
    """Availability rail: tape coverage of the book, status counts, winner/loser missingness gap, sanity ratio."""
    has = x.status.isin(['ok', 'early', 'no_cross'])
    m1 = x.M1.notna()
    win = x.net_R > 0
    log(f'[rail {label}] n {len(x)}; tape found {has.mean():.1%} (src {x.tape_src.value_counts().to_dict()}); '
        f'status {x.status.value_counts().to_dict()}')
    log(f'[rail {label}] M1 defined {m1.mean():.1%}; M1 missing winners {1 - m1[win].mean():.1%} vs losers '
        f'{1 - m1[~win].mean():.1%} (gap {abs(m1[win].mean() - m1[~win].mean()) * 100:.1f} pp); tape found winners '
        f'{has[win].mean():.1%} vs losers {has[~win].mean():.1%}')
    for s in SPLITS:
        xs = x[x.split == s]
        log(f'[rail {label} {s}] n {len(xs)}, M1 defined {xs.M1.notna().mean():.1%}, early {(xs.status == "early").mean():.1%}')
    ok = x[x.status.isin(['ok', 'early'])]
    log(f'[sanity {label}] tape minute volume / bar v_T median {(ok.vol_min_tape / ok.v_T).median():.3f} '
        f'(IQR {(ok.vol_min_tape / ok.v_T).quantile(.25):.3f}-{(ok.vol_min_tape / ok.v_T).quantile(.75):.3f}); '
        f'sec to cross median {ok.sec_to_cross.median():.1f}; M2 defined {x.M2.notna().mean():.1%}; '
        f'M3 defined {x.M3.notna().mean():.1%}; corr(M1,M2) {x.M1.corr(x.M2, method="spearman"):.2f}, '
        f'corr(M2,VT) {x.M2.corr(x.VT, method="spearman"):.2f}, corr(M1,VT) {x.M1.corr(x.VT, method="spearman"):.2f}')


def main():
    """Fetch/resume bar and tape fields, then V1-V3 and the pass bar."""
    rng = np.random.default_rng(1731)
    log(f'==== cell 1,731 run {time.strftime("%Y-%m-%d %H:%M:%S")}')
    x = load_population()
    bf = bar_fields(x)
    tf = tape_fields(x, bf)
    x = add_measures(x, bf, tf)
    c = pd.read_csv(os.path.join(ROOT, 'research/hod_entry/causal_arming_causal.csv'), usecols=['day', 'split'])
    weeks = {('TRAIN-H2' if s == 'TRAIN' else 'VAL'): pd.to_datetime(g.day).dt.to_period('W-FRI').nunique()
             for s, g in c.groupby('split')}
    log(f'[weeks] {weeks}')
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 500)
    books = {'B1': x[(x['pop'] == 'B1') & (x.d_bps <= CAP)].copy(), 'A': x[x['pop'] == 'A'].copy()}
    vols, filts = [], []
    for pop, xb in books.items():
        coverage(xb, pop)
        q = v1_quintiles(xb, pop)
        f = v2_filter(xb, pop, weeks, rng)
        vols.append(q)
        filts.append(f)
        log(f'V1 {pop}:\n' + q.round(3).to_string(index=False))
        log(f'V2 {pop}:\n' + f.round(3).to_string(index=False))
        pb = pass_bar(f, q)
        for mcol, (ok, why) in pb.items():
            log(f'PASS-BAR {pop} {mcol}: {"PASS (pending live sign)" if ok else "FAIL"} {why}')
    pd.concat(vols).to_csv(VOL_CSV, index=False)
    pd.concat(filts).to_csv(FILT_CSV, index=False)
    log('V3 live OOS (cuts from the primary book B1):')
    j = live_join()
    v3 = v3_live(j, books['B1'], rng)
    pd.DataFrame(v3).to_csv(os.path.join(HERE, 'live_1731.csv'), index=False)
    log('==== done')


if __name__ == '__main__':
    main()
