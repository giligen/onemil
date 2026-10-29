"""Cell 1,713 -- sleeve #4 weekly short-term reversal (PREREG_STREV_20261029), Stage 1, rejection-only.

Run:  nice -n 19 python3 research/sleeves/strev.py        (writes weekly_1713.csv, prints the evaluation)

Rule: each Friday close (last session of the ISO week) the universe = top-500 by 60-session median dollar volume,
price >= $10, name-pattern + fund-name exclusions; signal = prior 5-session return C[t]/C[t-5]-1; long the 20 LOWEST,
equal weight, enter the first session of the next week at the open, exit the first session of the following week at
the open.  Primary cost charges the PREREG's full weekly turnover (20 bps round trip x 100 %); 40 bps and the
name-overlap (actual-turnover) cost are reported beside it.  Sensitivities: top-50, price >= $20, Thursday signal,
causal jump screen (>50 % one-day close move in the 5-session signal window).  TEST 2023+ is sealed: the parquet scan
stops at 2022-12-31, and a week whose exit open would fall after that is dropped.
"""
from __future__ import annotations

import logging
import sys
import time

import numpy as np
import pandas as pd

from fiftytwo_wh import CACHE_DIR, DV_WINDOW, MIN_DV, OUT_DIR, ROOT, SEAL_END, load_panel, symbol_screen
from lowvol import block_ci, fund_name_mask, max_dd, split_metrics, split_of

log = logging.getLogger('strev')
N_UNIV, K, PRICE_MIN, COST_RT, N_NULL, LOOK = 500, 20, 10.0, 20.0, 1000, 5


# ----------------------------------------------------------------------------- the rule
def week_index(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """First and last session index of every ISO week present in the panel."""
    iso = dates.isocalendar()
    key = iso['year'].to_numpy().astype(int) * 100 + iso['week'].to_numpy().astype(int)
    s = pd.Series(np.arange(len(dates)))
    g = s.groupby(key)
    return g.first().to_numpy(), g.last().to_numpy()


def week_returns(O: np.ndarray, C: np.ndarray, ent: np.ndarray, ext: np.ndarray) -> np.ndarray:
    """R[i] = open(exit session i) / open(entry session i) - 1 per name; exit open missing -> last close before it (WARNING)."""
    R = np.full((len(ent), O.shape[1]), np.nan)
    n_fb = 0
    for i, (a, b) in enumerate(zip(ent, ext)):
        o0, o1 = O[a].astype(np.float64), O[b].astype(np.float64)
        r = o1 / o0 - 1
        miss = np.isfinite(o0) & ~np.isfinite(o1)
        if miss.any():
            last = pd.DataFrame(C[a:b]).ffill().to_numpy()[-1]
            r = np.where(miss, last / o0 - 1, r)
            n_fb += int(miss.sum())
        R[i] = r
    if n_fb:
        log.warning('exit open missing for %d name-weeks; fell back to last close before the exit', n_fb)
    return R


def universe_at(C, V, t, allowed, entry_open, n_univ=N_UNIV, price_min=PRICE_MIN, jump=None) -> np.ndarray:
    """Indices of the top-`n_univ` names by 60d median dollar volume passing the PREREG filters at close t (knowable at t)."""
    dv = C[t - DV_WINDOW + 1:t + 1].astype(np.float64) * V[t - DV_WINDOW + 1:t + 1]
    ok_n = np.isfinite(dv).sum(axis=0) >= 50
    with np.errstate(all='ignore'):
        med = np.nanmedian(dv, axis=0)
    ok = allowed & ok_n & (med >= MIN_DV) & (C[t] >= price_min) & np.isfinite(C[t - LOOK]) & np.isfinite(entry_open)
    if jump is not None:
        ok &= ~jump
    idx = np.flatnonzero(ok)
    return idx[np.argsort(-med[idx], kind='stable')][:n_univ]


def jump_in_window(C: np.ndarray, t: int, lo: int) -> np.ndarray:
    """True where a name has a >50 % one-day close move in sessions (lo, t] -- unadjusted split / reverse-split guard."""
    with np.errstate(all='ignore'):
        j = np.abs(C[lo + 1:t + 1].astype(np.float64) / C[lo:t] - 1) > 0.5
    return np.nan_to_num(j).any(axis=0)


def drift_weights(w: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Weights after one week of buy-and-hold drift (NaN returns treated as 0)."""
    g = w * (1 + np.nan_to_num(r, nan=0.0))
    return g / g.sum() if g.sum() > 0 else g


def eq_weights(names: np.ndarray, n_sym: int) -> np.ndarray:
    """Equal weights over `names` as a full-length vector."""
    w = np.zeros(n_sym)
    w[names] = 1.0 / len(names)
    return w


def simulate(P, allowed, first, last, k, price_min, signal_lag, jump_screen, collect_null, seed, S) -> dict:
    """Run the weekly rule (week w: signal at last[w]-lag, entry first[w+1], exit first[w+2]); weekly rows + picks + null."""
    O, C, V, dates, symbols = P['open'], P['close'], P['volume'], P['dates'], P['symbols']
    nw = len(first)
    ent, ext = first[1:nw - 1], first[2:nw]          # week w in 0..nw-3
    R = week_returns(O, C, ent, ext)
    spy = symbols.index('SPY')
    rng = np.random.default_rng(seed)
    rows, picks, null_rows = [], [], []
    w_s, w_u = np.zeros(S), np.zeros(S)
    n_nan = 0
    for w in range(len(ent)):
        t = last[w] - signal_lag
        if dates[ent[w]] < pd.Timestamp('2017-01-01') or t < 80:
            continue
        jm = jump_in_window(C, t, t - LOOK) if jump_screen else None
        uni = universe_at(C, V, t, allowed, O[ent[w]], price_min=price_min, jump=jm)
        with np.errstate(all='ignore'):
            sig = C[t].astype(np.float64) / C[t - LOOK] - 1
        cand = uni[np.isfinite(sig[uni])]
        sel = cand[np.argsort(sig[cand], kind='stable')][:k]
        if len(sel) < k:
            log.warning('%s: only %d of %d loser names', dates[ent[w]].date(), len(sel), k)
        n_nan += int((~np.isfinite(R[w][sel])).sum())
        gs, gu = np.nan_to_num(R[w][sel], nan=0.0).mean(), np.nan_to_num(R[w][uni], nan=0.0).mean()
        ws_new, wu_new = eq_weights(sel, S), eq_weights(uni, S)
        to_s, to_u = 0.5 * np.abs(ws_new - w_s).sum(), 0.5 * np.abs(wu_new - w_u).sum()
        w_s, w_u = drift_weights(ws_new, R[w]), drift_weights(wu_new, R[w])
        rows.append({'entry': dates[ent[w]], 'month': dates[ent[w]].strftime('%Y-%m'), 'n_univ': len(uni), 'n_longs': len(sel),
                     'gross': gs, 'univ_gross': gu, 'turnover': to_s, 'univ_turnover': to_u, 'spy': R[w][spy]})
        for n in sel:
            picks.append((dates[ent[w]], symbols[n], float(sig[n]), float(C[t, n]), float(R[w][n])))
        if collect_null:
            pk = np.argpartition(rng.random((N_NULL, len(uni))), k, axis=1)[:, :k]
            null_rows.append(np.nan_to_num(R[w][uni[pk]], nan=0.0).mean(axis=1))
    if n_nan:
        log.warning('%d picked name-weeks had no exit price at all; counted as 0 return', n_nan)
    df = pd.DataFrame(rows)
    df['net'] = df['gross'] - COST_RT / 1e4                        # PREREG: full weekly turnover charged
    df['net40'] = df['gross'] - 2 * COST_RT / 1e4
    df['net_actual'] = df['gross'] - df['turnover'] * COST_RT / 1e4  # diagnostic: only the replaced fraction pays
    df['univ_net'] = df['univ_gross'] - df['univ_turnover'] * COST_RT / 1e4
    df['split'] = df['month'].map(split_of)
    df['year'] = df['month'].str[:4].astype(int)
    out = {'df': df, 'picks': pd.DataFrame(picks, columns=['entry', 'symbol', 'sig5d', 'price', 'ret'])}
    if collect_null:
        out['null'] = np.array(null_rows) - COST_RT / 1e4          # random books also pay the full-turnover cost
    return out


# ----------------------------------------------------------------------------- aggregation + metrics
def to_months(d: pd.DataFrame) -> pd.DataFrame:
    """Compound weekly columns into entry-month returns (months with < 3 weeks only at the series edges)."""
    cols = ['gross', 'net', 'net40', 'net_actual', 'univ_gross', 'univ_net', 'spy']
    m = d.groupby('month').agg(**{c: (c, lambda x: float(np.prod(1 + x) - 1)) for c in cols}, weeks=('net', 'size'),
                               turnover=('turnover', 'mean'), split=('split', 'first'), year=('year', 'first'))
    return m.reset_index()


def week_pass_bar(wk: pd.DataFrame, mo: pd.DataFrame, null: np.ndarray, col='net') -> dict:
    """Frozen VAL bar: net excess > 0 with year-block CI > 0; max DD <= 20 %; green weeks >= 52 % and above the null."""
    m = split_metrics(mo.assign(net=mo[col]))
    green = float((wk[col] > 0).mean())
    ng = (null > 0).mean(axis=0)
    r = {'excess_gt0': m['ex_univ'] > 0, 'excess_CI_gt0': m['ex_lo'] > 0, 'dd_le_20': m['dd'] <= 0.20,
         'green_ge_52': green >= 0.52, 'green_above_null_mean': green > ng.mean(), 'green_above_null_p95': green > np.percentile(ng, 95)}
    r['PASS'] = r['excess_CI_gt0'] and r['dd_le_20'] and r['green_ge_52'] and r['green_above_null_p95']
    r['UNDETERMINED'] = (not r['PASS']) and r['excess_gt0']
    r['VERDICT'] = 'PASS' if r['PASS'] else ('UNDETERMINED' if r['UNDETERMINED'] else 'REJECT')
    return {**r, 'green': green, 'null_green_mean': float(ng.mean()), 'null_green_p95': float(np.percentile(ng, 95)),
            'p_null_ge': float((ng >= green).mean()), 'metrics': m}


def fmt(m: dict) -> str:
    """One-line print of a split_metrics dict (percent units), no CAPM."""
    p = lambda k: f'{m[k] * 100:+.2f}'
    return (f"n={m['n']} mean {p('mean')} univ {p('univ_mean')} spy {p('spy_mean')} | ex_univ {p('ex_univ')} CI[{p('ex_lo')},{p('ex_hi')}] "
            f"ex_spy {p('ex_spy')} | sharpe {m['sharpe']:.2f} DD {m['dd'] * 100:.1f}% (univ {m['univ_dd'] * 100:.1f}, spy {m['spy_dd'] * 100:.1f}) "
            f"| ex-top5% {p('ex_top5')} | green-mo {m['green'] * 100:.0f}% (univ {m['univ_green'] * 100:.0f})")


def mom_overlap(wk: pd.DataFrame, mo: pd.DataFrame) -> dict:
    """MOM ledger (monthly entries, no pick names): weeks sharing an ISO week with a MOM entry, monthly net correlation."""
    mom = pd.read_csv(ROOT / 'research/momentum_weekly/1700c_monthly.csv', parse_dates=['entry_date'])
    mom = mom[mom['entry_date'] <= SEAL_END].copy()
    iso = lambda s: s.dt.isocalendar().year.astype(int) * 100 + s.dt.isocalendar().week.astype(int)
    coincide = wk['entry'].pipe(iso).isin(iso(mom['entry_date']))
    mom['month'] = mom['entry_date'].dt.strftime('%Y-%m')
    j = mo.set_index('month')[['net', 'split']].join(mom.set_index('month')['net_ret'].rename('mom'), how='inner').dropna()
    corr = {s: (round(float(np.corrcoef(g['net'], g['mom'])[0, 1]), 2), len(g)) for s, g in list(j.groupby('split')) + [('ALL', j)]}
    return {'weeks_sharing_week_with_mom_entry': int(coincide.sum()), 'weeks': len(wk), 'monthly_corr_with_MOM': corr,
            'note': 'MOM ledger has no pick names, so name-level position overlap cannot be computed'}


# ----------------------------------------------------------------------------- driver
def main() -> None:
    """Frozen rule + sensitivities on TRAIN/VAL; write weekly_1713.csv; print the evaluation."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    CACHE_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    P = load_panel('2016-01-01', '2022-12-31')
    assert P['dates'].max() <= SEAL_END
    allowed_aw, _ = symbol_screen(P['symbols'])
    fund = fund_name_mask(P['symbols'])
    allowed = allowed_aw & ~fund
    log.info('allowed as-written %d, after fund-name screen %d (%d fund-like removed) / %d symbols', allowed_aw.sum(),
             allowed.sum(), (allowed_aw & fund).sum(), len(allowed))
    first, last = week_index(P['dates'])
    run = lambda **kw: simulate(P, kw.pop('allowed', allowed), first, last, kw.pop('k', K), kw.pop('price_min', PRICE_MIN),
                                 kw.pop('signal_lag', 0), kw.pop('jump_screen', False), kw.pop('collect_null', False),
                                 1713, len(P['symbols']))
    base = run(collect_null=True)
    wk, picks, null = base['df'], base['picks'], base['null']
    sens = {'top50': run(k=50)['df'], 'price20': run(price_min=20.0)['df'], 'skipLast': run(signal_lag=1)['df'],
            'jumpSig': run(jump_screen=True)['df'], 'asWritten': run(allowed=allowed_aw)['df']}
    for name, d in sens.items():
        wk[f'net_{name}'] = d.set_index('entry')['net'].reindex(wk['entry']).to_numpy()
        wk[f'gross_{name}'] = d.set_index('entry')['gross'].reindex(wk['entry']).to_numpy()
    wk['excess'] = wk['net'] - wk['univ_net']
    wk.to_csv(OUT_DIR / 'weekly_1713.csv', index=False)
    picks.to_parquet(CACHE_DIR / 'picks_1713.parquet')
    np.save(CACHE_DIR / 'null_1713.npy', null)
    log.info('wrote weekly_1713.csv (%d weeks) in %.0fs', len(wk), time.time() - t0)
    mo = to_months(wk)
    mo_all = {}
    for sp in ('TRAIN', 'VAL'):
        sel = (wk['split'] == sp).to_numpy()
        w, m = wk[sel], mo[mo['split'] == sp]
        print(f"\n==== {sp} {w.month.iloc[0]}..{w.month.iloc[-1]} weeks={sel.sum()} months={len(m)} universe median {int(w.n_univ.median())} "
              f"min {w.n_univ.min()} longs min {w.n_longs.min()}")
        bar = week_pass_bar(w, m, null[sel])
        print('NET RT20 (full turnover):', fmt(bar['metrics']))
        print('  gross:', fmt(split_metrics(m.assign(net=m['gross']), bench='univ_gross')))
        print('  gross/week %+.3f%%  net/week %+.3f%%  univ net/week %+.3f%%  cost/week 0.200%%' % (
            w.gross.mean() * 100, w.net.mean() * 100, w.univ_net.mean() * 100))
        print('  turnover: one-way fraction replaced/week %.2f (strat) | %.3f (universe EW); full-turnover charge = 1.00' % (
            w.turnover.mean(), w.univ_turnover.mean()))
        print('NET RT40:', fmt(split_metrics(m.assign(net=m['net40']))))
        print('NET actual-turnover cost:', fmt(split_metrics(m.assign(net=m['net_actual']))))
        for name in sens:
            ws_ = w.dropna(subset=[f'net_{name}'])
            ms = to_months(ws_.assign(net=ws_[f'net_{name}'], gross=ws_[f'gross_{name}'])[['entry', 'month', 'split', 'year', 'turnover', 'gross',
                           'net', 'net40', 'net_actual', 'univ_gross', 'univ_net', 'spy']])
            print(f'SENS {name:9s}: gross/wk {ws_[f"gross_{name}"].mean() * 100:+.3f}% net/wk {ws_[f"net_{name}"].mean() * 100:+.3f}% |',
                  fmt(split_metrics(ms)), '| green-wk %.1f%%' % (100 * (ws_[f'net_{name}'] > 0).mean()))
        print('PASS BAR line by line:', {k: v for k, v in bar.items() if k != 'metrics'})
        for c, lab, adj in (('net40', 'RT40', -COST_RT / 1e4), ('net_actual', 'actual-turnover', 0.0)):
            b2 = week_pass_bar(w, m, null[sel] + adj, col=c)     # null re-charged at the same cost model as the strategy column
            print(f'  verdict under {lab} cost:', b2['VERDICT'], 'excess %.2f%% CI lo %.2f%%' % (
                b2['metrics']['ex_univ'] * 100, b2['metrics']['ex_lo'] * 100))
        nl = null[sel]
        print(f'  null random-{K} weekly: net mean {nl.mean() * 100:+.3f}%/wk vs strat {w.net.mean() * 100:+.3f} | green mean '
              f'{(nl > 0).mean() * 100:.1f}% vs actual {bar["green"] * 100:.1f}%')
        top = np.sort(w.net.to_numpy())[::-1]
        print('  weekly net: ex-top-1%% mean %+.3f%%, ex-top-5%% mean %+.3f%%, best %+.1f%%, worst %+.1f%%' % (
            top[int(np.ceil(.01 * len(top))):].mean() * 100, top[int(np.ceil(.05 * len(top))):].mean() * 100, top[0] * 100, top[-1] * 100))
        yr = m.groupby('year').agg(strat=('net', 'mean'), univ=('univ_net', 'mean'), spy=('spy', 'mean'))
        print('  per-year mean monthly % (strat/univ/spy):', {int(y): tuple(np.round(r * 100, 2)) for y, r in yr.iterrows()})
        pk = picks[picks.entry.isin(w.entry)]
        bad = pk[(pk.ret > 5) | (pk.ret < -0.8)]
        print(f'  price-scale: {len(pk)} name-weeks, {len(bad)} beyond +500%/-80%; ret max {pk.ret.max() * 100:+.0f}% min {pk.ret.min() * 100:+.0f}%; '
              f'median pick 5d signal {pk.sig5d.median() * 100:+.1f}%, median price {pk.price.median():.1f}, distinct names {pk.symbol.nunique()}; '
              f'top symbols {pk.symbol.value_counts().head(6).to_dict()}')
        mo_all[sp] = (w, m)
    print('\nOverlap with MOM:', mom_overlap(wk, mo))
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
