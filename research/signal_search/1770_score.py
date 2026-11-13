"""Cell 1,770 scorer (PREREG_1770, frozen 2026-11-13 21:20 UTC): the 1,769 primary trades (E1 short 09:35, stop = first-5-min high
+50 bp, cover 10:30) re-charged with MEASURED NBBO from 1770_quotes.parquet, never a band.
Variants (all on the same trades; the 1769 stop / cover decisions come from the minute bars):
  Q   PREREG primary: sell at the BID of the last valid quote in 09:35:00-09:35:59; cover at the ASK of the last valid quote in
      10:30:00-10:30:59, or for a stop-hit trade at the first ask >= stop level in the stop minute (measured slip = that ask vs the
      stop level); + 2 bp per leg + borrow 50 bp.
  B   bar-price gross of 1769 minus the MEASURED half-spreads (entry + cover/stop quote) instead of the band: the clean paired delta.
  P   like Q but the quotes struck BEFORE the 09:35 / 10:30 instant (last quote 09:34:55-59 / 10:29:55-59) -- sensitivity: Q's quotes
      are up to 59 s after the instant at which the 1769 bar price is struck.
Day-clustered t / ex-top-5 % / worst-5 % day / weekly P10 / MDE come from 1769_score.stats (same definitions).
Usage: cd /home/ec2-user/onemil && ulimit -v 2500000 && nice -n 19 timeout 600 python3 research/signal_search/1770_score.py
"""
import os, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FEE_RT, BORROW = 4.0, 50.0                      # 2 bp per leg x 2 legs; borrow bp per day
TIERS = [0, 10, 20, 50, 1e9]; TIER_NAMES = ['<10', '10-20', '20-50', '50+']


def log(m):
    print(m, flush=True)


def load_1769():
    """Import 1769_score.py (module name starts with a digit) for stats / cl_t / cost_bands."""
    sp = importlib.util.spec_from_file_location('s1769', os.path.join(HERE, '1769_score.py'))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def hs_bp(bid, ask):
    """Half-spread in bp of mid: (ask - bid) / (ask + bid)."""
    return (ask - bid) / (ask + bid) * 1e4


def sanitize(S):
    """Post-hoc data-cleaning rule (NOT in the PREREG; reported as a deviation): a quote is a bad NBBO print and is set missing when its
    half-spread > 1000 bp of mid (crossed-looking / 0.01 bids, 2x-wide asks) or its mid is outside [0.5, 2] x the 1769 bar price.
    A stop quote (first ask >= stop level) failing it falls back to the minute's last ask if that is within 1.5 x the stop level, else missing."""
    S = S.copy(); ref = {'e': S.price, 'ep': S.price, 'c': S.model_fill, 'cp': S.model_fill}
    for k in ('e', 'ep', 'c', 'cp'):
        mid = (S[f'{k}_bid'] + S[f'{k}_ask']) / 2
        bad = S[f'{k}_bid'].notna() & ((hs_bp(S[f'{k}_bid'], S[f'{k}_ask']) > 1000) | (mid / ref[k] < 0.5) | (mid / ref[k] > 2))
        log(f'  sanitize {k}: {int(bad.sum())} bad quotes of {int(S[f"{k}_bid"].notna().sum())} set missing')
        S.loc[bad, [f'{k}_bid', f'{k}_ask']] = np.nan
    badt = S.ask_trig.notna() & ((hs_bp(S.bid_trig, S.ask_trig) > 1000) | (S.ask_trig / S.trig > 2))
    log(f'  sanitize stop: {int(badt.sum())} bad first-ask>=stop quotes of {int(S.ask_trig.notna().sum())} -> fall back to the last ask')
    S.loc[badt, ['ask_trig', 'bid_trig']] = np.nan
    badl = S.ask_last.notna() & (S.ask_last / S.trig > 1.5)
    log(f'  sanitize stop last ask: {int(badl.sum())} > 1.5x stop set missing')
    S.loc[badl, 'ask_last'] = np.nan
    return S


def attach_quotes(S, Q):
    """Merge the reduced quote rows (entry, cover, entry_pre, cover_pre, stop) onto the trade frame as e_/c_/ep_/cp_ columns, then sanitize."""
    for kind, pre in (('entry', 'e'), ('cover', 'c'), ('entry_pre', 'ep'), ('cover_pre', 'cp')):
        q = Q[(Q.kind == kind) & (Q.n > 0)][['day', 'symbol', 'bid', 'ask']].rename(columns={'bid': f'{pre}_bid', 'ask': f'{pre}_ask'})
        S = S.merge(q, on=['day', 'symbol'], how='left')
    q = Q[(Q.kind == 'stop') & (Q.n > 0)][['day', 'symbol', 'bid', 'ask', 'ask_trig', 'bid_trig', 'ask_last']].rename(
        columns={'bid': 's_bid0', 'ask': 's_ask0'})
    return sanitize(S.merge(q, on=['day', 'symbol'], how='left'))


def measured(S, entry, cover):
    """Add the per-variant columns: entry bid, cover ask (stop trades: first ask >= stop level, else last ask with a WARNING count),
    required-quote flag, gross (bp of entry bid), half-spreads and measured stop slip.  entry/cover = column prefixes."""
    S = S.copy(); st = (S['stop'] == 1).to_numpy()
    ask_stop = S.ask_trig.where(S.ask_trig.notna(), S.ask_last)
    hs_stop = hs_bp(S.bid_trig, S.ask_trig).where(S.ask_trig.notna(), hs_bp(S.s_bid0, S.s_ask0).where(hs_bp(S.s_bid0, S.s_ask0) <= 1000))     # fallback: first quote of the minute
    S['x_ask'] = np.where(st, ask_stop, S[f'{cover}_ask'])
    S['ok'] = S[f'{entry}_bid'].notna() & S.x_ask.notna()
    S['e_bid_v'] = S[f'{entry}_bid']
    S['hs_e'] = hs_bp(S[f'{entry}_bid'], S[f'{entry}_ask'])
    S['hs_x'] = np.where(st, hs_stop, hs_bp(S[f'{cover}_bid'], S[f'{cover}_ask']))
    S['gross_q'] = (S.e_bid_v - S.x_ask) / S.e_bid_v * 1e4
    S['slip_bp'] = np.where(st, (S.x_ask - S.trig) / S.e_bid_v * 1e4, np.nan)
    S['slip_model_bp'] = np.where(st, (S.model_fill - S.trig) / S.e_bid_v * 1e4, np.nan)
    S['fallback_last_ask'] = st & S.ask_trig.isna().to_numpy() & S.ask_last.notna().to_numpy()
    return S


def coverage(S, tag, log_fn=log):
    """Required-quote coverage and the winner / loser missingness gap (winner = 1769 band net > 0)."""
    w = S.net > 0
    cov = S.ok.mean(); cw, cl = S.ok[w].mean(), S.ok[~w].mean()
    gap = (1 - cw) * 100 - (1 - cl) * 100
    log_fn(f'{tag}: coverage {cov:.1%} (n {len(S)}); missing winners {(1 - cw):.1%} vs losers {(1 - cl):.1%} -> gap {gap:+.1f} pp; '
           f'stop trades w/o ask>=stop quote (fallback last ask) {int(S.fallback_last_ask.sum())}')
    return dict(split=tag, n=len(S), coverage=cov, miss_winners=1 - cw, miss_losers=1 - cl, gap_pp=gap, void=bool(cov < 0.9 or abs(gap) > 5))


def run_variant(S1, name, split, Dy, M):
    """Stats of one variant on the covered trades: Q / P (quote prices), B (bar gross - measured hs), band (1769 net, same trades)."""
    s = S1[S1.ok].copy()
    out = []
    for vname in name:
        t = s.copy()
        if vname == 'band1769':
            pass                                                     # net, gross, hs_bp already the 1769 columns
        elif vname == 'B':
            t = t[t.hs_e.notna() & t.hs_x.notna()].copy()
            t['hs_bp'] = t.hs_e + t.hs_x; t['net'] = t.gross - FEE_RT - t.hs_bp - BORROW
        else:
            t['gross'] = t.gross_q; t['hs_bp'] = t.hs_e + t.hs_x; t['net'] = t.gross_q - FEE_RT - BORROW
        a, b = ('2021-01-01', '2023-12-31') if split == 'VAL' else ('2016-01-01', '2020-12-31')
        dd = Dy[(Dy.date >= a) & (Dy.date <= b)]
        if split == 'TRAIN':                                         # 1,000-trade sample: weeks come from the sampled days only
            dd = dd[dd.date.isin(set(t.day))]
        st = M.stats(t, len(dd) / 5.0, set(dd.wk), np.random.default_rng(1770))
        out.append(dict(sample=split, variant=vname, **st, mean_hs_total=float(t.hs_bp.mean()), mean_slip_bp=float(t.slip_bp.mean()),
                        median_slip_bp=float(t.slip_bp.median()), mean_slip_model_bp=float(t.slip_model_bp.mean()),
                        net_pairedΔ_vs_band=float((t.net - s.net.loc[t.index]).mean()) if vname != 'band1769' else 0.0))
    return out


def hs_tables(S):
    """Median / mean measured half-spread at 09:35 and 10:30 by price tier vs the 1769 band total (hs_bp = entry + exit band)."""
    S = S.copy(); S['tier'] = pd.cut(S.price, TIERS, labels=TIER_NAMES, right=False)
    rows = []
    for split, g in S.groupby('split'):
        for tier, h in list(g.groupby('tier', observed=True)) + [('all', g)]:
            e, c = h.e_hs_all.dropna(), h.c_hs_all.dropna()
            rows.append(dict(split=split, tier=tier, n=len(h), n_e=len(e), med_e=e.median(), mean_e=e.mean(), n_c=len(c), med_c=c.median(),
                             mean_c=c.mean(), med_e_plus_c=(h.e_hs_all + h.c_hs_all).median(), band_total_med=h.hs_bp.median(), band_total_mean=h.hs_bp.mean()))
    return pd.DataFrame(rows)


def main():
    """Coverage rail -> half-spread tables -> TRAIN sample, VAL stats -> kill checks."""
    M = load_1769()
    S = pd.read_parquet(os.path.join(HERE, '1770_sample.parquet'))
    Q = pd.read_parquet(os.path.join(HERE, '1770_quotes.parquet'))
    assert S.day.max() < '2024-01-01', 'TEST leaked'
    Dy = pd.read_parquet(os.path.join(HERE, '1769_days.parquet')); Dy['wk'] = pd.to_datetime(Dy.date).dt.strftime('%G-%V')
    log(f'sample {len(S)} trades, quote rows {len(Q)}; quote kinds {Q.kind.value_counts().to_dict()}')
    S = attach_quotes(S, Q)
    S['e_hs_all'] = hs_bp(S.e_bid, S.e_ask); S['c_hs_all'] = hs_bp(S.c_bid, S.c_ask)
    tabs = hs_tables(S); tabs.to_csv(os.path.join(HERE, '1770_hs_tiers.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
    log('HALF-SPREAD (bp of mid) by price tier: measured vs the 1769 band total (entry+exit)\n' + tabs.round(1).to_string())
    M.cost_bands()                                                   # prints the 1748 per-leg band medians for reference
    SQ = measured(S, 'e', 'c'); SP = measured(S, 'ep', 'cp')
    cov_rows = []
    for sp in ('VAL', 'TRAIN'):
        cov_rows.append(dict(variant='Q', **coverage(SQ[SQ.split == sp], f'{sp}/Q')))
        cov_rows.append(dict(variant='P', **coverage(SP[SP.split == sp], f'{sp}/P')))
    pd.DataFrame(cov_rows).to_csv(os.path.join(HERE, '1770_coverage.csv'), index=False)
    rows = []
    for sp in ('TRAIN', 'VAL'):                                      # TRAIN table written before VAL is read
        a = SQ[SQ.split == sp]; b = SP[SP.split == sp]
        rows += run_variant(a, ['band1769', 'Q', 'B'], sp, Dy, M) + run_variant(b, ['P'], sp, Dy, M)
        if sp == 'TRAIN':
            pd.DataFrame(rows).to_csv(os.path.join(HERE, '1770_summary.csv'), index=False); log('TRAIN written; reading VAL once')
    R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, '1770_summary.csv'), index=False)
    keep = ['n', 'n_wk', 'mean_net', 'day_t', 'hit', 'stop_hit', 'worst5_day', 'ex_top5', 'y2021', 'y2022', 'y2023', 'wk_p10', 'MDE80', 'gross',
            'mean_hs_total', 'mean_slip_bp', 'median_slip_bp', 'mean_slip_model_bp', 'net_borrow100', 'net_pairedΔ_vs_band']
    log(R.set_index(['sample', 'variant'])[keep].round(1).to_string())
    v = R[(R['sample'] == 'VAL') & (R.variant == 'Q')].iloc[0]; tr = R[(R['sample'] == 'TRAIN') & (R.variant == 'Q')].iloc[0]
    kills = {'VAL net<=50bp': v.mean_net <= 50, 'day-t<2': v.day_t < 2, 'ex-top5<=0': v.ex_top5 <= 0, 'worst5 day<-800bp': v.worst5_day < -800,
             '2022<0': v.y2022 < 0, 'net@borrow100<=0': v.net_borrow100 <= 0, 'TRAIN sample opposite sign': np.sign(tr.mean_net) != np.sign(v.mean_net),
             'coverage VOID': any(r['void'] for r in cov_rows if r['variant'] == 'Q')}
    log(f'KILL CHECKS {kills} => {"KILL" if any(kills.values()) else "SURVIVE"}')


if __name__ == '__main__':
    main()
