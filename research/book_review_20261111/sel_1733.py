"""Cell 1,733 -- ORB selection parity: BT top-K vs live provisional/final/submitted/fills per day.

Read-only. Rebuilds the BT day ranking from the nightly candidate sim with the SAME helpers the pipeline uses
(study_orb_filter.composite_score, study_orb_sizing.assign_quintile, Q_ORDER, family/super-group dedup, top-8),
parses the live journal extracts (scratch dir J) and writes selection_1733.csv + a printed summary.
"""
import glob
import json
import os
import re
import sys

import pandas as pd
import yaml

sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
from trading.orb_csv import read_orb_csv  # noqa: E402
from study_orb_filter import FILTER_FEATURES, composite_score  # noqa: E402
from study_orb_sizing import assign_quintile  # noqa: E402
from study_orb_correlation_filter import symbol_family, symbol_super_group  # noqa: E402

J = sys.argv[1]
OUT = 'research/book_review_20261111/selection_1733.csv'
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
D0, D1 = '2026-09-29', '2026-11-10'
FEAT_LIVE = {'gap': 'gap_pct', 'rtv': 'range_total_volume', 'rabr': 'range_avg_bar_range_pct',
             'rs': 'range_size_pct', 'p20h': 'price_vs_20d_high_pct', 'pdcp': 'prev_day_close_position',
             'rcp': 'range_close_position'}


def load_bt():
    """BT candidate sim scored and ranked exactly like study_orb_pipeline_static_lock (top-K pre-veto)."""
    cfg = yaml.safe_load(open('orb.yaml'))
    feats = cfg['filter']['features']
    params = {f: {'mean': float(feats[f]['mean']), 'std': float(feats[f]['std']), 'sign': int(feats[f]['sign'])}
              for f, _ in FILTER_FEATURES}
    cuts = [float(x) for x in cfg['quintile_cutoffs']]
    thr = float(cfg['filter']['threshold'])
    k = int(cfg['sizing'].get('max_concurrent', 8))
    df = read_orb_csv('analysis_results/orb_features_20261110_2203.csv')
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df = df[(df['date'] >= D0) & (df['date'] <= D1)].copy()
    need = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=need + ['pnl', 'pnl_pct', 'range_size_pct', 'entry_price']).copy()
    df['bt_comp'] = composite_score(df, params)
    df['bt_q'] = assign_quintile(df['bt_comp'], cuts)
    hist = [json.loads(l) for l in open('logs/orb_refit_history.jsonl')]

    def pit(day):
        """The refit record in force on `day` (latest Sunday refit strictly before it)."""
        best = None
        for h in hist:
            if h['ts'][:8] < day.replace('-', ''):
                best = h
        return best

    df['bt_comp_pit'] = float('nan')
    df['bt_q_pit'] = None
    for day, d in df.groupby('date'):
        h = pit(day)
        pp = {f: {'mean': float(h['params'][f]['mean']), 'std': float(h['params'][f]['std']),
                  'sign': int(params[f]['sign'])} for f in params}
        df.loc[d.index, 'bt_comp_pit'] = composite_score(d, pp)
        df.loc[d.index, 'bt_q_pit'] = assign_quintile(df.loc[d.index, 'bt_comp_pit'], [float(c) for c in h['cutoffs']])
    rows = []
    for day, d0 in df.groupby('date'):
        rec = {}
        for tag, cc, qc in (('', 'bt_comp', 'bt_q'), ('_pit', 'bt_comp_pit', 'bt_q_pit')):
            d = d0[(d0[cc] >= thr) & (d0[qc] != 'Q1')].copy()
            d['_qr'] = d[qc].map(Q_ORDER)
            d = d.sort_values(['_qr', cc], ascending=[True, False])
            seen_f, seen_s, n = set(), set(), 0
            for rank, (_, r) in enumerate(d.iterrows(), 1):
                fam, sup = symbol_family(r['symbol']), symbol_super_group(r['symbol'])
                dup = bool((fam and fam in seen_f) or (sup and sup in seen_s))
                top = False
                if not dup and n < k:
                    n += 1
                    top = True
                    if fam:
                        seen_f.add(fam)
                    if sup:
                        seen_s.add(sup)
                rec.setdefault(r['symbol'], {})[f'bt_rank{tag}'] = rank
                rec[r['symbol']][f'bt_topk{tag}'] = top
                rec[r['symbol']][f'bt_dedup{tag}'] = dup
        for sym, v in rec.items():
            rows.append(dict(date=day, symbol=sym, **v))
    rk = pd.DataFrame(rows)
    df = df.merge(rk, on=['date', 'symbol'], how='left')
    return df, thr, cuts, params


def parse_live():
    """Per-day live selection facts from the journal extracts."""
    rx_sc = re.compile(r'(\d\d):(\d\d):(\d\d) \|.*ORB SCORED: (\S+) comp=([-\d.]+) (Q\d) \| gap=([-\d.]+) '
                       r'rtv=(\d+) rabr=([-\d.]+) rs=([-\d.]+) p20h=([-\d.]+) pdcp=([-\d.]+) rcp=([-\d.]+) .*pool=(\S+)')
    rx_bt = re.compile(r'ORB: (\S+) below filter threshold — comp=([-\d.]+)')
    rx_pv = re.compile(r'provisional top-\d+ @ [\d:.]+ ET: (.*)$')
    rx_fn = re.compile(r'reconcile final top-\d+ of (\d+) ranked: \[(.*?)\]')
    rx_sub = re.compile(r'(\d\d:\d\d:\d\d) \|.*ORB ENTRY SUBMITTED: (\S+) ')
    rx_veto = re.compile(r'\[ORB\] (PDR VETO|G1 VETO|RANGE-SIZE VETO):? (\S+?):? ')
    live = {}
    for f in sorted(glob.glob(os.path.join(J, '2026-*.txt'))):
        day = os.path.basename(f)[:10]
        L = {'scored': {}, 'below': {}, 'prov': [], 'final': [], 'n_ranked': None, 'sub': {}, 'veto': {}}
        for line in open(f, errors='replace'):
            m = rx_sc.search(line)
            if m:
                hh = int(m.group(1))
                if hh <= 15:  # opening window only (13:3x EDT / 14:3x EST UTC)
                    L['scored'][m.group(4)] = dict(live_comp=float(m.group(5)), live_q=m.group(6),
                                                   **{f'live_{k}': float(m.group(7 + i)) for i, k in
                                                      enumerate(FEAT_LIVE)}, live_pool=m.group(14))
                continue
            m = rx_bt.search(line)
            if m:
                L['below'][m.group(1)] = float(m.group(2))
            m = rx_pv.search(line)
            if m:
                L['prov'] = re.findall(r'(\S+?)\(rh=', m.group(1))
                L['prov'] = [s.strip(', ') for s in L['prov']]
            m = rx_fn.search(line)
            if m:
                L['n_ranked'] = int(m.group(1))
                L['final'] = re.findall(r"'(\S+?)'", m.group(2))
            m = rx_sub.search(line)
            if m:
                L['sub'].setdefault(m.group(2), m.group(1))
            m = rx_veto.search(line)
            if m and 'provisional' not in line:
                L['veto'][m.group(2)] = m.group(1).split()[0]
        live[day] = L
    return live


def main():
    bt, thr, cuts, params = load_bt()
    book = pd.read_csv('analysis_results/orb_bplus_book.csv')
    book = book[(book.date >= D0) & (book.date <= D1)][['date', 'symbol', 'entered', 'exit_reason', '_sized_pnl']]
    book = book.rename(columns={'entered': 'bt_entered', 'exit_reason': 'bt_exit', '_sized_pnl': 'bt_pnl'})
    book['bt_book'] = True
    fills = pd.read_csv('research/book_review_20261111/stops_1732.csv')
    fills = fills[['date', 'symbol', 'pool', 'fill', 'fill_min_et', 'live_exit_reason', 'live_pnl']]
    fills['live_fill'] = True
    live = parse_live()
    lrows = []
    for day, L in live.items():
        syms = set(L['prov']) | set(L['final']) | set(L['sub'])
        for s in syms:
            r = dict(date=day, symbol=s, live_prov=s in L['prov'],
                     live_final_pos=(L['final'].index(s) + 1) if s in L['final'] else None,
                     live_sub_ts=L['sub'].get(s), live_veto=L['veto'].get(s), live_n_ranked=L['n_ranked'])
            r.update(L['scored'].get(s, {}))
            lrows.append(r)
    lv = pd.DataFrame(lrows)
    keys = pd.concat([bt[(bt['bt_topk'] == True) | (bt['bt_topk_pit'] == True)][['date', 'symbol']], book[['date', 'symbol']],  # noqa: E712
                      lv[['date', 'symbol']], fills[['date', 'symbol']]]).drop_duplicates()
    cols = ['date', 'symbol', 'bt_rank', 'bt_topk', 'bt_dedup', 'bt_comp', 'bt_q', 'bt_rank_pit', 'bt_topk_pit',
            'bt_comp_pit', 'bt_q_pit', 'gap_pct', 'range_size_pct',
            'entry_price', 'prev_day_range_pct', 'range_total_volume', 'range_avg_bar_range_pct',
            'price_vs_20d_high_pct', 'prev_day_close_position', 'range_close_position', 'entered', 'exit_reason']
    out = keys.merge(bt[cols].rename(columns={'entered': 'sim_entered', 'exit_reason': 'sim_exit'}),
                     on=['date', 'symbol'], how='left')
    out['bt_feat_present'] = out['bt_comp'].notna()
    out = out.merge(book, on=['date', 'symbol'], how='left').merge(lv, on=['date', 'symbol'], how='left')
    out = out.merge(fills, on=['date', 'symbol'], how='left')
    # live scored for every name (also BT-only ones not in prov/final/sub)
    for i, r in out.iterrows():
        L = live.get(r['date'])
        if L is None:
            continue
        sc = L['scored'].get(r['symbol'])
        if sc and pd.isna(r.get('live_comp')):
            for k2, v in sc.items():
                out.at[i, k2] = v
        if pd.isna(r.get('live_comp')) and r['symbol'] in L['below']:
            out.at[i, 'live_comp'] = L['below'][r['symbol']]
            out.at[i, 'live_q'] = 'below_thr'
        if pd.isna(r.get('live_veto')) and r['symbol'] in L['veto']:
            out.at[i, 'live_veto'] = L['veto'][r['symbol']]
        out.at[i, 'live_log'] = True
    for b in ['bt_book', 'live_fill', 'live_prov']:
        out[b] = out[b].astype(object).where(out[b].notna(), False).astype(bool)
    out['bt_fill'] = out['bt_entered'].fillna(0).astype(int) == 1
    out['comp_diff'] = out['live_comp'] - out['bt_comp']
    out['comp_diff_pit'] = out['live_comp'] - out['bt_comp_pit']
    # largest per-feature z-contribution difference live - BT
    def top_feat(r):
        best, bv = None, 0.0
        for k2, f in FEAT_LIVE.items():
            lvv, btv = r.get(f'live_{k2}'), r.get(f)
            if pd.isna(lvv) or pd.isna(btv):
                continue
            dz = (lvv - btv) / params[f]['std'] * params[f]['sign'] / len(params)
            if abs(dz) > abs(bv):
                best, bv = f, dz
        return f'{best}:{bv:+.3f}' if best else None
    out['top_feat_dz'] = out.apply(top_feat, axis=1)
    out = out.sort_values(['date', 'bt_rank', 'symbol'])
    out.to_csv(OUT, index=False)
    print(f'wrote {OUT} rows={len(out)} thr={thr} cuts={cuts}')
    show = ['date', 'symbol', 'bt_rank', 'bt_topk', 'bt_q', 'bt_rank_pit', 'bt_topk_pit', 'bt_q_pit', 'bt_book', 'bt_exit', 'bt_pnl',
            'live_comp', 'live_q', 'live_prov', 'live_final_pos', 'live_n_ranked', 'live_sub_ts', 'live_veto',
            'live_fill', 'live_pnl', 'pool']
    div = out[(out['bt_fill'] != out['live_fill']) | (out['live_sub_ts'].notna() & ~out['live_fill'])]
    pd.set_option('display.width', 320)
    print(div[show].to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
