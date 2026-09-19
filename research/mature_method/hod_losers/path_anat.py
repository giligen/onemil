#!/usr/bin/env python3
"""hod_losers PART 1 §4-§6 — the loser PATH, the LEVEL, the CONTEXT. Descriptive only."""
import os, sys
import numpy as np, pandas as pd

ROOT = '/home/ec2-user/onemil'
os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_break')
sys.path.insert(0, f'{ROOT}/research/mature_method/hod_filter_stack')
import score as S            # noqa: E402
import score2 as S2          # noqa: E402

D = f'{ROOT}/research/mature_method/hod_losers'
OUT = open(f'{D}/part1b.txt', 'w')
MK = (1, 3, 5, 10, 15, 30)


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); OUT.write(s + '\n'); OUT.flush()


def load():
    pop = S2.load_pop(); S.build_impute(pop)
    pa = pd.read_csv(f'{D}/path.csv', dtype={'symbol': str, 'day': str, 'why2': str},
                     keep_default_na=False, na_values=['']).drop_duplicates(
                         ['day', 'symbol', 'entry_m'])
    out = {}
    for nm, kw in (('B0', S2.BASES['B0']), ('B2', S2.BASES['B2'])):
        sg = S2.sig_set(pop, **kw)
        sg = sg[sg.split.isin(('TRAIN', 'VAL'))]
        sg = sg.merge(pa.drop(columns=['tag', 'break_m']), on=['day', 'symbol', 'entry_m'],
                      how='left')
        bk = S.apply_book(sg, 12, 4)
        out[nm] = (sg, bk)
    return out


def main():
    B = load()
    # ---- parity gate -----------------------------------------------------------------
    sg = B['B0'][0]
    ok = sg.why2.notna()
    p(f'PARITY: path-walk exit vs pop.csv exit — why match {(sg.why[ok] == sg.why2[ok]).mean():.6f}, '
      f'minute match {(sg.exit_m[ok] == sg.exit_m2[ok]).mean():.6f}, rows matched '
      f'{ok.sum()}/{len(sg)}')

    p('\n' + '=' * 100)
    p('4. LOSING TRADES — the path.  W = trade ended > 0R, L = trade ended <= 0R (booked, B0+B2)')
    p('=' * 100)
    for nm in ('B0', 'B2'):
        bk = B[nm][1]
        for sp in ('TRAIN', 'VAL'):
            d = bk[(bk.split == sp) & bk.mfe_r.notna()].copy()
            d['W'] = d.rr > 0
            g = d.groupby('W')
            p(f'\n[{nm} {sp}] n {len(d)}  winners {int(d.W.sum())}  losers {int((~d.W).sum())}')
            p(f'   {"":26s} {"LOSERS":>10s} {"WINNERS":>10s}')
            rows = [('MFE (R)', 'mfe_r', 'median'), ('MFE (R) mean', 'mfe_r', 'mean'),
                    ('minutes to MFE', None, None), ('MAE (R)', 'mae_r', 'median'),
                    ('hold minutes', None, None),
                    ('entry-bar close pos', None, None),
                    ('breakout-bar close pos', 'range_pos', 'median')]
            for lbl, col, how in rows:
                if col is None:
                    if lbl == 'minutes to MFE':
                        a = (d[~d.W].mfe_m - d[~d.W].entry_m).median()
                        b = (d[d.W].mfe_m - d[d.W].entry_m).median()
                    elif lbl == 'hold minutes':
                        a = (d[~d.W].exit_m - d[~d.W].entry_m).median()
                        b = (d[d.W].exit_m - d[d.W].entry_m).median()
                    else:
                        rng = (d.e_high - d.e_low).replace(0, np.nan)
                        pos = (d.e_close - d.e_low) / rng
                        a, b = pos[~d.W].median(), pos[d.W].median()
                else:
                    a = getattr(g.get_group(False)[col], how)()
                    b = getattr(g.get_group(True)[col], how)()
                p(f'   {lbl:26s} {a:10.3f} {b:10.3f}')
            p(f'   {"MFE ladder (median R)":26s}')
            for k in MK:
                p(f'     at +{k:2d} min          {d[~d.W][f"mfe{k}"].median():10.3f} '
                  f'{d[d.W][f"mfe{k}"].median():10.3f}')
            p(f'   {"MAE ladder (median R)":26s}')
            for k in MK:
                p(f'     at +{k:2d} min          {d[~d.W][f"mae{k}"].median():10.3f} '
                  f'{d[d.W][f"mae{k}"].median():10.3f}')
            st = d[d.why == 'stop']
            p(f'   stops {len(st)} ({len(st)/len(d)*100:.0f}%); of them WICK (bar closed back above '
              f'the stop) {st.stop_wick.mean()*100:.0f}%; median MFE before the stop '
              f'{st.mfe_r.median():+.3f}R at +{(st.mfe_m - st.entry_m).median():.0f} min; '
              f'median time to stop {(st.exit_m - st.entry_m).median():.0f} min')
            # the ORB question: does an early no-go predict the loss?
            for k in (3, 5, 10):
                for thr in (0.0, 0.25, 0.5):
                    lo = d[d[f'mfe{k}'] < thr]; hi = d[d[f'mfe{k}'] >= thr]
                    if len(lo) < 20 or len(hi) < 20:
                        continue
                    p(f'     MFE@{k:2d}min < {thr:.2f}R : n {len(lo):4d} gross {lo.rr.mean():+.3f} '
                      f'net {lo.net.mean():+.3f} WR {(lo.rr>0).mean()*100:4.1f}%  |  '
                      f'>= : n {len(hi):4d} gross {hi.rr.mean():+.3f} net {hi.net.mean():+.3f} '
                      f'WR {(hi.rr>0).mean()*100:4.1f}%')
            # the entry bar itself
            dd = d.copy()
            dd['e_pos'] = (dd.e_close - dd.e_low) / (dd.e_high - dd.e_low).replace(0, np.nan)
            dd['d_rev'] = (dd.e_low - dd.next_open) / (dd.next_open - dd.stop)   # R below entry
            for lbl, col, cuts in (('entry-bar close pos', 'e_pos', (0.3, 0.5, 0.7)),
                                   ('breakout-bar close pos (range_pos)', 'range_pos', (0.3, 0.5, 0.7)),
                                   ('entry-bar low, R below entry', 'd_rev', (-0.75, -0.5, -0.25))):
                p(f'   -- {lbl} --')
                for cu in cuts:
                    lo = dd[dd[col] < cu]; hi = dd[dd[col] >= cu]
                    if len(lo) < 20 or len(hi) < 20:
                        continue
                    p(f'     < {cu:+.2f}: n {len(lo):4d} gross {lo.rr.mean():+.3f} net {lo.net.mean():+.3f} '
                      f'WR {(lo.rr>0).mean()*100:4.1f}%  |  >= : n {len(hi):4d} gross {hi.rr.mean():+.3f} '
                      f'net {hi.net.mean():+.3f} WR {(hi.rr>0).mean()*100:4.1f}%')

    p('\n' + '=' * 100)
    p('5. THE LEVEL — touches, consolidation duration, volume, multi-day highs (PRE-BOOK signal set)')
    p('=' * 100)
    for nm in ('B0', 'B2'):
        sg = B[nm][0]
        sg = sg[sg.mfe_r.notna()].copy()
        sg['at_pdh'] = (sg.pdh_ratio >= 1.0).astype(float)
        sg['at_h5'] = (sg.h5_ratio >= 1.0).astype(float)
        for f, cuts in (('touch_n', (1, 2, 3, 5)), ('consol_bars', (5, 8, 12, 20)),
                        ('bar_vol_ratio', (0.5, 1.0, 2.0)), ('at_pdh', (1.0,)), ('at_h5', (1.0,))):
            p(f'\n  [{nm}] {f}')
            for sp in ('TRAIN', 'VAL'):
                d = sg[sg.split == sp]
                line = []
                for cu in cuts:
                    lo = d[d[f] < cu]; hi = d[d[f] >= cu]
                    if len(lo) < 30 or len(hi) < 30:
                        continue
                    line.append(f'>={cu}: n{len(hi)} g{hi.rr.mean():+.3f} vs n{len(lo)} '
                                f'g{lo.rr.mean():+.3f} (sep {hi.rr.mean()-lo.rr.mean():+.3f})')
                p(f'     {sp:5s} ' + ' | '.join(line))
        p(f'\n  [{nm}] distribution: touch_n median {sg.touch_n.median():.0f} '
          f'p90 {sg.touch_n.quantile(.9):.0f}; consol_bars median {sg.consol_bars.median():.0f} '
          f'p90 {sg.consol_bars.quantile(.9):.0f}; at_pdh {sg.at_pdh.mean()*100:.0f}%; '
          f'at_h5 {sg.at_h5.mean()*100:.0f}%')

    # ---- 6. the coh_by_t question -----------------------------------------------------
    p('\n' + '=' * 100)
    p('6. CONTEXT — why coh_by_t was degenerate, and the family/anchor cohort')
    p('=' * 100)
    pop = S2.load_pop()
    c = pd.to_numeric(pop.coh_by_t, errors='coerce')
    p(f'  coh_by_t: cov {c.notna().mean():.3f}  nunique {c.nunique()}  '
      f'value counts {c.value_counts().head(6).to_dict()}')
    OUT.close()


if __name__ == '__main__':
    main()
