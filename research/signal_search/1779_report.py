"""Write RESULT_1779.md from 1779_summary.csv and 1779_placebo.csv (PREREG_1779 Output)."""
import pandas as pd
OUT = '/home/ec2-user/onemil/research/signal_search'
S = pd.read_csv(f'{OUT}/1779_summary.csv'); PC = pd.read_csv(f'{OUT}/1779_placebo.csv').set_index('rung')
g = lambda r, s, c: S[(S.rung == r) & (S.split == s)][c].iloc[0]
L = ['# RESULT 1,779 GAPSHORT frequency frontier (marginal rows per rung, VAL 2026-01..05, cost 40.8 bp stack)', '',
     '| rung | names/wk added (med) | names/day after (med) | VAL net bp | day-t | ex-top-5% | worst-5% | wkP10 comb (R) | placebo p95 (i / ii) | PASS |', '|---|---|---|---|---|---|---|---|---|---|']
for r in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
    pl = f"{PC.loc[r,'pl1_p95']:+.0f} / {PC.loc[r,'pl2_p95']:+.0f}" if r in PC.index else 'n/a (VAL<=0)'
    L.append(f"| {r} | {g(r,'VAL','names_wk_med'):.1f} | {g(r,'VAL','names_day_after_med'):.0f} | {g(r,'VAL','mean_net'):+.1f} | {g(r,'VAL','day_t'):.2f} | "
             f"{g(r,'VAL','ex_top5'):+.0f} | {g(r,'VAL','worst5_pct'):.1f}% | {g(r,'VAL','wkP10_comb'):.2f} | {pl} | FAIL |")
L += ['', 'Pass bar: net >= +50 bp, t >= 2, ex-top-5% > 0, worst-5% >= -12%, both TRAIN halves same sign, beats both p95s, >= 3 names/wk. No rung passes (every rung fails net, ex-top-5% and worst-5%).', '',
      '## TRAIN reads (marginal rows)', '| rung | TRAIN net | n | H1 net | H2 net | same sign |', '|---|---|---|---|---|---|']
for r in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
    h1, h2 = g(r, 'TRAIN_H1', 'mean_net'), g(r, 'TRAIN_H2', 'mean_net')
    L.append(f"| {r} | {g(r,'TRAIN','mean_net'):+.1f} | {g(r,'TRAIN','n_valid'):.0f} | {h1:+.1f} | {h2:+.1f} | {'yes' if h1*h2>0 else 'no'} |")
L += ['', '## Details (VAL)', '| rung | rows | valid | coverage | miss gap pp | names/wk p10 | rung wkP10 R | combined wk mean R |', '|---|---|---|---|---|---|---|---|']
for r in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']:
    L.append(f"| {r} | {g(r,'VAL','n_rows'):.0f} | {g(r,'VAL','n_valid'):.0f} | {g(r,'VAL','coverage'):.0%} | {g(r,'VAL','miss_gap_pp'):.1f} | {g(r,'VAL','names_wk_p10'):.1f} | {g(r,'VAL','wkP10_rung'):.2f} | {g(r,'VAL','wk_mean_comb'):.2f} |")
L += ['', f"R5 secondary (cover at measured ask1100, rows with it): n={g('R5','VAL','n1100'):.0f} mean {g('R5','VAL','net1100_mean'):+.1f} bp; R6 n={g('R6','VAL','n1100'):.0f} mean {g('R6','VAL','net1100_mean'):+.1f} bp.", '',
      '## Notes / caveats',
      '* Base book = 1776R book (1,087 rows, reproduced exactly). L1 tercile cutoffs follow the 1776R convention (ALL TRAIN population rows, not top-10 rows): T3 >= 7,224,163 (matches 1776R); T2 >= 945,387. Top-10 TRAIN-only T3 cutoff would be 16,970,246 (not used).',
      '* Quote source: quotes.jsonl first, else 1774 bid/ask1030 (agreement on 1,085 overlapping rows: 100 % exact); ~656 missing rows fetched (appended to quotes.jsonl). Rows with no/invalid quote are dropped (coverage column).',
      '* Placebo run only for rungs with VAL mean > 0 (R1 only). Placebo (i) pool = a 600-row sample of same-day non-base non-rung quoted VAL rows (not the full population); (ii) = same symbol +-5 trading days. 200 draws.',
      '* Rungs R5/R6 enter at bid - 10 bp, cover at the close; rows lacking `close` are invalid. R5/R6 VAL means are negative before placebos.',
      '* Gap rank is computed on the 1776R universe (population CSV + daily_bars gap, gap <= 200 %), identical to the base book definition.',
      '', 'CHILDREN: (1) is the R1 +3 bp mean a placebo-level result, i.e. is any L1 gating adding information beyond gap rank; (2) short side of the 10:30 extended survivors is negative, so test the continuation (long/hold-through) mirror only under its own PREREG with GROSS edge first; (3) frequency from other signal families (non-gap) rather than widening gap/L1 gates.']
open(f'{OUT}/RESULT_1779.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
