# DECOMP.md -- the owner's question: "entering an R earlier"

PREREG.md "Exhibits first" #2. Base entries (the signal shared by cells 1,400-1,402) vs a same-name-day counterfactual HOD-break entry (next open after the first close>H_t following the base entry, same stop B_t, same C1 target+2R exit). TRAIN/VAL only, TEST sealed. Both legs use the PROXY cost (no measured NBBO for these minutes).

## TRAIN (n=3958)

- Share of base entries whose stock later closes above H_t (breaks): 76.1%
- Mean C1 net R, base entries that break: 0.254 (n=3012)
- Mean C1 net R, base entries that never break: -1.060 (n=946)
- Mean C1 net R, ALL base entries (unconditional -- prices in the failed bases): -0.060 (n=3958)
- Breaking subset ONLY, paired same-name-day comparison (n=2999): base-entry C1 R = 0.252 vs HOD-break-equivalent-entry C1 R = -0.001 -- "entering an R earlier" delta = 0.253 R

## VAL (n=2289)

- Share of base entries whose stock later closes above H_t (breaks): 81.6%
- Mean C1 net R, base entries that break: 0.390 (n=1868)
- Mean C1 net R, base entries that never break: -1.127 (n=421)
- Mean C1 net R, ALL base entries (unconditional -- prices in the failed bases): 0.111 (n=2289)
- Breaking subset ONLY, paired same-name-day comparison (n=1842): base-entry C1 R = 0.383 vs HOD-break-equivalent-entry C1 R = 0.065 -- "entering an R earlier" delta = 0.318 R


Stop-at-open base entries (entry bar opened at/below the base low) are excluded from this exhibit and scored in the cells at -cost: {'TRAIN': 0, 'VAL': 0}.
