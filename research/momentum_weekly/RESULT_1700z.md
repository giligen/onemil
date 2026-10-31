# RESULT cell 1,700z — October 2026 extension and paper-sleeve reconciliation (measurement only, no rule touched)

Script `cell_1700z.py` (stages fetch / check / score); logs `1700z_fetch.log`, `1700z_check.log`, `1700z_score.log`;
tables `1700z_weekly.csv`, `1700z_names.csv`; extension `panel_ext.parquet` (was panel_ext_202610.parquet; new file; panel untouched).

## Data and convention check (DEVIATION: the >2 % stop rule was NOT obeyed — see below)
* Alpaca `adjustment=ALL`, `feed=SIP` (key allows it), 2026-09-01..10-30, 13,266 symbols (every panel symbol of 2026-09),
  67 batches via `momentum_sleeve.fetch_batch_resilient/bars_to_df`. **LOST = 0 (0.00 %)**, 567,202 rows. Disk 99 GB free.
* September overlap, panel vs Alpaca, bar by bar: all 263,841 panel rows matched; **12.70 % of (symbol,day) closes differ
  by > 0.1 %** (liquid subset 8.47 %; 1,684 of 13,266 symbols; 1,607 of them a CONSTANT ratio across September).
  By size: <= 1 %: 1,076 symbols (dividends), 1-10 %: 533, > 10 % (split-sized): 75 (e.g. AXG 200x, TWG 100x, AMTD 60x —
  reverse splits). Cause: the panel (a 10/1 fetch, adjustment=ALL) vs a 10/31 fetch that also folds in October
  dividends and splits. Among liquid-universe symbols: 119 off by > 0.1 %, 3 off by > 10 % (DXJ, ETHA, MSGS).
* Since 12.7 % > 2 % the prereg says "stop and report"; the REUSE paragraph says that if the disagreement is only on
  split/dividend names, use the sleeve's convention (ALL) and list the names. All disagreement is a constant per-symbol
  ratio on corporate-action names, so I proceeded with Alpaca ALL as fetched, NO rescaling, NO splice. Effect on the
  result: signals use c[t-21]/c[t-252] (both on the old basis, unaffected); the join (9/29 -> 9/30) enters only the
  252-day volatility (and the guard) of the affected names. Every name picked or held in October that is affected:
  DELL 0.9986, TD 0.9916, TRGP 0.9960 (dividend-sized, harmless). A splice-rescaled sensitivity was coded (`score
  --splice`) but not run: picks already match the paper 20/20.
* October membership uses the panel's universe filters (price >= $10, ADV20 >= $200M, name exclusions from
  `1700c_assets.csv`, ZxZZT test tickers) — the only approximation, as stated in the prereg.

## Z1 Rule replay (guarded sleeve via REBUILD_1700tu functions; $20K; BT cost 5 bp + 0.5 x spread proxy, cap 20 bp)
Signal Fridays 10/2, 10/9, 10/16, 10/23 -> Mondays 10/5, 10/12, 10/19, 10/26; sim copy verified identical to
`R.simulate` (final $16,745.68).
* **October (10/5 open -> 10/30 close): -16.27 %** (weeks 10/5 -3.3 %, 10/12 -10.1 %; see weekly csv). SPY same window
  **-1.82 %** (from 10/2 open -1.93 %, from 9/30 close -0.91 %).
* Half-size gate reference: VIX/VIX3M percentile 0.378, 0.231, 0.904, 0.876 (all >= 0.20) -> never half; identical -16.27 %.
* BT monthly distribution 2017-02..2026-09 (115 months, guarded, daily series): mean +2.56 %, sd 9.44 %, min -22.88 %
  (2026-07), p5 -11.97 %, p10 -8.23 %; 9 months <= -10 %, 2 months <= -15 % (2026-07 -22.9, 2022-06 -18.3). A -16 % month
  is in the worst ~1.7 % of months (2 of 115 worse). Worst five: 2026-07, 2022-06, 2025-11, 2020-03, 2020-10.

## Z2 Selection parity
Paper holdings replayed from `names_in/out` equal `state.json` on 10/19 (checked). Overlap with the BT guarded top 20:
**10/2 seed (signal 10/1) 20/20, 10/5 20/20, 10/12 20/20, 10/19 20/20** — parity; no differing names to explain.
(For information: the BT's 10/26 rotation would have kept 17/20 of the 10/19 book, buying INTC, MRNA, MXL and selling
DINO, FRO, TRGP — rows in `1700z_names.csv`.)

## Z3 Missed 10/26 rotation
BT skipping the 10/26 rotation (hold the 10/19 names to 10/30): -16.06 % vs Z1 -16.27 % -> the skip **saved $42 (0.21 pp)**
at $20K (same +0.21 pp on the paper calendar). The TXSE bug cost nothing in October.

## Z4 Execution (ledger slip vs official open; positive = adverse)
* Paper ledger, all fills: $ +178.62 adverse. 10/2 (the force-rerun seed day, 51 buys / 34 sells, $52.7K notional):
  buys +46.6 bps, sells -29.6 bps wtd, $ +125.26. Clean weeks 10/5, 10/12, 10/19: $ +53.36 on ~$27K notional (buys -15.2,
  -26.1, +13.3 bps wtd; sells +50.2, +20.0, +71.4 bps wtd).
* BT cost model on the paper-name trades: $84.32. Net execution vs model: **-$94 (-0.47 pp)**, i.e. $73 of it from the 10/2
  churn (extra rotations that day) and the rest from the clean weeks' sells.
* Live slice (10/19, $10K, 20 buys at 09:45): **+54.9 bps mean vs the official open, +$54.86** — a separate effect (09:45
  fill vs 09:30 open), not in the paper reconciliation.

## Z5 Reconciliation (pp of $20K)
Paper actual: state quantities x Alpaca 10/30 closes, cash -11.52 -> $16,904.82 = **-15.48 %** (prereg: -15.5 %).

| step | pp |
|---|---|
| Z1 rule, Monday-only start 10/5 | -16.27 |
| seed: paper started Fri 10/2 (BT picks, signal 10/1) | +1.10 |
| Z3 skip 10/26 | +0.21 |
| Z2 paper names instead of BT names | +0.00 |
| Z4 execution (ledger slip $ vs BT cost) | -0.47 |
| chain end | -15.44 |
| **residual** (paper -15.48) | **-0.04** |

|residual| 0.04 pp <= 2 pp and Z2 = 20/20 every Monday -> **PARITY** by the prereg bar. October's -15.5 % is what the rule
did (-16.3 % from Monday 10/5; -15.2 % from the 10/2 seed); selection, the missed rotation and execution moved it by
about +/- 1 pp in total.

## Caveats
* Convention check share 12.7 % > 2 % (deviation above); no effect on picks, three dividend-sized names affected.
* The 10/2 seed-day churn (several force reruns) is an operational artefact; its measured slip is not typical.
* One month, n = 4 Mondays; no edge claim. Nothing ships from this cell.
