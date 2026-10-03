# DIVE: BT zero picks 9/24, 9/26(weekend), 9/29, 10/2 (2026-10-03)

## Verdict: ZEROS ARE GENUINE ON THE BT'S OWN CANDIDATES (PDR/G1 veto, no refill) - but the BT candidate set differs from the engine's. Parity is NOT proven; 9/29 and 9/30 diverge. No producer defect found in the stage chain; a CANDIDATE-SET/UNIVERSE divergence is open.

## Q1 Producer
ExecStart `/usr/bin/python3 orb_backtest.py`, timer Mon-Fri 16:30 America/New_York (= 20:30 UTC, after close), TimeoutStart 5400 s, MemoryMax 4G.
Journal since 9/23 holds only 9/30, 10/1, 10/2 runs (older rotated). Each: start 20:30:4x, finished 21:05:47 / 21:19:24 / 21:18:32 UTC
(35 / 49 / 48 min), "Deactivated successfully", no ERROR. WARNINGs: ~hundreds of "winner-stack floor fail-open (degenerate/no_atr)" (ATR14 hits 13108/13438),
"missing trading days: 1" each night (fetch of that day, then daily-fill 51-64K rows, intraday-fill 25-36 missing pairs, PM$/news fetched). RESIM n_missing_bars=0.
Ran after the close, bars fetched same run (no early-run issue seen). Book stays "644 picks" 3 nights running: book max date = 9/30 (no 10/1 or 10/2 rows) = those days have 0 picks, indistinguishable from "not computed".
Config printed: N=8, threshold 0.01208, pdr>=11, G1 (rv20 7.106 / pdr 9.226), range-size<=2.221, skip_q1, catalyst_veto=False, dedup family+super_group.

## Q2 Stage table (replica of study_orb_pipeline_static_lock.py main(), live yaml params, features csv 20261002_2057; script in scratchpad stages.py)
date | feat rows | entered=1 | >=thr | not Q1 | top8+dedup | PDR | G1 | range-size | book picks (actual book)
9/22 | 20 | 10 | 11 | 8 | 8 | 3 | 3 | 3 | 3 (CRCA,BIAF,VNCE)
9/23 | 19 | 5 | 6 | 4 | 4 | 1 | 1 | 1 | 1 (GDXD)
9/24 | 12 | 6 | 5 | 3 | 3 | 0 | 0 | 0 | 0
9/25 | 24 | 8 | 13 | 10 | 8 | 3 | 2 | 2 | 2
9/28 | 22 | 13 | 16 | 14 | 8 | 1 | 1 | 1 | 1 (BEZ)
9/29 | 30 | 14 | 19 | 14 | 8 | 0 | 0 | 0 | 0
9/30 | 22 | 4 | 13 | 11 | 8 | 3 | 3 | 3 | 3 (NBIL,CRWG,NEBX; all entered=0 no-fill)
10/1 | 12 | 6 | 7 | 5 | 5 | 1 | 0 | 0 | none (book ends 9/30)
10/2 | 36 | 24 | 27 | 16 | 8 | 0 | 0 | 0 | none
Replica matches the book on every day it has rows. The stage that empties it: the post-ranking PDR veto (prev_day_range <= 11 %), no refill: 9/24 3->0, 9/29 8->0, 10/2 8->0, 10/1 5->1->G1 0.
(9/26 is a Saturday; 9/24 zero is genuine.)

## Q3 Engine vs BT (session_archive, trades.db orb)
9/22: engine CRCA, VNCE traded; BT CRCA, BIAF, VNCE (engine had BIAF scored Q3, not traded).
9/23: GDXD both. 9/24: engine KYTX, BEZ, SKDD all PDR-vetoed -> zero; BT zero (agree, but BT row set has none of KYTX/SKDD as top-3? BT top3 vetoed too).
9/29: engine PDR/G1-vetoed GDXD, JDST, KOLD, BNC, HYLN, SANG, AMPX and SUBMITTED AXTL (7.9 s). BT: zero. AXTL is BT rank 11 (Q2, c=0.177); BT top-8 contained
XNDU, GLWG, KOLD, BNC, BFLY, STAA, SANG, HYLN; engine's top-8 contained GDXD, JDST, AMPX, which are NOT in the BT features. => candidate-set mismatch, so BT zero vs engine 1 pick.
9/30: engine ASTX + AEHG submitted; BT book NBIL, CRWG, NEBX (all no-fill rows). ASTX/AEHG are ABSENT from the 9/30 features (22 rows). Daily-bar gap (open vs prev close) = +1.9 % for both
(ASTX 9.95/9.76, AEHG 9.27/9.10) vs engine min_gap 5 %; engine gap came from resolve_gap_input (minute-bar open / snapshot). NBIL/NEBX were only HOD-dry armed in the engine, never ORB scored.
10/1: engine scored WVE, CBRX, LPA, IBX, RKLX... all vetoed -> zero; BT 0 (agree). 10/2: engine no decision; BT 0.

## Q4 Input comparison (3 names)
AXTL 9/29: BT entry 3.458, gap 7.414, range_total_vol 58,022, range_size 7.288 %, PDR 18.69, rv20 13.96, entered=1; engine entry 3.46, stop 3.21 (range_low), ATR14 0.4914 -> agree on price/stop.
Engine ORB SCORED INFO lines are absent in the 9/29-9/30 archives (only 9/22, 9/24, 10/1-10/2 have them), so gap/RVOL/spread engine values could not be printed.
ASTX, AEHG 9/30: no BT feature row (see above). Spread is not a features column (BT has no spread gate). I could not obtain the engine's logged gap input for ASTX/AEHG (no fallback/gap-input lines in the 9/30 archive).

## Q5 Verdict + actions
Zeros are genuine given the BT inputs (every stage accounted for, replica = book). The defect candidate is upstream of selection: BT candidate set (daily-open gap, cache.db) vs engine candidate set
(minute-bar/snapshot gap input) differ on 9/29 (GDXD, JDST, AMPX engine-only) and 9/30 (ASTX, AEHG engine-only; 10.35 entry vs prev close 9.76 = ~6 % from 09:35 price). Suspect: engine gap measured off a post-open
price/snapshot where BT uses the 09:30 daily open. Unverified; needs engine gap_input dump for ASTX/AEHG (resolve_gap_input, trading/orb_engine.py:1448) vs study_orb_features.py:305.
Spec fix paragraph (not applied): make the parity read compare engine picks to the BT book only after verifying each engine pick has a features row; log the engine's gap_input (open, prev_close, source) at INFO on SCORED lines for 9/29-30 style days.
Parity reader (scripts/eod_sections.py): a BT-zero day where the engine picked a symbol absent from the features CSV must be NO-DATA (candidate-set miss), not zero; days after book max date (10/1, 10/2 now) are NO-DATA until the book has a row or a "ran, 0 picks" marker.
Not done: engine ORB SCORED values for 9/29-30, intraday bars for ASTX/AEHG (cache.db has none for 9/30), older journal runs (rotated).
