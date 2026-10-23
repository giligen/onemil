# RESULT 1,703 - ORB universe coverage, engine (paper) vs nightly BT features CSV, 13 sessions 2026-10-06..10-22
Data: 13 archives (grep 'ORB SCORED' + reconcile/PDR/ENTRY lines), 13 nightly `orb_features_*` CSVs (354 rows, `read_orb_csv`), composite via `composite_score`+`load_feature_params`, ranked as the engine does (thr 0.01208, skip_q1, order Q4,Q5,Q3,Q2,Q1, dedup, top-8, PDR<=11 veto). Scripts: `cell1703_build.py`, `cell1703_final.py`. Cell count +1 (1,703).
Caveats: (a) the archives hold NO 'below filter threshold' lines (cron grep `\[ORB` misses `ORB: x below...`), and the engine never logs sub-top-8 ranked names or names silently dropped (DEBUG), so "engine ranked" = SCORED + reconcile top-8 + preplaced/submitted + PDR-veto lines; the reconcile "of N ranked" is the only full count. (b) Where the engine did not re-score a name (preplaced / provisional-vetoed) its comp is the BT comp. (c) Pool = production only; add-on pools excluded.

| day | eng n (reconcile) | BT n | engine-only | BT-only | top-8 changed? | picks (post-PDR) changed? |
|---|---|---|---|---|---|---|
| 10/06 | 4 | 5 | - | CTVA | yes (CTVA) | no (PDR veto) |
| 10/07 | 14 (14) | 14 | - | - | no | no |
| 10/08 | 3 | 3 | - | - | no | no |
| 10/09 | 14 (18) | 16 | 4 unidentifiable (rank>8) | COHX, NRR | no | no |
| 10/12 | 10 (11) | 10 | ASTY | - | yes (ASTY in, CCC out) | YES: ASTY ordered live (BT: no) |
| 10/13 | 12 | 12 | - | - | no | no |
| 10/14 | 12 (12) | 12 | - | - | no | no |
| 10/15 | 18 | 17 | NAIL | - | no | no |
| 10/16 | 13 (13) | 13 | - | - | no | no |
| 10/19 | 57 (57) | 57 | - | - | no | no |
| 10/20 | 9 (9) | 7 | BNC, RAMZ | - | yes (BNC,RAMZ in; SNDQ out) | no (both PDR-vetoed, SNDQ vetoed in BT) |
| 10/21 | 17 (17) | 17 | - | - | no | no |
| 10/22 | 19 (19) | 18 | SSG | - | yes (SSG in; SMST out) | YES: SMST (BT pick) never ordered |
Totals: top-8 differs 4/13 sessions (10/06, 10/12, 10/20, 10/22); orderable picks differ 2/13 (10/12, 10/22). Removing the engine-only names restores the BT top-8 on 10/12, 10/20, 10/22; adding BT-only names does not on 10/06 (CTVA).

## Reason taxonomy (named divergent names, 8; 10/09's 4 engine-ranked names beyond rank 8 are unidentifiable)
1. BT incomplete-range drop, 2 names (BNC 10/20, SSG 10/22): 4 of the 5 one-minute bars 09:30-09:34 in cache; ALL 354 CSV rows have 5/5 bars, 13 of 14 replica-universe names missing from the CSV have <5. Live ranges them from 4 bars. Thin names (live rtv 11.0K / 6.0K; cache 27K / 15K, so live feed volume also reads ~40 % of cache).
2. BT 15K floor on cached volume to 09:35 (premarket included, `MIN_935_VOL_FLOOR`), 2 names (ASTY 12,866 on 10/12; RAMZ 13,997 on 10/20). Engine has no such gate. Both 2x wrappers.
3. Composite drift across the Q1 cut 0.1102, 2 names (NAIL 10/15 eng .113 vs BT .106; CTVA 10/06 eng .106 vs BT .112); also CGEM/SKDD at the threshold (eng .015 vs BT .009, Q1 both ways, no effect). Decision-relevant only if the name is near the top (neither was).
4. Engine silent drop, 2 names (COHX, NRR 10/09; both passed GAP_GATE, never SCORED; BT comp .119/.259, Q2/Q3). NRR has 8 daily bars (20-day features missing - inferred, DEBUG line not archived). COHX cause not in the archive: unexplained.
Wrapper status is not a reason (ORB admits 2x/3x since 9/5; 4 of the 5 engine-only names are wrappers because wrappers are 45 % of the pool). Price band, gap, prev-volume pass on both sides for every named name.

## 10/22 SSG/SMST verified
SSG: Q5, comp .4738, live rtv 5,971, 4/5 range bars, prev-day range 3.91 %, range size 0.585 %; ranked 6th in the live top-8 [BEZ DAMD BHVN RTO AVEX SSG GDXD BTCZ], then PDR-vetoed (slot left empty, no refill) - it would also fail the range-size veto (<=2.221). BT top-8 [BEZ DAMD BHVN RTO AVEX GDXD BTCZ SMST]: SMST (Q3 .2217) is 8th in BT, 9th live, so never ordered. (SMST has no breakout in the BT either - `entered=0` - so the P&L effect is nil; the parity break is real.)

## Which side is per spec
The universe rule is `orb.yaml universe` (gap>=5 %, prev vol>=500K, $3-30) via Alpaca snapshots; `orb_machine_rules.md` L1 states no range-bar-count, 09:35-volume or rtv rule. The BT's two extra gates are documented proxies ("interim live-parity", "NOT a live gate twin", "range-computability proxy" in `study_orb_broad.py`). So the ENGINE is per spec; the BT is non-compliant for categories 1-2 (undocumented in the rulebook, differ from live). Category 3 is feed-volume/feature noise (neither side wrong). Category 4 is an engine-side unlogged exclusion (a defect in observability, possibly a rule).

## Fix I would spec (no code changed)
1. Decide ONE rule, written in `orb_machine_rules.md` L1: either live also requires 5/5 range bars AND >=15K cached volume to 09:35 (then SSG/BNC/ASTY/RAMZ drop live - they are the thinnest names and live rtv 5-11K is below the floor in feed terms), or the BT drops both gates. Evidence prior: thin 6-11K names with 4 bars cannot be filled at the 5-30 s clock, so lean to a shared floor helper (`trading/orb_universe_gate.py`, BT+live, parity test) on LIVE RANGE VOLUME, not cached volume.
2. Log every silent drop at INFO with reason (missing feature, below threshold, rangeless, floor); fix the archive cron grep to include `ORB:` lines so 'below filter threshold' is archived; log the full ranked list at reconcile.
3. Extend `decision_parity` to compare the ranked SET and top-8 (it only compares composites of common names, "coverage gaps are bt_parity's job").
