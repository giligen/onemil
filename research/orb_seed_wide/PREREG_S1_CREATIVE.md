# PREREG — S1 "attention, supply, conditions" pass + the combined-seed rung. Cells 1,322–1,327

Owner 2026-09-21: "think on how to make this positive, be creative". Committed BEFORE scoring. Population and
book as in `PREREG_S1_EXIT.md` (S1 = gap 3–5 %, open $3–30, production selection, 8 slots, R = $375, cost M;
book `out/runS1_true.csv`, entered rows: TRAIN 210, VAL 141). TEST sealed.

## Why these and not more loser-anatomy cuts
`REPORT_S1_FILTERS.md`: cuts learned from S1's own TRAIN losers do not transfer. The features below are NOT
from that scan; each is a canonical gapper-conviction mechanism (Cameron: attention, pre-market participation,
supply) or an owner-requested condition (regime / day), all knowable at 09:30 ET.

| cell | rule (frozen) | mechanism | data |
|---|---|---|---|
| 1,322 GAP-RANK | keep iff the symbol is in the day's **top-10 by gap_pct** among the wide universe that day (gap ≥ 3 %, $3–50, prev vol ≥ 500K) | pre-market gainer lists are rank-ordered; a 3–5 % gap on a quiet day gets the crowd's attention, on a busy day it does not | wide features CSV |
| 1,323 PM-$VOL | keep iff pre-market dollar volume (04:00–09:30 ET) ≥ the **TRAIN median** of the S1 book (fixed once from TRAIN, applied to VAL) | a 3–5 % gap without pre-market participation is drift, with it a catalyst | `scripts/orb_premarket_backfill.py` into `intraday_bars_1min`; **VOID if coverage < 80 % or winner/loser missingness gap > 5 pp** |
| 1,324 RVOL-5m | keep iff `range_total_volume / avg_daily_volume_20d` ≥ TRAIN median | first-5-minute relative volume = in-play (Zarattini's RVOL gate) | book columns |
| 1,325 HMM-CALM | keep iff the SPY HMM state on the trade date is the calm state (lowest mean vol20 of the 3 states; labels causal, fit 2025) | owner: skip weeks likely to lose; S1 breakouts need a quiet tape | `research/regime/hmm_labels.csv` |
| 1,326 SPY-GAP-UP | keep iff `spy_gap_pct ≥ 0` | beta tailwind for a breakout | book column. **Disclosure:** seen in the TRAIN anatomy (tercile t 2.4) — a peeked feature; it must clear the full bar incl. both halves |
| 1,327 COMBINED SEED | pipeline walk of production ∪ S3 ∪ S1(gap ≥ 4) = rows with (open ≤ 30 & gap ≥ 4) or (30 < open ≤ 50 & 3 ≤ gap < 5), production parameters, 8 shared slots | the frequency rung: S1 pays only as an add-on if it does not displace production picks | wide features CSV → `out/runCOMB_features.csv` → `out/runCOMB_true.csv` |

Diagnostics (report-only, never pass/fail): float_shares from `cache.db::universe` (current, NOT point-in-time);
price sub-buckets $3–10 / $10–30; top-5 / top-20 rank variants.

## Pass bar
Filter cells 1,322–1,326 (each): kept mean R ≥ +0.15 TRAIN and VAL; VAL day-clustered t ≥ 2; TRAIN halves ≥ 0;
dropped cohort mean R < 0 both splits; kept VAL fills/wk ≥ 3 (C5). **Exploration tier** (owner 9/18, reported
separately, not a pass): kept R > 0 on TRAIN, both halves and VAL, ≥ 3 fills/wk, dropped cohort ≤ 0 on VAL.
Cell 1,327: VAL and TRAIN total $ ≥ `runB_true` (production alone) and weekly MDD ≤ 1.25× it; ≥ 90 % of
production's filled (date, symbol) preserved; fills/wk ≥ 3; cadence block printed.

## Decision rule
A filter pass → independent rebuild, then it becomes a seed rule in a live PREREG. Exploration-tier only → put in
front of the owner as such, with the quarter's resolving power. 1,327 pass → the live seed widening PREREG
(engine universe query + config + tests + dry day). Nothing passes → S1 closed as a standalone; frequency comes
from S3 + the corpse-gate fix (9/21), which itself widens the live universe tomorrow.

## Not allowed
Re-cutting thresholds after VAL; combining cells; scoring float or price buckets as cells; touching orb.yaml.
Scorer `score_s1_creative.py`; builder `build_combined_seed.py`.

## Addendum 2026-09-21 (after 1,327 was scored — disclosed as a follow-up, cell 1,328 PRIORITY SLOTS)
1,327 failed on the pre-registered "production picks preserved ≥ 90 %" (78 % / 67 %) and TRAIN MDD, while $ rose
on both splits and the added trades were positive on both. The failure mode is slot competition: the wider pool
displaces production picks in the shared ranking. Cell 1,328 changes ONLY the allocation rule, not selection or
exit: **production candidates (gap ≥ 5 %, open $3–30) have first claim on the 8 slots; add-on candidates (S3 and
gap 4–5 %) fill leftover slots that day in composite order.** Reconstructed from the two walked books
(`runB_true` for production picks — they had all 8 slots alone — and `runCOMB_true` for the add-on picks, each
trade's own walked `_sized_pnl`; the compounding-sizing drift is the known approximation, reported). Pass bar:
production picks preserved = 100 % by construction; total $ ≥ production on both splits; weekly MDD ≤ 1.25×
production on both splits; fills/wk ≥ 3; added cohort mean R > 0 on both splits. A pass → pipeline knob
`ORB_BT_PRIORITY_STRATUM` + live implementation under a live PREREG, never a live change from this
reconstruction. Scorer `score_priority.py`.
**Pre-scoring correction (same day, before `score_priority.py` was run):** the walked books show at most 4
(production) / 6 (combined) picks per day — the 8 slots never bind, so "slot competition" is NOT the displacement
mechanism; a pool-dependent veto is (identified in the report). Cell 1,328 is therefore scored as what the
reconstruction actually is: the **UNION of two independently evaluated pools** (production rules on the production
pool, unchanged; the add-on pool evaluated on its own), sharing 8 slots. Same pass bar. That is also the only
live design that leaves the production book untouched.
