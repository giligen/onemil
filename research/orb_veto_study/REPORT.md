# ORB V1 veto study — REPORT

## Run 1 (2026-09-06 21:20 UTC) — INVALID, discarded
The first pass used `research/orb_signal_study/features_base.csv` as the exit-resim dump. Its `pnl` is the features CSV's fixed +2R / −1R exit (the trap CLAUDE.md warns about), not the shipped static lock: the baseline came out $3,244 / 130 picks against the honest $6,085 / 130, with `exit_reason=target` rows. The pipeline accepted it because the keys matched. Nothing from that pass is evidence. Fix: a full bar-walk dump with `ORB_BT_DUMP_CANDIDATES` (static-lock exits for every candidate), baseline must reproduce $6,085 / 130 before any veto row counts. Era columns now read `_sized_pnl` (stage scale, same as the monthly table).

## Run 2 (2026-09-06 20:50 UTC) — VALID: static-lock dump, baseline reproduces the honest book
Baseline through `candidates_static_lock_dump.csv`: **$6,085 / 130 picks / 73 fills / MDD −$685 / 6 red months / worst −$236** = the honest 9/5 reference. Stage sizing from orb.yaml ($10K / 3 / $375). Pass rule as pre-registered (DESIGN.md).

| run | picks | fills | total | MDD | red | worst mo | 25H1 | 25H2 | 2026 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 130 | 73 | 6,085 | −685 | 6 | −236 | 2,442 | 1,135 | 2,509 | |
| V1a range_size ≤ 2.221 | 120 | 70 | 6,256 | −620 | 6 | −185 | 2,442 | 1,241 | 2,574 | **KEEP** |
| V1b adr20 ≤ 5.01 | 123 | 69 | 6,425 | −551 | 6 | −185 | 2,442 | 1,341 | 2,643 | **KEEP** |
| V1c spy_3d ≥ 1.484 | 118 | 67 | 5,675 | −647 | 6 | −236 | 2,131 | 1,258 | 2,286 | REJECT (2025H1 −311, 2026 −223) |
| V1d bar_range ≤ 0.943 | 110 | 65 | 5,480 | −685 | 6 | −236 | 2,442 | 650 | 2,389 | REJECT (2025H2 −485) |
| V1e retvol20 ≤ 3.798 | 123 | 69 | 6,425 | −551 | 6 | −185 | 2,442 | 1,341 | 2,643 | KEEP — identical picks to V1b |
| V1all a+b+c | 103 | 61 | 6,120 | −513 | 6 | −185 | 2,131 | 1,569 | 2,421 | REJECT (c inside) |
| **V1ab a+b** (post-hoc combo of the two keeps) | 114 | 67 | **6,531** | **−551** | 6 | **−185** | 2,442 | 1,341 | 2,643 | passes the rule |

### What the two keeps actually remove — and a reinterpretation the data forces
- **V1a (opening range ≤ 2.2% of price)**: 10 picks — 3 fills, all losers (−$171), 7 no-fills that burned a slot. Mechanism: a range that tight makes the +30 bps stop-limit a noise trigger.
- **V1b**: every one of the 7 removed picks has `avg_daily_range_pct_20d == 0.0` (and `return_volatility_20d == 0.0`, which is why V1e is the same set). That is **missing 20-day history** (new listings / fresh wrappers), not "quiet names". No pick in the book had 0 < adr20 ≤ 5.01. So the honest rule is **"no 20-day history → no trade"**: 4 fills, all losers (−$340: FJET, PS, CBRG, SSPC), 3 no-fills. It is also a deliberate-rules fix: the composite z-scores a 0.0 as if it were a real (extremely quiet) value — accidental behaviour today.
- Combined: 16 picks out, **6 fills, 0 winners** (a coin flip would have given ~6 × 40% ≈ 2–3 winners), +$446 (+7.3%), MDD −20%, worst month −22%, every era not worse, all at 21 months / $10K.

### Honest size of this
At the $10K stage these are hundreds of dollars over 21 months; the value is the drawdown and worst-month shape plus zero winners among the removed. At a $100K book the same rules are worth ≈ +$4K / 21 months and a fifth off the drawdown. It does not change ORB's 2026 return class (+25% on stage capital → +26%); ORB's 2026 is the ramp's speed, not a missing rule.

### Proposal (joint decision — a new live rule; NOT shipped)
Add two no-refill post-selection vetoes to the ORB pipeline and live engine as ONE shared helper (the PDR-veto form): `range_size_pct <= 2.221` and `avg_daily_range_pct_20d` missing/0 (no 20-day history). Config `orb.yaml::filter.{min_range_size_pct, require_20d_history}`, env kill switches, parity tests, green-check line. Rollback = flags off.

## Shipped 2026-09-07 (owner GO 9/6 "yes for both"), live from the Tuesday 9/8 boot
- Range-size veto: `trading/orb_range_size_veto.py` + `orb_engine._range_size_veto_reject` + pipeline block; `orb.yaml::filter.range_size_veto {enabled: true, min_range_size_pct: 2.221}`.
- Short-history: `g1_veto.short_history_veto: true` (the rv20==0.0 marker is vetoed; the 8/15 fail-open is the `false` setting).
- Parity check: the pipeline on the shipped yaml knobs, no experiment flag, reproduces the a+b run to the dollar ($6,531 / 114 picks / 67 fills). NB the pipeline's per-veto print counts rows at the point of application (before slot/no-fill accounting), so "dropped 33 … +1,584" is not the book-level effect; the book diff (16 picks, −$445, 0 winners) is.
- Monday 9/7 is Labor Day (market closed); first live session Tuesday 9/8 — same day as BF P1's first session. Separate strategies, separate log lines (`RANGE-SIZE VETO`, `G1 VETO`).

## Re-verification on the news-complete features (2026-09-08 23:30 UTC, `rerun_20260908/`)
The nightly's PM$/news append had been crashing (NaN 'NA' ticker); the BT catalyst veto failed open on ~5.6K symbol-days. Fixed 9/8, history backfilled, features 20260908_2049, fresh static-lock dump (12,897 candidates). Both vetoes still pass the pre-committed rule; each alone and together:

| variant | picks | fills | total | MDD | red | worst | 25H1 | 25H2 | 2026 |
|---|---|---|---|---|---|---|---|---|---|
| vetoes off | 85 | 66 | 6,220 | −551 | 6/21 | −198 | 2,442 | 1,135 | 2,643 |
| range-size only | 81 | 63 | 6,391 | −485 | 6/20 | −187 | 2,442 | 1,241 | 2,709 |
| short-history only | 82 | 63 | 6,522 | −454 | 6/21 | −187 | 2,442 | 1,341 | 2,740 |
| **both (shipped)** | 79 | 61 | **6,627** | **−454** | 6/20 | −187 | 2,442 | 1,446 | 2,740 |

The honest ORB reference is now **$6,627 / 79 picks / 61 fills / MDD −$454 / worst −$187 / 2026 YTD +$2,740** (nightly `analysis_results/orb_bplus_book.csv`, reproduced by the side walk to the dollar).
