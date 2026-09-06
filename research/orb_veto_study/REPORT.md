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
