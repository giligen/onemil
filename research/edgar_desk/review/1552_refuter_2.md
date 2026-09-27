# 1552 refuter 2: statistics, execution and multiplicity (second full-file pass, 2026-09-27)

**Inputs.** `cell_1552_events.csv` has 144,264 leg-rows (TRAIN 105,746; VAL 38,518). The latest date is 2024-06-27. There
are 0 TEST rows, which is asserted in the script. I also used the raw Alpaca daily bars to get ADV, the intraday-low SSR
and the price scale. TEST was not read.

**Scripts and outputs.** `1552_refuter_2b.py` produces `1552_refuter_2b_stats.csv` and `1552_refuter_2b.log`. The
price-scale checks are in `1552_refuter_2b_pricescale.csv` and `1552_refuter_2b_splitlike_E5.csv`. The earlier pass on the
same lens is kept in `1552_refuter_2_recompute.{py,csv}` and `1552_refuter_2_placebo_exec.*`, and it agrees with this pass
to 0.1 bps.

## Verdict: the result (0 of 9 cells pass) stands, and no cell's verdict changes
Named leg on VAL, deduplicated on (date, symbol). The builder's value is in brackets.

| cell | leg | net bps | t (day-clustered) | ex-top-5 % | ex-top-1 % | drop best 2 sessions | cap ±20 % | ev/wk (calendar) | chance pass* |
|---|---|---|---|---|---|---|---|---|---|
| 1552 OFFERING S | E1 | −6.8 (−5.4) | −0.57 | −60 | −25 | −10 | +2.5 | 40.8 | 0.00 |
| 1553 SHELF S | E1 | −0.3 (−1.3) | −0.02 | −52 | −17 | −9 | +0.2 | 14.6 | 0.00 |
| 1554 REV SPLIT S | E5 | +22.4 (+21.4) | 0.85 | −94 | −19 | +12 | +18.5 | 21.0 | 0.00 |
| 1555 AUDITOR S | E1 | −99.7 (−78.0) | −0.90 | −177 | −132 | −130 | −17 | 2.4 | 0.00 |
| 1556 NON_REL S | E5 | +186 | 1.77 | +65 | +147 | +109 | +166 | **0.86** (builder 1.68) | 0.01 |
| 1557 LATE_FIL S | E5 | +156 | 1.35 | −7 | +90 | +48 | +166 | **2.58** (builder 4.67) | 0.02 |
| 1558 OFFICER S | E1 | +1.9 (+2.3) | 0.29 | −36 | −11 | −0.3 | +3.8 | 114.5 | 0.00 |
| 1559 CONTRACT L | E5 | −42.5 (−44.8) | −1.41 | −228 | −127 | −55 | −81 | 38.7 | 0.00 |
| 1560 ACTIVIST L | E5 | −32.4 (−101.3) | −0.52 | −208 | −94 | −61 | −35 | 6.8 | 0.00 |

*Chance pass: a day-cluster sign-flip null on the demeaned leg, 400 draws. It counts the share of draws that clear the
mean ≥ 15, t ≥ 2.5, ex-5 % > 0 and cap > 0 conditions.

**Multiplicity.** Nine pre-directed cells, with the leg named on TRAIN, so the leg choice adds no VAL inflation. The t ≥ 2.5
condition has a chance rate of about 0.6 % per cell under the null. The sign-flip rates above are between 0 and 2 %.
Across 9 cells that gives about 0.03 to 0.06 expected chance passes, or roughly a 3 to 5 % chance of at least one. The
observed count is 0. The programme total is 1,561 cells.

**Bar-breaking leg choice still fails.** Suppose the best leg is picked on VAL, which the bar forbids:
- 1552 E5 is +47.7 bps with t 2.05, ex-5 % of −88 and TRAIN of −2.6.
- 1560 E1 is +68.5 bps with t 1.85, ex-5 % of −46 and TRAIN of −21.7, which has the wrong sign.

## Checks this lens asked for
- **Leg-naming order.** `cell_1552.py` (l.619–641) names the leg from TRAIN rows only. It picks the higher day-clustered
  TRAIN t, and a tie goes to E1. It logs and flushes the name before any VAL statistic for that cell. The log times are
  04:56:59 to 05:00:57, and the stats were written at 05:01:30. Deduplication does not change any named leg.
  - Caveat: the max-t naming rule is a code choice. It is not in the PREREG prose.
- **Year signs (TRAIN, named leg).**
  - OFFERING E1 is positive in 3 of 4 years (2019: −15).
  - SHELF E1 is positive in 3 of 4 (2020: −8).
  - REV SPLIT E5 is positive in 1 of 4. It is positive only in 2022 (+112), and 2022 carries 30× the net TRAIN sum. That is
    the 2022 small-cap bear drift, not an event effect.
  - AUDITOR is positive in 2 of 4.
  - NON_REL is positive in 3 of 4 (2020: −697).
  - LATE_FIL is positive in 3 of 4.
  - OFFICER is positive in 3 of 4.
  - CONTRACT is positive in 2 of 4.
  - ACTIVIST is positive in 3 of 4, but 2022 is −188.
- **Placebo and null (universe drift).** On SHORT cells the universe rose in both splits: universe_bps runs from −3 to −42.
  The placebo margin and the null ≥ 99 are therefore nearly automatic. OFFICER_EXIT E1 VAL nets +2 bps, yet it scores
  null 100 with placebo t 3.2. These two conditions screen nothing on SHORT cells in a rising tape. No LONG cell has a
  positive edge that 2020–21 beta could explain:
  - CONTRACT's 2020–21 contribution is −0.56 of its TRAIN sum.
  - ACTIVIST's 2022 is −188.
  - A size-matched placebo (from the earlier pass) shows that SHORT-class names lag their size peers by 15–76 bps. However,
    the naked short nets about 0 because the matched basket rises.
- **Borrow and locate.** Borrow is charged at a flat 3 %/yr. Offering, reverse-split and late-filing names typically cost
  30–100 %+ to borrow, or have no locate at all. Re-pricing at those rates:

  | leg (VAL) | at 30 % | at 100 % |
  |---|---|---|
  | REV SPLIT E5 | −29 | −164 |
  | NON_REL E5 | +134 | ≈ 0 |
  | LATE_FIL E5 | +104 | −30 |
  | OFFERING E1 | −14 | −33 |

  In the large cells every SHORT E5 leg is negative at 30 %.
- **SSR.** The builder tests the prior *close* ≤ −10 %. Rule 201 triggers on the intraday low. The prior-session-low version
  flags 95 OFFERING VAL events and 94 OFFICER VAL events, which the builder keeps. Removing them moves net by −1 to −3 bps.
  Another 72–82 events per large cell gap down more than 10 % at the entry open, which is ambiguous for a MOO short.
- **MOO fill for $3K.** Prior 20-day dollar volume is below $1M for 0 % of events, so the gate works. It is below $3M for
  11–17 % of the large cells and 23–35 % of AUDITOR, NON_REL, LATE_FIL and ACTIVIST. The median $3K order is 1–4 % of an
  opening auction sized at 1 % of ADV, which is fine at the median but material in the thin third. The Alpaca daily open is
  the first print, not the official cross.
- **Price scale (new in this pass).** The raw bars are unadjusted, so an E5 window can straddle a split. Among ±50 % events
  in the named legs, 15 straddle a split-ratio overnight gap. Examples:
  - AMZN's 20:1 split (2022-05-31) credits a REV SPLIT short with +94.6 %.
  - CHDN's 4:1 split credits a short with +65 %.
  - WHWK (2021-08-24, 13.6× gap) credits a CONTRACT long with +10,733 %.
  - CIM's 1:3 split credits a long with +155 %.

  A scan of every E5 event for split-ratio gaps flags 0–9 per cell/split. Excluding the flagged events:
  - REV SPLIT E5 VAL goes from 22.4 to 16.6.
  - CONTRACT E5 VAL goes from −42.5 to −33.3.
  - AUDITOR E5 TRAIN goes from −78.7 to +5.3 (one event). It stays below E1's t, so the named leg is unchanged.

  No verdict moves. E1 legs are intraday and immune.

## Defects (none changes a verdict today; each must be fixed before any rerun or promotion)
1. `events_wk` divides by the number of weeks that contain an event, not by calendar weeks. Because of this, 1557 appears
   to clear the ≥ 3/wk bar when its true rate is 2.58/wk.
2. `passes_bar` omits two conditions: "TRAIN same sign t ≥ 1" and "≥ 3 of 4 TRAIN years positive".
3. Duplicate (date, symbol) events are scored separately. This mostly affects ACTIVIST, where group 13D filers turn 530
   events into 609 rows and E5 VAL moves from −32 to −101.
4. SSR is tested on the close instead of the intraday low.
5. Borrow is a flat 3 % with no locate model. The null uses 5 borrow days while events use about 7 calendar days (minor).
6. Unadjusted splits fall inside E5 windows. The ±50 % inspection in the PREREG is needed and was not run by the builder.
7. `RESULT_1552.md` is stale: it says the stats were not produced. It must be rewritten with the 0/9 table before anything
   reaches the owner.
8. The builder and the rebuild disagree badly: event Jaccard is 0.24–0.77 against the PREREG bar of 0.99, and the named leg
   differs on 3 of 9 cells. Both builds return 0/9, so the null does not depend on this. The numbers above are still not
   reportable as precise per-cell estimates.

**Adequacy (MDE at t 2.5, VAL).** About 30 bps for E1 in the large cells and about 60–100 bps for E5. The small cells are
powered only for about 250–350 bps, and they fail the frequency bar in any case.
