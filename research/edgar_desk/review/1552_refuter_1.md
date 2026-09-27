# 1552 refuter 1 (full population, pass 2): timing, price scale, survivorship, item mapping, serial cap

Subject: the builder's full-population result, `cell_1552_stats.csv` (finished 05:01 UTC 9/27; 0 of 9 computable
cells pass). The rebuild's result (`REBUILD_1552.md`) also has 0 passes. The earlier sample-rebuild notes in this
file (9/26 21:44) are superseded by this pass.
Code: `review/refuter1_recompute.py`, which reuses the builder's own loaders and `score_event`, so each variant
changes only the rule under test. Outputs: `review/refuter1_{stats,events,surv}.csv`.
- Variant A is the builder's timing. It reproduces `cell_1552_stats.csv` exactly: same n and bps, t within 0.003.
- Variant B uses ET-correct timing.
- Variant C is B minus the E5 windows that contain a split-like overnight gap (open/prev close ≥ 1.8 or ≤ 0.55).

**Verdict: the result stands (0 passing cells). The defects below are real, but none of them changes a cell's
verdict.**

## 1. Timing: DEFECT (conservative, no look-ahead), shared by builder AND rebuild
- `acceptanceDateTime` is true UTC. 8-K acceptances run 10Z–03Z with a gap at 03–05Z. At 23Z the filingDate is
  the next day for 5,768 of 5,911 filings (after the 17:30 ET cutoff), and at 20–21Z it is the same day. This
  fits UTC and does not fit ET.
- Both implementations read the UTC wall clock as ET. The builder uses `ts.time()` on a UTC timestamp, and the
  rebuild uses `tz_localize(None)`. So the independent rebuild cannot catch this bug.
- Effect: acceptances between 05:00 and 09:00 ET (09–13Z) get the next session instead of the same-day MOO.
  - That is about 20 % of events (OFFERING 791 of 3,338 in VAL; OFFICER_EXIT 1,576 of 9,066).
  - No acceptance at 09:00 ET or later ever gets a same-day open, so there is no look-ahead.
- ET-correct timing (B), VAL named leg, bps / t:

  | cell | class | leg | bps | t |
  |---|---|---|---|---|
  | 1552 | OFFERING | E1 | +3.3 | 0.28 |
  | 1553 | SHELF | E1 | +8.1 | 0.56 |
  | 1554 | REVERSE_SPLIT | E5 | +30.7 | 1.06 |
  | 1555 | AUDITOR | E1 | −4.6 | −0.11 |
  | 1556 | NON_RELIANCE | E5 | +198 | 1.78 (0.86 ev/wk) |
  | 1557 | LATE_FILING | E5 | +155 | 1.46 (2.6 ev/wk, ex-top-5 % −14) |
  | 1558 | OFFICER_EXIT | E1 | +4.8 | 0.77 |
  | 1559 | CONTRACT | E5 | −18.8 | −0.58 |
  | 1560 | ACTIVIST | E5 | −63.5 | −0.91 |

  No leg reaches t ≥ 2.5. The best unnamed leg is OFFERING E5 VAL +54 bps, t 2.18, ex-top-5 % −85, so it fails
  even if it had been named. The TRAIN-named legs under B are unchanged.

## 2. Price scale: raw bars confirmed, artifacts are rare in VAL and correcting them lowers SHORT E5
- The Alpaca parquet is RAW. GE went 12.95 → 104.48 at its 1:8 reverse split (2021-08-02), and AAPL 499 → 127 at
  its 4:1 split (2020-08-31). E1 (open→close of one session) is scale-free.
- E5 windows containing a split-like gap in VAL: 0–5 per cell.
  - Removing them (C) LOWERS every SHORT E5 mean. For example, LATE_FILING goes from +155 to +124 bps, because a
    forward split looks like a short win.
  - The REVERSE_SPLIT short artifact (a −900 % "loss") does not occur: the $5 prior-close floor removes
    pre-split sub-$1 names.
- ±50 % events on VAL named legs: 68, all listed from `refuter1_events.csv`. They are dominated by CONTRACT E5
  LONG small caps (MINM, SCNX, LUNR +314 %, CIFR, CLSK, CIM split-flagged). CONTRACT fails anyway.
  - The other large moves are real crashes or squeezes: BETR, TYGO, LUNR, OCEA.
  - None was filtered. None flips a sign that matters.
- Known TRAIN split artifact: WHWK CONTRACT E5 +1,073 % (2021-08-24). It inflates a LONG cell that is negative
  anyway.

## 3. Survivorship: DEFECT in TRAIN, bounded, cannot flip
- `symbol_cik_map`: 5,033 eligible symbols are "lost" (no CIK). They are 33 % of eligible ≥$5 sessions, but they
  are mostly ETFs, preferreds and units (AGG, ACWI, AGNCP, …).
- TRAIN: 94 % of the sessions of names that die before 2024-06 are unmapped, so their filings are absent. These
  dead names have a WORSE unconditional E5 short: −24.9 bps against −6.9 for mapped names. The missing
  population does not favour the SHORT cells.
- VAL: dead-in-window names are 0.2 % of eligible sessions. The price source also lacks some failures (SIVB and
  FRC have no rows).
- Bound: even 5 % missing events at +500 bps E5 each moves LATE_FILING to about +172 bps with t about 1.6. It is
  still under 2.5, and still under 3 events/week.

## 4. Item / form mapping: impure classes, spec-level caveat, not a verdict change under the frozen spec
Sampled from the `submissions/` JSON: 20 per class plus the item co-occurrence tables.
- **5.03 (REVERSE_SPLIT):** mostly bylaw amendments and preferred designations (3.03+5.03) at large caps: Robert
  Half, Public Storage, BD, Baker Hughes, NOV, Berkley, Viatris, Vistra, Steelcase. Only about 4 of 20 plausibly
  announce a reverse split. The cell measures "8-K 5.03 at ≥ $5", not reverse splits.
- **424B3:** mostly resale supplements (Ginkgo, ChargePoint, Janus, Amprius, Allurion de-SPAC resales), closed-end
  funds, and DRIP or plan prospectuses. Top OFFERING names are REIT and utility ATMs (SUI 44, O 34, PG 30).
- **1.01 (CONTRACT):** often credit-agreement amendments (PRA Group, Movado, Sonic, Public Storage), i.e.
  financings. The 2.03 exclusion catches only some of them.
- **4.01 (AUDITOR):** includes mechanical de-SPAC super-8-K auditor swaps (1.01+2.01+3.02+3.03+4.01+5.01+…).
  BKKT appears in 1552, 1554, 1555 and 1560 on the same day.
- **SC 13D (ACTIVIST): filer-side attribution defect.** The submissions feed lists a 13D under the FILER's CIK as
  well as the subject's. BAC 32, RILY 19, JPM 10, WFC 9, GS 8 and CG 8 are priced on the filer's own stock (about
  9 % of events). Removing symbols with ≥ 6 filings: TRAIN E1 −47.5 / E5 −47.9, VAL E1 +91.7 / E5 −84.6. Still a
  fail: TRAIN is negative, so "same sign t ≥ 1" fails.
- NT 10-K/Q and 4.02 match their definitions.

## 5. Amended serial-issuer cap: works as intended
- Big-bank share of OFFERING events after the cap is 0.3 %, and of SHELF 0.1 %.
- The count uses all five 424B subtypes, and so does the rebuild.
- The early-2019 undercount (no pre-2019 history) affects TRAIN only.

## 6. Additional defect found (not in lens, recorded)
- The builder's `events_wk` divides by the number of distinct ISO weeks that contain an event, not by calendar
  weeks.
- Calendar weeks give LATE_FILING 2.58/wk VAL, where the builder shows 4.67. AUDITOR is 2.39 (builder 2.95) and
  NON_RELIANCE 0.86 (builder 1.68).
- The builder figure would wrongly pass the ≥ 3/wk bar for LATE_FILING. Today it changes no verdict, because
  those cells fail on t. It must be fixed before any re-use.

---
# Pass 3 (2026-09-27, independent re-check of passes 1-2 through the same lens)

**Verdict: the result stands.** 0 of the 9 computable cells pass, and no defect found changes any cell's verdict.
The defects below are all real. New scripts: `refuter1_surv_bound.py` and `refuter1_surv_sim.py`.

## (a) Timing: re-confirmed
- `acceptanceDateTime` is true UTC, despite the builder docstring saying ET. The evidence is the hour histogram of
  8-K/NT/13D/424B5 filings:
  - The peak is at 20–21Z, which is 16–17 ET after the close.
  - The filingDate rolls to the next day for 94 % of 23Z acceptances (after the 17:30 ET cutoff).
  - For the 00–02Z acceptances it rolls only about 11 % of the time (weekends).
- The builder reads the UTC wall clock through `ts.time()`, and the rebuild does the same thing. So:
  - About 9.4 % of filings, the 05:00–09:00 ET pre-market acceptances, enter one session late.
  - Look-ahead cases (an acceptance at or after 09:00 ET given a same-day open) = **0**.
- 20 sampled events all have the entry open after the acceptance. CIFR 08:35 ET, TCMD 07:00 ET and OC 06:49 ET
  show the day-late entry.
- ET-correct timing (variant B), VAL, TRAIN-named leg: the named legs are unchanged under B, and the best t is 1.78
  (NON_RELIANCE, at 0.86 events/week). Table in pass 2 above.

## (b) Price scale: re-confirmed RAW
- Examples of raw prices across splits:
  - CIM goes 4.39 → 12.75 at its 1:3 reverse split on 2024-05-22.
  - MULN goes 0.064 → 1.46 (2023-05-04) and 0.08 → 8.00 (2023-12-21).
- The builder VAL list has 105 event-legs beyond ±50 %. None was filtered. Inspected:
  - CIM CONTRACT E5 +155 % is a split artifact (LONG, and CONTRACT fails anyway).
  - LUNR, TYGO, BETR, AVTE, MINM, CLSK and OCEA are real moves.
  - RDNW = RMBL and AVTE = JBIO are the same series under two tickers. These are duplicate events from renamed
    tickers in the Alpaca parquet. They inflate n but leave t roughly unchanged because they fall in the same
    cluster.
- E1 is scale-free. Removing E5 split windows lowers the SHORT E5 means (for example LATE_FILING goes from +155 to
  +124 bps).

## (c) Survivorship: a VAL hole exists, and it cannot flip a verdict
- The yearly death rate of eligible (≥ $5, ≥ $1M) names in the Alpaca parquet is 2.9 / 2.4 / 3.4 / 2.5 % for
  2019–2022, but only **0.5 % in 2023**. The Databento 2025 panel shows 3.3 %.
- On XNAS in VAL, 894 dead eligible symbols appear. Only 56 of them are in `symbol_cik_map` (SIVB, FRC and PACW
  are all missing).
  - That is 2.2 % of eligible sessions.
  - They are mostly SPACs, preferreds and acquired names. Their unconditional E5 short is −103 bps, so the
    missing population is net adverse to SHORT cells.
- 43 of them are real failures, with an E5 short of +557 bps on their ≥ $5 sessions.
- Re-adding VAL filings that have no Alpaca bar but do have an XNAS bar gives 0 events for cells 1554–1557 and 1
  event for 1558.
- Realistic LATE_FILING bound: one NT per failure name, at a random session of its final 250, run for 2,000 draws.
  - Result: +5 events, 2.69 events/week, t 1.56 (max 2.26), P(pass) = 0.
  - Only a perfect-foresight worst case, where each name contributes its single best session, reaches t 5.0. That
    case is not a realistic bound.
- NON_RELIANCE (0.86/wk) cannot reach 3/wk. REVERSE_SPLIT stays under t 2 even with all 43 names at +945 bps.

## (d) Item mapping: 20 raw filings per class from `submissions/` (1,500 random CIKs)
- **5.03:** 0 % of the descriptions mention a split, 24.6 % co-occur with item 3.03, and 21 % with 5.07.
  - The sample includes 3M, WesBanco ("8-K ON BYLAWS CHANGE"), Magnera ("AMENDED BY-LAWS"), O'Reilly
    (annual-meeting charter amendment), and the Coterra and Serina mergers.
  - About 3–5 of 20 plausibly announce a reverse split. The cell is "5.03", not "reverse split".
- **424B3:** resale prospectuses (Oculis "final resale"), supplements wrapping a 10-Q or 8-K (Talkspace,
  Amplitude), Toronto-Dominion structured-note pricing supplements (filed as 424B3, not 424B2), closed-end funds,
  and a commodity ETF (UGA).
  - 31 % of OFFERING events have only a 424B1 or 424B3 trigger.
  - Restricting OFFERING to 424B4/424B5 plus 3.02 gives TRAIN E1 +0.1 bps (t 0.02) and E5 −18.5 bps, so E1 is
    the named leg. VAL E1 is +6.2 bps (t 0.48), and VAL E5 is +57.8 bps (t 1.99) but TRAIN-negative. **FAIL in
    both legs.**
- **SC 13D:** the filer-side attribution is confirmed.
  - OPKO filed about Xenetic, and Carlyle, Vodafone, Telefonica and Cosan appear as filers: about 5 of 20.
  - Symbol concentration: BAC 32, RILY 19, JPM 10, WFC 9, GS 8, CG 8.
  - Pass 2's filer-excluded ACTIVIST is TRAIN-negative → FAIL.
- **1.01:** financings and credit agreements appear, but so do real contracts.
- **4.01:** includes de-SPAC super-8-Ks (Mirion).
- **NT 10-K/Q and 4.02:** clean.

## (e) Serial cap and 424B2 exclusion
- Both are implemented as written: `FORM_424B_OFFERING` excludes 424B2, and the cap counts all five 424B subtypes
  over a trailing 365 days, with more than 12 → excluded from OFFERING and SHELF.
- The residual hole: Toronto-Dominion-style notes filed as **424B3** are removed only through the cap.

## Other, carried from pass 2
- `events_wk` in the builder divides by the number of weeks that contain an event. Examples: LATE_FILING shows
  4.67/wk against 2.58 in calendar weeks, and NON_RELIANCE 1.68 against 0.86.
- This must be fixed before the scorer is reused. It changes no verdict today.
