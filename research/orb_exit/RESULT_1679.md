# RESULT 1,679 — the ORB give-back: exit rules on real ledgers, R-floored

PREREG: `research/orb_exit/PREREG_1679.md` (FROZEN 2026-09-30 16:10 UTC). Script: `1679_orb_exit.py`
(0 ERROR/Exception lines in `1679_orb_exit.log`). Full reads: `1679_reads.csv` (56 rows). Per-fill detail:
`1679_per_fill.csv` (601 rows). Anatomy detail: `1679_summary.json`.

## Coverage and reconstruction agreement
- **L1 LIVE** (`data/trades.db`, strategy=orb, account live, 2026-05-19..09-23): **123/123 usable (100%)**
  after backfilling 32 missing symbol-days (32,982 bars) into `bars_sip.db` via the scratchpad wrapper
  pattern (`research/bf_zero/backfill_bars_sip.py`, FEATURES/STATE monkey-patched to scratchpad paths;
  the production bar store itself was only ever appended, never rewritten).
- **L2 BT** (`analysis_results/orb_bplus_book.csv`, entered==1): **478/483 usable (99.0%)**; 5 symbol-days
  have no bars (SKBL 2025-03-10, STSS 2025-06-06, ZGM 2025-09-10, UGRO 2026-04-10, AGH 2026-05-04) —
  left alone (out of this cell's write scope; well above the 80% availability rail).
- **Reconstruction agreement**: L2's entry-minute rule (`find_breakout_bar_ts`: first bar in [09:35,10:35)
  ET with high > the 09:30-09:34 range high) applied to L1's own bars, vs L1's real `filled_at`, on n=45
  overlapping (symbol,date) pairs: **median |Δ| = 0.0 min, mean |Δ| = 1.98 min, 80% exact, 91% within 1
  min**. The reconstruction is trustworthy.
- R floor (<0.5% of entry): L1 1/123, L2 0/478 fail — excluded from R-unit reads only; dollar reads use the
  full population.

## Give-back anatomy (R-floored; "ever" = MFE from entry to the ACTUAL exit bar)
| Ledger/half | n | share ever≥1R & closes≤0 | R given back (ever≥1R) | min peak→exit | closed by stop/target/FC |
|---|---|---|---|---|---|
| L1 whole  | 122 | 9.0%  (share≥0.5R 18.9%, ≥1.5R 0.8%) | 1.62 R | 107 | 60 / 38 / 25 |
| L2 2025   | 212 | 4.2% | 1.25 R | 128 | 114 / 49 / 49 |
| L2 2026   | 266 | 6.0% | 1.51 R | 122 | 98 / 92 / 76 |
| L2 whole  | 478 | 5.2%  (share≥0.5R 14.9%, ≥1.5R 0.8%) | 1.40 R | 125 | 212 / 141 / 125 |

**Correction to the lead number.** Cell 1,678's crude ORB reconstruction (touch-tolerance entry, price-touch
exit, no real times) read "21% of fills reach +1R and close ≤0, giving back 1.88R" — flagged not a claim.
On validated real times the number is **5–9%**, giving back **1.3–1.6R**. The mechanism is real but 1,678
overstated its frequency by roughly 2-4x.

## Read 2 — HOD-trained exit models (1,678), unchanged, applied to the real ledgers
Rule X(c=0.20) / X+ (both scoring-direction models, both featsets, 8 combos) **reverses** 1,678's
"every cell passed" read: on **L2 both years** mean ΔR is **-0.28 to -0.75 R**; the whole-sample day-clustered
t is ≤ -2.05 on all 8 combos (2025-only: all ≤ -2.24; 2026-only: -0.79 to -2.99, 6/8 ≤ -1.5) — consistently,
often strongly, **negative**. $ effect at $375 risk: -$106 to -$282/fired trade. L1 is weakly positive
(+0.05 to +0.18 R, |t| < 0.9, n fired 18-59) and does not rescue it. **Read 2 fails the pass bar outright**
(wrong sign on L2 in both years).

## Read 3/4 — mechanical alternatives vs the actual exit, give-back-saved / continuation-forgone
Rules (a)(b)(c)(e) fully replace the post-entry exit mechanism (independent of touchgo Rule M/D, which
stays as recorded in the actual exit, ~30% of L2 fills); (d)'s runner leg does too; (f) overlays one fixed
checkpoint on the actual exit, like Read 2.

| Rule | L1 whole ΔR (t, $/trade) | L2 2025 ΔR (t) | L2 2026 ΔR (t) | L2 whole ΔR (t, $/trade) | saved/forgone (L2 whole) |
|---|---|---|---|---|---|
| (a) lock 0.5R@1R      | +0.173 (0.97, +$69) | -0.184 (-2.25) | -0.208 (-2.32) | -0.198 (-3.09, -$74) | 0.78 / 0.91 |
| (b) lock BE@1R        | +0.175 (1.59, +$69) | -0.099 (-1.35) | -0.050 (-0.76) | -0.072 (-1.39, -$27) | 1.11 / 0.72 |
| (c) trail MFE-1R      | +0.138 (1.86, +$55) | -0.192 (-2.07) | -0.294 (-3.28) | -0.249 (-3.87, -$93) | 0.53 / 0.95 |
| (d) 50%@1R + live     | **+0.223 (3.59, +$75)** | +0.012 (0.29) | +0.024 (-0.58†) | +0.019 (-0.23†, +$7) | 0.76 / 0.63 |
| (e) live-lock ref     | +0.117 (1.26, +$46) | -0.018 (-0.72) | +0.026 (0.33) | +0.007 (-0.14, +$3) | 0.98 / 0.62 |
| (f) time-stop 60m<.5R | -0.001 (-0.11, -$18; fired 26/122) | -1.05 (-2.46; fired 27/212) | -0.40 (-2.04; fired 45/266) | -0.64 (-3.17, -$241; fired 72/478) | 0.58 / 1.57 |

† day-clustered t's sign differs from the fill-weighted mean's sign here — a handful of days concentrate the
fills; flagged, not hidden.

**(e) is the sanity check** — it should ≈ the ledger's own actual exit. L2 whole ΔR = +0.007R (~0, as
expected: the walker's mechanics reproduce production). L1 ΔR = +0.117R (t=1.26) reflects the ATR-floor +
40%@3R scale-out the real book runs that this simplified 1.75R/0.5R-only walker doesn't reproduce
(disclosed scope, not a bug — the L2 check confirms the walker itself is correct).

## Pass bar (PREREG's own): **nothing passes**
Required: paired ΔR ≥ +0.05R, day-clustered t ≥ 2.5, ex-top-5% > 0 on L2 in BOTH years, same-signed on L1, $
positive both, R-floored.
- **(a)(b)(c)(f)**: L2 negative (several significant) — fail outright, several with the OPPOSITE sign from L1.
- **(d)**: the only rule positive on both ledgers, and significant+tail-robust on L1 (t=3.59, ex-top5=+0.13),
  but L2 both years are flat/insignificant (|mean|<0.05R, |t|<0.6) — **closest candidate, does not clear the
  L2 leg**. Worth a follow-up PREREG once L1 has more fills (n=122 today); not shippable now.
- **(e)**: reference only ("what already happened"), not a candidate.
- **Read 2 (X/X+, 8 combos)**: wrong sign on L2 in both years — fails.

## Files
`research/orb_exit/{RESULT_1679.md, 1679_reads.csv, 1679_per_fill.csv, 1679_orb_exit.py,
1679_orb_exit.log, 1679_summary.json}`.

## Disclosed scope / caveats (read as an adversary)
1. Fills use touch-based semantics throughout (stop/lock/target trigger at the level touched, +10bps slip),
   matching `study_orb_pipeline_static_lock.py`'s OWN production convention — not a new obtainability
   violation, but not independently re-verified against NBBO quotes (none in `bars_sip.db`).
2. L2's stop = the book's range_low (per PREREG); L1's stop = the ledger's real `stop_loss_price`
   (occasionally a few cents off range_low via the ATR floor) — a deliberate, disclosed difference in "R"
   between ledgers.
3. Rules (a)(b)(c)(e) ignore touchgo Rule M/D (tag_bb/tag_b1, ~30% of L2 exits) by construction — they
   simulate "the entire post-entry hold used this rule," not "this rule layered on touchgo." Rule (d)'s
   runner leg shares that scope.
4. `scale_lock`/`scale_eod` (L2, 14% of fills) collapse the blended actual P&L onto one reconstructed exit
   bar via final-leg touch-search — the +3R scale-out itself is not separately modeled.
5. Read 2 uses the persisted 1,678 models UNCHANGED (no refit) on both ORB ledgers as fully out-of-sample
   data.
6. Multiplicity: 56 reads total (32 Read-2 combos + 24 Read-3/4 rule×slice rows); c fixed at 0.20 (no
   sweep), no lock/trail level tuned — within the PREREG's "≈54, no tuning" bound.
