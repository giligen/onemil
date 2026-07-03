# ORB price-band boundary studies — consolidated verdicts (2026-07-05)

All run on the validated defended-pipeline replica (frozen H1-2025 fit,
static_lock + touchgo + PDR veto), pre-declared ship bars. Scripts:
research/scripts/orb_band_study.py (env-parameterized: ORB_BAND_MIN/MAX/
GAP_CAP/ENTRY_SLIP_BPS/EXIT_SLIP_BPS).

## $30-60 (raise upper bar): NO-GO
See research/orb_band_30_60_verdict.md. B1 −$6.7K era-inconsistent;
B2 +$8.9K/18mo. Monster tail absent above $30.

## $2-3 (lower the floor to $2): NO-GO, decisively
B1 (frozen fit, merged, slippage stress applied to the sliver only):
- 30/10 bps (validated model): **−$43,918** vs base
- 60/20 bps (penny-friction):  **−$70,982**
- 100/30 bps (conservative):   **−$75,238**
Fails at even the friendliest tier, before the friction that makes it
worse. 893 sliver candidates broke out (51% — entry dynamics fine); the
frozen composite mis-ranks them and they displace better small-caps.
Reports: /tmp/orb_band_study_report_2_3_slip{30,60,100}.txt.

## $3→$4 floor raise: NO-SHIP (fragile despite +$40K headline)
min>4.0 shows TOT $249.7K (+$40K, +19%), both eras better, MDD better —
BUT: threshold grid is a staircase of single trades (>4.05 fine, >4.15
crashes $33K because ANNA — the biggest trade in the book — entered at
$4.11); monthly delta only 11/19 ≥0 with $30K of the $40K in 2026-01
alone. Fails monotone-dose-response and month-consistency, the tests the
PDR veto passed (18/19). The sub-$4 toxicity DIRECTION is noted (71
post-veto trades, both fine buckets negative) — revisit with more data,
never as a hard threshold pennies below a monster.

## Standing conclusion
**$3-30 is not an accident.** Three boundary moves tested with the full
pipeline + pre-declared bars; all three fail. The universe knob is done.

## Related evidence recorded the same day
- WR-vs-literature counterfactual: our exact 537 post-veto trades under
  a +0.5R target = 63.7% WR, +1R = 50.5% (literature range) vs our 40.2%
  — the WR gap is exit design (tail preservation), not a defect. Bounded
  EV at +1R ≈ +0.07R/trade vs actual $368/trade with top-5 = 74%.
- Monster-detection: April 2026 campaign catalog (pre-trade classifier
  OOS −$30K; post-+1R classifier < always-stay; add-at-3R fails
  hero-removal; docs/orb_research_apr_2026.md). Only untried channels:
  FLOAT (needs point-in-time data; current float_cache has lookahead)
  and PREMARKET VOLUME (needs new bar fetches). Everything else refuted.
