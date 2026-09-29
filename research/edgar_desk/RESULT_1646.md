# RESULT — cells 1,646–1,648: pre-announcement run-up from the firm's own 8-K history — FAIL (closed)

Judged 2026-09-29 06:05 UTC by the main session from `RESULT_1646_build.md` (Sonnet, `cell_1646.py`) and `REBUILD_1646.md`
(independent rebuild from the prose, `rebuild_1646.py`). Programme count: 1,648. Free data (EDGAR 8-K 2.02 cache, daily bars).

## Agreement
Firm-quarter sets: Jaccard 0.68 (below the 0.98 bar) — traced to defensible population choices (8-K/A amendment merging,
gap and split guards), not to a bug; on the shared keys the through-event P&L matches to 1e-13 and the pre-window P&L
correlates 0.98 (early-arrival handling). Both builds reach the same verdict on every item. One shared residual risk
noted by the rebuild: neither split guard catches a plain unadjusted 2:1 split inside a hold (it would fabricate a
+100 % winner; the ex-top-5 % lines are the defence and they are negative).

## Numbers (VAL 2023–2024H1; net bps per event; build / rebuild)
| item | 1,646 pre-window (E−5 → E−1) | 1,647 through event (→ E+1) | bar |
|---|---|---|---|
| n (VAL) | 11,478 / similar | same | — |
| events/week in season | 147 | 147 | ≥ 20 ✓ |
| expected-date hit rate [E−1, E+1] | 47 % / 51 % (±3 sessions: 74 %; median gap 1–2 sessions) | same | ≥ 70 % ✗ |
| mean net | +10.8 / +12.6 | +25.3 / +28.0 | ≥ +40 ✗ |
| day-clustered t | 0.62 / 0.70 | 1.10 / 1.17 | ≥ 2.5 ✗ |
| ex-top-5 % | −71 / −72 | −98 / −93 | > 0 ✗ |
| SPY-adjusted | −1 / −2 | −16 / −15 | ≥ +25 ✗ |
| volume-tercile table | VAL rising, TRAIN not (U-shaped) | | both halves ✗ |

## Adequacy
MDE ≈ 43 bps at t 2.5 against a 40-bps bar: the test could see the effect it was built for. What it saw is a mean
carried entirely by the top 5 % of events (ex-top-5 % ≈ −70 to −100 bps on every cell and split) and ≈ 0 against SPY.
The estimator's precision (half the actual releases within a day, three quarters within three days) dilutes any
pre-announcement effect by about half; a paid calendar would sharpen the window but cannot turn a tail-carried,
market-adjusted zero into a book. The earnings-announcement premium of the literature does not survive on this
population, this window and auction costs.

## Verdict
FAIL on the frozen bar, both builds. Closed with the tercile tables on record. No paid calendar is requested.
