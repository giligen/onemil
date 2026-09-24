# REPORT — live ORB on 2023-01 .. 2024-06, point-in-time universe (cells 1,418–1,419)

## Cell 1418 (production seed gap ≥ 5 %, $3–30)

- fills 106 (of 171 picks, no-fill share 38%) over 89 days, 1.4 fills/week (78 weeks)
- net R/fill +0.089, t (day-clustered) +1.55, ex-top-5 % +0.027, total $+3,542
- halves: 2023 +0.124 R/fill, 2024H1 +0.039 R/fill
- months: worst -703 (2024-03), best +871 (2023-09), green months 78% of 18
- **VERDICT (frozen): FLAT**

- **Pooled out-of-regime ORB (2023-01 .. 2024-12, same code): n 165, +0.055 R/fill ± 0.046 (t +1.19)** — vs 2025 +0.272 (n 127)

## Cell 1419 (addon_p30 gap 3–5 %, $30–50)

- fills 10 (of 10 picks, no-fill share 0%) over 10 days, 0.1 fills/week (78 weeks)
- net R/fill -0.210, t (day-clustered) -1.42, ex-top-5 % -0.354, total $-786
- halves: 2023 -0.252 R/fill, 2024H1 -0.205 R/fill
- months: worst -470 (2024-04), best +279 (2024-02), green months 17% of 6
- **VERDICT (frozen): NEGATIVE**


## Main-session review and the pre-committed consequences (2026-09-24)
* Data: XNAS.ITCH point-in-time tickers ($5.51) → consolidated Alpaca daily (12,891 of 12,976 tickers, 0 lost after the
  fetch-loop fix) → 15,612 candidates, 99.4 % with minute bars; same code as the 2024H2 run.
* **1,418 production seed: FLAT** (+0.089 R/fill, t 1.55, n 106; 2023 +0.124, 2024H1 +0.039; 78 % green months).
  Pooled out-of-regime 2023-01..2024-12: **+0.055 ± 0.046 R/fill, n 165** — vs +0.272 in 2025.
  → **Consequence applied:** `scripts/orb_ramp_check.py` ADVANCE now needs 40 live fills at the stage
  (`min_fills_advance`); the BT-band DEMOTE keeps its 8-fill trigger. Tests 33/33.
* **1,419 addon_p30: NEGATIVE** (−0.210 R/fill, n 10). With 2024H2 (+0.455, n 4): out-of-regime ≈ −0.02 R on 14 fills.
  → the $30–50 add-on pool is **no longer order-eligible**; it stays dry-only (PREREG_LIVE_UNION amended).
Programme count 1,419.
