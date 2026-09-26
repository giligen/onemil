# RESULT — information events (cells 1,474–1,477): FAIL on VAL, population closed — judge's verdict 2026-09-26

| cell | TRAIN 2016-21 (n, mean net %/20 sess, t) | VAL 2022-23 (builder) | VAL (independent rebuild) | book VAL %/mo | verdict |
|---|---|---|---|---|---|
| I1 cluster ≥ 2 insiders ≥ $100K | 2,553 · +1.83 · 3.10 | 1,103 · +0.12 · 0.19 | +0.18 | −0.34 | FAIL |
| I2 officer/director ≥ $50K | 5,114 · +1.31 · 2.55 | 2,101 · +0.25 · 0.58 | +0.23 | +0.38 | FAIL |
| I3 opportunistic ≥ $25K | 4,665 · +1.60 · 3.06 | 1,767 · +0.61 · 1.28 (null pctile 92) | +0.60 | +1.18 | FAIL (closest) |
| I4 initial 13D | 951 · +0.74 · 0.92 | 286 · −1.81 · −0.98 | +0.06 (rebuild applies the panel's `etb` gate; 211 extra builder signals fail it) | −1.36 | FAIL under both readings |

* Independent rebuild (`rebuild.py`, from the PREREG prose only): trade returns agree on 100 % of common Form 4 trades
  and 96 % of 13D trades; signal-set Jaccard 0.85 / 0.93 / 0.94 / 0.85 — below the 0.99 gate because the prose left the
  cluster-window anchor, the officer-title rule and the 13D eligibility gate under-specified. The VAL means agree to
  within 0.07 pp on I1–I3, so the FAIL does not depend on the reading; on I4 the two readings differ by 1.9 pp and BOTH fail.
* Survivorship arm VOID for every cell: the multiday panel holds no delisted-name prices for any symbol that produced a
  signal, so TRAIN's +1.3–1.8 % is upward-uncertain (delisted small caps are the losers this signal would have bought).
* The 10b5-1 exclusion is a no-op before 2023 (the SEC field did not exist), disclosed. TEST 2024-01 → 2026-09 stays sealed.
* Reading: an insider-purchase effect of roughly +0.2–0.6 % per 20-session trade with t ≈ 1 on 2022–23 is consistent with
  the published magnitude in a bear-and-recovery window, and it is worth ≈ +1 %/month at 20 slots × $3.3K at best — the
  same MDE wall the multi-day programme hit. Not a book at this account. The data build (402,664 purchases, 1,376 initial
  13Ds) stays for any future information cell.
