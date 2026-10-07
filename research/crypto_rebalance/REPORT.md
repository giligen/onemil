# REPORT — leveraged-crypto-ETF close rebalance flow in BTC spot (cell 1, 2026-10-07)

PREREG: `PREREG.md` (frozen before scoring). Script `run.py`; per-day rows `results.csv` (BTC), `results_eth.csv`;
all cells × costs × slices `results_stats.csv`; `completeness.csv`. Data: Alpaca crypto 1-min bars, cached `data/`.

## Verdict: FAIL at every cost; the gross effect is absent (point estimate wrong-signed)
Rule: position = sign(09:30→15:30 ET BTC move), held 15:30→16:00 ET, NYSE trading days.

| cell | cost | mean/day | t | n |
|---|---|---|---|---|
| primary 2024-06-04..2026-10-06 | gross | −3.0 bp | −1.92 | 578 |
| primary | 5 bp | −8.0 bp | −5.1 | 578 |
| primary | 30 bp (Alpaca spot) | −33.0 bp | −21.1 | 578 |
| primary halves (gross) | | −0.9 bp / −5.1 bp | −0.4 / −2.5 | 289 / 289 |
| primary terciles by |r_day| (gross) | | low +1.6, mid −1.5, high −9.1 bp | 0.7 / −0.6 / −2.8 | ~193 each |
| primary ex-top-5 % days (gross) | | −7.3 bp | | 578 |
| placebo PERIOD 2022-01-01..2024-06-03 (no leveraged BTC ETF) | gross | +4.0 bp | +1.83 | 606 |
| placebo WINDOW 11:30→12:00, primary period | gross | +1.5 bp | +0.76 | 585 |
| reversal 16:00→16:30, primary / placebo period | gross | −3.2 / −2.2 bp | −2.15 / −1.15 | |
| ETH/USD primary | gross | −2.7 bp | −1.42 | |
| next-bar-open entry (executability) | gross | −2.9 bp | −1.86 | |

MDE at t 2.5: 3.9 bp/day (30-min sd 37.7 bp, n 578). Completeness: primary 578/587 (98.5 %), placebo period 606/607,
placebo window 585/587 — gate (≥ 95 %) passes.

## Reading
- No rebalance-flow signature: the direction-of-day drift into the close is ≤ 0 in the leveraged-ETF era, the pre-ETF
  period shows the opposite sign at the same |t| ≈ 1.8 (noise band ±4 bp), and the post-close reversal is the same size
  in both eras — nothing is tied to the products' existence.
- Economic adequacy: the 2× funds' rebalance (≈ 2 × AUM × |r_day|, order $10⁸ on a 3 % day) is < 1 % of BTC's daily
  spot+futures volume, so the plausible impact is 1–2 bp — BELOW the 3.9 bp MDE. The test cannot see an effect that
  small, and an effect that small cannot be traded at ≥ 5 bp cost. Either way: not a book for us.
- The high-|r_day| tercile is −9 bp gross (t −2.8): big days REVERSE into the close. That is a contrarian cell, NOT
  pre-registered, read here only as a caveat; it would need its own PREREG on a fresh period before anyone believes it.
- Caveats: rule-based NYSE calendar, early-close days kept; zero-volume minutes forward-filled ≤ 5 min; bar-close
  entry vs next-bar-open entry agree.

## Programme count
Cell 1 of the crypto-ETF line. Closed on this mechanism; the decay-harvest (short both legs) mechanism is untested and
blocked on a borrow quote.
