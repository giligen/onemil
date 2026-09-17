# Stage K diagnostics — controls on the scorer

Not pre-registered cells; they cannot promote anything. They exist because a NULL needs the same scrutiny as a finding (`feedback_independent_check_before_claims`).

## D1 — market control: RANDOM symbol-days from the same universe, same machinery

| control | split | n | gross bps | net bps (PREREG cost) | net bps (auction cost) | t |
|---|---|---:|---:|---:|---:|---:|
| random, hold 1 | TRAIN | 22,122 | +2.3 | -42.1 | -7.7 | -22.84 |
| random, hold 1 | VAL | 10,488 | +10.2 | -33.4 | +0.2 | -12.30 |
| random, hold 2 | TRAIN | 22,050 | +7.5 | -36.8 | -2.5 | -13.41 |
| random, hold 2 | VAL | 10,497 | +16.9 | -26.8 | +6.9 | -6.07 |
| random, hold 3 | TRAIN | 22,085 | +16.2 | -28.3 | +6.2 | -7.84 |
| random, hold 3 | VAL | 10,539 | +22.8 | -21.1 | +12.8 | -3.72 |
| random, hold 5 | TRAIN | 22,194 | +25.8 | -18.5 | +15.8 | -4.14 |
| random, hold 5 | VAL | 10,426 | +22.4 | -21.5 | +12.4 | -2.94 |
| random, hold 10 | TRAIN | 22,059 | +50.4 | +5.9 | +40.4 | +0.84 |
| random, hold 10 | VAL | 10,502 | +61.7 | +17.8 | +51.7 | +1.64 |

Read: this is what a coin flip earns on the K universe under the exact fills, exits and costs of the 20 cells. It is the zero line every declared family must beat.

## D3 — K2 (new 250-day high on >=2x volume) decomposed against daily_addons M41

`research/lit_review_2026/daily_addons.md` reports M41 (new 252d high on >=1.5x volume, top-4 per day, NO stop, entry at the SIGNAL CLOSE, hold 10) at TRAIN net +103.6 bps / VAL +389.3 / TEST -183.2. Stage K's K2 is TRAIN net -78.8. The difference is entirely in the three conventions below, each switched one at a time on the SAME signal set.

| K2 variant | split | n | gross bps | net bps (PREREG cost) | net bps (auction cost) | t |
|---|---|---:|---:|---:|---:|---:|
| as declared: next-open entry + 7% stop, hold 10 | TRAIN | 1,647 | +69.5 | +23.8 | +59.5 | +0.79 |
| as declared: next-open entry + 7% stop, hold 10 | VAL | 1,174 | +108.2 | +64.1 | +98.2 | +1.44 |
| next-open entry, NO stop, hold 10 | TRAIN | 1,647 | +87.0 | +41.3 | +77.0 | +1.20 |
| next-open entry, NO stop, hold 10 | VAL | 1,174 | +143.3 | +99.2 | +133.3 | +2.07 |
| signal-CLOSE entry (M41 convention), NO stop, hold 10 | TRAIN | 1,647 | +113.1 | +67.4 | +103.1 | +1.93 |
| signal-CLOSE entry (M41 convention), NO stop, hold 10 | VAL | 1,174 | +159.1 | +115.0 | +149.1 | +2.36 |
| signal-CLOSE entry, 7% stop, hold 10 | TRAIN | 1,647 | +89.4 | +43.7 | +79.4 | +1.44 |
| signal-CLOSE entry, 7% stop, hold 10 | VAL | 1,174 | +133.6 | +89.5 | +123.6 | +1.96 |

All four are the WHOLE signal set (no book, no slot limit) — the book layer is isolated in D2 below.

## D2 — the book layer on random signals (does first-come slotting destroy value?)

| book | split | n | gross bps | net bps | net bps (auction) | t |
|---|---|---:|---:|---:|---:|---:|
| random hold 3, 10 slots | TRAIN | 800 | +6.5 | -37.6 | -3.5 | -1.90 |
| random hold 3, 10 slots | VAL | 340 | -8.1 | -52.2 | -18.1 | -1.60 |
| random hold 3, 20 slots | TRAIN | 1,600 | +19.6 | -24.9 | +9.6 | -1.86 |
| random hold 3, 20 slots | VAL | 680 | -7.9 | -51.4 | -17.9 | -2.40 |
| random hold 5, 10 slots | TRAIN | 480 | +11.8 | -30.5 | +1.8 | -0.99 |
| random hold 5, 10 slots | VAL | 210 | -49.6 | -94.4 | -59.6 | -1.91 |
| random hold 5, 20 slots | TRAIN | 960 | +15.1 | -27.9 | +5.1 | -1.23 |
| random hold 5, 20 slots | VAL | 420 | +44.1 | +0.2 | +34.1 | +0.00 |

## D4 — SPY buy and hold, the scale reference

- SPY TRAIN (2025-01-02..2025-12-31, 250 days): +16.64%  = +6.7 bps/day
- SPY VAL (2026-01-02..2026-05-29, 102 days): +10.73%  = +10.5 bps/day

