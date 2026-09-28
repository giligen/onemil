# PREREG — cells 1,640–1,642: VIX BASIS CARRY with a DEFINED LOSS (the roll yield of the VIX futures curve)

FROZEN 2026-09-28 20:00 UTC before any number. Programme count: 1,639 → 1,642. Rank #2 of `research/ideas_web/
RANKED_20260928.md` (academic scan #4: term-structure carry, walk-forward IR 0.4–2.3 on futures).

## Mechanism (documented: Simon & Campasano 2014 "The VIX futures basis"; Eraker & Wu 2017)
VIX futures trade above spot most of the time (contango); a short front-month position earns the roll-down as the
contract converges to spot. The premium is compensation for the vol-spike tail (Feb 2018: XIV −96 % in a day; Aug 2024:
SVIX ≈ −40 %). The basis itself predicts the futures return (the documented timing signal): carry when the basis is
wide, stand aside when it is flat or inverted. Retail expression with a DEFINED loss: long a short-vol ETP (loss ≤ the
notional) or, later, a put spread on a long-vol ETP (loss ≤ the premium). This cell tests the timing mechanism on the
ETPs; the option expression is a later cell that needs OPRA. Honest expectation up front: at a notional sized so that
a −60 % day costs ≤ $450, the carry is ≈ $5–15/month — this is a MECHANISM check feeding the option sleeve, not a
money book on its own.

## Data (free)
CBOE daily VIX futures settlements per expiry (cboe.com historical VX CSVs) → the constant-maturity 30-day futures
level F30 by linear interpolation of the two nearest expiries; VIX spot close (CBOE VIX_History.csv); Alpaca daily
bars for SVXY (2011-10→; −1x until 2018-02-27, −0.5x after), SVIX (2022-03→, −1x), UVXY (1.5x since 2018; reverse
splits: adjusted series required — price-scale check mandatory), VIXY. Sample 2011-10 → 2026-09-04. TRAIN 2011–2019,
VAL 2020–2023, TEST 2024-01..2026-09 sealed (one read for the single best passing cell).

## Signal and trades
Basis b_t = F30_t / VIX_t − 1 from settlements at the close of day t. ENTER long the short-vol ETP at the open of day
t+1 (auction price, 5 bps) when b_t ≥ +3 % and b_{t−1} ≥ +3 % (two closes, pre-declared). EXIT at the next open when
b_t ≤ 0 (backwardation) or VIX_t > 1.25 × its 20-day mean (the spike guard). No intraday stops (the ETP loss is
bounded by the notional; the daily reset ETP cannot go negative).
* 1,640 SVXY, basis-gated, whole sample (report the −1x era and the −0.5x era separately; the −0.5x P&L is the live
  expression).
* 1,641 SVIX (−1x, 2022→), basis-gated (short sample, report-only beside 1,640's −1x era).
* 1,642 ALWAYS-IN SVXY (no gate; report-only): the gate must beat it on drawdown and on the worst day (the mechanism).
Report per cell and split: days in market (share), mean net bps/day in market, Newey-West t (5 lags) of the daily
P&L, annualised return on notional, worst day, worst month, max drawdown, the basis-decile table of next-day ETP return
(the mechanism: return rising in the basis), per-year, ex-worst-5-days, the P&L at $750 notional (the $450 tail budget)
and at $5K notional (the tail in dollars beside it).

## Pass bar (frozen; VAL = 2020–2023, cell 1,640 −0.5x era)
Net ≥ +4 bps/day in market, NW t ≥ 2.5, ≥ 40 % of days in market, TRAIN (2011–2019) same sign t ≥ 1 in the −1x era,
the basis-decile table monotone on both halves, worst day ≥ −25 % of notional, max drawdown ≥ −35 %, and the gate's
worst day and drawdown both better than always-in (1,642).

## Independent check and consequences
Rebuild from the prose (in-market day set Jaccard ≥ 0.98, bps within 1); refuters: settlement time vs ETP close
(both 16:00 ET — the signal uses day t, the entry day t+1's open; nothing from t+1), the F30 interpolation at expiry
rolls (the Wednesday expiry), ETP reverse splits and the 2018 leverage change (price-scale check on every jump > 40 %),
survivorship (XIV's death is in SVXY's own −1x history: 2018-02-05 must be in the sample as a −90 % day), the two-close
rule vs one, tails (ex-worst-5-days). PASS → the option expression cell (UVXY put spread / SVIX call spread, OPRA
pull ≈ $30) is frozen next; the ETP itself is run on paper at $750 notional as the forward instrument. FAIL → closed;
the option sleeve is not pursued on this mechanism.

## Not allowed
Tuning the +3 %, the spike guard or the exit after a number; adding leverage; any intraday version.
