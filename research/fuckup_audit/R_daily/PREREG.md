# Stage R_daily — the K families on eight years of daily history (pre-registered 2026-09-18, before any return was computed)

Owner approved the XNAS.ITCH daily purchase on 2026-09-18 to give the multi-day family real
statistical power. Stage K and Stage N2 both died on POWER, not on a finding: a half-year TRAIN split
at 6–19 trades/week could not have seen a per-trade effect smaller than **80–205 bps**
(`N_databento/N2/REPORT.md` §1), and a multi-day book plausibly earns 20–60. This stage re-runs the
SAME five families on a panel that starts in 2018.

**Written before the run. Nothing below was chosen after seeing a return on this panel.**

## 1. Data — one continuous daily panel 2018-05 → 2026-09, Nasdaq-listed only

| span | source | on disk / bought |
|---|---|---|
| 2018-05-01 … 2023-12-29 | XNAS.ITCH `ohlcv-1d` ALL_SYMBOLS | already on disk (`N3/xnas_daily.parquet`, N3 spent $19.38 + $0.30) |
| 2024-01-02 … 2024-06-28 | XNAS.ITCH `ohlcv-1d` ALL_SYMBOLS | **bought today, priced $1.8155 before the first byte** (stop was $10) |
| 2024-07-01 … 2026-09-04 | EQUS.SUMMARY `ohlcv-1d` (consolidated tape) | already on disk |

**Every result of this stage is a NASDAQ-LISTED-UNIVERSE result.** XNAS.ITCH is the Nasdaq venue
tape; for a Nasdaq-listed name its open/close are the Nasdaq opening/closing crosses (the official
prints) and its volume is the Nasdaq share of consolidated volume. For an NYSE-listed name the same
fields would be arbitrary off-primary prints, so NYSE/ARCA/BATS listings are excluded on BOTH sides
of the seam and no claim here extends to them.

Membership (`research/scripts/pit_listings.py`, EQUS.SUMMARY definitions, 2024-07 … 2026-09):
`exchange == 'XNAS'` and `security_type == 'C'`, test tickers (`^Z[A-Z]ZZT`) removed → 3,257 distinct
symbols, median 2,551 per month. EQUS era: membership per month. ITCH era: the union of those months
— **a survivorship filter on 2018–2024 that is stated, not hidden** (a name delisted before 2024-07
is absent; for a long-only continuation book that biases results UP, so a positive TRAIN result is an
upper bound and a negative one is safe). The secondary universe (every symbol in the files, PIT
membership NOT required, test tickers still out) is carried beside it as the survivorship control.

**Volume across the seam** (`seam.md` §2): on the one overlapping month (2024-09, bought by N3) the
median ratio ITCH volume / consolidated volume for liquid Nasdaq-listed commons is **0.3694**
(p25 0.3016, p75 0.4332, n = 15,430). Every ITCH-era volume is multiplied by **1/0.3694 = 2.707** so
that ONE `20-day median dollar volume ≥ $10M` rule means the same thing on both sides. Signals are
suppressed on the 20 sessions after 2024-07-01, the only days whose 20-day volume window straddles
the seam.

**Price scale** (`seam.md` §3): ITCH vs EQUS on the overlap month, 52,802 Nasdaq-listed keys — median
absolute difference 0.0000%. `cache.db::daily_bars` holds 19 rows before 2024-07, so the ITCH era has
no direct Alpaca overlap; the chain is ITCH → EQUS (52,802 keys here) → Alpaca (N2: 200 keys, 99.0%
within 0.01%). Both files are UNADJUSTED: 1,872 overnight close ratios ≤ 0.55 or ≥ 1.80 on
Nasdaq-listed commons in the ITCH era (568 within 5% of an exact 2:1 or 3:1) are flagged in
`split_candidates.csv` and reported, never silently adjusted; a control book drops every trade whose
hold window contains one.

**Known measurement deviation, declared before the run.** A Databento `ohlcv-1d` bar on XNAS.ITCH
aggregates the whole ITCH session (04:00–20:00 ET), so in the ITCH era `open` is the first print of
the extended session rather than the 09:30 auction. Audited free against N3's minute tape (the true
09:30 print), 2021–2022, 350,190 liquid Nasdaq-listed keys: **signed median 0.0 bps**, median
absolute 26 bps, and on K1-style signal days −53 bps (2021, n = 41) / −0.0 bps (2022, n = 31) — i.e.
symmetric noise, and what little conditional tilt there is makes the simulated entry *more* expensive
than the obtainable one. Consequence, carried into every number: the ITCH era's per-trade SD (and
therefore its MDE) is inflated relative to the EQUS era; no directional edge is created. Highs, lows
and volume in the ITCH era likewise include extended hours, which widens the day's range and lifts
rolling highs — both make K1/K2/K5 harder to trigger, never easier.

## 2. Families, holds, books — IDENTICAL to `K/PREREG.md`, no re-tuning

K1 gap ≥ +8% on ≥ 3× ADV with the close in the top third of the range, hold 5, stop = the gap day's
low on a close · K2 close at a new 250-day high with volume ≥ 2× ADV, hold 10, stop 7% on a close ·
K3 5-day return in the bottom decile of the universe with close > open, hold 3, no stop · K4 top
decile of the trailing 20-day mean overnight return, hold 1 · K5 50-day high within 10 days, 3-day
decline ≥ 5%, close > 20-day SMA, hold 5, stop 5% on a close. Long only, equal $ per position, one
open position per name, first-come by the family's declared strength key.

**Cells: 5 families × 2 holds (declared, and half of it — K4 extends to 2, the same single deviation
Stage K declared) × 2 books (10 / 20 slots) = 20.** Cumulative for the K line across the program:
20 (K) + 4 (N2) + **20 (here) = 44**; the permutation p is computed across this stage's 20.

Universe rule as K: 20-day median dollar volume ≥ $10M (consolidated-equivalent), close ≥ $5, common
stock (PIT `security_type == 'C'` where the window allows, the union rule before it), test tickers
out, early-close days are not signal days. Fill as K: the next session's OPEN, cost = half the
liquidity-band spread + 5 bps per side; exits at the close of the hold's last bar or at the first
close through the stop. Secondary cost models (daily-band, auction) reported beside the primary,
never substituted for it.

## 3. Splits, gates, power

**TRAIN 2019-01-01 … 2023-12-31 (5 years) · VAL 2024-01-01 … 2025-06-30 · TEST 2025-07-01 …
2026-09-04.** 2018-05 … 2018-12 is warm-up history only (no split label, never scored). The seam
2024-07-01 falls INSIDE VAL — deliberate, so that no split is defined by a change of data source.

Gates are PLAN §1 verbatim, executed by `K/report_k.py` unmodified: **G1** TRAIN mean net > 0,
t ≥ 2.0, ≥ 5 trades/week · **G2** VAL mean net > 0, t ≥ 1.0, ≥ 55% of weeks green, weekly mean above
(G1 survivors // 10) standard errors · **G3** TEST read ONCE, only for cells frozen in writing in
`R_daily/FREEZE.md`, reported whatever it says.

**MDE, stated before any return is looked at.** The pre-committed target is **< 30 bps per trade on
TRAIN**. With K's per-trade SD of 900–2,100 bps at these holds, a t = 2 detection needs
`MDE = 2·SD/√n`, so 30 bps needs n ≈ 3,600–19,600 booked TRAIN trades. Five years at K's booked rate
(10 slots, 5-day holds ≈ 250 trades/year/cell) gives ≈ 1,250 — an MDE of **51–119 bps**. The denser
cells (20 slots, short holds) reach ≈ 2,500–5,000 trades → **25–84 bps**. So: **the purchase buys a
2.5–3× reduction in MDE versus Stage K, and only the densest cells reach the 30 bps target; the
5-day/10-slot cells do not.** The realised MDE per split and per cell is printed in the tables, and
the verdict sentence names it.

Also reported for every cell: tail tests (top 1% / top 5% removed, winners capped at the 95th
percentile of winners), permutation p across the 20 cells (5,000 sign-flip draws, seed 23), the
availability audit per split for every field used, capacity at 1% of the 20-day median dollar volume
(position $, book $, $/month at a $50K book and at the capacity book, worst month), the split-control
book, and the secondary (survivorship-control) universe.

## 4. Decision rule

If no cell clears G1, TEST is NOT read and the stage reports a null with its MDE, in PLAN §1 phrasing
("no edge detectable in THIS universe / horizon / book / window / cost; the smallest effect the test
could have seen was X"). If a cell clears G1 + G2, it is frozen in `FREEZE.md` before TEST is read,
and — if TEST agrees — the report states the exact rule in prose for an independent rebuild and
records that **no daily-order engine exists** (Alpaca's `opg` / `cls` time-in-force is unused in this
repo: a build, not a knob).
