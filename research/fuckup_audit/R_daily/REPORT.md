# Stage R_daily — the five K families on eight years of Nasdaq-listed daily history

Pre-registered in `R_daily/PREREG.md` (2026-09-18, before any return on this panel). Nothing was
tuned: the families, holds, books, universe gate, fill, cost model, stops, gates, tails, permutation
and capacity are `K/build_k.py` + `K/report_k.py` executing **unmodified**; `run_r.py` repoints the
panel, the output directory, the splits and the early-close calendar, and swaps Stage K's 2026 asset
class map for point-in-time listings.

## 1. One page

**Does five years of TRAIN rescue any multi-day family? No. 0 of the 20 pre-registered cells pass G1,
G2 is unreachable, TEST (2025-07-01..2026-09-04) was NOT read.** One cell out of 40 cell-splits has a
positive mean (`K3_h1_n10` TRAIN, +7.6 bps at t 0.11) and it is **one bad print**: MPWR 2022-10-14
enters at $4.26 against a signal-day close of $310.05 (+7,178% on one trade). Drop the two such
trades and that cell reads **-61.5 bps at t -13.9** (`controls.csv`). Every other cell is negative on
TRAIN *and* on VAL, under the primary cost model, under both secondary cost models, with the top 1%
and top 5% removed, with winners capped, with splits removed, and on the union-rule control universe.

| cell | TRAIN n | tr/wk | **TRAIN net bps** | t | MDE/trade | VAL n | **VAL net bps** | t | VAL wk green% | G1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `K1_h5_n10` | 48 | 1.2 | **-428.0** | -1.75 | 490 | 149 | **-139.1** | -1.51 | 39.6 | fail |
| `K1_h5_n20` | 48 | 1.2 | **-428.0** | -1.75 | 490 | 163 | **-126.5** | -1.48 | 41.7 | fail |
| `K1_h2_n10` | 48 | 1.2 | **-394.8** | -2.13 | 370 | 161 | **-125.8** | -2.01 | 33.3 | fail |
| `K1_h2_n20` | 48 | 1.2 | **-394.8** | -2.13 | 370 | 165 | **-131.2** | -2.15 | 33.3 | fail |
| `K2_h10_n10` | 809 | 3.7 | **-101.8** | -2.49 | 82 | 322 | **-144.5** | -2.73 | 39.5 | fail |
| `K2_h10_n20` | 1,216 | 5.5 | **-91.0** | -2.81 | 65 | 525 | **-107.4** | -2.57 | 39.5 | fail |
| `K2_h5_n10` | 1,162 | 5.3 | **-94.9** | -3.20 | 59 | 491 | **-100.7** | -2.56 | 35.1 | fail |
| `K2_h5_n20` | 1,410 | 6.4 | **-80.4** | -2.96 | 54 | 648 | **-91.7** | -2.78 | 33.8 | fail |
| `K3_h3_n10` | 4,013 | 15.4 | **-40.7** | -3.01 | 27 | 1,168 | **-67.8** | -2.56 | 40.8 | fail |
| `K3_h3_n20` | 7,628 | 29.2 | **-38.4** | -4.12 | 19 | 2,286 | **-57.5** | -2.81 | 40.8 | fail |
| `K3_h1_n10` | 10,366 | 39.6 | **+7.6** [1] | 0.11 | 139 | 3,237 | **-67.9** | -8.48 | 22.4 | fail |
| `K3_h1_n20` | 16,186 | 61.8 | **-15.9** [1] | -0.36 | 89 | 5,369 | **-72.5** | -12.65 | 11.8 | fail |
| `K4_h1_n10` | 12,471 | 47.6 | **-85.3** | -17.76 | 10 | 3,520 | **-82.4** | -6.91 | 23.7 | fail |
| `K4_h1_n20` | 24,929 | 95.1 | **-79.5** | -25.16 | 6 | 7,040 | **-69.3** | -9.75 | 26.3 | fail |
| `K4_h2_n10` | 6,236 | 23.8 | **-75.9** | -7.91 | 19 | 1,760 | **-104.7** | -4.51 | 30.3 | fail |
| `K4_h2_n20` | 12,470 | 47.6 | **-74.9** | -11.96 | 13 | 3,520 | **-79.5** | -5.58 | 28.9 | fail |
| `K5_h5_n10` | 2,279 | 8.9 | **-29.8** | -1.31 | 45 | 789 | **-76.3** | -1.73 | 40.3 | fail |
| `K5_h5_n20` | 3,438 | 13.4 | **-40.5** | -2.32 | 35 | 1,219 | **-91.6** | -2.78 | 39.0 | fail |
| `K5_h2_n10` | 3,482 | 13.6 | **-57.3** | -4.67 | 25 | 1,251 | **-103.6** | -4.83 | 44.7 | fail |
| `K5_h2_n20` | 4,612 | 18.0 | **-57.8** | -5.66 | 20 | 1,623 | **-96.6** | -5.33 | 43.4 | fail |

`tr/wk` is Stage-K's trades-per-ACTIVE-week (weeks with no trade are not counted): K1's 48
TRAIN trades are 0.18 per calendar week over five years, not 1.2.

[1] the MPWR bad print; both cells are -60 bps at t -14 / -18 once it is removed (`controls.csv`).
Full tables incl. tails, the two secondary cost models, capacity and the control universe:
`phaseA.md`, `cells_trainval.csv`, `controls.csv`, `perm_p.csv` (search-adjusted p = 0.98-1.00 for
every cell), `availability.md` (every field >= 99.08% covered on every split — its closing paragraph
about the class map is inherited Stage-K boilerplate and does NOT apply here).

**Power — the point of the purchase, stated before the returns were read (PREREG §3: target < 30 bps
on TRAIN) and now realised.** The MDE (the smallest per-trade mean that would have produced t = 2) is
**6.3 bps** in the densest cell and under **30 bps in 8 of the 20 cells**; Stage K's best was 80 bps
and N2's 96 bps on a half-year TRAIN. So for K3, K4 and K5 at the short holds this is **not a
low-power null**: the books are measurably negative at t = -3 to -25, 30-100 bps per trade below
break-even, on 4,000-25,000 booked trades. K1 (48 TRAIN signals, MDE 370-490 bps) and K2 (MDE 54-82
bps) remain low-power; their TRAIN and VAL signs agree, both negative.

**Where the money goes.** Gross means run -29 to +14 bps outside the bad-print cell; the primary cost
model charges 50-60 bps round trip. Even at the **auction** cost model (10 bps round trip, no quoted
spread — the most generous defensible assumption for an opening-auction fill) only four TRAIN cells
turn positive (K3_h3 +2 to +4, K5_h5_n10 +13, K3_h1 +50 with the bad print) and **every one of them
is negative on VAL**. The families are not being killed by the cost contract; they have no gross edge
to protect.

**Capacity is not the constraint.** One position at 1% of the name's 20-day median dollar volume is
**$170K-$433K**, i.e. a **$1.7M-$8.2M book** at 10-20 slots. At the owner's $50K book every cell is
1/70th of capacity, and the best TRAIN cell's mean month is **-$340 to +$645 per month at $50K**
(worst month -$3,800 to -$8,700 at that size). Nothing here is worth an engine.

**Cells looked at: 20** here (5 families x 2 holds x 2 books). Cumulative for the K line across the
program: 20 (K) + 4 (N2) + 20 (R_daily) = **44**.

**Data spent today: $1.8155** (`metadata.get_cost` before the first byte; stop was $10). Cumulative
daily-history spend on this line: $19.38 + $0.30 (N3) + $2.22 (N2) + $1.82 = **$23.72**.

## 2. The panel

`fetch_gap.py` bought the ONE missing slice — XNAS.ITCH `ohlcv-1d` ALL_SYMBOLS 2024-01-01 -> 2024-07-01,
**1,160,261 rows, 11,661 symbols, 124 sessions 2024-01-02..2024-06-28** — at $1.8155. `build_panel_r.py`
then joined it with the ITCH file already on disk (2018-05-02..2023-12-29) and the two EQUS.SUMMARY
files (2024-07-01..2026-09-04): **3,566,236 rows · 2,278 symbols · 2,098 sessions ·
2018-05-02 -> 2026-09-04**, in the exact layout `K/build_k.py::load_panel` reads.

Universe: point-in-time Nasdaq-listed common stock (`pit_listings`, `exchange == XNAS` and
`security_type == C`, test tickers out) — 3,257 distinct symbols, 2,278 of which ever reach ten
sessions of $10M+ dollar volume (the panel's symbol set, a strict superset of the universe: a 20-day
median of $10M needs ten $10M days inside the window). **Every result here is a Nasdaq-listed-universe
result** and none of it extends to NYSE/ARCA listings, whose ITCH open and close are off-primary
prints. Seam handling, the venue-share constant (ITCH volume x 2.707), the price-scale check
(52,802 overlapping keys, median absolute difference 0.0000%) and the split scan are in `seam.md`.

## 3. The three honest limitations, all declared before the run

1. **Survivorship on the ITCH era.** There is no point-in-time listing source before 2024-07, so
   2018-2024 membership is the union of the bought months. Measured attrition of the Nasdaq-listed
   common universe is **11.7%/year** (2,708 symbols in 2024-07, 684 of them gone by 2026-09), so the
   pre-2024 era is missing roughly that cohort of delisted names each year. The bias is **upward** for
   a long-only book — and every cell is negative anyway, so the conclusion is safe in the direction
   that matters.
2. **The ITCH daily bar is an extended-session aggregate.** Its `open` is the first print of the
   04:00-20:00 ITCH session, not the 09:30 auction. Audited free against N3's minute tape (2021-22,
   350,190 liquid keys): signed median **0.0 bps**, median absolute 26 bps, and -53 / -0.0 bps on
   K1-style signal days (n = 41 / 31) — symmetric noise, conservative if anything, but it widens the
   ITCH era's per-trade SD. Its visible cost: **K1 fires 0.18 times a week on TRAIN against 2.1 on
   VAL** (48 signals in 5 years vs 167 in 18 months) because a 4 a.m. first print rarely shows the
   8% gap the 09:30 auction shows. K1's TRAIN column is therefore weak evidence; K2/K3/K5 (close- and
   high-based) and K4 are not affected this way, and K4's TRAIN and VAL numbers agree closely
   (-85 vs -82 bps) across the data-source seam.
3. **Unadjusted files.** 1,872 flagged overnight ratios <= 0.55 or >= 1.80 (568 at an exact 2:1/3:1);
   43 trades touched one. Removing them moves no cell by more than 11 bps and flips none
   (`controls.csv`, control A). Bad prints (control B) matter in exactly one place, §1's footnote.

## 4. Verdict

**No edge was detectable for the five pre-registered multi-day long-only families — gap continuation,
52-week-high breakout on volume, short-term reversal, overnight continuation, uptrend pullback — in
the Nasdaq-listed US common-stock universe with a 20-day median dollar volume >= $10M and price >= $5,
at 1-10 session holds, in a 10-20 position equal-$ book, over 2019-01-01..2025-06-30, at a cost of
half the liquidity-band spread + 5 bps per side on a market-on-open fill; and the smallest per-trade
effect the test could have seen was 6-30 bps in 8 of the 20 cells and 35-490 bps in the rest.** In the
eight powered cells the result is stronger than a null: the books are negative by 30-100 bps per trade
at t = -3 to -25, and they stay negative on the out-of-sample half. TEST was not read and no cell was
frozen, so `FREEZE.md` is not written and the daily-order engine question does not arise (for the
record: Alpaca's `opg`/`cls` time-in-force is still unused in this repo — it would be a build, not a
knob).

What the purchase bought is worth keeping: **the multi-day direction is now closed with power, not
with a shrug.** The K line ends here unless a *selection* layer is proposed on top of one of these raw
families — the same lesson the BF and ORB books already carry (the raw rule is edgeless; the
selection was the edge).

## 5. Files

`fetch_gap.py` · `seam_check.py` -> `seam.md`, `pit_xnas_common.csv`, `split_candidates.csv` ·
`build_panel_r.py` -> `daily_panel_2018_2026.parquet` (122 MB, gitignored) · `run_r.py` (drives
unmodified `K/build_k.py` + `K/report_k.py`) -> `phaseA.md`, `cells_trainval.csv`, `perm_p.csv`,
`signal_counts.csv`, `availability.md`, `trades/*.csv`, `split_control.csv` · `controls.py` ->
`controls.csv` · `phaseA.log`. The two bought parquets (`xnas_daily_2024H1.parquet`, 24 MB; the raw
DBN, 33 MB) are gitignored.

Note on `signal_counts.csv`: its `train_per_week` column is Stage-K's hardcoded `n / 52`, which for a
five-year TRAIN reads 5x high — the honest per-week figures are in the `tr/wk` column of §1, computed
from the booked trades' own week count.
