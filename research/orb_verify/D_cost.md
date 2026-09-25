# Lens D — Cost model check on the ORB OOS +0.077R claim

Claim under test: OOS mean +0.077R/fill, n=554 (2023-01..2024-12 + 2025-07..2026-09-23),
R = `_sized_pnl / 375`, entered==1 rows in book_1418.csv / book_1415.csv / book_2025_26.csv.

## 1. What the model actually charges (grep, not assumption)

**Entry — flat band, NOT measured NBBO.** `trading/orb_planner.py:89-151` (live) and
`study_orb.py:46` (`ENTRY_SLIP_BPS_DEFAULT = 30.0`, imported by `study_orb_features.py` and
used to build `research/orb_2023/build_features_2023.py` / `..._2024.py`'s entry_price):
```
entry_price = range_high * (1.0 + entry_slip_bps / 10000.0)   # entry_slip_bps = 30 (orb.yaml:61)
```
One constant applied to every symbol/date regardless of quoted spread, price, or liquidity.

**Exit — flat band.** `study_orb_pipeline_static_lock.py:75` `EXIT_SLIP_BPS = 10.0`, applied as
`exit_px * (1 - 10/10000)` at every exit path (tag_bb, tag_b1, stop/lock, mid_kill, eod — lines
289-449). Same constant in `orb.yaml:330 exit_slip_bps: 10` (live) and `orb.yaml:409-412`'s
reference slippage block (30bps/10bps, "matches BT assumptions").

**Verdict on mechanism:** this is exactly the banded model CLAUDE.md's cost rule forbids
("measured per-trade NBBO... never a band" — the same class of error that turned HOD-break's
published book from +0.3R to -0.62R). BT and live share the identical constants (30/10bps), so
there is no BT/live *parity* defect — but neither side is measured cost. This is a real process
gap, independent of which direction it biases the number.

## 2. Cross-check #1 — `research/exec_cost/REPORT_RECAL.md` (2026-09-20, B+ book, 165 filled, $10K stage)

Its own ORB table (ll.30-40): current flat model "P" = 13.5bps entry vs recalibrated-toward-
realized "O" = 3.1bps. Moving all the way to O changes TRAIN $6,662→$6,953 (+4.4%) and VAL
$6,386→$6,525 (+2.2%) — **immaterial and in the favorable direction** (realized cost is *lower*
than modeled, so the modeled number is conservative, not inflated). The report's own caveat
(l.46-47): "ORB has no per-trade spread column; entry cost is a flat bps on notional" — i.e.
even this "recalibrated" number is still a flat aggregate, not true measured NBBO. **A genuine
measured-per-trade-NBBO ORB cost model has never actually been built** — only flat-vs-flat
comparisons exist.

## 3. Cross-check #2 — live fills in `data/trades.db` (strategy='orb', ran via
`scripts/analyze_orb_slippage.py`, which already implements exactly this comparison)

169 orb rows, 123 filled, 98 round-tripped, 2026-05-18 → 2026-09-23 (the live $10K-stage book):

| leg | BT assumption | live mean | live median | live p90 |
|---|---|---|---|---|
| entry slip | 30 bps | 15.9 bps | 16.3 bps | 32.0 bps |
| exit slip | 10 bps | -21.0 bps | 0.4 bps | 44.7 bps |
| round-trip | 40 bps | -5.2 bps (i.e. 45.2bps *cheaper* than modeled) | — | — |

98/123 entries and 66/98 exits were **better** than the flat-band assumption. Median-based
reading (mean is outlier-driven — one HPQ exit at -1090bps and a few others look like
`exit_trigger_price` logging artifacts, not real fills) says the same thing as the recal
report: **the flat band overstates realized cost, it does not hide a loss.**

## 4. Cross-check #3 — direct join, book_2025_26.csv modeled entry_price vs trades.db fill_price

Joined on (date, symbol), entered==1 rows only (CSV read with `keep_default_na=False`,
ticker 'NA' preserved — none hit in the matched set). 45 of 473 entered-book rows had a live
counterpart (live orb trading only started ~2026-05, so most of the 2025-07-onward book predates
it — expected, not a defect).

```
mean(live_fill - book_modeled_entry) = -10.9 bps
median = -5.5 bps      n = 45
```

Same sign again: the pipeline's modeled entry (range_high × 1.003) sits *above* what live orders
actually paid, on average. Consistent across three independent measurements (recal report, full
slippage script, direct join).

## Bottom line for this lens

- **No evidence the cost model manufactures the +0.077R edge.** Every available check (recal
  report, live-fills summary, direct entry-price join) points the same direction: the flat
  30bps/10bps band is mildly *conservative* vs what the live book actually paid, not permissive.
  Recomputing the claim under "realized" cost would move the point estimate **up**, not down —
  cost is not the mechanism that would refute +0.077R.
- **But the required artifact doesn't exist.** CLAUDE.md's rule is "measured per-trade NBBO,"
  and that has never been built for ORB — only flat-band-vs-realized-live-fills comparisons.
  Treat "cost is fine" as *not disproven*, not as *proven*.
- **Coverage gap is the real caveat.** All three empirical checks above are drawn from the
  2026-05→09 live book only. That validates cost realism for, at most, the tail of the claim's
  2025-07..2026-09-23 segment. It says **nothing** about 2023-01..2024-12 (289 of the claim's
  554 fills, 52% of n) — a different liquidity/spread regime, no live fills exist to check
  against, and small/illiquid-name spreads in 2023-24 could plausibly be wider than what a
  $10K-stage 2026 book with tighter selection sees. The cost model for the majority of the
  sample is unverified, not verified-bad.
- Does not touch the claim's other stated weakness (top-5% tail dependence, ex-top-5% mean
  -0.013R) — that's a separate lens; a conservative-not-permissive cost model does nothing to
  rescue a tail-carried book.

**Threat to claim from Lens D specifically: minor.** Direction of known bias favors the claim
(modeled cost ≥ realized cost); the unverified 52% of the sample is a real gap but not evidence
of overstatement.
