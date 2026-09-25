# Lens C — Obtainability of the ORB out-of-sample tail

**Claim under test:** live ORB config, 554 fills (2023-01..2024-12 ex-Q4'24... actually
2023-01..2024-06 + 2024-07..2024-12) + 2025-07..2026-09-23, mean +0.077R, ex-top-5% mean
-0.013R, i.e. top 5% (28 fills) carries the whole book.

**Reproduced first, exactly.** `trading/orb_csv.read_orb_csv` on `research/orb_2023/book_1418.csv`
(106 entered) + `research/orb_2024/book_1415.csv` (59 entered) + `research/thermo/book_2025_26.csv`
filtered to `date >= 2025-07-01` (389 entered) = **554 fills**, R = `_sized_pnl/375` on `entered==1`
rows: mean **+0.0769R**, total **$15,969.10**, ex-top-5% (28 dropped) mean **-0.0130R**. All match
the claim to 3 decimals — no coding-error in the headline number itself.

## Verdict: the tail is entry-obtainable but NOT clean. Two real defects found, one material.

### 1. Entries: obtainable for all 28 (no evidence of unobtainable fills)
Rebuilt each of the 28 trades' opening range (09:30-09:35 ET) and post-range bars directly from
the source DBs (`research/orb_2023/bars.db` for the 5 pre-2024-07 names, `data/cache.db` for the
23 names from 2025-2026, per task cutoff — all reads done by 11:53 UTC, before the 13:25 UTC
bar). Entry rule confirmed from `trading/orb_engine.py:3222` + `orb.yaml:61`: buy-stop-limit at
`range_high * 1.003` (entry_slip_bps=30). For every one of the 28: the breakout bar (first bar
with `high > range_high`) or the bar immediately after had `high >= entry_price`, i.e. a resting
limit order would have filled inside a printed bar, always within 1 minute of the range close and
inside the pipeline's 60-min entry window. No gap-through, no off-bar fill, no fill before 09:35.
Participation (implied shares from `_rp_position/entry_price` vs `range_total_volume`) was ≤2.7%
of the opening 5-min volume on every trade (median ~0.3%) — no fill would have moved the tape.
EOD closing prints (the exit price for `eod`/`scale_eod` rows) all landed on bars with real
volume (100-255,648 shares), not stale single-tick prints. Full per-trade table (symbol, date,
entry_price, range_high/low, breakout ts, fill ts, obtainable, scale-target-hit, eod_close,
shares, participation%) is in the run log; happy to re-dump to CSV on request.

### 2. MATERIAL: 5 of the 28 tail trades did not go through the live production exit simulator
`study_orb_pipeline_static_lock.py` always sources bars from **`data/cache.db`** (line 553-554),
never from the point-in-time archival DBs the 2023/2024 studies were built on. Its own run log
(`research/orb_2023/run_all.log:157`) prints **"ATR14 available for 0/4567 symbol-days"** and
**"floor bound on 0 resimmed rows; 0 rows scaled 40%@+3.0R"** — for both cell 1418 and 1419. That
is a smoking gun: `data/cache.db` has zero minute-bar coverage back to 2023-2024H1, so the resim
loop's fallback branch (`if bars is None or bars.empty: keep row['pnl']/row['exit_reason']
unchanged`) fired for effectively **every one of the 106 entered trades in book_1418** — not just
the tail. None of them ran through the live static-lock / touchgo / 3R-scale / 15:45-force-close
exit logic. They instead carry whatever `study_orb_features.py::simulate_orb_trade(entry_mode=
'touch', stop_mode='range_low', target_mult=2.0, time_stop_minutes=60)` produced — the exact
"older `orb_features_*.csv::pnl` scripts are NOT production-parity" case CLAUDE.md already warns
about, just not previously traced to this specific book.

Affected tail trades (all `exit_reason` ∈ {target, eod}, all from book_1418, all pre-2024-07):

| symbol | date | entry_price | sized_pnl | exit_reason | pnl_pct |
|---|---|---|---|---|---|
| SIDU | 2024-01-03 | 8.1444 | $703.68 | target | +21.68% |
| NRGV | 2023-01-18 | 4.3330 | $478.55 | target | +14.36% |
| LIFW | 2023-11-06 | 3.2898 | $452.31 | target | +13.57% |
| SLNO | 2023-09-28 | 24.3829 | $449.55 | eod | +13.49% |
| BFRG | 2023-04-14 | 6.7201 | $442.57 | target | +13.28% |

The `pnl_pct` values (13.3-21.7%) are consistent with a fixed-2R target against `range_low`, not
with the live rule's 3R scale/touchgo/15:45 close. (For context: LIFW's raw 1-min tape that day is
a real, gradual, high-volume breakout — 100K-800K shares/min, no discontinuity — so the underlying
price action isn't fabricated; it's the **exit rule applied to it** that isn't the one running
live.) `book_1419`, `research/orb_2024/book_1415` (2024H2) and `research/thermo/book_2025_26.csv`
(2025-2026, `ATR14 available 12953/13280`, `489 rows scaled`) **did** resim correctly — this is
isolated to the 2023–2024-06-28 leg (cell 1418/1419), which is 106 of the 554 fills (19%).

**Impact on the headline number:** dropping these 5 trades (not the whole leg — just the ones that
happen to be in the reported top-28) from the full 554-fill pool:
- mean R: 0.0769 → **0.0653** (n 554→549, total pnl $15,969→$13,442)
- ex-top-5% mean R (recomputed on the 549-trade pool, new top-27 threshold): **-0.0214**, worse
  than the reported -0.0130 — because removing 5 big winners promotes 27 new trades into the "top
  5%" bucket and the body of the distribution is still slightly net-negative.

This does not, by itself, flip the sign of the pooled edge (+0.065R is still positive), but it
means the reported "out-of-regime" 2023-2024H1 leg (n=106, +0.089R/fill, from the same CLAUDE.md
line this claim traces to) needs the same audit before it's relied on — its P&L was never run
through the pipeline that's actually live.

### 3. Soft finding: tail concentration in newly-issued 2x single-stock wrapper ETFs
Cross-checked all 28 tail symbols against `data/research/orb_asset_class_map_20260711.csv`.
**11 of 28 (39%) are leveraged single-stock wrapper ETFs**, not the underlying operating company:
RGTZ (2x short RGTI), NBIG/NEBX (2x long NBIS) + NBIL (2x long NBIS, different issuer) + NBIZ (2x
short NBIS), OKLL (2x long OKLO) + OKLS (2x short OKLO), CRCG (2x long CRCL) + CRCA (ProShares
Ultra CRCL) + CCUP (T-Rex 2x long CRCL), GLXU (2x long GLXY), CWVX (2x long CRWV), LUNL (2x long
LUNR). These are IN by design (`daily_bars` universe note: "2x wrappers IN since 9/5 for ORB") —
not a rule violation — but they collapse the effective independence of the tail: the 4 NBIS
wrappers (NBIG/NBIL/NBIZ/NEBX) all fired on **the same underlying's move on the same day**
(2026-07-30, $463+454+448+447 = $1,812 combined, 11% of total tail P&L), and the 3 CRCL wrappers
(CRCG/CRCA/CCUP) all fired on **2025-10-02** ($561+559+556 = $1,676, another 10%). So "28
independent tail trades" is really closer to **~23 independent underlying-day events** once the
same-day same-underlying duplicates are collapsed, and roughly a fifth of the tail's dollars come
from two single-underlying days on brand-new, thin, correlated derivative products. Not disqualifying
by itself, but it sharpens Lens 5 (tail dependence / lottery-ticket check) beyond what a flat
"top-5%-of-fills" framing shows, and it means a bad print or a halt in NBIS or CRCL on either of
those two days would have hit 3-4 "trades" at once, not one.

## Bottom line
- Fill mechanics are real: all 28 tail entries are obtainable off the actual minute tape, low
  participation, no stale-print exits.
- The claimed +0.077R / -0.013R-ex-tail numbers reproduce exactly from the CSVs as given.
- But 5 of the 28 tail trades (SIDU, NRGV, LIFW, SLNO, BFRG — and, more broadly, all 106 trades in
  the 2023–2024-06-28 leg) were priced by a non-production-parity simulator because the resim step
  silently no-ops when `data/cache.db` has no coverage for the date. Excluding just those 5: mean
  R 0.077→0.065, and the tail gets *more* negative-carrying (ex-top-5% -0.021 vs -0.013 claimed),
  not less.
- Recommend: (a) re-point `study_orb_pipeline_static_lock.py`'s bars source (or add a fallback) so
  the 2023-2024H1 leg actually resims through the live exit rule before it's cited again, (b) when
  reporting tail dependence, collapse same-day same-underlying wrapper duplicates first.
