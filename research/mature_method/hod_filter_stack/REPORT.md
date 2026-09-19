# HOD-break filter stack — REPORT (2026-09-19)

Cells exactly as declared in `PREREG.md` (committed `731bb28` **before any cell was scored**).
Artifacts: `pass2.py` → `sig2.csv` (153,473 signal rows) + `blite.csv` (2,254,214 raw break bars) ·
`feats.py` → `pop.csv` (147,196 rows after membership, 75 columns) · `repro.py` → `repro.log` ·
`score2.py` → `score2.log` · `learn.py` → `learn.log` · `supp2.py` → `supp2.log` · `ofi.py` +
`arms.py` → `arms.log`. One python process at a time, `nice -n 10`, `ulimit -v 3000000`;
`bars_sip.db`, `data/cache.db`, `data/trades.db` opened **read-only**. No config, `orb.yaml`,
systemd unit, cron, order or cache was written. The dry run was not touched.

---

## VERDICT — **STAY DRY AS INSTRUMENT**

*The money the owner is pointing at is real and this pass measured it again: on the corrected base
population the best in-sample feature splits the book by **+0.68 R a trade** (entry minute, t 15.5,
sign-consistent across both halves of 2025). **None of it survives to the next period** — the same
cut reads **+0.06 R** on VAL. The strongest day-level gate — SPY's own 09:30→10:00 direction — looked
era-consistent at +0.156 R (TRAIN, t 2.49) / +0.264 R (VAL, t 2.81) and then **failed its own
causality trace**: it applies a 10:00 fact to trades that fire at 09:31–09:59 (43 % of the
population). Made causal it is +0.028 R on TRAIN (t 0.42). A purged, embargoed walk-forward learner
on 35 features **loses to the unfiltered base on both splits and is beaten by its own shuffled-label
control**. The CKS order-flow arm was priced, pulled for $12.73 at 74.7 % coverage, and its TRAIN
tercile spread (+0.124) **reverses on VAL** (−0.172). 0 of 25 declared cells clear the claim bar;
0 clear the live-exploration bar; **TEST was never opened**.*

The instrument stays because `dry_run: true` places zero orders and is the only source of forward
data in the programme that cannot be a look-ahead. Not SHIP-TO-DRY: the dry run already runs the
shipped rule, and this pass supplies no `HodBreakParams` change worth making.

---

## 1. Reproduction gate — EXACT — and the independent-pass check

`score2.sig_set` at the shipped knobs on THIS study's own bar pass, vs `hod_break/REPORT.md` §6:

| split | this pass | §6 reference | verdict |
|---|---|---|---|
| TRAIN | 1,688 tr · 31.8/wk · gross −0.027 · net −0.088 · green **41.5 %** · **−$14,835** | 1,688 · 31.8 · −0.027 · −0.088 · 41.5 · −14,835 | **MATCH** (Δn 0, Δ$ 0) |
| VAL | 820 tr · 35.7/wk · gross +0.016 · net −0.050 · green **43.5 %** · **−$4,128** | 820 · 35.7 · +0.016 · −0.050 · 43.5 · −4,128 | **MATCH** (Δn 0, Δ$ 0) |

**Independent-pass check.** `pass2.py` re-derives every signal from the bars and is a *different*
pass from `hod_break/pass_breaks.py` (different seeding, different stop set, different emission
rule). On the shared population: **15,653 of 15,653 symbol-days shared, 100.0000 % identical entry
minute, max |Δ rr| = 4.4e-16.** The two passes are the same book to floating point.

## 2. The four base populations (stage 2)

`B0` shipped · `B1L` loose consolidation (K=3 / X=8 %) · `B1` **consolidation filter OFF** (same
last-5-bar-low stop, no HOD-proximity test) · `B2` = B1 + **rv upper cut OFF** · `B3` = B2 + spread
≤ 8 % of R. Book 12/day, 4 concurrent, $100 risk. `imp` = share of rows whose spread is imputed.

| base | split | n | /wk | **gross R** | net R | net(band) | t | **green %** | rs | worst $ | **total $** | MDD $ | gr mo % | imp % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0** | TRAIN | 1,688 | 31.8 | −0.027 | −0.088 | −0.089 | −2.64 | **41.5** | 7 | −2,718 | **−14,835** | −17,209 | 25.0 | 1 |
| **B0** | VAL | 820 | 35.7 | +0.016 | −0.050 | −0.051 | −1.06 | **43.5** | 6 | −2,076 | **−4,128** | −7,519 | 40.0 | 1 |
| B1L | TRAIN | 1,475 | 27.8 | −0.108 | −0.175 | −0.189 | −4.88 | 34.0 | 6 | −2,130 | −25,788 | −25,788 | 16.7 | 18 |
| B1L | VAL | 797 | 34.7 | +0.031 | −0.041 | −0.061 | −0.83 | 43.5 | 4 | −1,673 | −3,254 | −4,846 | 40.0 | 26 |
| **B1** | TRAIN | 1,700 | 32.1 | −0.039 | −0.102 | −0.115 | −3.17 | 35.8 | 10 | −2,695 | −17,276 | −20,180 | 33.3 | 22 |
| **B1** | VAL | 777 | 33.8 | +0.044 | −0.024 | −0.047 | −0.52 | 43.5 | 4 | −1,578 | −1,876 | −6,281 | 40.0 | 34 |
| **B2** | TRAIN | 1,622 | 30.6 | −0.039 | −0.107 | −0.140 | −3.42 | 32.1 | 5 | −2,231 | −17,346 | −18,929 | 25.0 | 49 |
| **B2** | VAL | 706 | 30.7 | **+0.083** | **+0.013** | −0.026 | +0.27 | 43.5 | 4 | −957 | **+893** | −2,940 | 40.0 | 57 |
| **B3** | TRAIN | 1,204 | 22.7 | −0.074 | −0.114 | −0.127 | −3.13 | 26.4 | 13 | −1,403 | −13,723 | −14,063 | 8.3 | 28 |
| **B3** | VAL | 596 | 25.9 | +0.047 | +0.005 | −0.013 | +0.10 | 47.8 | 6 | −1,450 | **+313** | −5,019 | 60.0 | 38 |

Pre-book gross R (no book, no slots), TRAIN / VAL: B0 +0.007 / +0.001 · B1L −0.009 / +0.011 ·
**B1 +0.017 / −0.003** · **B2 −0.002 / +0.015** · B3 −0.046 / +0.013.

**What the correction buys.** Removing the wrong-side consolidation rule (B1) is the one change that
improves the book relative to the shipped stack at the SAME frequency (32.1 vs 31.8 a week):
pre-book TRAIN gross rises from +0.007 to +0.017 and VAL's booked total from −$4,128 to −$1,876.
Dropping the `rv < 5` cut on top (B2) flips VAL to +$893 and +0.083 gross and costs TRAIN. The
loose-consolidation form (B1L) is the worst of the four — so it is the *proximity test*, not the bar
count, that is wrong-side. **None of this creates an edge.** Every base is still ~0.2 R a trade
short of its **+0.2151 R** measured cost, and B2's gross is −0.002 / +0.015, i.e. zero.

**Caveat that belongs beside every B2/B3 number**: the measured per-trade NBBO was collected at B0's
signal minutes, so **49–57 % of B2's rows carry an imputed spread** (the declared price-band ×
hour-band median). `net(band)` is the conservative arm and is 0.03–0.04 R worse. Every separation
below is therefore quoted on **gross** as well as net, so that no conclusion rests on the imputation.

## 3. Availability audit — three families declared and dropped, none silently

100 % coverage on `breadth_*`, `cohort_rank_dist`, `rs_vs_cohort`, `spy_dist_hod_pct`,
`spy_ret_open_sig`, `ret_open_sig`, `slope5`, `vwap_dist_pct`, `hod_age_bars`, `prev_close_pos`,
`dow`, `spy_vol20`, `breadth_by_1000`, `sym_prior_n/meanR` and every day-level field. Definitional
NaNs (`ret_open_1000`, `ret_1000_sig`, `spy_ret_30m` ≈ 55 % — a break before 10:00 has no 10:00
anchor) are **outcome-balanced to within 1.3 pp** and were kept.

- **`pm_vol` (premarket volume ≤ 09:30): DECLARED AND DROPPED.** `data/cache.db::intraday_bars_1min`
  is RTH-only (391 bars, first bar 09:30); `bars_sip.db` carries premarket only for the thin-tape
  refetch subset. This was D1's leak on ORB. Not available here, and not imputed.
- **QQQ intraday: DECLARED AND DROPPED.** 3,807 1-min rows (~10 sessions) against 425 needed.
- **FINRA short interest: DECLARED AND DROPPED.** `cdn.finra.org` returns HTTP 403 from this node
  and no other free dissemination-dated source is reachable. The field was never fabricated.
- **The first build of the day-level block FAILED its own audit** (`gap_pct`, `prev_range_pct`,
  `prev_close_pos` missing on 13.2 % of winners vs 28.6 % of losers) because they were taken from the
  screened universe file (~1,560 names a session). Rebuilt from the full `daily_bars` table they are
  **100 % covered with a 0.0 pp outcome gap**. The rule caught a construction defect, which is what
  it is for — and it means PDR *is* in this study, properly.
- `sym_prior_meanR` was 74.6 % covered with a 14 pp outcome gap until "no prior history" was encoded
  explicitly as 0 — a state fully known at the decision instant, not an imputation.

## 4. The separation table — NEW families vs the 13 already tried (stage 4)

Terciles cut on **TRAIN** edges, read on both splits, on **gross** R. `winsor` = the same spread with
TRAIN rr clipped at its 1st/99th percentile (a spread living in a tail shows up here). `halves` =
same sign in H1-2025 and H2-2025. Full tables: `separation.csv`, `separation_b0.csv`, `score2.log`.

### 4a. On B2 (34 features) — the top of the table

| feature | fam | TRAIN spread | winsor | t | halves | **VAL spread** | selectable |
|---|---|---|---|---|---|---|---|
| `entry_m` | OLD | **+0.681** | +0.681 | +15.51 | yes | **+0.060** | yes |
| `dow` | NEW | +0.545 | +0.545 | +12.67 | no | +0.037 | no |
| `gap_pct` | OLD | +0.368 | +0.368 | +8.47 | no | −0.045 | no |
| `sym_prior_meanR` | NEW | +0.288 | +0.288 | +6.61 | no | −0.079 | no |
| `spy_ret_30m` | NEW | +0.283 | +0.284 | +5.49 | no | n/a (56 % cov) | no |
| `rv_profile` | OLD | +0.280 | +0.279 | +6.38 | yes | **−0.096** | yes |
| `spy_dist_hod_pct` | NEW | +0.235 | +0.234 | +5.32 | yes | **−0.014** | yes |
| `prev_range_pct` (PDR) | OLD | +0.187 | +0.188 | +4.13 | no | +0.109 | no |
| `breadth_min` | NEW | +0.178 | +0.177 | +3.96 | yes | +0.062 | no |
| `prev_close_pos` | NEW | +0.171 | +0.171 | +3.86 | no | +0.030 | no |
| `rs_vs_cohort` | NEW | +0.147 | +0.148 | +3.36 | yes | +0.007 | no |
| `breadth_15m` | NEW | +0.144 | +0.144 | +3.26 | yes | +0.041 | no |
| `spy_vol20` | NEW | +0.140 | +0.139 | +3.19 | no | +0.151 | no |
| `spy_5m_ret` | OLD | +0.132 | +0.132 | +3.05 | yes | +0.115 | no |
| `breadth_day` | NEW | +0.105 | +0.105 | +2.47 | yes | −0.153 | no |
| … 19 more, every TRAIN spread ≤ 0.102 … | | | | | | | |

**Top 5 NEW features by TRAIN separation: `dow` +0.545, `sym_prior_meanR` +0.288, `spy_ret_30m`
+0.283, `spy_dist_hod_pct` +0.235, `breadth_min` +0.178. Their VAL spreads are +0.037, −0.079, n/a,
−0.014, +0.062.** Not one carries. The continuous `vwap_dist_pct` (+0.082) is no better than the
binary it replaced; `cohort_rank_dist` (+0.065), `hod_age_bars` (+0.046), `slope_accel` (+0.048),
`vol3_over_10` (+0.044) and `sym_prior_n` (+0.029) are noise. Winsorising changes nothing anywhere —
no spread in this table lives in a tail.

### 4b. On B0, where the four un-rebuildable OLD features are complete (38 features)

| feature | fam | TRAIN spread | t | halves | VAL spread |
|---|---|---|---|---|---|
| `entry_m` | OLD | **+0.518** | +9.21 | yes | −0.006 |
| `drive_min` | OLD | **+0.462** | +8.13 | yes | +0.081 |
| `bar_vol_x` | OLD | **+0.302** | +5.29 | yes | +0.022 |
| `ret_1000_sig` | **NEW** | +0.257 | +3.80 | yes | −0.075 |
| `rv_profile` | OLD | +0.251 | +4.40 | yes | −0.115 |
| `ret_open_1000` | **NEW** | +0.212 | +3.15 | yes | −0.099 |
| `dow` | NEW | +0.210 | +4.02 | yes | +0.053 |
| `breadth_day` | NEW | +0.206 | +3.65 | yes | **−0.225** |
| `rs_vs_cohort` | NEW | +0.190 | +3.33 | yes | −0.108 |
| `breadth_min` | NEW | +0.184 | +3.02 | yes | −0.072 |

**The direct answer to "are the new families better than the 13 already tried?" is NO.** On the
population where the comparison is clean, the **four best separators are all OLD**, and the top three
were already inside the causal-filter study's 17. The best NEW feature (`ret_1000_sig`, the
acceleration split) ranks 4th and its VAL spread is −0.075. `breadth_day` is the sharpest NEW
in-sample number in the pass and has the sharpest VAL reversal in the table.

### 4c. The one number that settles the pass

`entry_m` deciles on B2, cut on TRAIN, read in both splits (gross R):

```
TRAIN  [577,578)+0.04  [578,586)-0.07  [586,594)-0.01  [594,605)-0.02  [605,615)-0.36
       [615,621)-0.69  [621,649)-0.24  [649,710)+0.09  [710,803)+0.60  [803,840)+0.55
VAL    [577,578)+0.14  [578,586)-0.01  [586,594)+0.11  [594,605)+0.13  [605,615)-0.10
       [615,621)-0.14  [621,649)-0.04  [649,710)-0.02  [710,803)-0.15  [803,840)-0.25
```

**1.29 R of in-sample dispersion (−0.69 → +0.60) becomes 0.39 R of noise with the sign of the two
best buckets REVERSED.** The 10:15–10:21 hole that `hod_break` and the causal-filter study both
independently found at −0.69 R reads −0.14 R five months later. This is the same ~1 R that
`bf_zero` §6b measured between end-of-day cohorts. It is there. It is not observable at 10am by
anything in this feature set, and the largest thing that looks like it swaps sign between years.

## 5. Rule cells (stage 4, 10 declared)

The pre-committed rule (spread ≥ 0.20 R, n ≥ 300 each side, same sign in both TRAIN halves, max 5)
selected **three** features: `entry_m`, `rv_profile`, `spy_dist_hod_pct`.

| cell | split | n | /wk | gross R | net R | t | **green %** | rs | worst $ | **total $** | MDD $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 `entry_m` T2 | TRAIN | 842 | 15.9 | +0.049 | −0.010 | −0.20 | 35.8 | 5 | −1,382 | −801 | −5,355 |
| R1 `entry_m` T2 | VAL | 470 | 20.4 | −0.010 | −0.073 | −1.19 | 43.5 | 3 | −1,189 | −3,432 | −4,877 |
| R2 `rv_profile` T0 | TRAIN | 816 | 15.4 | −0.020 | −0.084 | −1.80 | 35.8 | 8 | −1,490 | −6,894 | −7,634 |
| R2 `rv_profile` T0 | VAL | 510 | 22.2 | +0.000 | −0.068 | −1.16 | 34.8 | 4 | −1,500 | −3,483 | −4,406 |
| **R3 `spy_dist_hod_pct` T2** | TRAIN | 612 | 11.5 | +0.033 | −0.035 | −0.67 | **45.3** | 5 | −1,947 | −2,131 | −6,706 |
| **R3 `spy_dist_hod_pct` T2** | VAL | 351 | 15.3 | +0.097 | **+0.027** | +0.40 | 43.5 | 4 | −840 | **+956** | −2,244 |
| R-AND2 | TRAIN | 523 | 9.9 | +0.028 | −0.031 | −0.52 | 43.4 | 4 | −787 | −1,620 | −3,854 |
| R-AND2 | VAL | 314 | 13.7 | −0.029 | −0.093 | −1.24 | 47.8 | 3 | −1,064 | −2,908 | −4,124 |
| R-AND3 | TRAIN | 96 | 1.8 | −0.055 | −0.113 | −0.84 | 20.8 | 4 | −391 | −1,084 | −2,039 |
| R-AND3 | VAL | 64 | 2.8 | −0.251 | −0.316 | −2.01 | 26.1 | 5 | −353 | −2,024 | −2,308 |

**The decile middle-bucket veto the causal-filter study forbade itself** (edges printed before
scoring, `score2.log`):

| cell | worst TRAIN decile | TRAIN | VAL |
|---|---|---|---|
| V1 `entry_m` d5 = **[615, 621) = 10:15–10:21 ET**, −0.690 R on 345 rows | green 35.8 %, **−$15,619** | green **56.5 %**, **+$1,505** |
| V2 `dow` d0 = Monday, −0.216 R on 1,864 rows | green 28.3 %, −$16,045 | green 43.5 %, −$577 |
| V3 `gap_pct` d1 = **[−5.73 %, −3.59 %)**, −0.423 R on 457 rows | green 34.0 %, −$17,820 | green 43.5 %, +$90 |

**Best rule cell: R3** — the break level in the top tercile of "how far SPY is below its own HOD" —
the only rule cell with positive dollars anywhere: **+$956 on VAL at 15.3 trades a week and −$2,131
on TRAIN**. `R-AND3` is the frequency floor's warning made flesh: 1.8 trades a week, −0.25 R on VAL.

## 6. The learner (stage 5) — beaten by its own shuffled control

35 features, purged and embargoed walk-forward (182-day window, 5-session embargo, ≥ 50 training
sessions, `n_jobs=1`, no random K-fold), target `P(net R > 0)`. **Purged training rows: 0, by
construction** — an HOD trade closes in its own session. 6,251 of 7,027 rows scored across 17 test
months, so 40 TRAIN weeks / 22 VAL weeks are scorable and the base is re-quoted on those months.

| cell | split | n | /wk | gross R | **net R** | t | **green %** | worst $ | **total $** |
|---|---|---|---|---|---|---|---|---|---|
| B2 (scored months) | TRAIN | 1,255 | 31.4 | −0.041 | −0.109 | −3.06 | 32.5 | −2,231 | −13,666 |
| B2 (scored months) | VAL | 706 | 32.1 | +0.083 | **+0.013** | +0.27 | 45.5 | −957 | **+893** |
| L-k4 top-4/day | TRAIN | 740 | 18.5 | −0.046 | −0.114 | −2.55 | 32.5 | −1,430 | −8,443 |
| L-k4 top-4/day | VAL | 408 | 18.5 | +0.011 | −0.058 | −0.91 | 50.0 | −1,122 | −2,349 |
| L-k8 top-8/day | TRAIN | 1,100 | 27.5 | −0.047 | −0.115 | −3.05 | 27.5 | −1,871 | −12,704 |
| L-k8 top-8/day | VAL | 595 | 27.0 | −0.038 | −0.109 | −2.15 | 31.8 | −1,582 | −6,486 |
| L-k12 top-12/day | TRAIN | 1,173 | 29.3 | −0.046 | −0.114 | −3.12 | 35.0 | −2,004 | −13,351 |
| L-k12 top-12/day | VAL | 642 | 29.2 | −0.055 | −0.127 | −2.64 | 27.3 | −1,652 | −8,179 |
| **SHUFFLED LABEL top-8** | TRAIN | 1,108 | 27.7 | −0.022 | **−0.089** | −2.30 | 37.5 | −1,820 | −9,857 |
| **SHUFFLED LABEL top-8** | VAL | 614 | 27.9 | −0.028 | **−0.097** | −1.90 | 27.3 | −1,523 | −5,938 |

**The declared top-k cells are themselves a look-ahead in slot allocation** — "the top 4 of the day"
is not knowable at 09:45 — so the causal twin (probability threshold fit on TRAIN only) is the form
that could ship: `L-t4` VAL **−$4,284**, `L-t8` VAL **−$4,643**, `L-t12` VAL **−$507**. All negative.

**Did the learner help? No — and it did not beat noise.** The real model's top-8 book is −0.115 /
−0.109 R; the **shuffled-label** model's top-8 book is −0.089 / −0.097 R, i.e. the shuffled control
is *better than the real model on both splits*. Per-family ablation (`learn.log`) moves VAL net
between −0.044 and −0.121 R with no family standing out, and dropping the whole day-level family
gives the **best** VAL number. The regime the ORB meta-label study lacked (0 R/pick, ~1 R of hidden
dispersion, more rows) turned out not to be the binding constraint: there is nothing here to learn.

## 7. Day-level cells (stage 6) — and the look-ahead inside my own best result

| cell | split | n | /wk | gross R | net R | t | **green %** | flat % | worst $ | **total $** | MDD $ | gr mo % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D-a breadth TOP tercile | TRAIN | 531 | 10.0 | −0.033 | −0.107 | −2.01 | 32.1 | 28.3 | −1,916 | −5,670 | −7,538 | 41.7 |
| D-a breadth TOP tercile | VAL | 450 | 19.6 | +0.087 | +0.017 | +0.29 | 43.5 | 4.3 | −715 | +754 | −2,059 | 40.0 |
| D-b breadth BOT tercile | TRAIN | 547 | 10.3 | −0.130 | −0.194 | −3.57 | 30.2 | 20.8 | −1,592 | −10,620 | −11,008 | 25.0 |
| D-b breadth BOT tercile | VAL | 74 | 3.2 | −0.081 | −0.154 | −1.02 | 13.0 | 60.9 | −1,052 | −1,137 | −2,139 | 25.0 |
| **D-c SPY 09:30→10:00 up** | TRAIN | 875 | 16.5 | +0.034 | −0.035 | −0.82 | 37.7 | 3.8 | −1,617 | −3,077 | −6,400 | 33.3 |
| **D-c SPY 09:30→10:00 up** | VAL | 372 | 16.2 | **+0.207** | **+0.137** | **+2.08** | **73.9** | 4.3 | −827 | **+5,112** | −1,510 | **80.0** |
| D-d `spy_vol20` BOT tercile | TRAIN | 531 | 10.0 | −0.057 | −0.127 | −2.38 | 13.2 | 64.2 | −2,136 | −6,749 | −7,507 | 28.6 |
| D-d `spy_vol20` BOT tercile | VAL | 293 | 12.7 | +0.076 | +0.008 | +0.11 | 13.0 | 56.5 | −765 | +233 | −1,090 | 33.3 |
| D-e `spy_vol20` TOP tercile | TRAIN | 527 | 9.9 | −0.088 | −0.154 | −2.77 | 17.0 | 58.5 | −2,231 | −8,108 | −9,581 | 42.9 |
| D-e `spy_vol20` TOP tercile | VAL | 135 | 5.9 | +0.105 | +0.036 | +0.32 | 13.0 | 78.3 | −467 | +480 | −467 | 100.0 |
| D-f skip Monday | TRAIN | 1,309 | 24.7 | −0.054 | −0.123 | −3.56 | 28.3 | 1.9 | −2,116 | −16,045 | −17,868 | 25.0 |
| D-f skip Monday | VAL | 597 | 26.0 | +0.060 | −0.010 | −0.19 | 43.5 | 4.3 | −1,037 | −577 | −3,339 | 60.0 |
| **D-g SPY up AND breadth ≥ mid** | TRAIN | 677 | 12.8 | +0.038 | −0.033 | −0.68 | 41.5 | 11.3 | −1,617 | −2,213 | −5,323 | 41.7 |
| **D-g SPY up AND breadth ≥ mid** | VAL | 358 | 15.6 | **+0.204** | **+0.135** | **+2.01** | **78.3** | 4.3 | −893 | **+4,840** | −1,990 | **80.0** |
| D-h = D-g ∧ D-c | — | identical to D-g (D-g implies D-c) | | | | | | | | | |

`D-d` / `D-e` carry a 56–78 % **flat-week** share, so their green-week ratios are mostly no-trade
weeks and are not comparable to the rest; the flat column exists to say so.

**D-c/D-g looked like the find of the pass.** SPY-up mornings vs SPY-down mornings, kept-minus-
rejected net R, with the gross number beside it so it cannot be the cost model talking:

| base | TRAIN sep (t) | VAL sep (t) |
|---|---|---|
| B0 | −0.022 (−0.33) | +0.264 (+2.81) |
| B1 | +0.053 (+0.81) | +0.338 (+3.63) |
| **B2** | **+0.156 (+2.49)**, gross +0.159 | **+0.264 (+2.81)**, gross +0.263 |
| **B3** | **+0.223 (+3.07)**, gross +0.224 | **+0.321 (+3.16)**, gross +0.320 |

On the corrected bases that is the first gate in the whole programme with the **same sign and t ≥ 2.5
in both years**, a stateable mechanism (a long-only breakout book continues when the tape is bid and
fails when it is offered), gross ≈ net so it is real selection and not arithmetic, and it cuts the
TRAIN loss from −$17,346 to −$3,077 (B2) and −$13,723 to −$1,038 (B3) while producing the VAL
positive. Weekly dollars, D-g: TRAIN 22/53 green, −$2,213; **VAL 18/23 green, +$4,840, worst −$893**.

**It fails its causality trace.** The 09:30→10:00 SPY return is known only at 10:00, and **43 % of
B2's signals fire before 10:00**. As scored above, the gate retro-applies a 10:00 fact to trades
taken at 09:31–09:59. The two live-implementable forms:

| form | base | TRAIN sep (t) | VAL sep (t) | TRAIN book | VAL book |
|---|---|---|---|---|---|
| **C1** pre-10:00 taken unconditionally, gate applies from 10:00 | B2 | **+0.028 (+0.42)** | +0.225 (+2.46) | −$11,238, 32.1 % green | +$4,437, 65.2 % green |
| | B3 | **+0.018 (+0.23)** | +0.288 (+2.77) | −$7,765, 39.6 % green | +$3,319, 52.2 % green |
| **C2** no entries before 10:00 AND SPY-up morning | B2 | +0.090 (+1.51) | **−0.052 (−0.61)** | −$2,908, 49.1 % green | −$2,420, 34.8 % green |
| | B3 | +0.091 (+1.26) | **−0.011 (−0.12)** | −$3,880, 39.6 % green | −$295, 39.1 % green |
| C3 reference: post-10:00 only, no SPY gate | B2 | — | — | −$9,688, 43.4 % green | −$6,988, 34.8 % green |

**Causal, the era-consistency is gone**: C1 is +0.028 R on TRAIN (t 0.42), and C2 *changes sign*
between the years. The whole TRAIN half of the apparent effect lived in declining EARLY signals on
days the market would later turn out to have sold off — unknowable at 09:45. This is the same class
of error as `bf_zero` §6b and D1, found here by the pre-committed causality trace rather than by the
owner, which is the only reason it appears in this report as a negative instead of as a headline.

## 8. The two labelled-subset arms

### 8a. CKS order-flow imbalance (arm O) — priced, pulled, and dead

Priced first, as declared. `mbp-1` costs $0.36 per 20 symbols per 90 min ≈ **$450** for this study —
**over the $60 cap, NOT pulled**. `bbo-1s` costs $0.026 for the same block; the actual pull was
**$12.73** over 344 sessions on `EQUS.MINI`, well inside the cap. OFI is computed from best-quote
changes alone, which `bbo-1s` supplies at 1-second resolution.

**Coverage, reported before the arm is scored: 7,027 rows pulled, 7,001 merged onto B2 (79.1 %), and
6,617 (74.7 %) carry ≥ 20 quote updates in the 5-minute window** — above the 60 % threshold, so the
arm is scored on B2 proper and not on a rump. Two standing caveats: 1-second sampling is a **proxy**
(intra-second quote events are lost) and `EQUS.MINI` is **one publisher** carrying a single-digit
share of SIP volume.

| field | TRAIN terciles (gross R) | TRAIN spread | t | halves | **VAL spread** | best-tercile book |
|---|---|---|---|---|---|---|
| `ofi_5m` (depth-normalised, 5 min into the break) | +0.051 / +0.016 / −0.074 | **+0.124** | +2.72 | yes | **−0.172** | T0: TRAIN −$11,240 (41.5 % green) · VAL −$7,183 (21.7 % green) |
| `ofi_1m` (the break minute only) | −0.036 / +0.009 / +0.020 | +0.056 | +1.21 | yes | +0.019 | T2: TRAIN −$8,983 · VAL −$1,351 |

`ofi_5m` separates in TRAIN at t 2.72 — and **the best TRAIN tercile is the LOWEST OFI** (the most
sell-side pressure into the break, a contrarian reading with no mechanism behind it), the spread is
below the 0.20 R selection floor, and **VAL reverses to −0.172**. Its best-tercile book loses $11.2 K
and $7.2 K. `ofi_1m` is flat. **The order-flow filter the 9/13 plan asked to be remembered has now
been measured on this book at 74.7 % coverage and is not a filter.** A future pass would have to
argue for `mbp-1` depth (a different instrument at 35× the price), not for re-running this one.

### 8b. News recency — underpowered, reported as such

`news_recency_min` (minutes since the last own-ticker premarket headline; `has_news` was rejected by
the causal-filter study, recency had never been tried). **Coverage on B2 is 15.7 % (1,392 rows)**
because the news backfill was run for B0's population only; TRAIN has 675 rows, below the 900-row
floor the separation table applies, so it is reported as **UNDERPOWERED, not as a null**. On the
covered subset: TRAIN gross +0.057 / net −0.003; VAL gross +0.040 / net −0.021. `news_n` (73.4 %
covered) is degenerate at tercile edges — most rows are 0 articles — with covered-subset gross
+0.042 / +0.071 and net −0.020 / +0.005. Neither is a filter at this resolution; a proper test needs
the backfill re-run over B2's population and is a separate pre-registration.

## 9. Count-matched permutation null (2,000 draws, per-week pick count fixed)

42 declared cell × split nulls (`nulls.csv`). **Three sit outside their band, and not one is
favourably outside on BOTH splits**: `B1L` TRAIN 34.0 vs [18.9, 30.2] **above**, `R1 entry_m T2`
TRAIN 35.8 vs [39.6, 54.7] **below**, `D-g`/`D-h` VAL 78.3 vs [52.2, 73.9] **above**. The shipped
book sits inside its own band on both splits (41.5 vs [28.3, 41.5]; 43.5 vs [30.4, 47.8]) — exactly
what `hod_break` and `orb_gates2` found independently: **green weeks on this book are bought with
pick count, not with week-level timing skill.** `D-g`'s VAL 78.3 % is the one genuinely outside
reading in the pass, and its TRAIN twin is 41.5 % against a null mean of 40.2 %, i.e. nothing.

## 10. Both bars

**Claim bar — 0 of 25 declared cells pass G1** (TRAIN net R > 0 with t ≥ 2.0, ≥ 10 trades/week, and
selected-subset TRAIN gross ≥ +0.25 R). Every TRAIN mean net R is negative except `R1` (−0.010, so
not even that), and **no cell's TRAIN gross exceeds +0.049 R** against the +0.25 R the bar asks and
the **+0.2151 R** the book must clear to break even. G2 was therefore never evaluated and, per
PREREG §7, **TEST was never opened** — `FREEZE.md` records it.

**Live-exploration bar — not met.** It requires a positive point estimate on green weeks *and* on
dollars at $100 risk on **both** splits. The three cells with positive VAL dollars — `R3` (+$956),
`D-c` (+$5,112), `D-g` (+$4,840) — are −$2,131, −$3,077 and −$2,213 on TRAIN; the two day cells fail
their causality trace on top of that; and `D-g` is outside a null band on one split only.

**Frequency floor (≥ 10/wk).** `R3` keeps 11.5–15.3/wk, `D-c` 16.2–16.5/wk, `D-g` 12.8–15.6/wk — all
clear it. `R-AND3` (1.8/wk) and `D-e` (5.9/wk) fail it as well as failing on their numbers.

**Adequacy review.**

1. *Did we test what the book actually IS, and the corrected version of it?* Yes. The reproduction
   gate is exact to the dollar, and two independent bar passes agree at max |Δ rr| = 4.4e-16. The two
   gates the gate map called wrong-side were removed and the result measured (§2). Standing caveats
   unchanged: the universe file is the range ≥ 5 % daily screen, made superset-exact by the causal
   +5 % floor; 1,140 delisted symbol-days have no consolidated tape; touch-only breaks (2.1 % of live
   signals) are outside the population.
2. *Is the cost model right for its venue?* It is the measured per-trade SIP NBBO `hod_break`
   validated, but **49–57 % of B2/B3 rows are imputed** because the NBBO was collected at B0's signal
   minutes. Every separation above is quoted on gross as well as net precisely so that no conclusion
   rests on the imputation; the `net(band)` arm is 0.03–0.04 R worse and changes no verdict.
3. *Does a caveat of our own explain the headline?* No. The one that would have — the D-c look-ahead
   — was found by the pre-committed trace and is reported as the reason the headline is a negative.
4. **What is the MDE?** Per trade, pre-book: **0.051 R** (B2 TRAIN), 0.071 R (B2 VAL), 0.066 / 0.088 R
   (B0), 0.072 / 0.109 R (B3). On the green-week share: **±18.0 pp** over 53 TRAIN weeks and
   **±28.9 pp** over 23 VAL weeks. So the test could see a per-trade effect of 0.05–0.11 R while the
   book needs +0.215 R — **a powered rejection of the effect the book requires, not an underpowered
   null.** It could *not* have resolved a green-week improvement smaller than ~18 pp (TRAIN) or
   ~29 pp (VAL), which is why no cell in §5–§7 is separable from B0 on the primary metric, and that
   limit is itself part of the answer.
5. *Tail dependence.* B2 net −0.107 → ex-top-1 % −0.129 → ex-top-5 % −0.218 (TRAIN); +0.013 → −0.010
   → −0.093 (VAL). B0: −0.088 → −0.109 → −0.198 and −0.050 → −0.073 → −0.158. The book is not a
   lottery ticket whose edge hides in a tail — it has no edge to concentrate.

**Multiplicity — the honest count.** Declared decision cells **25** (4 base + 10 rule + 3 learner +
8 day); actually scored **27** (a 5th base `B1L`, and the selection rule filled only 3 of its 5 rule
slots, so R1–R3 + 2 ANDs). Plus **3** supplementary causal-threshold learner twins, **14** post-hoc
causality/robustness cells (§7's base grid and the C1/C2/C3 forms — none used as a claim), **72**
screening cells (34 features on B2, 38 on B0) and **4** arm cells.
**Programme cumulative: 439 + 23 (hod_break) + 25 + 3 + 14 + 72 + 4 = 580.** Expected largest |t|
under a pure null over 54 decision cell × splits ≈ 2.8–3.0; the largest favourable t on any *causal*
cell in this pass is **+2.08** (`D-c` VAL), on one split.

---

## What the owner asked, answered in four lines

1. **"The money is there."** Yes — measured a third way: 1.29 R of in-sample dispersion across the
   entry-minute deciles on the corrected population (§4c).
2. **"It's the filter that you need to find."** It is not in these 30 new fields. Of the four best
   separators on the clean comparison population, **all four are features the causal-filter study had
   already tried**, and the best new one reverses sign on VAL.
3. **"For weeks or days or trades we should not take."** The day family holds the strongest number in
   the pass — SPY's own morning direction, VAL 73.9–78.3 % green weeks, +$4,840–5,112 — and it is the
   number that failed the causality trace. Its live-implementable form is +0.028 R on TRAIN.
4. **"Ultrathink new combinations."** Two selection methods on the same 35 fields: hand-cut terciles
   found three selectable features whose VAL spreads are +0.060, −0.096 and −0.014; a purged
   walk-forward learner was beaten by its own shuffled-label control on both splits.

## What would change this verdict

1. **A causal, era-consistent GROSS separator.** The bar is +0.25 R of gross on both splits. The best
   causal candidate found here is +0.028 R. Cost-side gates are exhausted — every one of them earns
   70–85 % of its number from the cost term (`hod_break` §5).
2. **The stop, not the filter.** `hod_break` named it and this pass strengthens it: every cell above
   keeps a ~40 % win rate against a 2:1 payoff, and the ex-tail ladder says the losses are broad, not
   tail-driven. Re-shaping the stop is a new pre-registration.
3. **A day-regime field known BEFORE 09:35.** The one day-level effect that appeared is real in
   direction and unavailable in time. A field with the same mechanism computable at the open —
   overnight index futures, the opening auction imbalance, the pre-open NYSE TICK — would be a
   legitimate new pre-registration. It is NOT tested here, and it is the single most promising
   direction this pass produces.
4. **`mbp-1` depth, not `bbo-1s` snapshots** — if anyone wants to reopen order flow. That is a
   different instrument at ~35× the cost, and nothing in §8a argues for paying it yet.

**Recommended action: NONE.** `config.yaml hod_break` stays exactly as the owner set it
(`enabled: true, dry_run: true`). Going live still needs a new pre-registration and the owner's
word, and this study supplies no grounds for one.
