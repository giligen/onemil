# Lens B — in-sample audit of the ORB "+0.08R out-of-sample" claim

**Verdict: the claim's "out-of-sample" window is not out-of-sample.** Excluding 2025H1 (the
composite z-params' TRAIN window) is the only exclusion the claim makes. But every other live
parameter — both vetoes, the catalyst-veto on/off switch, touchgo, the spread gate, the entry
buffer, and the position-sizing/slot count that defines the R=$375 denominator itself — was
chosen or validated using data **inside** 2025H2–2026-09, which is the great majority of the
claimed "out-of-sample" 554-fill book. The only window nothing was ever fit or validated on is
**2023-01 to 2024-12** (n=165 of the 554 fills). Recomputed on that window alone: mean R =
**+0.055**, day-clustered t = **1.55**, n=165/133 days — below the pre-registered t≥2 "EDGE" bar
and matching `research/orb_2023/PREREG.md`'s own pre-committed "FLAT" verdict. The pooled t=2.71
the claim reports is manufactured almost entirely by the contaminated 2025H2-2026 slice.

## 1. Reproduction check (before auditing — confirms I'm reading the right data)
Independently rebuilt the claim's population from the raw CSVs via `trading/orb_csv.read_orb_csv`,
`entered==1`, R=`_sized_pnl`/375, day-clustered t = t-test on per-day mean R:

| slice | n | days | mean R | day-t | total $ | ex-top-5% mean R |
|---|---|---|---|---|---|---|
| 2023-01..2024-06 (`orb_2023/book_1418.csv`) | 106 | 89 | +0.089 | 1.62 | $3,542 | — |
| 2024-07..2024-12 (`orb_2024/book_1415.csv`) | 59 | 44 | -0.007 | 0.33 | -$147 | — |
| thermo ≥2025-07-01 (`thermo/book_2025_26.csv`) | 389 | 201 | +0.086 | 2.63 | $12,574 | -0.012 |
| **pooled (claim as stated)** | **554** | **334** | **+0.0769** | **3.05** | **$15,969.10** | **-0.0130 (28 dropped)** |

Matches the claim's stated $15,969 total, +0.077 mean R, and -0.013 ex-top-5% almost exactly (my
day-t=3.05 vs claim's 2.71 — SE methodology differs, not material to what follows). The claim's
own numbers are internally reproducible; the problem is what window they're computed on.

## 2. Every live-config parameter, its ship date, and what it was fit/validated on
(orb.yaml comments + linked reports; `git log`/dates not independently re-verified beyond what's
quoted in-file — budget did not allow re-deriving each report from scratch, only tracing its
stated window.)

| parameter | value | shipped | fit/validation sample (as stated in the source doc) | touches claimed OOS window? |
|---|---|---|---|---|
| `filter.threshold`, `quintile_cutoffs`, 7 z-score features | 0.01208…, [.107,.202,.298,.423] | 2026-08-15 (B+ restart) | TRAIN **2025-01-01→2025-06-30** only (`orb_bplus_frozen_params_aug2026.yaml` `meta.train_window`; matches code default `ORB_BT_TRAIN_START/END`) | **No — genuinely clean**, and the claim already excludes this window |
| `adaptive_mults` (all 1.0, uniform) | 2026-08-15 (neutralized 7/17) | compared "old / shipped-7-10 / flat" books on era splits (2025H1/H2/2026 implied) for "leave-out-top-5 + era-consistency" | **Yes** (2025H2/2026 eras scored) |
| `prev_day_range_veto.min_prev_day_range_pct = 11.0` (was 8.0) | 2026-08-15 | `orb_drag_program_aug2026.md`: combo (incl. PDR) **"chosen on TRAIN+VAL only"** = TRAIN 2025H1 / VAL **2025H2** ("Drag by window... TRAIN→VAL→OOS", eras H1/H2/26); **2026 (Jan-Aug) "OOS unveiled once"** and reported before the owner's GO | **Yes — 2025H2 used to pick the value; 2026 Jan-Aug looked at pre-ship** |
| `g1_veto.return_volatility_20d_min=7.106`, `prev_day_range_pct_min=9.226` | 2026-08-15 | same synthesis, same TRAIN/VAL/OOS split | **Yes** |
| `g1_veto.short_history_veto = false` (rv20==0 fail-open) | 2026-09-13 | orb.yaml comment: "fails its own **2025->2026** check (2025 n=30 -0.26R, 2026 n=29 +0.12R)" — decision explicitly conditioned on 2025H2+2026 outcomes | **Yes** |
| `range_size_veto.min_range_size_pct = 2.221` | 2026-09-07/08 | `orb_veto_study/REPORT.md` accept/reject table has explicit **`25H1 \| 25H2 \| 2026`** columns per variant | **Yes** |
| `catalyst_veto.enabled = false` | 2026-09-19 | `orb_gates2/REPORT.md` §"Population... 2025-01-02 → 2026-09-16" — **TRAIN = 2025 (53 wk, i.e. all of 2025 incl. H2)**, **VAL = 2026-01..05 (22 wk)**, TEST = 2026-06+ (16 wk, sealed) | **Yes — 2025H2 is literally TRAIN; 2026 Jan-May is VAL for this exact switch** |
| `touchgo` (rule_m thresh 0.5, rule_d revert 0.75/exit -0.5) | default-on, unchanged | `docs/CLAUDE_HISTORY.md`:348 — "walk-forward **Jan'25-May'26** (924 trades, 8/11 OOS months helped)" | **Yes — spans nearly the entire pre-June-2026 portion of the claimed OOS window** |
| `entry.max_spread_bps = 300` (was 150) | 2026-07-04 | `orb_spread_gate_verdict.md`: "+$24.7K/**18mo**" back from 7/4/2026 ≈ 2025-01→2026-06 | **Yes (2025H2 + 2026H1)** |
| `entry.stop_limit_buffer_bps = 50` (was 30) | 2026-09-18 | `fuckup_audit/Q_fill/REPORT.md`: "P&L **21 mo**" w/ TRAIN/VAL/TEST columns; explicitly validates fill quality on **"2026-05-19 → 2026-09-17"** | **Yes — reaches to 6 days before the claim's own end date (2026-09-23)** |
| `sizing.max_concurrent=8`, `risk_per_trade_usd=375`, `account_budget_usd=26666.67` (3→8 slots) | 2026-09-17 | `fuckup_audit/D1_orb/REPORT.md`: population **2025-01-02 → 2026-09-16**, splits **"2025 (12mo) / 2026-01..05 / 2026-06+"** | **Yes — the very denominator (375) used to compute the claim's "R" was picked on ~the whole claimed OOS window** |
| `range_minutes`, `entry_mode`, `time_stop_minutes`, `last_entry_submit_time_et`, `dedup.*`, `kill_rails`, `pdt_guard`, `force_close_time_et` | various | structural/correctness fixes (e.g. `last_entry_submit_time_et` fixed a QBTZ/BATL late-fill bug), not P&L-tuned on any split | not applicable — excluded from this audit as non-data-fit |
| `filter.skip_q1 = true` | pre-8/15 | orb.yaml comment cites "VAL +$5,151, HOQ1+ +$3,405" via `study_orb_q1q2_filter.py` — exact VAL/HOQ1+ date ranges **not independently confirmed this session** (budget) | **Flagged, unconfirmed — likely yes given the naming pattern elsewhere in the same file** |

## 3. Which claimed-OOS periods were actually used to pick a parameter
**Virtually all of it.** Between 2026-07-04 and 2026-09-19 (i.e. up to 6 days before the claim's
own 2026-09-23 cutoff), six separate studies used 2025H2 and/or 2026 data as TRAIN, VAL, or a
pre-ship "OOS unveil" to select: the PDR veto value, the G1 veto values, the catalyst-veto
on/off switch, the touchgo parameters (validated through May 2026), the spread gate (300bps),
the entry buffer (50bps), and the slot count/per-trade risk that defines R itself. Only
**2023-01 to 2024-12** — 165 of the 554 fills — was never touched by any of these studies.

## 4. Recomputed claim on the truly-untouched window only
2023-2024 fills only (`orb_2023/book_1418.csv` + `orb_2024/book_1415.csv`), same R=`_sized_pnl`/375,
day-clustered t:

**n=165, 133 trading days, mean R = +0.055, day-clustered t = 1.55, total $3,395.14,
ex-top-5% mean R = -0.006 (8 trades dropped).**

This misses the pre-registered `research/orb_2023/PREREG.md` EDGE bar (net R/fill ≥ +0.10 AND
day-clustered t ≥ 2 AND ex-top-5% > 0 AND both calendar halves positive — 2024H2 alone is -0.007R)
and lands squarely in that PREREG's own pre-committed **FLAT** bucket (|R|<0.10, t<2). It is
*consistent with* the claim's headline sign but does **not** clear the significance bar the claim
implies with t=2.71.

For completeness, even the most literally sealed slice *inside* the contaminated window — thermo
dates ≥2026-06-01, roughly matching `orb_gates2`'s sealed TEST split and postdating Stage-Q's
2026-09-17 tuning cutoff — is also sub-bar: n=124, 54 days, mean R=+0.067, **t=1.05**.
**No window anywhere in the ORB history that was not used to tune some live parameter clears
t≥2.** The claim's t=2.71-3.05 exists only because the pooled book folds in the same data that
repeatedly served as TRAIN/VAL for the vetoes, touchgo, spread gate, entry buffer, and sizing.

## 5. Scope / budget caveats
- Did not re-derive each cited report's numbers from scratch (e.g. did not rerun
  `orb_drag_program_aug2026.md`'s synthesis or `orb_gates2`'s 40-cell grid) — traced only the
  *window* each report states it used, per the independent-reimplementation-catches-coding-errors
  vs causality-trace distinction (CLAUDE.md checks #1 vs #2); this is a causality-trace on
  parameter provenance, not a full rebuild of every upstream study.
- `filter.skip_q1`'s exact VAL/HOQ1+ date range is unconfirmed (flagged above, not counted as
  clean or contaminated).
- Did not check `data/cache.db` or any bars DB (rule; also would have been stale past 13:25 UTC —
  request started 11:41 UTC so bars DB access was in-window but avoided anyway per instructions
  to prefer the pre-committed book CSVs).
- Tail dependence (ex-top-5% flips negative on the pooled book, -0.013R) is a separate red flag
  already disclosed in the claim's own text; not re-litigated here as it's outside Lens B's scope
  (in-sample audit), but it corroborates the same conclusion: the pooled book's positive mean is
  fragile, both to which 28 trades you keep and to which years you're allowed to look at twice.
