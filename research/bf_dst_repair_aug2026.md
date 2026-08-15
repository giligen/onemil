# Bull Flag Cache DST Repair — Aug 2026

**Status: HALTED at pre-mv gate. Production cache UNCHANGED. Owner decision required.**
Date: 2026-08-15. Cache: `data/bull_flag_cache_e50_x30.csv` (md5 `62f92667d031f90437094b7b654ef06e`,
2,348 data rows). Backup: `data/bull_flag_cache_e50_x30.csv.bak.pre_dst_repair_20260815` (same md5, DO NOT DELETE).

## What was ordered
Splice freshly-rebuilt EST-season rows over the 1,001 phantom-contaminated EST rows
(old code fetched premarket-shifted winter windows), then validate + re-verdict BF features.
Rebuild inputs (fresh API bars, current code): `scratchpad/bf_est_w1.csv` (14 rows, Jan–Mar'25),
`scratchpad/bf_est_w2.csv` (28 rows, Nov'25–Mar'26). 42 rows total.

## STEP 1 — Splice (BUILT, NOT DEPLOYED)
- EST windows: `2025-01-02..2025-03-07` and `2025-11-03..2026-03-06`.
- Verified **1,001** EST rows in prod cache (436 in W1 window + 565 in W2 window). Matches order.
- Non-EST rows = 1,347. Splice target = 1,347 + 42 = **1,389** data rows.
- Files are **CRLF**; splice preserves CRLF and header byte-identically; non-EST rows byte-identical
  (multiset) and within-date order preserved (stable sort by date). Final file is fully date-sorted.
- **Candidates staged in scratchpad (NOT in `data/`, prod untouched):**
  - `candidate_spliced_42rows.csv` (1,389 rows — spec-literal)
  - `candidate_spliced_40rows_clean.csv` (1,387 rows — 2 premarket rows dropped; **recommended**)

## THE ANOMALY — pre-mv RTH gate FAILED (2 of 42 rebuild rows are pre-market)
| symbol | date | entry_time_et | pnl | in gate 09:30–16:00? |
|---|---|---|---|---|
| LE | 2025-02-25 | 09:02:00 | −64 | NO |
| PHGE | 2025-12-08 | 08:58:00 | −528 | NO |

Evidence they are phantom:
- `backtest.py LE 2025-02-25` and `backtest.py PHGE 2025-12-08` → **0 setups** under current code,
  including with `--include-premarket`. Irreproducible under any flag.
- Both rows exist **identically in the backup AND the rebuild** (see provenance below) — the rebuild
  *inherited* them; it did not freshly validate them. So the rebuild's own premarket leak persists
  from the old contamination (reduced 23 raw premarket rows → 2, not eliminated).

Per protocol (`On ANY failure: Telegram loudly, leave production cache untouched`) the splice was
**halted before mv**. Loud `[BF REPAIR]` Telegram sent.

## STEP 2 — Validation (on the clean-40 candidate)
- **Phantom absence confirmed**: old cache had `LAES 2025-01-08 @08:50`, `ARQQ 2025-01-17 @08:43`,
  `KPLT 2025-12-12 @09:15` (all premarket) — **all ABSENT** from candidate.
- **Spot-reproduction** (current `backtest.py`, entry/exit to the cent):
  - DOMH 2025-01-31 → entry $3.62 @10:23 ET, exit $3.80 @10:39, trail_stop. **MATCH.**
  - RYM 2025-02-05 → entry $29.35 @09:40, exit $28.19 @10:03, stop, pnl −395.25. **MATCH.**
  - (DTCK/KPTI single-symbol differ only because cache stores RAW Stage-1 trades while `backtest.py`
    applies full production filters — expected, not an error.)

## Provenance: backup EST vs rebuild EST
- Backup EST = **999 unique (symbol,date) pairs** (1,001 rows); rebuild = 42 pairs.
- **35 pairs in both — entry-time AND price match to the cent on all 35 (0 diffs).**
- 964 pairs only in backup = raw phantom setups current code does not reproduce (this is the
  "1001 → 42" reduction). **7 genuinely new** rebuild pairs: CSAI, MEG, RYM, DOGZ, DLLL, MRVU, PLU.
- Backup EST row timing was **978 RTH / 23 premarket** — i.e. the "96% phantom" framing was about the
  raw *count* (964 extra raw pairs), NOT timing; most backup EST rows were already RTH & correct.

## STEP 3 — Clean baseline (Stage-2, full period 2025-01-01 → 2026-08-13, all via `--cache-file`)
| Cache | Trades | WR | Total P&L |
|---|---|---|---|
| Pre-repair (backup, contaminated) | 78 | 52.6% | **$+33,754** |
| Repaired clean-40 (2 premarket dropped) | 79 | 53.2% | **$+36,232** |
| Repaired spec-42 (incl 2 premarket) | 80 | 52.5% | $+35,572 |

**Repair delta = +$2,478 (+7.3%).** NOT material by STEP-5 thresholds (not >30%, no half-year sign flip).
Net effect = **3 surviving trades**: drop PHGE phantom (−660), add real RTH CSAI (−715) + MEG (+2533).
Reason the 964 phantom raw rows barely move the total: Stage-2's conviction/volume/intraday gates
filter nearly all of them; only PHGE (premarket) survived pre-repair.

Per-month: every month byte-identical pre vs repaired **except** Feb 2025 (+$1,818, CSAI+MEG) and
Dec 2025 (−660 → $0, PHGE dropped). Half-years: 2025H1 22,316→24,135; 2025H2 2,622→3,282;
2026H1 11,006 (unchanged); 2026H2 −2,191 (unchanged). **No sign flips.**

## STEP 4 — Feature re-verdicts (provisional, on clean-40; A = $36,232 / 79tr)
| Feature | Testable in Stage-2? | Result | Verdict |
|---|---|---|---|
| Regime sizing (BT_REGIME_SIZING) | YES (post-cache sizing) | ON $36,232 vs OFF $25,707 = **+$10,525 (+41%)** | **HOLDS strongly** |
| Two-tier filter | YES (post-cache filter) | ON $36,232 vs OFF $36,590 = **−$358** | WEAKENED / flat (was +$224 on backup; sub-1% noise) |
| Marginal scaling | YES (post-cache sizing) | ON == OFF (no trades in band) | HOLDS (keep OFF) |
| V-reversal bonus | **NO** — baked into cached `conviction_mult` | toggle inert ($0) | **DEFERRED** (needs cache rebuild A/B) |
| Vol-confirmed trail | **NO** — baked into cached exit/`pnl` (Stage-2 reads pnl, no re-sim) | toggle inert ($0) | **DEFERRED** (needs cache rebuild A/B) |
| pole=2 (BF_MIN_POLE_CANDLES) | **NO** — build-time detector param | not runnable on pre-built cache | **DEFERRED**; repair left 2026-Jan–Apr months (its rejection hinge) byte-identical → verdict very likely HOLDS |

**Methodology note:** on a pre-built cache, Stage-2 A/B only meaningfully tests *post-cache* levers
(two-tier, regime, marginal). v_reversal / vol_conf / pole are *build-time* levers whose effect is
frozen into the cache columns; re-verdicting them requires rebuilding the cache with the feature
on/off (the expensive fresh-API step), not a config toggle.

## STEP 5 — Decision
- BF P&L picture is **NOT materially changed** (+7.3% total, no half-year sign flip). No BF
  re-derivation warranted on this evidence.
- **Regime sizing** — the dominant BF lever — HOLDS strongly (+41%). No shipped-flag change indicated.
- The repair is correct and mildly beneficial; its whole value is 3 EST trades.

### Recommendation (NOT executed — owner call)
1. **Resolve the 2 premarket rows.** Preferred: **DROP them** (deploy `candidate_spliced_40rows_clean.csv`,
   1,387 rows) — honors the RTH invariant and is the correct artifact; both are losers, net +$588 vs spec-42.
   Alternative: regenerate the rebuild so LE/PHGE are cleanly re-derived (they should yield 0 setups).
2. If approved, `cp candidate_spliced_40rows_clean.csv data/bull_flag_cache_e50_x30.csv` (backup already exists).
3. Optionally rebuild the EST cache with v_reversal / vol_conf / pole=2 on/off to complete their
   re-verdicts — low urgency given the immaterial baseline change.

## Constraints honored
- Production cache never overwritten (md5 unchanged throughout).
- `.bak.pre_dst_repair_20260815` never touched. `cache.db` bars never touched.
- config.yaml (gitignored) toggled for A/B then restored to original (true/true/false/true) — verified
  via `Config()`. **Caveat during work:** `git checkout config.yaml` is a no-op because config.yaml is
  gitignored; an intermediate corruption occurred and was caught + fully reversed (baseline reproduces
  79tr/$36,232). Lesson: restore gitignored config via explicit value set, never `git checkout`.
- Nothing committed to git.
