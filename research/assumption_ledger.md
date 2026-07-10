# Assumption ledger — every parameter, its evidence status (2026-07-04)

Owner mandate: "don't take any assumption for granted — fine-tune the
entire machine." This ledger is the standing inventory. Every knob gets a
status; NEVER-TESTED knobs are debt. Update whenever a knob is validated,
shipped, or its evidence goes stale (>2 quarters old on a regime-sensitive
knob = re-check).

Legend: ✅ validated (walk-forward or live-data study, cited) ·
🔧 shipped this cycle · ⚠ never-tested (debt) · 💀 dead knob (proven inert) ·
📌 untestable (capture bias / policy) — monitored, not tuned.

## ORB — entry chain
| knob | value | status | evidence |
|---|---|---|---|
| range_minutes | 5 | ✅ | Apr'26 variant sweep (5 beat 15/30) |
| universe gap min | 5% | 📌 | untestable below 5 (capture bias); monitored via selection audit |
| universe prev_vol | 500K | 📌 | same capture bias; monitor |
| price band | $3–30 | ✅ | Jul'26: $2–3 NO-GO (−$44..75K), $30–60 NO-GO, $4 floor fragile — orb_price_band_verdicts_jul2026.md |
| stop_limit_buffer | 30bps | ✅ | Jul'26 quote-path study: 97% of BT fills genuinely fillable at +30bps, ALL top-10 monsters fillable (min-ask 35-315bps BELOW trigger), knob insensitive 10-150bps. BT fill assumption VALID. /tmp/orb_entry_buffer.csv |
| spread gate | 300bps | 🔧 | NBBO study Jul'26: 150 skipped monsters (BKKT 153, XNDU 267); orb_spread_gate_verdict.md. NEVER tighten <150 |
| entry window (time_stop) | 60min | ✅ | Apr'26 (30 worse) + Jul'26 re-sweep (90 worse by $6.3K, late breakouts net-negative) |
| 9:35 one-shot top-K | — | ✅ | BT-parity by construction; race-fixed 7/3-4 |
| composite threshold | 0.0 | ✅ | original TRAIN sweep (−1.5..0.5 grid) |
| frozen H1-25 z-fit | — | ✅ | refits $34-47K WORSE OOS (audit #4); mandate cancelled |
| Q1 filter | on | ✅ | +$8.5K OOS, study_orb_q1q2_filter |
| PDR veto | ≤8% | 🔧 | +35% P&L, MDD −31%, 18/19mo — orb_pdr_veto.py docstring |
| slots (max_concurrent) | 4 | ✅ | W1: 3/5 worse-or-noise |
| family/super dedup | on | ✅ | W1: loosening ≈ flat |
| quintile mults | Q2 1.5/Q3 1.4/Q4 0.5/Q5 0.5 | 🔧 Jul'10: CORRECTED — old fit (Q4=1.842) was on 15:59-exit physics live never traded; inverts under correct 15:45 exits (leave-out-stable, era-consistent; old config 2026=−$21K). SHRUNK clip [0.5,1.5] per Q5-cap doctrine. research/orb_selection_reaudit_jul2026.md |
| quintile ranking order | Q4-first | ✅ Jul'10 re-audit: ordering is NOT the lever (slots rarely bind); Q2-first doubles MDD. Unchanged. |
| BT pipeline mult source | orb.yaml literals | 🔧 Jul'10 parity fix: per-run refit silently diverged from live after the 15:45 fix. BASELINE REBASED: ~$251K/18mo shipped-config (retire the $344,766 refit-mults number). |

## ORB — exit chain
| knob | value | status | evidence |
|---|---|---|---|
| lock arm / lock stop | 1.75R / 0.5R | ✅ | 5/8 sweep, Pareto vs 1.5/1.0 on holdout |
| touchgo rule M / D | 0.5 / 0.75R→0.5R | ✅ | walk-forward, stable across rolling windows |
| no profit target | — | ✅ | Apr'26 50-variant sweep; +1R target = 50.5% WR but amputates the tail (Jul'26 counterfactual) |
| force-close time | 15:45 | ✅🔧 | Jul'26 sweep 15:00/30/45/59: peak AT 15:45 (live accidentally optimal). BT parity fix shipped — BT last-bar exit understated ~$20K/18mo |
| BT entry slip model | 30bps | ✅ | conservative: live actual 11–21bps by bucket (analyze_orb_slippage) |
| BT exit slip model | 10bps | ✅ | conservative: live exits median 16bps BETTER than bid (spread-aware pricing) |
| exit_min_offset / spread_offset_factor | 0.01 / 0.30 | ✅(design) | FABC fix; live telemetry confirms good fills; not swept (low $) |
| min_stop_pct | 1.0 | 💀 | inert 0.5–2.0 (no 5%-gapper ranges <2%) |
| safety bracket SL/TP | 10% / 3x | 📌 | belt-and-suspenders only |

## ORB — risk/policy (owner's domain, not edge-tunable)
| daily loss limit | stage table | 📌 policy |
| risk/trade, budget | ramp stages | 📌 policy (Jul'26 gates: operational + loss floor) |

## Bull flag (recently churned; supply-constrained — lower tuning ROI)
| conviction floor | 1.8 | 🔧 audit Jul'26 |
| intraday threshold | 20% | 💀 (10–15% band inert through TTF stack) |
| TTF / V-rev bonus / per-tier MACD / regime mults | various | ✅ (2026 Q2 ships, each with BT + parity tests) |
| pattern detector (pole 3, retrace 50, …) | — | ✅ rejected changes 5/15 (recency-negative) |
| trail vol guard min_vol_ratio | — | ✅ Exp D |
| BF trail activate_at_r | 2.0 | 🔧 Jul'26: 1.5→2.0 (reverses Apr'13 ship — population changed). 13/13 improved, 0 hurt, 19/19 months, monotone both eras, ≈+35% book. Revert trigger in config comment. |
| BF conviction rule weights | hand-tuned | ✅ Jul'26: hand weights BEAT ridge/sign-ensemble OOS at every pass rate (n=249 w/ components) — do not ML-refit at this sample size |
| BF detector strictness | pole3/retr50/… | ✅ Jul'26 twin-rebuild: loosening REFUTED (incrementals +$21/trade raw, negative 2026, displace winners in Stage-2). Strict detector is load-bearing. |
| BF max_pullback_candles | 10 | 🔧 Jul'26: 5→10 single-knob twin (+$3.8K/18mo, both eras +, harm-free). Was 66% of 2026 rejections + NEVER tested alone (bundling lesson). Most long consolidations still die — modest supply, not a jackpot. |
| BF supply ceiling | — | ✅ Jul'26: MARKET-limited, not code — eligible movers GREW (2068→2702/mo) while flag-formation collapsed 8%→1.4%. Monitor monthly via research/scripts/bf_rejection_histogram.py. |
| BF detector timeframe | 1-min | ✅ Jul'26: multiframe test (1/2/5-min) — the literature's ~10 raw flags/day EXIST at every TF but carry ZERO edge (WR 32-37%, avg R −0.28..0.00; 5-min WORST). The shape has no edge; selection is the edge. |
| BF price_max | 30 | 🔧 Jul'26: 23→30 twin (+$6.4K/18mo, BOTH eras +, 22 incr @41% WR). SUPERSEDES the old "$24+ death zone" note (pre-modern-stack measurement). Combo with pullback-10 not twin-tested — expect ≈additive, verify live. |
| BF entry slip model | — | ✅ Jul'26: live +40-54bps vs 50 model (n=29, recheck n>=60) |
| BF Stage-1 cache reproducibility | — | 🚨 Jul'26: IRREPRODUCIBLE — mover screen float gate time-travels (current floats on history). Prod cache = point-in-time truth, NEVER rebuild. Studies must use rebuild twins. |
| BF detector era-fit | strict rules | ⚠ TESTING — twin-build preview: loose=strict in Mar-25, loose 3x strict in Jun-26; strict rules may bind exactly in the 2026 regime |

## Standing debt — CLEANUP PASS 2026-07-04 (all four attacked)
1. time_stop 60 vs 90 — ✅ CLOSED, 60 STAYS: full-universe rediscovery
   (11,491 symbol-days) found +282 late breakouts (65-95min) but the
   pipeline is $6.3K WORSE with them ($85.1K→$78.8K, no era-consistent
   gain). Late breakouts are net-negative; GLXU 6/8 was an outlier, not
   a signal. (Within-methodology comparison; reports
   /tmp/orb_band_study_report_3_30_slip30_w{60,90}.txt)
2. BF entry-slip: ✅ VALIDATED — live median +40-54bps vs 50bps model
   (n=29, thin); re-check at n>=60
3. Monster-detection channels:
   - premarket volume: ✅ **SHIPPED as SIZING (2026-07-04)** — the veto
     form tested the wrong end; the monotone-positive gradient's value is
     at the TOP. Upsize-only x1.5 above the TRAIN-frozen upper tercile
     ($5.82M): +$76.8K/18mo, ALL eras positive, 0 giants downsized,
     worst month −$2.6K, corr(PM$, composite)=0.05 (orthogonal).
     trading/orb_pm_mult.py; sizing.pm_dollar_vol_mult; ORB_PM_MULT=0.
     Data: data/research/orb_premarket_dollar_vol_20260704.csv.
   - float: ✅ RESOLVED WITHOUT PURCHASE — SEC EDGAR shares-outstanding
     (free, point-in-time by filing date) decontaminated the test:
     LOW-shares cohort is real (+$1,301/trade, 9/26 monsters — the
     low-float thesis holds point-in-time) but the gradient is only
     monotone in 2026 (2025 muddled: Q3 −$317, Q4 +$356) → fails
     era-consistency, NOT shippable. The dirty read's "high float bad"
     was partially lookahead (PIT Q4 = +$269, not negative). NO paid
     float data needed — EDGAR proxy suffices and the answer is 'not
     yet'. Do NOT stack a shares-based mult on the PM mult without
     joint validation. Data: /tmp/orb_pit_shares.csv,
     /tmp/edgar_shares_hist.csv (fetcher /tmp/edgar_shares_study.py).
4. Stage-2 regime sizing: ✅ SHIPPED — day-level A/B/C1/C2 mult + C2 skip
   in batch_backtest Stage-2 (BT_REGIME_SIZING=0 = pre-fix, verified
   byte-identical). A/B on 2025-01→2026-07: $31.9K/74tr → $35.6K/67tr.
   UD scaling remains live-only (documented).

## Placement-latency residual (2026-07-10, quantified live)
IQMX 7/9: broke out during the 44s ranking window; the buy-stop guard
correctly refused to chase (ask ran 60bps past limit). BT fills these
(assumes resting order from 9:35:00); measured tail ≈3% of picks
(entry-buffer study). Cost of THIS instance: BT made +$5 on it. The
substitution channel (freed slot refilled with CCXI, never triggered,
$0) is self-limiting. ACCEPTED live-only divergence — documented, not a
defect. Guard-skips do NOT consume daily slots (unlike PDR vetoes)
because they are transient microstructure, not deterministic selection.

## Live-telemetry learnings (2026-07-09 read, ~2.5wk of quote data)
- Exit ESCALATIONS: ✅ STUDIED 2026-07-10 — **NO-SHIP.** NBBO-tape replay
  of ALL 7 events ever: the re-price ladder (limit at bid−0.3×spread with
  6s patience before market) recovers just **+$133 TOTAL, with 4/7 events
  made WORSE** — the tape keeps falling during the extra seconds, so the
  'tax' vs the T+10 bid is mostly REAL market movement, not recoverable
  spread-crossing. Expected value ≈ $19/event optimistic-fill-assumption,
  against added complexity in the single most safety-critical path.
  The 10s→market ladder STAYS AS IS. research/scripts/
  orb_escalation_reprice_study.py (rerunnable as events accumulate;
  re-look only if event count 5x's or per-event tax structurally grows).
- FAST fills (<2min after 9:35) avg −$215/trade vs RESTING fills −$36
  (n=19/10, buggy-era sample): immediate triggers may be spent momentum —
  consistent with touchgo philosophy. CANDIDATE, needs proper BT test
  (evaluate marketable-path entries as a cohort); do NOT act on n=29.
- Tag exits systematically BEAT their next-bar-open estimates (PLTZ est
  −$21→actual +$87; AAOX est −$30→+$449): the sell limit fills at
  market-or-better in fast tape. Estimate copy fixed 7/9.
- entry_quote_ofi: NO signal at n=29 (corr −0.07) — honest null, keep
  collecting.
- Entry spreads median 15bps, p90 92bps — the 300bps gate rarely binds
  (as intended); ask runs +43-67bps submit→fill (momentum confirmation,
  not a cost — resting stop-limits fill AT plan price).

## Methodology notes (hard-won)
- **Same-day BT counterfactuals are INVALID before the ~20:30 UTC nightly**
  regen: intraday daily-bar fetches are incomplete (2026-07-06: 16:33 read
  showed 0 candidates; 20:30 full data showed 4 — all would have stopped
  out, so the outage that day accidentally saved ~$362 at Stage-0).
- Rebuild-vs-prod-cache comparisons are invalid (float-gate time travel);
  twins only.
- The most-binding gate in any funnel must get a SOLO test cell — bundles
  shadow it (BF pullback lesson).

## Rules of the ledger
- A ship without a ledger row update is incomplete.
- Any knob touched by a fix gets its status re-checked, not assumed.
- Baseline note: the authoritative defended-book baseline is
  **$344,766** (Jan'25–Jul'26: veto + 15:45 parity + PM sizing mult,
  pipeline-confirmed 2026-07-04). Decomposition: $209,734 (veto,
  old 15:59 convention) → $258,298 (+15:45 parity) → $344,766 (+PM mult).
  (was $209,734 under the old last-bar/15:59 convention — the fix helps twice, both live-true: EOD
  holds skip the closing fade AND stops in the 15:45-15:59 window never
  fire live). Studies dated before 2026-07-04 quote the 15:59 convention;
  reproduce them with ORB_BT_FORCE_CLOSE_ET=15:59.
