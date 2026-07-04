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
| entry window (time_stop) | 60min | ✅(dated) | Apr'26 variants (time_30 worse); GLXU 6/8 ran post-window → re-sweep 60/90 QUEUED |
| 9:35 one-shot top-K | — | ✅ | BT-parity by construction; race-fixed 7/3-4 |
| composite threshold | 0.0 | ✅ | original TRAIN sweep (−1.5..0.5 grid) |
| frozen H1-25 z-fit | — | ✅ | refits $34-47K WORSE OOS (audit #4); mandate cancelled |
| Q1 filter | on | ✅ | +$8.5K OOS, study_orb_q1q2_filter |
| PDR veto | ≤8% | 🔧 | +35% P&L, MDD −31%, 18/19mo — orb_pdr_veto.py docstring |
| slots (max_concurrent) | 4 | ✅ | W1: 3/5 worse-or-noise |
| family/super dedup | on | ✅ | W1: loosening ≈ flat |
| quintile mults + Q5 cap 1.5 | frozen | ✅ | anti-overfit guard; do not touch |

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
| BF entry slip model | — | ⚠ stale — recalibrate from trades DB quarterly (README note) |

## Standing debt — CLEANUP PASS 2026-07-04 (all four attacked)
1. time_stop 60 vs 90 — sweep RUNNING (background; verdict pending)
2. BF entry-slip: ✅ VALIDATED — live median +40-54bps vs 50bps model
   (n=29, thin); re-check at n>=60
3. Monster-detection channels:
   - premarket volume: TESTED — real gradients (high-PM$ +$792/trade vs
     low −$17; all-era-consistent tercile). Walk-forward veto
     (pm_dollar_vol <= TRAIN-p20 ≈ $357K): +$15.7K/18mo, all eras >=0,
     0 monsters hit — but only 11/19 months and best-month = 44% of
     benefit → **PARKED as validated-direction candidate** (PDR bar is
     18/19). Live-implementable via 9:30 snapshot volume. Re-evaluate
     next quarterly review or if live PM telemetry accumulates.
     Data: /tmp/orb_premarket_features.csv (100% coverage).
   - float (dirty read, current-float contaminated): RUNNING background.
     Point-in-time float remains a purchase decision.
4. Stage-2 regime sizing: ✅ SHIPPED — day-level A/B/C1/C2 mult + C2 skip
   in batch_backtest Stage-2 (BT_REGIME_SIZING=0 = pre-fix, verified
   byte-identical). A/B on 2025-01→2026-07: $31.9K/74tr → $35.6K/67tr.
   UD scaling remains live-only (documented).

## Rules of the ledger
- A ship without a ledger row update is incomplete.
- Any knob touched by a fix gets its status re-checked, not assumed.
- Baseline note: with the 15:45 parity fix the defended-book baseline is
  **$258,298** (Jan'25–Jul'26, veto + parity; was $209,734 under the old
  last-bar/15:59 convention — the fix helps twice, both live-true: EOD
  holds skip the closing fade AND stops in the 15:45-15:59 window never
  fire live). Studies dated before 2026-07-04 quote the 15:59 convention;
  reproduce them with ORB_BT_FORCE_CLOSE_ET=15:59.
