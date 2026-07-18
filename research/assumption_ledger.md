# Assumption ledger — every parameter, its evidence status
   - Bar-level 1R-moment predictors (2026-07-11, owner insisted): ❌ NO
     PREDICTOR SURVIVES correct path accounting. First pass showed
     speed/pullback/greenness separating monsters — an ARTIFACT of two
     path bugs (tag-exited trades counted as alive; stop-before-1R
     counted as reachers). Corrected (212 clean 1R-reachers):
     P(>=3R | impulse@1R) = P(>=3R | grind@1R) = 16%, era-flipping.
     At 1R, monsters are still camouflaged — in speed, volume, pullback,
     and bar-shape space. Volume era-flips (again).
   - BYPRODUCT — BE-stop-at-1R for grinders (first real consistency
     lever found): flat-exit grinders that round-trip to entry before
     arming. −$8.8K/18mo total (−3%) for MDD −$18.2K→−$14.6K (−20%) and
     worst month −$7.3K→−$4.8K (−34%); 46 round-trips flatted, 3
     monsters killed incl CMCT 31.8R. Unconditional version much worse
     (−$60K, kills BNAI 12.6R). FAILS the profit-up rule → NOT shipped;
     documented as the honest consistency price list (owner preference
     call). Needs BE-fill slippage modeling + StopMonitor wiring if ever
     approved. /tmp/monster_1r_v2.csv, research/scripts pending. (2026-07-04)

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
   - Z-params + quintile cutoffs BT source (2026-07-17): 🚨 P0 PARITY FIX —
     pipeline refit z-params/cutoffs per run from the features CSV whose
     TRAIN slice GREW (band-study full-universe rebuilds: 1,836 rows vs
     original fit) → BT scored on different params than live's frozen
     orb.yaml constants (ASPI 7/14: identical features, comp 0.410 live vs
     0.317 BT → quintile flip → false RED day). Pipeline now reads
     filter.features + quintile_cutoffs from orb.yaml (live parity),
     same pattern as the 7/10 mult fix. CONSEQUENCES:
     (a) live-parity book = $210,190 shipped-mults / MDD −$14,727
     (retire $295,896 — drifted);
     (b) the Jul-10 QUINTILE MULT CORRECTION IS INVALIDATED: derived on
     drifted quintile labels. Under live cutoffs NO mult scheme passes
     leave-out (old +$207K→−$11K@top5; flat-vs-shipped +$83K→+$1.5K@top5;
     era ex-top3 flips everywhere). The quintile layer carries no robust
     sizing signal — monsters move buckets with the param set.
     FLAT 1.0 = the null: $293,568 / MDD −$16,334.
     (c) news×PM gate evidence SURVIVES (quintile-free cells);
     (d) monster/window/session studies' conclusions survive (label-free);
     the July universe CSV quintile columns are drifted — do not reuse.
     RESOLVED 7/17: owner approved FLAT mults (live same day).
   - Parameterless (per-day percentile) selector (2026-07-17, owner
     mandate): ❌ REFUTED — pure day-rank $161K / MDD −$67K (no absolute
     floor → trades best-of-bad-lot days); with ≥0.5 floor $226K / MDD
     −$25K; baseline frozen-composite stack $293,568 / MDD −$16,334 wins
     era-consistently and leave-out-robustly at every depth. The frozen
     ABSOLUTE quality scale is load-bearing; relative ranking destroys
     it. Drift risk is instead covered by: yaml-pinned BT params,
     hard-fail on silent refits (BT_ALLOW_REFIT), and the nightly
     field-level decision-parity gate. CANONICAL BASELINE: $293,568
     (frozen z + thr 0 + Q1 skip + flat mults + news-gated PM 2.0).
     research/scripts/orb_dayrank_selector_study.py.
   - "Turn July-2026 positive" challenge (2026-07-18, owner): ❌ CLOSED —
     NO real setup flips it. (a) EXITS: bar-accurate ladder sim on all
     680 trades — July reach-1R rate 36% < the 50% a 1:1 target needs;
     best ladder (tgt1R+BE@0.5) July −$6.5K, and ladders destroy the
     18mo book ($288K→$22-61K, every monster → ~$1K). (b) ENTRY subsets:
     combo-only July −$3.7K (JLHL 7/17 was a 2.0x combo LOSER); stocks-
     only/news-only worse; July winners are feature-camouflaged (NBIZ =
     Q1!). (c) ORACLE best-4/day hindsight = +$41.9K — the month was
     winnable only with information our ~30 tested features don't carry.
     BYPRODUCT — the researched DROUGHT-MODE candidate: COMBO-ONLY
     (news×PM×stock cell as sole strategy): $202K/18mo (70% of book) from
     62 trades (3.4/mo), WR 52%, MDD −$11.0K, worst month −$6.6K; cuts a
     July-like month's bleed 77%; keeps the news monsters (AMCI/ANNA/
     CRNC/QCLS/BNAI), forfeits newsless ones (CMCT class, −$86K/18mo).
     NOT SHIPPED — parked as the deliberate consistency-mode option.
   - ORACLE FEATURE HUNT (2026-07-18, owner mandate): 🎯 FOUND the missing
     information class — UNDERLYING-COMPLEX CO-MOVEMENT (und_cohort =
     count of same-underlying wrappers qualifying the same morning;
     mechanical, point-in-time, no curation). Universe-level: uc>=2 mean
     +$353/+$150/+$94 per era (vs −$25..−$178 for uc=1), monster rate
     2-3x base, ERA-CONSISTENT — first new all-era feature since PM$.
     Hand-curated SECTOR buckets era-flip (25H1 inverts) → sector-aware
     forms NOT robust; underlying-level is. Oracle days explained: 7/16 =
     four 2X-SHORT wrappers of one crashing complex; 7/06 = 3 long-IREN
     wrappers won together while the book picked losers.
     SHIPPABILITY: sizing boost on selected uc>=2 FAILS leave-out (n=54);
     bench promotion FAILS era (25H2 benched −$30 mean). PARKED as
     taxonomy, not a lever. CATALYST TAXONOMY (selected book, 18mo):
     news $232.6K/162tr (81% of book) · newsless+complex $21.3K/54tr ·
     newsless+ALONE $33.9K/464tr (12% of book from 68% of trades, mean
     $38-110, holds ASST/PONY/HERE/BKKT monsters). Veto-newsless-alone
     book = $253.9K from 216 trades — the fine-grained drought-mode.
     Note for sector-news idea: news does NOT transfer via underlying
     (refuted 7/11); PRICE confirmation (uc>=2) is what transfers.
     /tmp/theme_feature_study.py -> research/scripts.
   - BOTTOM-UP MARKET-MONSTER CENSUS (2026-07-18, owner: "market has
     monsters weekly — find them"): CONFIRMED — 2,744 market monster-days
     in 19mo (open→high ≥30%, $2-50, ≥$5M day-vol) = ~34/WEEK, every week.
     FUNNEL LEAK TABLE: universe screen admits only 16% (gap≥5 kills 63%
     — HALF of all market monsters open flat/down and build intraday);
     of 428 admitted: 331 became candidates, 34 selected = $279K = 97% of
     the whole book. 297 BENCHED candidate-monsters worth +$700K raw —
     BUT unreachable: per-gate full-cohort economics (live-parity):
     threshold-failers −$323K net (KEEP), PDR cohort −$162K (KEEP),
     Q1 re-admit +$4.9K but MDD +50% (KEEP filter), slots 4→5/6 WORSE
     (cap dilution). Selection layer fully re-validated post-param-fix;
     the gapper-architecture book ≈$293K is at its information ceiling.
     THE STRUCTURAL ANSWER: the untouched pool = FLAT-OPEN INTRADAY
     IGNITERS: 1,380 monsters (~17/wk), median run +42%, median day
     $vol $24M — invisible to ORB by design (gap gate) and to BF (flags
     stopped forming). 'Monsters weekly' requires a THIRD detector:
     intraday ignition breakout (any-hour consolidation→volume-surge).
     PROPOSED as the next major program. /tmp/market_monsters.csv,
     /tmp/flat_open_monsters.csv.
     /tmp/exit_ladder_study.py, orb_dayrank_selector_study.py dump.

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
| BF last_entry 10:45 + midday skip + EOD-flat | session structure | ✅ Jul'11 VALIDATED (first time): 18mo afternoon twin (last_entry 15:30, midday on) — post-10:45 = −\$12.7K raw at WR 28-32%, NEGATIVE both eras, in EVERY sub-session (pre-midday −\$4.7K, midday −\$1.5K, afternoon −\$6.5K); conviction cannot rescue (conv≥2.2 still −\$6.6K). Morning open session = +\$383/trade. The 10:45 wall is load-bearing. Twin: /tmp/bf_twin_afternoon.csv; BF_LAST_ENTRY/BF_SKIP_MIDDAY env overrides. |
| BF re-entry (multi/day) | off | ✅ Jul'11 RE-REFUTED under current stack: 2nd entries −\$81/trade (n=30), 3rd −\$89 (n=3). Confirms −\$1,299/yr finding. |
| BF conviction floor | 1.8 | ✅ Jul'11 re-validated: 1.5-1.7 floors add 2025-only money, 2026 WORSENS (recency-negative, pole=2 pattern). |
| BF trade caps (5/day,3conc,DLL) | — | ✅ Jul'11: NEVER BIND on shipped book — not the supply constraint. |
| BF regime C2 skip | 0.0x | ✅ Jul'11: half-size C2 adds +\$3.3K but ALL 2025, 2026 flat — skip stays. |
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
   - pre-market NEWS (2026-07-10, owner-prompted): 🎯 STRONGEST separator
     found since PM$ — but ONLY as the interaction news×pm_hi. Combo cell
     per-trade mean +$1,580/+$1,569/+$935 across TRAIN/25H2/2026 (era-
     consistent); monster rate 28/15/13% vs 6/7/8% rest; survives the
     continuous-PM$ control. News alone = negative; pm_hi WITHOUT news
     ≈ flat (the shipped PM mult's lift lives in the news subset).
     Zero lookahead (all articles ≤09:30 ET). LLM/keyword catalyst-quality
     classification REFUTED for longs (recaps = catalysts; recap-only holds
     AMCI+BNAI $36.6K — do NOT port stupid-money's classifier). Slot
     promotion refuted (benched 2026 combo −$3K). Sizing variant A2
     (combo-only 2.0×) passes owner rule: TOT $250K→$301.5K, all eras +,
     MDD −$18.8K→−$18.2K — but lift is top-5-concentrated & combo big-loser
     rate rising (3→8→10%/era). ✅ **SHIPPED 2026-07-10 (owner: "flip
     both")** — news_gate in trading/orb_pm_mult.py (high_mult 1.5→1.0,
     high_mult_news 2.0), live news fetch at 9:31 prefetch (fail-open),
     BT pipeline models it (legacy byte-identical via ORB_PM_NEWS_GATE=0:
     $250,276; gated: $301,518 verified). EoD green-check now HARD-gates
     on pm_mult recompute drift + soft-flags news drift + tracks combo
     cumulative. Live 2026-07-13 with the quintile-mult correction.
     research/orb_news_catalyst_jul2026.md;
     data/research/orb_news_catalyst_20260710.csv.
   - ETF→underlying news mapping (2026-07-11, owner-prompted): ❌ REFUTED.
     45% of universe = leveraged wrappers (own-ticker news rate 1.1%);
     mapping to the underlying (40.2% news rate) INVERTS the signal —
     und-news×pm_hi is NEGATIVE for wrappers in all 3 eras (−324/−125/−27
     per trade), gate-with-mapping worse in every era. Wrapper monsters
     are mostly NEWSLESS momentum (only 3/11 had und news). The news edge
     is a COMMON-STOCK edge; the shipped own-ticker gate is correct as-is.
     Do NOT re-propose underlying or industry mapping without new
     evidence. Parked: wrapper-with-und-news VETO (negative cell, needs
     no-refill study). research/orb_news_catalyst_jul2026.md addendum.
   - Asset-class rule (2026-07-11, deliberate-rules mandate): ✅ SHIPPED —
     news boost requires POSITIVE stock identification
     (trading/orb_asset_class.py; lev sets → 33K map → API → unknown
     never boosts). Book $301,518→$295,896 (−$5.6K = 4 lucky wrapper
     recap-tag trades; MDD identical) — the price of immunity to a
     Benzinga tagging change flipping the gate into the crowding cell.
     Full machine rulebook: research/orb_machine_rules.md.
   - News WINDOW study (owner's prev-day question, 2026-07-11): fresh
     premarket news (today 4:00-9:35) is the payload (+$1.8K/+$1.6K/
     +$1.0K per era, monster 31/12/11%). Yesterday-SESSION news as boost
     extension: ❌ NO-SHIP — payload = ASST+PONY (H1-25), 2026 n=4 +$54
     (pole=2 recency pattern). Wrappers: NO window works in any direction
     (prev-day und-catalyst does NOT carry day-2: −$129/−$177).
     PM$ remains TODAY 4:00-9:29 premarket $ (deliberate: is the crowd
     here THIS morning); prev-day-run is PDR's job, not PM's.
   - Monster bottom-up study + two-regime design (2026-07-11, owner-
     prompted "predict monsters, consistent otherwise"): ❌ REFUTED both.
     (a) PREDICTION CEILING: all 47 selected-book monsters dossiered —
     at entry they are feature-camouflaged (medians ≈ rest: gap 6.3 vs
     6.9, PM$ 2.7M vs 2.1M, composite 0.33 vs 0.32). Best deliberate
     flag union captures 83% of monsters only by flagging 71% of the
     book (no discrimination). The predictable subset (fresh-news×PM$
     stocks, 21% capture at 22% precision) is EXACTLY what the shipped
     2.0x gate already boosts. Ceiling reached.
     (b) TWO-REGIME (target exits on unflagged): CANNOT deliver its
     goal — capping winners doesn't touch red months. Even ALL-target
     +1.5R (max consistency, −63% total to $110K) keeps IDENTICAL
     negMo 6/19, worst month −$7,291, MDD −$18,174. Consistency in
     this machine is LOSS-SIDE-bound and the loss side is already
     optimized (stops/touchgo/vetoes/Q1/PDR). Do not re-propose
     conditional exits without new loss-side evidence.
     Consistency levers that remain: portfolio allocation (BF corr
     +0.21, 0/19 joint-negative months) + monster-rate tripwire
     (18 monsters/18mo, median gap 19d, max 127d — drought vs broken).
   - In-flight monster prediction at 1R/1.5R (2026-07-11, owner-prompted):
     ✅ CONFIRMED as the best signal in the trade's life — P(>=3R) jumps
     ~7% unconditional -> 26% once armed (~1.75R), 55% for 2.0x-flagged
     stocks (runners mean +10.6R true-R). BUT the machine already acts on
     exactly this: the static lock converts 'armed' into a free-roll
     (locked floor, uncapped top) — the arm IS the monster-recognition
     trigger. Conditional lock levels per cohort: ❌ DIES with true-R
     units (initial analysis had a scaling error: 1R != \$3K when the
     \$25K position cap binds; true median risk \$843). Wrapper armed
     continuation is 23% >=3R (not ~0 as mis-scaled) — tightening their
     lock bounds NET NEGATIVE (-\$24K worst / +\$18K impossible-best).
     Era-stable continuation: unflagged stocks 21/21/19%, wrappers
     42/24/16%. PARKED (needs bar-level resim + appetite decision):
     add-to-winners at arm on 2.0x-flagged only (n=20/18mo, rough EV
     +\$40-60K model) — NOTE it INCREASES skew, anti-consistency; a
     news-gated variant of the Apr-2026 parked add-to-winners.
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
