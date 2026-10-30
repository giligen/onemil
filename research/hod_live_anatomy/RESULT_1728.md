# RESULT 1,728 — EXPLORATORY (no PREREG): why do live HOD fills "beat" the BT population?

Scripts: `cell_1728.py` (steps 1–4, output `cell_1728_out.txt`, per-fill `cell_1728_bt.csv` / `cell_1728_live.csv`),
`cell_1728b.py` (shortlist + live gate audit, output `cell_1728b_out.txt`). Daily context for BOTH books from
`data/cache.db daily_bars` opened read-only (`mode=ro&immutable=1`), coverage BT 99 % / live 100 %.
Read count in this cell ≈ 40 (13 dimension tables, 5 gate rows, 17 rules, 4 shortlist reads) — every number below is
a selected read, not a confirmed one.

## 1. There is no gap on a common basis — the "+0.2 R" was the cost model plus the R floor

| book | n | gross at the LEVEL (R = level − stop) | net (measured cost) | net (1,438 banded cost_R) |
|---|---|---|---|---|
| BT TRAIN-H2 (1663, R ≥ 1.5 %) | 2,349 | **+0.072** (t 1.34) | −0.021 | −0.160 (cost_R 0.199) |
| BT VAL (1663, R ≥ 1.5 %) | 3,157 | **+0.061** (t 1.19) | −0.030 | −0.183 (cost_R 0.213) |
| LIVE paper (fills_1721) | 231 | **+0.075** (95 % CI −0.09..+0.24, day-t 0.81) | −0.026 | — |
| LIVE, gate-compliant (level ≥ $20 AND R ≥ 1.5 %) | 110 | +0.024 | −0.023 | — |

Decomposition of the quoted gap (10/23: live −0.04 vs "all-armed −0.21"):
* **cost model ≈ 0.15 R**: cell 1,438's `causal_arming_causal.csv` charges `cost_R` 0.20–0.24 R/fill (the old banded
  table); the same 9,911 fills under the measured cost (fills_1658: entry 7 / stop 6 / EOD 11 bps) are −0.062 / −0.066.
* **R floor ≈ 0.04 R**: the 0.75–1.5 % R bucket (4,403 fills, −0.11 net) is in the 9,911 but not in the 5,506.
* **residual ≈ 0.00 R net, +0.01 R gross.** BT total cost from the level = 0.09 R, live = 0.10 R. Exit mix matches
  (stop 56 % / target 27 % / EOD 16 % BT vs 54 / 23 / 23 live). BT fills already respect the +15 bps cap (fill − level
  max 15.0 bps, 2,010 of 11,921 crossings = 17 % were NO-fill in the BT), so gap-through fills are not the difference;
  BT entry slip 6.4 bps vs live 8.1 bps (≈ 0.01 R).

**Nothing about which breaks fill live selects better trades. The lemon is dry on the evidence available.**

## 2. Parity defect found: the live arm has NO min_price and NO min_r_pct gate

`trading/hod_break.py::arm_state` (used by `_evaluate_resting` → `_arm_live_order`) returns level/trigger/limit/stop
with no price floor and no R floor. The research twin `research/hod_entry/causal_arming.py::arm_state` line 81 rejects
`trigger < min_price or (trigger − stop)/trigger < min_r_pct`. `config.yaml hod_break.min_r_pct: 1.5` and
`min_price: 20` only reach the scan-hook admission (`_admit_movers`, line 709) and the non-resting path (line 2293);
the streamed universe admits prev close ≥ $15. Live consequence (9/29–10/30, 231 fills):
* **87 fills (38 %) with R < 1.5 % of price**, as tight as 0.24 % (SMCI 10/27 35.03/34.945, SHBI, STZ, THRM, UNP…):
  gross +0.164, net +0.036. The other 144: gross +0.020, net −0.064.
* **45 fills (19 %) with level < $20**: gross +0.149.
* Tight-R fills are sized by the $4,000 notional cap, so their dollar risk is a fraction of $50 — the per-fill R mean
  over-weights them relative to dollars.
The live paper book is therefore NOT the 1,663 population; the gate-compliant live subset (n 110) reproduces the BT
(−0.023 vs −0.026 net). `tests/test_hod_resting_entry.py` claims parity but does not cover these two gates.
Decision for the main session (I changed no code): either add both gates to the live arm (restores ONE spec) or, if the
tight-R / sub-$20 fills are wanted, write them into the spec and BT them (the BT's own 0.75–1.5 % bucket is −0.11 net —
it disagrees with the live +0.04 on n 87, CI ≈ ±0.3).

## 3. Where the populations differ (quantiles in `cell_1728_out.txt` Step 2)

| dimension | BT q25/q50/q75 | live q25/q50/q75 | BT R by bin (TRAIN-H2 vs VAL) | gross shift if BT reweighted to live mix |
|---|---|---|---|---|
| fills per day / rank in day | 20/28/44, rank 7/14/24 | 10/11/12, rank 3/6/8 (cap 12 binds) | no monotone pattern; signs flip by split | caps 4/12 first-come on the BT: net −0.022 / −0.025 (= ALL) |
| ATR14 % (prior days) | 6.9/9.8/14.8 | 5.1/7.7/11.0 | ≤ 4.5 %: TR −0.07, VAL −0.12; ≥ 9 %: TR −0.04, VAL +0.03 | +0.066 → +0.046 (live sits where BT earns LESS) |
| prior-day return % | −3.5/+0.6/+5.7 | −3.0/−0.1/+3.1 | ≥ 3 %: TR +0.05/+0.03, VAL +0.02/+0.19; −3..0: TR −0.10, VAL −0.18 | +0.066 → +0.041 (live sits where BT earns LESS) |
| extension of level above open % | 5.3/5.8/7.3 | 5.1/5.4/6.1 | flat (≤ 6 %: −0.001 / −0.029) | none |
| gap % | −1.9/+0.3/+2.5 | −2.4/−0.1/+1.6 | flat; live gap ≥ 2 % fills −0.52 gross (n 49) not reproduced in BT | none |
| price (level) | 27/40/71 | 22/31/83 | flat | none |
| time of fill | 9:53/10:12/10:54 | 9:45/10:06/10:41 | sign-split across bins | none |

Every live-vs-BT difference that the BT can price moves the BT expectation DOWN or not at all — none explains a live
advantage. `n_cross` (BT net +0.86 at ≥ 8 crossings vs −0.85 at ≤ 1) is the count of armed crossings over the WHOLE
day (`causal_arming.py:269`) — look-ahead, excluded.

## 4. Shortlist for a confirmation PREREG (Fable writes it; none is confirmed here)

| rule | mechanism | TRAIN-H2 n / net / t / ex-top-5 % / A│B | VAL n / net / t / ex-top-5 % / A│B | live kept / gross / net |
|---|---|---|---|---|
| **R1** prior-day close-to-close return ≥ +3 % | name already in play, multi-day momentum continuation; knowable pre-open | 793 / **+0.040** / 0.47 / −0.062 / +0.073│+0.009 (drop −0.052) | 1,169 / **+0.086** / 1.34 / −0.014 / +0.139│+0.022 (drop −0.097) | 26 % (59) / +0.195 / +0.176 (drop +0.034) |
| **R2** Monday only | weekend news accumulation; weakest mechanism | 461 / **+0.098** / 1.00 / −0.004 / +0.144│+0.049 | 587 / **+0.135** / 1.13 / +0.036 / +0.347│−0.087 | 14 % (33) / +0.350 / +0.348 |
| **R3** parity: level ≥ $20 AND (level − stop)/level ≥ 1.5 % at arm, live | ONE spec for BT and live; not an edge rule | = the 1663 population: −0.021 / −0.030 | — | 48 % (110) / +0.024 / −0.023 |

Both R1 and R2 are positive and same-signed in TRAIN-H2 and VAL on net and on both alternating-week halves except
R2-VAL-B; neither has ex-top-5 % > 0 in both splits; winner-capped (+1.5 R) is negative for R1 in both splits.
Both were chosen AFTER looking at the bins (R1 from the prior-day table, R2 from the weekday table) — the BT reads are
in-sample selections; the live column is the only quasi-independent check (n 59 / 33, 5 weeks). MDE for a ~30 % cut
at day-clustered SE ≈ 0.065 R ≈ 0.18 R — both lifts (+0.07..+0.17 vs dropped) are under it. A PREREG must hold
2024H2 (bars_sip.db does not cover it — would need a fetch) or the forward paper ledger as the sealed test.

## 5. Adversarial notes

* **The live +0.075 is noise-compatible with the BT**: sd 1.27 R, 95 % CI −0.09..+0.24, day-t 0.81, 5 weeks.
* **R unit**: live gross_R uses `risk_per_share`, which is 6 % larger than level − consol_low on average (depth_ratio
  1.06; live stop exits −0.936 R vs BT −1.025). Proportional, moves the mean < 0.01 R.
* **Live set is not the BT rule** (section 2): 38 % of the fills are outside the BT's R floor; any live-vs-BT comparison
  before this fix compares different populations.
* **Unsimulated live gates**: buying-power guard (resting notional ≤ 25 % of BP), `max_resting`, kill rails, symbols
  held by other strategies, and the live fill rate of crossings (not measured here). First-come caps 4/12 on the BT
  change nothing, so caps alone do not select.
* **Daily features**: same source both books; adv20/ATR use prior days only; today's open is known at 09:30 (fills
  ≥ 09:35). Look-ahead descriptors (`la_*`) were reported only, never used in a rule.
* **Multiplicity**: ~40 reads here on top of 1,727 cells; the shortlist is hypothesis generation only.
