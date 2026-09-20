# frames16 — INSTRUMENT CALIBRATION · THE LAMBDA RESIDUAL · F49 THE MIRROR AS A SHORT (2026-09-20)

## VERDICT: **STAY-DRY.**
**0 of 12 declared cells clears the claim bar. 0 clears the live-exploration bar. Nothing is
proposed for a dry run, no config change, no engine change.** Two secondary verdicts:
`hod_filter_stack` **arm O is downgraded from "measured and dead" to WEAKENED — its instrument
(EQUS.MINI) does not reproduce the relation it was measuring**, and **arm 2 is NOT RUN: the
affordable 1-second instrument cannot reproduce the quantity lambda is (shown by measurement, not
asserted).**

Pass 16. `PREREG.md` and `FREEZE.md` were committed (`d205778`) **before any cell was scored**.
Programme cell count **1,240 → 1,252** (12 declared: 8 scored in arm 3, 4 declared-and-not-run in
arm 2). Databento spend **$0.8225** of a $12.00 cap. **TEST was never opened.**

Artifacts: `price.py`→`price.csv` · `pull.py`→`raw/*.dbn.zst` (**the raw quotes are kept on disk** —
arm O discarded its own, which is why this calibration had to re-pull) · `calib.py`→
`calib_<day>.csv`, `calib_loss_<day>.csv` · `price2.py`→`price_arm2.csv` · `lam_check.py`→
`lam_check.csv` · `short_walk.py`→`sw_YYYY-MM.csv` (35,454 walked shorts) · `score3.py`→
`cells3.csv`, `cap_sens.csv` · `cost_check.py`→`cost_check.csv` · `nbbo.py`→`nbbo16.csv`
(6,458 measured leg-minutes, 98.5 % with quotes) · `score3b.py`→`cells3_measured.csv`.
One python process at a time, `nice -n 10`, `ulimit -v 3000000`; `bars_sip.db`, the frames15 panels
and `cache.db` opened read-only; nothing written outside `frames16/`; no config, `orb.yaml`, engine,
checker, systemd unit, cron or order touched; no subagents.

---

## 0. THE SENTENCE THIS PASS IS REQUIRED TO PRINT FIRST

**The discovered-mirror short is real on gross and is eaten by its own measured spread.** Shorting
the next bar after a name already ≥ 5 % above its open puts up a **> +2 % hour on ≥ 3× its own
hourly volume** is worth **+0.238 R TRAIN / +0.201 R VAL gross** (day-clustered t **+6.19 / +3.53**,
both TRAIN halves, 15–20 trades a week, 1,726 trades), and it **beats both of its pre-registered
placebos on both splits** — the first time in 1,252 cells that a placebo decomposition has come out
in favour of the signal. At the **measured** NBBO on those same trades' own minutes the round trip
costs **0.194 R TRAIN / 0.199 R VAL** — **1.54–1.56× what the frames14 F45 table charges** — leaving
net **+0.044 (t +1.10) / +0.002 (t +0.04)**, with **ex-top-5 % negative on both splits (−0.083 /
−0.162)**, which is pre-committed kill #3. The gross effect is far above the MDE (0.108 / 0.167 R):
**this is a cost verdict, not a power failure.**

---

## 1. RAILS — asserted in code, raising, before any cell was read

| id | gate | result |
|---|---|---|
| **R1** | frames15 intraday population 42,224 walked signals, `gate5` share 26.93 % | **MATCH** (`score3.repro` raises otherwise) |
| **R2** | frames15 **B12** TRAIN n 2,238 gross **−0.398** R (clustered t −13.33) | **MATCH** |
| **R3** | frames15 **B12** VAL n 1,363 gross **−0.323** R (t −10.51) | **MATCH** |
| **R4** | CKS increments: this pass's implementation vs `hod_filter_stack/ofi.py::cks` | **max\|diff\| = 0** over 4,999 synthetic events (the independent-rebuild rail) |
| **R5** | availability: borrow-flag coverage 100.0 %, prior-close coverage 100.0 %, winner-vs-loser gap **0.0 pp** on both | ok |
| **R6** | measured-NBBO coverage on the scored cells: entry leg **98.1–98.5 %**, exit leg 99.5–99.8 % (S1–S4); winner-vs-loser missingness gap **0.3 pp** | ok |
| **R7** | TEST | `FREEZE.md`, no exception taken; the builder cuts at `2026-06-01`, not the scorer |

---

# ARM 1 — INSTRUMENT CALIBRATION (diagnostic, 0 cells)

Nine instruments priced with `metadata.get_cost` before any pull (`price.csv`, **$0.4726** for all
nine on session 1), then re-priced and pulled for a **second** session as a replication
(**$0.3499**). Raw DBN kept under `raw/`. Method: non-overlapping **10-second** windows over RTH,
`dmid_w = a + b · OFI_w / D_w`, OLS, per name per instrument (PREREG §2).

### 1.1 R² of the contemporaneous relation — the CKS reference is 0.6–0.7 on liquid names with full L1

| dataset / schema | AAPL (large) | FSLY (mid) | USAX (HOD-band small) |
|---|---|---|---|
| | **03-11 / 05-11** | **03-11 / 05-11** | **03-11 / 05-11** |
| **XNAS.ITCH `mbp-1`** (full depth, one major venue) | **0.649 / 0.619** | **0.522 / 0.541** | 0.562 / 0.174 |
| XNAS.BASIC `cmbp-1` (consolidated) | 0.646 / 0.580 | 0.512 / 0.530 | 0.561 / 0.174 |
| XNAS.ITCH `bbo-1s` (1-second) | 0.401 / 0.331 | 0.366 / 0.334 | 0.406 / **0.070** |
| XNAS.BASIC `cbbo-1s` | 0.402 / 0.330 | 0.415 / 0.323 | 0.406 / 0.070 |
| XNYS.PILLAR `mbp-1` | 0.306 / 0.394 | 0.046 / 0.053 | — (24 records) |
| **EQUS.MINI `mbp-1`** — *the dataset arm O used* | **0.006 / 0.110** | 0.038 / 0.035 | 0.139 / 0.231 |
| **EQUS.MINI `bbo-1s`** — *the schema arm O used* | 0.182 / 0.265 | 0.174 / 0.106 | 0.384 / 0.603 |

### 1.2 Why: EQUS.MINI is not quoting the market

Mean quoted spread over the same session, same name (bps):

| | AAPL | FSLY | USAX |
|---|---|---|---|
| XNAS.ITCH `mbp-1` | **1.0 / 1.0** | 10.7 / 10.0 | 88.0 / 334.7 |
| EQUS.MINI `mbp-1` | **10.9 / 5.2** | 74.1 / 57.6 | **1,233.9 / 1,096.1** |

EQUS.MINI quotes AAPL **10.9 bps** where the market quotes **1.0**, and a HOD-band small cap at
**12.3 %** where the market quotes **0.88 %**. Its book is not stale by a tick; it is a different
object. **Standing consequence: EQUS.MINI quotes must not be used for any spread, cost or
order-flow measurement in this house.** (`data/research/databento` daily EQUS.SUMMARY and the PIT
definition feed are unaffected — this is about the QUOTE schemas only.)

### 1.3 Quote events lost at 1-second sampling (same name, same session, same venue)

| | AAPL | FSLY | USAX |
|---|---|---|---|
| XNAS.ITCH | 97.5 / 95.9 % | 91.8 / 91.9 % | 85.7 / 77.0 % |
| EQUS.MINI | 98.5 / 96.8 % | 92.5 / 94.1 % | 66.9 / 62.1 % |

### 1.4 The pre-committed decision rule, row by row

| rule row | fires? | reading |
|---|---|---|
| `mbp-1` < 0.30 on the large cap ⇒ **the code is suspect, debug first** | **FIRED** on EQUS.MINI (0.006) | **Debugged and the code is EXONERATED**: increments identical to `ofi.py::cks` (max\|diff\| 0), and the same code on XNAS.ITCH `mbp-1`, same name, same session, returns **0.649** — inside the published 0.6–0.7. The low number is the DATASET. |
| full-depth ≥ 0.30 AND same-name `bbo-1s` < 0.10 ⇒ **NOT-A-TEST** | does not fire literally | on EQUS.MINI the full-depth leg is 0.006, not ≥ 0.30; on XNAS.ITCH the 1-s leg is 0.40, not < 0.10 |
| both ≥ 0.30 ⇒ **arm O stands** | not applicable to arm O's own instrument | XNAS.ITCH (0.649 / 0.401) would satisfy it, but arm O did not use XNAS.ITCH |
| `mbp-1` ≥ 0.30 on the large cap, `bbo-1s` in [0.10, 0.30) ⇒ **WEAKENED, not void** | **FIRES** (best full-depth 0.649; EQUS.MINI `bbo-1s` on AAPL **0.182**) | **this is the verdict on arm O** |

**Verdict on `hod_filter_stack` arm O: WEAKENED — and for a reason the rule named only obliquely.
The dominant defect is the PUBLISHER, not the clock.** The 1-second clock costs about 35–40 % of R²
on a sound book (0.65 → 0.40, 0.52 → 0.37); EQUS.MINI costs 0.65 → 0.006 on the same name and
session. Arm O's `ofi_5m` (TRAIN +0.124 R spread, **VAL −0.172**) is therefore a **noisy** reading
of order flow, and regression dilution attenuates exactly toward the flat-and-sign-flipping shape it
showed. **Arm O's null on OFI is underpowered by measurement error and is no longer quotable as a
clean null.**

---

# ARM 2 — THE LAMBDA RESIDUAL (4 cells declared, **0 scored — NOT RUN**)

PREREG §3 made the arm conditional on arm 1. Arm 1 left the gate **ambiguous**: on the HOD-band
small cap the 1-second R² reads **0.406** on one session and **0.070** on the other. Rather than
resolve a coin-flip gate by judgement, the thing arm 2 would actually use was **measured**
(`lam_check.py`, no new data — it re-reads arm 1's stored DBN): build λ = Δmid / (OFI/depth) on the
SAME one-minute windows from `mbp-1` and from `bbo-1s`, and ask whether they are the same number.

| instrument | ρ(λ full-depth, λ 1-second) | sign agreement | ρ(flow full, flow 1s) | ρ(λ, return) full / 1s |
|---|---|---|---|---|
| XNAS.ITCH, 6 name-sessions | **+0.065 … +0.321** (median +0.261) | 53.7 – 82.3 % | +0.235 … +0.774 (median +0.590) | −0.15…+0.08 / −0.12…+0.07 |
| EQUS.MINI, 6 name-sessions | −0.047 … +0.604 | 48.2 – 83.1 % | +0.039 … +0.721 | — |

**The 1-second instrument reproduces the full-depth FLOW at ρ ≈ 0.59 and the full-depth LAMBDA at
ρ ≈ 0.26, agreeing even on the SIGN only 54–82 % of the time.** λ is a ratio with the noisy quantity
in the denominator; errors-in-variables on a ratio is the worst possible place to accept a 77–98 %
event loss. (ρ(λ, return) is near zero on the full instrument, so λ is a genuinely distinct object
and not a return in disguise — which makes measuring it properly more important, not less.)

**Arm 2 is declared UNRUNNABLE AT THIS PRICE. No λ number is reported** (PREREG §6: an unmeasurable
quantity is not reported as a null). The cells L1–L4 are declared and unscored.

**What a real λ test costs — measured, not quoted.** Priced over B2's own 344 sessions and its own
symbols (`price_arm2.csv`, a 1-in-4 session sample, exact `metadata.get_cost`):

| instrument | full B2 (TRAIN+VAL) | note |
|---|---|---|
| XNAS.ITCH `bbo-1s` | **$11.23** | carries flow at ρ 0.59, λ at ρ 0.26 — not worth it |
| **XNAS.ITCH `mbp-1`** | **$80.15** | the definitive instrument (R² 0.52–0.65) |
| EQUS.MINI `mbp-1` | *$450 (arm O's own quote)* | **5.6× too high AND the wrong dataset** |

**The owner's call: the real order-flow test on this book costs $80, not $450.**

---

# ARM 3 — F49: THE DISCOVERED MIRROR AS A SHORT (8 cells)

Signal: at a session-hour close, `hrv ≥ 3` (the hour's volume ÷ `adv20 × share_h`, the symbol's own
hour shape, built from prior sessions only) and the hour's return past the mirror cut, on a name
whose session high has **already** reached open × 1.05 by that hour's close (`gate5`, causal
membership at the decision bar). **MIR2** = `|hour_ret| > 2 %` (the frames15 B12 object verbatim);
**UP2** = `hour_ret > +2 %` (the natural short leg, declared in advance, one degree of freedom).
Entry: a resting **sell limit** at `ref × (1 − 0.6 %)` filling at the NEXT BAR'S OPEN iff the open is
at or above it — a bar that opens below is a SKIP at 0 P&L, never a loss. 35,454 shorts walked;
after gate5 · price ≥ $5 · ex-wrapper the population is 6,737; after ETB and SSR, **2,100 filled
MIR2 trades (1,726 for UP2)**.

### 3.1 The mirror is a real long loss on exactly this universe (the (a)-check)

| | TRAIN | VAL |
|---|---|---|
| B12 as published (all prices, wrappers in) | n 2,238, **−0.398 R** (t −13.33) | n 1,363, **−0.323 R** (t −10.51) |
| B12 restricted to the short arm's universe (≥ $5, ex-wrapper) | n 1,439, **−0.359 R** (t −8.98) | n 977, **−0.301 R** (t −7.82) |

The long loss is not a penny-stock or wrapper artefact — it survives the restriction almost intact.

### 3.2 The cells at the F45 IMPUTED cost (the reading that looked like a candidate)

| cell | split | n | gross R | % of price | t clust | cost R | net R | t clust | /wk | green % (null p95) | weekly $ | worst wk | ex-5 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **S1** MIR2 · 2 % stop · bare | TRAIN | 1,269 | +0.153 | +0.306 | +4.64 | 0.124 | +0.029 | +0.88 | 16.7 | 52.8 (58.5) | +$37 | −$790 | +0.025 |
| | VAL | 831 | +0.129 | +0.259 | +2.55 | 0.125 | +0.004 | +0.08 | 21.6 | 60.9 (60.9) | +$93 | −$1,228 | −0.032 |
| **S2** MIR2 · 2 % · +2R | TRAIN | 1,269 | +0.174 | +0.348 | +5.33 | 0.126 | +0.048 | +1.47 | 17.7 | 58.5 (60.4) | +$66 | −$852 | +0.173 |
| | VAL | 831 | +0.144 | +0.287 | +3.04 | 0.128 | +0.016 | +0.33 | 24.0 | 60.9 (60.9) | +$81 | −$1,193 | +0.144 |
| **S3** UP2 · 2 % · bare | TRAIN | 1,071 | **+0.238** | +0.475 | **+6.19** | 0.126 | **+0.112** | **+2.94** | 15.2 | 58.5 (67.9) | +$130 | −$737 | +0.110 |
| | VAL | 655 | **+0.201** | +0.401 | **+3.53** | 0.128 | +0.073 | +1.28 | 20.0 | 60.9 (69.6) | +$168 | −$1,259 | +0.038 |
| **S4** UP2 · 2 % · +2R | TRAIN | 1,071 | +0.260 | +0.521 | +6.85 | 0.128 | +0.132 | +3.52 | 15.7 | 62.3 (69.8) | +$168 | −$906 | +0.259 |
| | VAL | 655 | +0.194 | +0.389 | +3.73 | 0.131 | +0.064 | +1.22 | 21.7 | 56.5 (69.6) | +$147 | −$893 | +0.194 |
| **S5** MIR2 · hr-high stop · bare | TRAIN | 1,269 | +0.219 | +0.206 | +4.40 | 0.181 | +0.038 | +0.75 | 17.3 | 50.9 (56.6) | +$50 | −$1,807 | −0.009 |
| | VAL | 831 | +0.193 | +0.300 | +3.02 | 0.169 | +0.025 | +0.38 | 21.8 | 56.5 (69.6) | +$239 | −$1,120 | −0.061 |
| **S6** MIR2 · hr-high · +2R | TRAIN | 1,269 | +0.207 | +0.207 | +5.43 | 0.188 | +0.019 | +0.49 | 18.9 | 43.4 (52.8) | −$25 | −$1,751 | +0.206 |
| | VAL | 831 | +0.147 | +0.213 | +3.42 | 0.175 | −0.028 | −0.65 | 24.7 | 52.2 (56.5) | +$3 | −$1,422 | +0.147 |
| **P1** placebo: volume, NO price (`\|ret\| ≤ 1 %`) | TRAIN | 1,192 | **+0.025** | +0.050 | +0.87 | 0.110 | −0.084 | −2.91 | 15.3 | 28.3 (41.5) | −$128 | −$915 | −0.076 |
| | VAL | 751 | **+0.008** | +0.016 | +0.19 | 0.111 | −0.103 | −2.48 | 18.9 | 26.1 (39.1) | −$205 | −$1,190 | −0.110 |
| **P2** placebo: a RANDOM other hour, same name-days | TRAIN | 1,092 | **+0.004** | +0.008 | +0.13 | 0.108 | −0.104 | −3.16 | 14.7 | 30.2 (37.7) | −$146 | −$925 | −0.094 |
| | VAL | 684 | **+0.067** | +0.135 | +1.39 | 0.109 | −0.042 | −0.86 | 18.3 | 52.2 (60.9) | +$75 | −$753 | −0.078 |

**The placebo decomposition comes out for the signal, on both splits.** S3's gross (+0.238 / +0.201)
is 6–25× P1's (+0.025 / +0.008) and 3–60× P2's (+0.004 / +0.067). The F49 claim is specifically
about the MIRROR — volume WITH price — and it survives its own controls. That is new.

### 3.3 The cells at the **MEASURED** per-trade NBBO — this is the number that decides

`cost_check.py` on a 350-trade sample found the F45 table 1.80× too narrow in the mean on this
population, so `nbbo.py` measured **every leg of every MIR2 trade**: 6,458 distinct (day, symbol,
minute) SIP quote-minutes, 98.5 % with quotes, entry-leg coverage **98.1–98.5 %** on the scored
cells with a **0.3 pp** winner-vs-loser missingness gap. Two contracts reported side by side:
**half+half** (half the measured spread at each leg — what §3.2 charged, the conservative reading)
and **per-outcome** (the PLAN §2 score4 contract: entry half always, exit × {stop 0.875, eod 0.412,
target 0.0} — the favourable reading).

| cell | split | gross R | cost: imputed → **measured** (×) | net **half+half** (t) | net **per-outcome** (t) | green % (null) hh | weekly $ hh | **ex-top-5 % hh** | MDE |
|---|---|---|---|---|---|---|---|---|---|
| **S1** | TRAIN | +0.153 | 0.124 → **0.198** (1.60×) | **−0.045** (−1.31) | −0.017 (−0.51) | 47.2 (49.1) | −$82 | −0.173 | 0.101 |
| | VAL | +0.129 | 0.125 → **0.207** (1.66×) | **−0.077** (−1.44) | −0.052 (−0.96) | 52.2 (56.5) | −$41 | −0.241 | 0.154 |
| **S2** | TRAIN | +0.174 | 0.126 → 0.205 (1.63×) | −0.031 (−0.94) | +0.007 (+0.20) | 49.1 (50.9) | −$73 | −0.033 | 0.100 |
| | VAL | +0.144 | 0.128 → 0.217 (1.70×) | −0.074 (−1.46) | −0.032 (−0.63) | 56.5 (52.2) | −$104 | −0.074 | 0.143 |
| **S3** | TRAIN | +0.238 | 0.126 → **0.194** (1.54×) | **+0.044** (+1.10) | +0.072 (+1.82) | 50.9 (58.5) | +$30 | **−0.083** | 0.108 |
| | VAL | +0.201 | 0.128 → **0.199** (1.56×) | **+0.002** (+0.04) | +0.028 (+0.46) | 52.2 (60.9) | +$52 | **−0.162** | 0.167 |
| **S4** | TRAIN | +0.260 | 0.128 → 0.202 (1.58×) | +0.058 (+1.49) | **+0.098** (+2.48) | 54.7 (60.4) | +$53 | +0.056 | 0.107 |
| | VAL | +0.194 | 0.131 → 0.210 (1.61×) | **−0.016** (−0.30) | +0.025 (+0.45) | 47.8 (56.5) | −$1 | −0.016 | 0.153 |
| **S5** | TRAIN | +0.219 | 0.181 → 0.265 (1.47×) | −0.047 (−0.86) | −0.011 (−0.21) | 41.5 (49.1) | −$91 | −0.268 | 0.142 |
| | VAL | +0.193 | 0.169 → 0.234 (1.38×) | −0.040 (−0.61) | −0.013 (−0.20) | 47.8 (60.9) | +$122 | −0.291 | 0.202 |
| **S6** | TRAIN | +0.207 | 0.188 → 0.279 (1.48×) | −0.072 (−1.80) | −0.013 (−0.33) | 35.8 (43.4) | −$181 | −0.074 | 0.116 |
| | VAL | +0.147 | 0.175 → 0.235 (1.35×) | −0.089 (−2.02) | −0.049 (−1.09) | 47.8 (47.8) | −$122 | −0.089 | 0.153 |

TRAIN halves, net, half+half / per-outcome: S1 −0.037/−0.051 · −0.013/−0.021 · S2 −0.035/−0.029 ·
+0.002/+0.011 · **S3 +0.067/+0.025 · +0.091/+0.056** · **S4 +0.067/+0.051 · +0.105/+0.092** ·
S5 +0.010/−0.092 · S6 −0.029/−0.106.
*(S5/S6's exit legs are only 64 % / 48 % measured — their exit minutes differ from spec a's and were
outside the measured set — so their measured cost is partly imputed and is the FAVOURABLE bound.
P1/P2 are reported at the IMPUTED cost throughout, which is favourable to the placebos and therefore
conservative for the comparison the signal wins.)*

### 3.4 The rails, each switched off (S1, imputed cost)

| rail | share affected | S1 gross with the rail off |
|---|---|---|
| borrow (ETB ∧ shortable; absent ⇒ not shortable) | **7.2 %** unshortable | +0.177 TRAIN / **+0.116** VAL (vs +0.153 / +0.129) — the borrowable subset is neither the better nor the worse half, consistent with F2's "borrow is not the constraint" |
| SSR (Reg SHO 201 uptick) | **10.6 %** under SSR, **6.4 %** of fills blocked | +0.147 / +0.141 |
| SSR, conservative upper bound (prev session ≤ −7 %) | 12.5 % under SSR, 7.9 % blocked | +0.155 / +0.133 |
| no-chase cap 0.3 / 0.6 / 1.2 % / none | 2.9 % skipped at 0.6 % | +0.160/+0.136 · +0.153/+0.129 · +0.168/+0.123 · +0.167/+0.123 |

**Unfilled counterfactual (the halt-resume adverse-selection check, run because a passive entry is
structurally the object that selected badly there):** the signals the cap SKIPS would have returned
**+0.241 R** against **+0.151 R** on the filled set — the cap is costing us, i.e. this passive short
does **not** carry the dip-buy signature. It is a lead, not a defect (see F52).

### 3.5 Against the pre-committed bar and the pre-committed kills

Bar (BOTH splits): positive weekly $ · green ≥ 50 % · ≥ 10 trades/wk · day-clustered t ≥ 2 · TRAIN
halves same-signed · ex-top-5 % positive on the uncapped exit.

* **At the measured cost, no cell reaches day-clustered t ≥ 2 on BOTH splits.** The best is S4
  per-outcome (TRAIN +2.48, **VAL +0.45**); S3 half+half is (+1.10, **+0.04**).
* **Kill #3 fires on the best uncapped cell**: S3's ex-top-5 % is **−0.083 TRAIN / −0.162 VAL**. The
  edge lives in the same top 5 % that has killed every long cell in this programme.
* **Kill #1 does not fire** (7.0–7.2 % unshortable; borrow is not the constraint).
* **Kill #2 does not fire** (ETB intraday borrow fee assumed 0; HTB excluded, not estimated).
* **Kill #4 does not fire** (SSR blocks 5.4–6.4 % of fills and moves gross by ≤ 0.006 R).
* **Kill #5 does not fire** — the mirror DOES beat both placebos on both splits. The frame's
  discriminating prediction is the one thing that survived.
* **Green weeks sit INSIDE or BELOW the count-matched permutation band on every cell and both
  splits.** The week shape here is pick COUNT, not timing skill — the same standing finding as
  `hod_break` and `orb_gates2`.
* **Live-exploration bar: NOT met.** It needs a positive point estimate on green weeks AND dollars
  at live size on both splits; at the measured cost S3 is +$30/wk TRAIN and +$52/wk VAL with green
  weeks below their own null on both, and S4 is −$1/wk on VAL.

---

## 4. THE POWER STATEMENT (PLAN §1 phrasing)

> **No edge was detectable in the discovered-mirror short — in the `bars_sip.db` causal-superset
> universe restricted to `gate5` / price ≥ $5 / ex-wrapper / Alpaca-ETB names, at a same-session
> horizon, at 12-per-day / 4-concurrent book size, over 2025-01 → 2026-05, at the MEASURED per-trade
> SIP NBBO — and the smallest per-trade effect this test could have seen at 80 % power is
> 0.108 R on TRAIN and 0.167 R on VAL (day-clustered).**

The distinction that matters: **the gross effect (+0.238 / +0.201 R) is far ABOVE that MDE and is
measured with confidence**; what fails is the difference between it and a cost of 0.194 / 0.199 R
measured on the same trades' own minutes. This is a **cost verdict, not a null**, and the honest
reading is that the mirror short is an edge of roughly the size of its own round trip.

**Multiplicity: 1,252 cells** (1,240 prior + 12 declared here; 8 scored). The 8 scored cells overlap
heavily — S1/S2/S5/S6 share one population and S3/S4 are its up-leg — so they are not 8 independent
looks; but at this point in the programme a TRAIN gross t of +6.19 with a VAL net t of +0.04 is
exactly the shape the previous 1,240 produced.

---

## 5. CAVEATS, WRITTEN AS AN ADVERSARY WOULD

1. **The cost is the ENTIRE verdict, and one caveat alone could reverse it.** The measured **entry**
   leg is 0.379 % of price at the median (0.569 % mean) against an **exit** leg of 0.161 % — the
   entry is 2.4× the exit, because every fill in this programme is a *reacting* order at the next
   bar's open that crosses the spread by construction. A short is the one side that can *rest* a
   sell limit at or above the offer and be **paid** the spread rather than pay it, and §3.4's
   unfilled counterfactual (+0.241 R on skipped vs +0.151 R on filled) says the passive side here is
   selecting the BETTER half, not the worse one. **As written, this verdict holds only for a
   reacting order at the next bar's open.** That is why F52 is the first frame out of this pass.
2. **Halts and LULD are not modelled at all.** On a name already +5 % on the day making a > 2 % hour
   on 3× volume, an upward LULD halt is precisely the event that hurts a short most, and the walk
   covers at a bar's open as if the tape were continuous. Unmodelled tail risk, on the losing side,
   in a population selected for exactly that risk.
3. **The borrow list is TODAY's snapshot** (14,355 symbols). Names delisted since are absent and are
   treated as not shortable (conservative); names that were hard-to-borrow in 2025 but are ETB now
   are wrongly included (optimistic). The rail moves gross by ≤ 0.024 R either way.
4. **The SSR prior-day leg uses close-to-close ≤ −10 %**, which UNDER-counts triggers (a name that
   touched −10 % intraday and recovered is missed). The conservative ≤ −7 % bound moves S1 gross by
   +0.002 / +0.004 R — immaterial, but it is an approximation, not a measurement.
5. **The universe is a superset built from a day-level condition** (`bars_sip.db` holds symbol-days
   whose session high reached open × 1.05). `gate5` makes MEMBERSHIP causal at the decision bar, but
   a name that never ran 5 % is never walked. The claim is limited to names that had already run 5 %
   by the signal hour.
6. **P1/P2 carry the imputed (too narrow) cost** because the NBBO measurement was restricted to the
   MIR2 population to fit the node's time budget. That is favourable to the placebos, so the
   signal-beats-placebo ordering is conservative — but the placebos' *net* numbers in §3.2 are
   optimistic by roughly the same 1.5× the signal's were.
7. **Arm 1 is two sessions and three names.** The AAPL result (0.006 vs 0.649 on the same session) is
   two orders of magnitude and is corroborated mechanically by the quoted spread (10.9 vs 1.0 bps),
   which does not depend on the regression — but the small-cap tier is genuinely unstable across the
   two sessions (0.562 → 0.174 full depth), and that instability is itself part of why arm 2 was not
   run.
8. **This pass changed the cost basis mid-stream.** §3.2 was scored at the imputed cost and looked
   like a live-exploration candidate; §3.3 is the same cells at the measured cost and is not. Both
   are printed at the same length deliberately. The number that counts is §3.3.

---

## 6. THE SHIP DECISION

**STAY-DRY. Nothing is proposed for a dry run.** `config.yaml` and `orb.yaml` are exactly as the
owner set them (`trading.enabled: false`, `orb.yaml strategy.enabled: false`,
`hod_break.enabled: true, dry_run: true`, `red_to_green.enabled: false`). No `HodBreakParams`
change, no new engine, no short book. Two things the owner may want to decide:

1. **$80 buys the order-flow test the 9/13 plan asked for** (XNAS.ITCH `mbp-1` over B2's own
   windows), against the $450 arm O quoted for the wrong dataset. Not taken inside this pass's $12
   cap.
2. **EQUS.MINI quote schemas are retired** for spread/flow measurement anywhere in this house.

---

## 7. THE NEXT THREE FRAMES

**F52 — THE ENTRY IS THE COST: the mirror short on a RESTING limit.** This pass's killer is the
entry half-spread: 0.379 % of price at the median entry minute, 2.4× the exit leg, 1.54–1.66× what
the F45 table charges, on a book that is gross-positive by +0.24 R = +0.48 % of price. Every fill in
1,252 cells has been a reacting order at the next bar's open, which crosses the spread by
construction. The short side can rest a sell limit at or above the offer and **earn** the spread —
and §3.4 already measured that the passive side here selects the BETTER half (+0.241 R on skipped
vs +0.151 R on filled), the opposite of the halt-resume dip-buy signature. Frame: re-walk with a
resting sell limit at the offer / at the prior bar's high, take the fill only when a later bar
actually traded at or above the limit (obtainability, PLAN §1.1b), and price the same book earning
half the spread instead of paying it. The swing is 0.2–0.4 R per trade against a +0.24 R gross —
it is the only lever in this pass that can change the sign. Pre-commit: the fill rate and the
unfilled counterfactual are printed BEFORE the book, and a passive fill that only happens on days
the short loses is reported as adverse selection, not as an edge.

**F53 — THE MIRROR AS A VETO FOR THE LIVE BOOKS (no new data, no new book).** The same object is
**−0.359 R TRAIN / −0.301 R VAL as a LONG** on exactly the universe bull flag trades (≥ $5,
ex-wrapper, already ≥ 5 % above the open). BF detects 57.5 % of its setups in the 10:00–11:00 hour —
after `hrv` for the 09:30 hour exists — and has never been asked whether its own entry sits inside a
mirror hour. Frame: join `frames15/hourly15.parquet` onto BF's Stage-2 cache at BF's own decision
minute, score the mirror condition as a veto with the RUNBOOK step-5 gate-separation map (kept minus
rejected, n each side, t, per year and pooled, median position risk each side), and report the
frequency cost first. A veto is a config flag with zero state to unwind, and the live books are the
only place in this house where an edge of this size compounds. Pre-commit: ORB is **out of scope**
(it enters at 09:35, before any hour has closed) and the rule must be era-consistent in 2025 and
2026 separately before it is proposed.

**F54 — THE FULL-DEPTH ORDER-FLOW TEST, SCOPED AND PRICED.** Arm 1 establishes that this house has
never measured order flow on an instrument that reproduces the CKS relation, that the definitive one
(XNAS.ITCH `mbp-1`, R² 0.52–0.65) costs a **measured $80.15** for the whole B2 population, and that
EQUS.MINI quotes are unusable for any flow or spread work. Frame: an owner-authorised $80 pull, then
λ, the residual, OFI at 1 and 5 minutes, and the depth/queue features, on a sound instrument, with
arm 1's calibration re-run **per session as an acceptance gate** (a session whose large-cap R² is
below 0.30 is dropped, not scored). Pre-commit: the cells are `hod_filter_stack` arm O's plus this
pass's L1–L4 verbatim, so the result is directly comparable with the weakened reading; and the raw
DBN is kept on disk, which arm O did not do.

*(Run order: F52 first — it is the only lever that can flip a measured sign and it needs no new
data; then F53, which costs nothing and touches a live book; then F54, which spends money.)*
