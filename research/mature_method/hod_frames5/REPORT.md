# hod_frames5 — F16 the portfolio · F18 the population · F17 the horizon — REPORT (2026-09-19)

Pass 5 of the HOD-break frame programme. Cells exactly as declared in `PREREG.md`, **committed
`7144d3c` before any cell was scored**. Artifacts: `diag5.py` → `diag5.log`, `popsizes.csv` ·
`score16.py` → `score16.log`, `cells16.csv`, `nulls16.csv` · `build_todo5.py` → `nbbo5_todo.csv` ·
`fetch_nbbo5.py` → `nbbo5.csv` (**7,605 fresh Alpaca-SIP quote-minutes**, 23 min at 331/min) ·
`score18.py` → `score18_asis.log` / `score18_measured.log`, `cells18_{asis,measured}.csv`,
`nulls18_*.csv`, `f18_r1_*.csv` · `supp5.py` → `supp5.log` · `score17.py` → `score17.log`,
`cells17.csv`, `nulls17.csv`. One python process at a time, `nice -n 10`, `ulimit -v 3000000`;
`cache.db`, `daily_bars` and every store opened **read-only**. No config, `orb.yaml`, systemd unit,
cron, order or cache was written. The dry run was not touched. **TEST was never opened**
(`FREEZE.md`).

---

## THE SENTENCE THIS PASS PRE-COMMITTED TO PRINTING FIRST

**The base book's gross is −0.002 R (TRAIN) / +0.015 R (VAL) and the booked-set measured cost is
0.061–0.065 R. In 958 prior cells the best obtainable gross was about +0.1 R. No cost frame, no fill
frame and no sizing frame can rescue this book — even free fills leave it at zero.** The three
frames here were admitted to the queue only because each changes the OBJECT: which positions are
held together (F16), which instruments are in the universe (F18), how long the position is held
(F17). None of them changes that arithmetic, and this pass does not claim otherwise.

---

## VERDICT — **STAY DRY on all three.** 0 of 31 declared cells clear either bar.

1. **F16 — the frame's own premise is FALSE, measured before any cell.** The four concurrent slots
   are **not** routinely four members of one complex. Over 84,587 (TRAIN) / 38,050 (VAL) occupied
   slot-minutes, **only 8.8 % / 9.9 % have two open positions sharing an `underlying_anchor`**, the
   **maximum same-anchor concurrency ever reached is 2 (TRAIN) and 3 (VAL)** — never 4 — and only
   **7.1 % / 7.5 %** of booked trades ever have a same-anchor sibling open. There is no 4× bet to
   de-duplicate. Consistently: 1-per-anchor de-dup raises green weeks (32.1 → 34.0 TRAIN, 43.5 →
   47.8 VAL, both **inside** their nulls) and **loses money on both splits** (−$17,346 → −$18,633;
   +$893 → −$163) while making MDD **worse** on both (−$18,929 → −$19,734; −$2,940 → −$4,264).
   2-per-anchor is a no-op by construction. **20 of 20 nulls inside.**
2. **F18 — the population is NOT the problem, and the parity claim that motivated the frame is
   wrong.** `config.yaml hod_break.min_price = 20.0`: the live dry run is on the **same $20
   population as every study**, not the $5 one `FRAMES.md` asserted. The base gross at the signal
   level is **flat at every price floor from $5 to $50** (−0.012 … +0.033 R on the as-is cost model,
   +0.001 … +0.035 R on the measured one) and **flat across every spread rung** (+0.000 … +0.005 R)
   — the cascade is innocent; loosening the gates only buys more cost (0.037 → 0.151 R). **Pass 4's
   "common stock is era-consistently negative" does not survive a measured cost model** (§2.2):
   under the dedicated fetch common-only is H1 **+0.055** / H2 −0.027 / VAL −0.012 at the signal
   level. And the wrappers-only population, the pass's declared candidate, **fails its own
   pre-committed promotion rule on both cost arms** — H1 −0.117 (as-is) / −0.095 (measured, booked),
   VAL green weeks **below** its own count-matched null on the as-is arm.
3. **F17 — the horizon does not pay, and the overnight risk is a different book.** Carrying the
   27.9 / 33.0 % of trades that reach 15:55 alive: **not one of the eight horizon cells is
   era-consistent, all 18 nulls are inside, and net R is negative on every cell on both splits
   except `h2a` VAL (+0.035, MDE 0.172)**. The gain the frame hoped for is real but tiny and
   sign-unstable: gross rises on TRAIN at the next open (−0.039 → −0.003) and on VAL at +1 session
   (+0.083 → +0.114), and the MDE rises faster than the estimate (0.088 → 0.292 R at +5 sessions).
   The cost of getting there: **33.3 % of carried trades end worse than −1 R** (the intraday stop)
   at +5 sessions and the worst single trade is **−14.87 R**, against a book whose entire risk
   model is a 1 R stop. The wrapper decay was measured, not assumed: **−0.2423 %/day** = **0.083 R
   per calendar day held**, and it is already inside every multi-day number because those numbers
   are priced from the wrapper's own daily bars.

---

## 0. Reproduction gate — EXACT, plus a new parity assertion and two independent reproductions

| id | this pass | reference | verdict |
|---|---|---|---|
| R1 `B2` TRAIN | 1,622 · 30.6/wk · gross −0.039 · net −0.107 · 32.1 % · **−$17,346** | `hod_frames4` §0 | **MATCH** (Δ$ 0) |
| R2 `B2` VAL | 706 · 30.7 · +0.083 · +0.013 · 43.5 % · **+$893** | identical | **MATCH** |
| **R3 portfolio-slot parity** | `common5.book_portfolio(key=None)` vs `common4.book_ranked` | 2,328 vs 2,328, identical set | **asserted in code** (the run aborts otherwise) |
| R4 `F16-m1` = pass 4's causal `anchor_cohort` | −0.013 / +0.154 gross, −$2,284 / +$2,099, H1 −0.120, 6.4 / 9.7 tr/wk | `hod_frames4` §3.3 | **MATCH to the printed digit** |
| R5 `F18-p2` = pass 4's `F15-c1` (common only) | 1,276 / −$15,628 · 650 / −$8,234, −0.060/−0.046/−0.055 | `hod_frames4` §3.2 | **MATCH** |

R4 and R5 are reproductions of pass-4 cells from a **separately written** slot machine and a
separately written instrument join — the independent-rebuild rail applied to the two pass-4 numbers
this pass leans on.

The **measured-cost arm is a declared sensitivity arm, not the reference** (§2.5): with
`nbbo5.csv` merged, B2 reads 1,668 / −$12,627 / +$2,427. That is a different set of trades, not a
cheaper one — §2.5 decomposes the whole difference.

---

# F16 — THE PORTFOLIO (10 cells: 1 diagnostic + 9 scored)

## 1.1 F16-diag — the concentration diagnostic, and the premise it refutes

Printed before any cell, as declared.

| split | occupied slot-minutes | ≥2 share an **anchor** | ≥2 a **venue** | ≥2 an **ADV$ bucket** | max same-anchor concurrency | booked rows with a same-anchor sibling open |
|---|---|---|---|---|---|---|
| TRAIN | 84,587 | **8.8 %** | 71.5 % | 37.9 % | **2** | 115 / 1,622 (**7.1 %**) |
| VAL | 38,050 | **9.9 %** | 78.1 % | 61.8 % | **3** | 53 / 706 (**7.5 %**) |

Top booked anchors: TRAIN TSLA 48 / SMCI 37 / MSTR 33 / IONQ 32 / PLTR 30; VAL IREN 20 / ASTS 16 /
AXTI 15 / PLTR 13 / IONQ 13. The pre-book set is 38.6 % wrapper, 61.0 % stock, 0.5 % unidentified;
the booked set is 38.5 / 61.0 — the slot rule is not concentrating the book by instrument class
either.

**`FRAMES.md`'s F16 premise — "the four concurrent slots are routinely four members of one
underlying complex" — is false on this book.** Two slots on one anchor happen in under a tenth of
slot-minutes; three happened at all only in VAL; four never happened. The anchor clustering F15 saw
(TSLA 72, PLTR 52) is clustering *across the period*, not *within a minute*: the same underlying
supplies many candidates on many different days. The `venue` number (71.5 / 78.1 %) is a five-valued
field and is reported only so it is not mistaken for a finding.

## 1.2 The cells

Live sizing, $100 risk. Admission untouched in every row.

| cell | TRAIN /wk · gross · net · grn % · **$** · MDD | VAL /wk · gross · net · grn % · **$** · MDD | H1 / H2 / VAL |
|---|---|---|---|
| **F16-base** B2 | 30.6 · −0.039 · −0.107 · 32.1 · **−17,346** · −18,929 | 30.7 · +0.083 · +0.013 · 43.5 · **+893** · −2,940 | −0.077 / −0.003 / +0.083 |
| **F16-a1** 1/anchor, 4 slots | 30.5 · −0.047 · −0.115 · **34.0** · −18,633 · −19,734 | 30.9 · +0.068 · −0.002 · **47.8** · **−163** · −4,264 | −0.082 / −0.014 / +0.068 |
| **F16-a2** 2/anchor, 4 slots | 30.6 · −0.039 · −0.107 · 32.1 · −17,346 · −18,929 | 30.8 · +0.084 · +0.014 · 43.5 · +989 · −2,940 | −0.077 / −0.003 / +0.084 |
| **F16-a1n** 1/anchor, **no refill** | 30.5 · −0.048 · −0.117 · 34.0 · −18,806 · −19,907 | 30.6 · +0.073 · +0.002 · 47.8 · +157 · −4,094 | −0.085 / −0.014 / +0.073 |
| **F16-a6** 1/anchor, 6 slots | 39.7 · −0.049 · −0.118 · 26.4 · −24,833 · −24,833 | 43.2 · +0.058 · −0.013 · 52.2 · −1,258 · −5,367 | −0.087 / −0.015 / +0.058 |
| **F16-a8** 1/anchor, 8 slots | 44.1 · −0.045 · −0.114 · 30.2 · −26,516 · −26,516 | 49.6 · +0.036 · −0.035 · 30.4 · −3,950 · −8,391 | −0.077 / −0.017 / +0.036 |
| **F16-v2** 2/venue, 4 slots | 28.1 · −0.032 · −0.100 · **37.7** · −14,801 · −16,869 | 30.0 · +0.047 · −0.022 · 39.1 · −1,502 · −4,108 | −0.081 / +0.013 / +0.047 |
| **F16-b2** 2/ADV$ bucket | 30.4 · −0.053 · −0.121 · 34.0 · −19,498 · −19,968 | 30.8 · +0.071 · +0.001 · 47.8 · +86 · −3,712 | −0.077 / −0.031 / +0.071 |
| **F16-m1** CONCENTRATED (causal sibling) | 6.4 · −0.013 · −0.068 · 39.6 · −2,284 · −3,637 | 9.7 · +0.154 · +0.094 · 52.2 · +2,099 · −2,044 | −0.120 / +0.084 / +0.154 |
| **F16-m2** concentrated × 1/anchor | 4.7 · −0.058 · −0.113 · 35.8 · −2,823 · −3,477 | 8.0 · +0.099 · +0.040 · 47.8 · +733 · −1,507 | −0.131 / +0.003 / +0.099 |

**Best de-dup cell: `F16-a1`** — and it is the frame's own refutation. It moves the owner's primary
metric in the right direction on both splits (+1.9 pp TRAIN, +4.3 pp VAL green weeks), and both
readings are **inside the count-matched null** (TRAIN 34.0 vs 29.5 [22.6, 35.8]; VAL 47.8 vs 46.9
[34.8, 56.5]). It costs **$1,287 on TRAIN and $1,056 on VAL**, and it makes **MDD worse on both
splits** — which kills the frame's stated mechanism outright: a rule sold as *removing a risk
concentration* increases the drawdown, because the trade it removes is replaced by a later, worse
one. `F16-a1n` (ORB's no-refill invariant) removes the replacement and is worse still on TRAIN.

`F16-a2` is byte-identical to B2 on TRAIN and +$96 on VAL, exactly as the diagnostic predicted (the
constraint binds in almost no minute). More slots is strictly worse on both splits — a third
independent confirmation of F13's by-product.

**F16-m1, the mirror, reproduces pass 4's causal `anchor_cohort` number to the printed digit** and
is reported as a reproduction, not a discovery: at 6.4 / 9.7 trades a week it is under the
frequency floor, H1 is −0.120, its green weeks are inside its null on both splits (39.6 vs 40.6
[34.0, 47.2]; 52.2 vs 53.5 [43.5, 65.2]), and its net (−0.068 / +0.094) is below its MDE
(0.209 / 0.242) on both.

**Nulls: 20 of 20 INSIDE.** No cell reaches either bar. **F16 verdict: DEAD.**

---

# F18 — THE POPULATION (14 scored cells + 2 structural reports, on TWO cost arms)

## 2.1 The live-parity finding (reported regardless of everything else)

| knob | `config.yaml hod_break` | the study population |
|---|---|---|
| `min_price` | **20.0** | `next_open >= $20` |
| `max_spread_bps` | 100 | 100 |
| `max_spread_frac_r` | 0.15 | 0.15 |
| `max_per_day` / `max_concurrent` | 12 / 4 | 12 / 4 |
| `last_entry_minute` | 840 | 840 |
| `cap` | 0.006 | 0.006 |

**`FRAMES.md` F18's claim that "the live engine uses a $5 floor, so the study population is not even
the shipped one" is WRONG.** The dry run has run at `min_price: 20.0` throughout; the four sessions
of forward data are on the same population as every study. Two residual items, both reported:

* **A definitional difference of 0.3 %**: the study floors on `next_open` (the fill), the engine
  floors on the break **level** (`hod_break_engine.py:665`). On the $20 first-break stream that is
  **8 signals engine-only and 11 study-only out of 7,027**. Not material; not changed by this pass.
* **`CLAUDE.md` is stale**: its HOD-break section still says "price ≥ $5 (cost rule 9/14)", which is
  not what the shipped config does. Flagged, not edited — this pass writes no config and no
  operating doc.

## 2.2 F18-R1 — the base GROSS of every population choice, at the SIGNAL level, before any admission rule

The question `FRAMES.md` says nine passes never asked. As-is cost arm and measured cost arm;
"era-consistent" = gross same-signed positive in H1-2025, H2-2025 and VAL.

| population | as-is TRAIN / VAL gross | as-is H1 / H2 | measured TRAIN / VAL gross | measured H1 / H2 | measured era-consistent |
|---|---|---|---|---|---|
| **mixed $20 (B2)** | −0.002 / +0.015 | −0.020 / +0.021 | +0.016 / +0.031 | +0.011 / +0.023 | **True** |
| mixed $5 | −0.012 / +0.001 | +0.006 / −0.032 | +0.001 / +0.010 | +0.024 / −0.023 | False |
| mixed $10 | −0.006 / −0.001 | +0.002 / −0.014 | +0.014 / +0.008 | +0.022 / +0.006 | True |
| mixed $30 | +0.003 / +0.018 | −0.027 / +0.045 | +0.017 / +0.046 | +0.004 / +0.033 | True |
| **mixed $50** | +0.033 / +0.019 | +0.017 / +0.056 | +0.035 / +0.041 | +0.036 / +0.034 | **True** |
| WRAP $20 | +0.025 / +0.107 | −0.031 / +0.092 | +0.009 / +0.097 | −0.052 / +0.081 | False |
| COMMON $20 | −0.022 / −0.044 | −0.013 / −0.034 | +0.018 / −0.012 | **+0.055** / −0.027 | False |
| mixed $20 **gates OFF** | +0.005 / −0.032 | +0.004 / +0.007 | +0.003 / −0.028 | +0.005 / +0.001 | False |
| mixed $20 frac_r 0.08 | −0.046 / +0.013 | −0.105 / +0.047 | +0.001 / +0.048 | −0.028 / +0.039 | False |
| mixed $20 frac_r 0.25 | +0.005 / −0.001 | +0.006 / +0.005 | +0.010 / +0.008 | +0.021 / −0.001 | False |
| mixed $20 frac_r 0.40 | +0.000 / −0.015 | −0.000 / +0.001 | +0.006 / −0.004 | +0.012 / −0.001 | False |

**The answer to "is B2 the pattern?" is YES.** The raw break's gross is flat — between −0.012 and
+0.035 R — across a price band spanning a factor of ten, and flat across every spread rung including
both gates off. The gates select a **cheaper** population, not a **better** one: on the measured arm
the booked cost runs 0.037 (8 % of R) → 0.059 (the shipped 15 %) → 0.079 → 0.102 → 0.151 R (gates
off) while gross moves by hundredths. **Nine passes of verdicts are NOT conditional on a gate nobody
validated — they are conditional on a pattern that is edgeless everywhere in this price band.** That
is the structural deliverable of F18 and it makes every prior verdict stand harder, not softer.

Two corrections to the prior ledger fall out of it:

* the **$50 floor is era-consistent on gross on both cost arms** (+0.036 / +0.034 / +0.041
  measured) — and its net is −0.026 / −0.024: the gross is positive and *smaller than the cost*. It
  books −$4,797 / −$1,122 at 18.7 / 25.8 trades a week. A positive gross that is half the spread is
  not a book.
* **pass 4's "common stock alone is era-consistently NEGATIVE" is partly a cost-model artifact.**
  On the measured arm, common-only at the signal level is H1 **+0.055** / H2 −0.027 / VAL −0.012,
  and booked it is TRAIN −$9,998 (not −$15,628) with H1 +0.022. It is still not era-consistent and
  still loses money on both splits — the pass-4 verdict survives — but the phrase "negative in all
  three eras" does not, and it should not be repeated.

## 2.3 The cells — the floor ladder and the two populations (as-is arm; `f20` ≡ base, `w20` ≡ `p1`)

| cell | TRAIN /wk · gross · net · grn % · **$** | VAL /wk · gross · net · grn % · **$** | H1 / H2 / VAL | null |
|---|---|---|---|---|
| **F18-base** mixed $20 | 30.6 · −0.039 · −0.107 · 32.1 · **−17,346** | 30.7 · +0.083 · +0.013 · 43.5 · **+893** | −0.077/−0.003/+0.083 | in / in |
| **F18-p1** WRAPPERS $20 | 20.4 · −0.005 · −0.069 · 41.5 · **−7,434** | 25.8 · **+0.135** · +0.067 · 43.5 · **+3,958** | **−0.117**/+0.091/+0.135 | in / **below** |
| **F18-p2** COMMON $20 | 24.1 · −0.053 · −0.122 · 28.3 · −15,628 | 28.3 · −0.055 · −0.127 · 30.4 · −8,234 | −0.060/−0.046/−0.055 | in / in |
| F18-f5 mixed $5 | 36.5 · −0.032 · −0.103 · 26.4 · −19,873 | 34.5 · +0.013 · −0.059 · 39.1 · −4,646 | −0.036/−0.028/+0.013 | in / in |
| F18-f10 mixed $10 | 35.0 · −0.051 · −0.124 · 34.0 · −22,955 | 32.3 · +0.018 · −0.057 · 43.5 · −4,211 | −0.081/−0.023/+0.018 | in / in |
| F18-f30 mixed $30 | 25.5 · −0.006 · −0.075 · 34.0 · −10,128 | 29.3 · +0.028 · −0.045 · **52.2** · −2,996 | −0.070/+0.052/+0.028 | in / **ABOVE** |
| F18-f50 mixed $50 | 17.8 · +0.009 · −0.061 · 39.6 · −5,755 | 23.7 · +0.009 · −0.064 · 47.8 · −3,496 | −0.132/+0.117/+0.009 | in / in |
| F18-w5 WRAP $5 | 24.8 · −0.018 · −0.088 · 34.0 · −11,571 | 31.9 · +0.091 · +0.016 · 52.2 · +1,179 | −0.119/+0.064/+0.091 | in / in |
| F18-w10 WRAP $10 | 23.6 · −0.023 · −0.092 · 35.8 · −11,443 | 30.1 · +0.111 · +0.038 · 47.8 · +2,600 | −0.113/+0.049/+0.111 | in / in |
| **F18-w30** WRAP $30 | 15.7 · +0.021 · −0.046 · 39.6 · −3,862 | 20.7 · +0.126 · +0.056 · **56.5** · **+2,670** | **−0.106**/+0.125/+0.126 | in / in |
| F18-w50 WRAP $50 | 8.5 · −0.012 · −0.083 · 41.5 · −3,720 | 12.0 · +0.100 · +0.028 · 43.5 · +776 | −0.219/+0.114/+0.100 | in / in |
| F18-s08 spread ≤ 8 % of R | 22.7 · −0.074 · −0.114 · 26.4 · −13,723 | 25.9 · +0.047 · +0.005 · 47.8 · +313 | −0.132/−0.020/+0.047 | in / in |
| F18-s25 spread ≤ 25 % | 34.2 · −0.004 · −0.095 · 35.8 · −17,125 | 34.6 · +0.012 · −0.082 · 30.4 · −6,521 | −0.046/+0.037/+0.012 | in / in |
| F18-s40 spread ≤ 40 % | 37.0 · −0.055 · −0.168 · 30.2 · −32,926 | 37.1 · +0.022 · −0.091 · 30.4 · −7,781 | −0.073/−0.038/+0.022 | **ABOVE** / in |
| F18-soff both gates OFF | 37.6 · −0.041 · −0.181 · 26.4 · −36,021 | 38.7 · −0.003 · −0.163 · 26.1 · −14,506 | −0.050/−0.033/−0.003 | in / in |

**The wrappers-only population fails its pre-committed promotion rule.** `F18-p1` is not
same-signed positive across the three eras — **H1-2025 is −0.117** — so the ship-bar treatment was
run as declared and stops there: TRAIN −$7,434, VAL green weeks **43.5 % against a null band of
57.0 [47.8, 65.2]** (below its own null — the pick count would have done better shuffled), TRAIN
clustered t −1.31, VAL +0.98, and its VAL net (+0.067) is under its MDE (0.153). Its VAL weekly
path is 10 green weeks of 23 with one week (+$1,897) carrying 48 % of the total.

**The declared mechanism is not supported either.** The forced-buying story (a 2×/3× wrapper
rebalances into the close, so its own flow buys the break) predicts the edge concentrates in the
higher-leverage names and the late session. The population's own halves say something else: the
wrapper book is negative in H1-2025 **at every floor** and positive in H2-2025 and VAL **at every
floor** — a **regime** signature, not an instrument one, and the same H1-shaped failure that has
defeated every object in ten passes. The best wrapper rung (`F18-w30`, VAL +$2,670 / 56.5 % green)
runs at 15.7 TRAIN trades a week with H1 −0.106 and a VAL green share inside its null.

**The floor is not a lever.** No floor makes the mixed book era-consistent; $30 and $50 trade less
and lose less, which is arithmetic, not edge. **The two ABOVE-null readings on this arm are
`F18-f30` VAL and `F18-s40` TRAIN, and both cells lose money** — F7's lesson, fifth and sixth
appearance.

## 2.4 The spread gates, with the cost measured (the measured arm)

| rung | booked cost R | TRAIN gross · net · **$** | VAL gross · net · **$** | imputed % |
|---|---|---|---|---|
| `frac_r ≤ 0.08` | **+0.037** | −0.060 · −0.097 · −12,801 | +0.092 · **+0.054** · +3,274 | 0 % |
| **`frac_r ≤ 0.15` (shipped)** | **+0.059** | −0.017 · −0.076 · −12,627 | +0.097 · **+0.034** · +2,427 | 0 % |
| `frac_r ≤ 0.25` | +0.079 | −0.030 · −0.109 · −20,170 | +0.031 · −0.059 · −4,663 | 1 % |
| `frac_r ≤ 0.40` | +0.102 | −0.060 · −0.162 · −32,168 | +0.025 · −0.085 · −7,133 | 1 % |
| both gates OFF | **+0.151** | −0.055 · −0.206 · −41,514 | +0.010 · −0.165 · −14,383 | 1 % |

**The honest answer to the sub-question: the gates are NOT removing names whose measured spread is
fine.** With 99.9 % of the $20 gate-free stream's decision quotes now actually measured, the cost of
the rows the gates remove is real and large — going from 15 % to 40 % of R nearly doubles the booked
cost (0.059 → 0.102 R) and the gross does not move to pay for it. The gate chosen from a wrong cost
table happens to sit at the right place for the right reason. Tightening to 8 % buys a cheaper book
(0.037 R) and gives back the same amount in gross; it is the only rung that beats the shipped one on
VAL dollars (+$3,274 vs +$2,427) and it is worse on TRAIN, H1 −0.097, and not era-consistent. **No
rung clears either bar.**

## 2.5 What the dedicated fetch actually changed — the decomposition (`supp5.log`)

Declared in PREREG §4.3, and it matters, because the measured arm's B2 (−$12,627 / +$2,427) looks
like a $4,719 improvement on the reference and **is not a cost effect**:

* pre-book signals **7,027 → 6,552**: 1,391 dropped by measurement, 916 added, 5,636 common;
* **obtainability flips: 910 True→False and 0 False→True.** Measuring the decision quote can only
  ever make a capped-limit fill *less* obtainable (a missing quote defaulted to obtainable). 65.6 %
  of gate-free rows had a decision quote before the fetch; 99.9 % have one now;
* spread-gate flips on the rows both arms priced: 1,123 pass→fail, 951 fail→pass;
* **re-pricing on the 1,566 rows booked by BOTH arms is worth +$921 TRAIN and +$287 VAL** — mean
  cost 0.0667 → 0.0590 R, gross identical to four decimals. Everything else is a different set of
  trades (487 as-is-only rows worth −$6,881 out, 533 measured-only rows worth −$3,084 in).

**The measured cost confirms the 0.061–0.065 R booked-cost finding of `hod_fresh` and moves the book
by under a cent per R.** The reference stays the as-is arm, as declared.

## 2.6 F18-R2 — availability audit

`asset_class` identification: 99.1 % ($5 floor) → 99.5 % ($50), winner-vs-loser missingness gap
0.1–0.3 pp at every floor. **ok** everywhere; no cell voided on coverage in this frame.

---

# F17 — THE HORIZON (8 scored cells + 2 diagnostics)

Carried population: the 686 of 2,328 booked trades that reach 15:55 alive (**TRAIN 27.9 %, VAL
33.0 %**). Exit-reason mix of the booked set: stop 1,162 / eod 686 / target 480. Price-scale rail:
**1 of 686 carried rows dropped** (0.1 %) for a daily bar that does not bracket the intraday entry —
the daily/intraday scale agreement is clean on this book.

| cell | TRAIN gross · net · grn % · **$** · MDD | VAL gross · net · grn % · **$** · MDD | H1 / H2 / VAL | MDE (TR/VAL) |
|---|---|---|---|---|
| **h0** 15:55 flat (shipped) | −0.039 · −0.107 · 32.1 · **−17,346** · −18,929 | +0.083 · +0.013 · 43.5 · **+893** · −2,940 | −0.077/−0.003/+0.083 | 0.088 / 0.132 |
| **h1a** next session's OPEN | **−0.003** · −0.078 · 37.7 · −12,628 · −15,139 | +0.035 · −0.044 · 34.8 · −3,090 · −7,059 | −0.043/+0.036/+0.035 | 0.100 / 0.149 |
| **h1b** = h1a (see note) | identical | identical | identical | identical |
| **h2a** +1 session close | −0.062 · −0.137 · 35.8 · −22,246 · −22,955 | **+0.114** · **+0.035** · 43.5 · **+2,458** · −5,660 | −0.112/−0.014/+0.114 | 0.108 / 0.172 |
| **h2b** +1 session, prior-close stop | −0.026 · −0.101 · 35.8 · −16,362 · −18,432 | +0.058 · −0.021 · 39.1 · −1,450 · −6,801 | −0.073/+0.019/+0.058 | 0.101 / 0.160 |
| **h3a** +2 sessions | −0.058 · −0.133 · 39.6 · −21,557 · −22,117 | +0.067 · −0.012 · 34.8 · −830 · −9,878 | −0.117/−0.001/+0.067 | 0.126 / 0.217 |
| **h3b** +2, prior-close stop | −0.028 · −0.103 · 37.7 · −16,674 · −18,895 | +0.041 · −0.038 · 34.8 · −2,705 · −7,456 | −0.083/+0.025/+0.041 | 0.101 / 0.157 |
| **h5a** +5 sessions | −0.043 · −0.119 · **41.5** · −19,261 · −19,648 | +0.051 · −0.027 · 43.5 · −1,936 · **−12,109** | −0.070/−0.018/+0.051 | **0.164 / 0.292** |
| **h5b** +5, prior-close stop | −0.032 · −0.108 · 37.7 · −17,436 · −19,363 | +0.037 · −0.042 · 34.8 · −2,931 · −7,616 | −0.084/+0.017/+0.037 | 0.100 / 0.156 |

**Note on `h1b`**: the prior-close stop has no session in which to act between the entry day's close
and the next open, so `h1b` is identical to `h1a` **by construction** — 7 distinct books, not 8,
reported as a structural duplicate rather than quietly dropped.

**Charges applied, as declared**: the overnight gap is taken at the OPEN, never netted at the prior
close; the carried exit pays a full half-spread (ratio 1.0) against the shipped `eod` exit's 0.412,
so every carried book is charged **more** per trade than the shipped one (mean cost +0.075 vs
+0.068 R).

## 3.1 The overnight risk this buys

| cell | carried | worst single trade | best | share ending worse than −1 R (the intraday stop) |
|---|---|---|---|---|
| h1a / h1b | 685 | **−9.07 R** | +6.04 R | **6.6 %** |
| h2a | 685 | −6.00 R | +8.15 R | **16.8 %** |
| h3a | 685 | −9.20 R | +12.12 R | **25.3 %** |
| h5a | 685 | **−14.87 R** | +33.05 R | **33.3 %** |
| h2b / h3b / h5b (prior-close stop) | 685 | −9.07 R | +7.85 R | 6.6–6.9 % |

The prior-close stop does exactly what it should — it caps the tail at the first gap — and it takes
the upside with it: `h5b` gross +0.037 VAL vs `h5a` +0.051, and neither is positive net. **A book
whose entire risk model is a 1 R stop cannot hold a third of its trades past a −1 R outcome and call
it the same book.** At $100 risk per trade `h5a`'s VAL MDD is −$12,109 against the shipped book's
−$2,940.

## 3.2 Per unit of time

Carried trades are 28–33 % of the book, so the whole-book per-day denominator is small and the
honest reading is simply that stretching the horizon does not raise net R at all while raising the
MDE faster than any point estimate: `h5a` net −0.119 / −0.027 against an MDE of **0.164 / 0.292 R**.
Five days of overnight risk buys a test that can no longer see the effect it is looking for.

## 3.3 F17-D1 — the two-cohort diagnostic

| cell | TRAIN mover (≥10 % day range) gross | TRAIN quiet | VAL mover | VAL quiet |
|---|---|---|---|---|
| h0 (shipped) | **+0.262** (n 1,033) | −0.567 (n 589) | **+0.360** (n 469) | −0.466 (n 237) |
| h1a next open | +0.293 | −0.520 | +0.323 | −0.536 |
| h2a +1 session | +0.238 | −0.587 | +0.404 | −0.460 |
| h3a +2 sessions | +0.240 | −0.578 | +0.372 | −0.537 |
| h5a +5 sessions | +0.262 | −0.580 | +0.250 | −0.342 |

**The continuation claim is not supported by the horizon.** If the HOD break were a 1–5 day momentum
signal on the days that keep running, the mover cohort's gross would *rise* with the horizon. It
does not: flat at +0.24…+0.29 R on TRAIN and drifting *down* on VAL from +0.36 to +0.25 as the
horizon lengthens. The +0.8 R mover/quiet separation is the same one `hod_frames` §2.3 identified as
**downstream of the trade's own outcome** — the day range that defines the cohort is unknown at the
signal — and holding longer does not convert it into anything causal.

## 3.4 F17-D2 — the wrapper decay, measured

38,926 wrapper-days over 119 wrappers; wrapper daily log return minus stated leverage × the
underlying's:

| leverage | wrapper-days | mean daily drag |
|---|---|---|
| −2.0× | 5,174 | **−0.4280 %/day** |
| −1.5× | 430 | +0.1008 |
| −1.0× | 860 | +0.0081 |
| **+2.0×** | **32,032** | **−0.2331 %/day** |
| +3.0× | 430 | +0.4618 |
| **all** | 38,926 | **−0.2423 %/day** (median −0.1821, sd 8.547) |

The median R of a booked wrapper is **2.91 % of price**, so the measured drag is **0.083 R per
calendar day held** — larger than the entire booked-set cost of a trade (0.059–0.065 R) for every
day it is held. Every multi-day cell above is priced from the **wrapper's own daily bars**, so the
drag is already inside those numbers; it is measured here so that "the decay is charged" is a
measurement and not an assertion, and so that its size is on the record: **a +2× wrapper held five
sessions pays about 0.4 R in rebalancing drag alone**, which is why a 38 %-wrapper book cannot be a
multi-day book.

**Nulls: 18 of 18 INSIDE. F17 verdict: DEAD**, and the intraday line closes on a second, independent
axis: the signal is worth ≈ 0 gross at every horizon from 15:55 to +5 sessions, on the identical
population.

---

## BOTH BARS, the nulls, the MDE, the multiplicity

**Claim bar G1 — 0 of 31.** No cell has TRAIN net R > 0 with iid *and* clustered t ≥ 2 at
≥ 10 trades/week — in this pass **no cell has a positive TRAIN net at all**. G2 was never evaluated;
**TEST was never opened.**

**Live-exploration bar — 0 cells.** No cell is positive in dollars on both splits. The two cells
positive on VAL dollars with a ≥ 50 % VAL green-week share (`F18-w30`, `F18-f30`) are TRAIN-negative
and H1-negative.

**Nulls — 68 cell × split bands on the reference arm (20 F16 + 30 F18 as-is + 18 F17): 64 inside,
2 below, 2 ABOVE.** Both ABOVE readings lose money. On the measured sensitivity arm: 26 inside,
3 ABOVE, 1 below — the three ABOVE are the three loosest spread rungs, all of which lose money on
both splits.

**MDE per frame (80 % power, per trade, net, against the 0.061–0.065 R break-even):**
F16 **0.074–0.240 / 0.104–0.270 R** (the de-dup cells 0.088 / 0.132, the concentrated mirror
0.209 / 0.242); F18 **0.079–0.180 / 0.121–0.222 R** (wrappers-only 0.116 / 0.153, wrappers at $50
0.180 / 0.220); F17 **0.088–0.164 / 0.131–0.292 R**, rising monotonically with the horizon. **In
every frame the MDE is larger than every point estimate in it.** The day-level MDE from pass 4
($68 / $94 a day on the base) is unchanged.

**Multiplicity.** 31 declared decision cells (9 F16 + 14 F18 + 8 F17), realised as 10 + 14 + 8
scored book rows (F16's 10th row is the B2 reference; F17's `h1b` is a structural duplicate of
`h1a`); plus 5 reproduction rows, 1 parity assertion, 1 concentration diagnostic, 2 structural
reports (F18-R1, F18-R2), 2 F17 diagnostics and 1 supplementary decomposition (`supp5`) that carry
no decision. The measured-cost arm re-scores the same 14 F18 cells and is a **sensitivity arm, not
14 new cells**. **Programme cumulative: 958 + 31 = 989.** Expected largest |t| under a pure null
over 31 × 2 ≈ 2.9; the largest favourable TRAIN clustered t on a causal cell is **−0.66** (none).

## The adequacy review, answered in writing (RUNBOOK step 10)

* **Did we test what the book actually IS?** Yes, and more literally than any prior pass: the live
  config was read knob by knob (§2.1) and the study population *is* the shipped one.
* **Is the cost and fill model right for its venue?** Now measured, not modelled: 7,605 fresh SIP
  quote-minutes take the $20 gate-free stream's decision-quote coverage from 65.6 % to 99.9 %, and
  the cost re-pricing is worth **+0.008 R** (§2.5). The fill model is unchanged; every fill is the
  open of the minute after the break bar.
* **Does any caveat in our own report explain the headline?** The only headline that moved is the
  measured arm's B2, and §2.5 shows it is composition, not cost. Pass 4's "common stock is
  era-consistently negative" is partly a cost-model artifact and is corrected in §2.2.
* **What is the MDE?** Stated per frame above; it exceeds every point estimate in this pass.

## Known deviations, stated rather than buried

1. **`h1b` is identical to `h1a` by construction** — 7 distinct horizon books, not 8.
2. **The measured-cost arm is not the reference.** It changes membership (§2.5) and is the declared
   sensitivity arm; the as-is arm reproduces the reference to the cent.
3. **The F17 carried book re-uses the shipped book's selection.** A book built to be held overnight
   would plausibly select differently; that is a NEW population (a different admission), not a new
   cell, and it is named as the frame's live remainder rather than scored.
4. **`hod_frames3/nbbo3.csv` is still not merged** (pass 4 §0's declared population difference);
   `nbbo5.csv` is merged only in the declared measured arm.
5. **F16's venue diagnostic is a 5-valued field** and its 71.5 / 78.1 % "concentration" is
   arithmetic, reported only so it is not mistaken for a finding.

## VERDICT — **STAY DRY.** No `HodBreakParams` change, no `run_book` change, no universe change, no engine change.

`config.yaml hod_break` stays exactly as the owner set it (`enabled: true, dry_run: true`);
`trading.enabled` and `orb.yaml` untouched. There is no SHIP-TO-DRY diff. For the record, the diffs
that would have been written: **F16** → an `underlying_anchor` de-dup in the engine's per-minute
candidate admission (`trading/hod_break_engine.py`) plus the same key in
`trading/hod_break.py::run_book`; **F18** → `hod_break.min_price` and/or an `asset_class` universe
filter in the engine's candidate screen; **F17** → the bracket legs and the 15:55 flat in
`hod_break_engine` (carry instead of close). None is built and, on this evidence, none should be.

**The next three frames are F19, F20 and F21, appended to `FRAMES.md`.**
