# PREREG — ORB pipeline mirrored to the SHORT side

Written 2026-09-20, **before any scoring run**. Cells **1,271 (Stage A)** and **1,272 (Stage B)** on the
programme counter. Every additional cell looked at is counted in REPORT.md §Multiplicity.

## 0. Question

Does the ORB machinery, mirrored, make money on 09:35 gap-DOWN names shorting a break BELOW the opening-range
low, **net of measured cost**, at **≥ 3 fills/week**?

## 1. Population (causal, point-in-time)

Source: `data/cache.db` (read-only) — `daily_bars` + `intraday_bars_1min`. Mirror of
`study_orb_broad.load_broad_universe`:

| knob | long study | this study |
|---|---|---|
| gap (today.open vs prev close) | **>= +5.0 %** | **<= −5.0 %** (mirrored) |
| prev-day volume | >= 500,000 | >= 500,000 |
| today's open | $3 – $30 | **$5 – $30** (floor lifted to $5 for borrow realism) |
| 1-min bars cached | required | required |
| RTH volume by 09:35 | >= 15,000 | >= 15,000 (range-computability proxy) |

Standing exclusions: `^Z[A-Z]ZZT$` test tickers; any symbol absent from `daily_bars`.
**Wrappers (2x/inverse single-stock ETFs): INCLUDED**, flagged via `trading/orb_asset_class`, and the book is
reported **with and without** them.

Every field in the membership rule is known before 09:35 on the day. Counted before this file was written:
**3,466 candidate symbol-days in 2025, 3,229 in 2026** (Jan–Sep).

## 2. Signal / entry / exit — the mirror

- **Opening range** = the five 1-min bars 09:30–09:34 ET. `range_high`, `range_low`.
- **Order** (resting from 09:35, like live's pre-placed stop-limit): stop-limit **SELL SHORT** at
  `L = range_low × (1 − 30 bps)`.
- **Trigger bar** = first 1-min bar with ts >= 09:35 whose `low <= L`. *(Deviation, declared here before
  scoring: the task prose said "low < range_low". A resting stop-limit at `L` triggers at `L`, not at
  `range_low`; using `range_low` would count triggers the live order never saw. `low <= L` is the obtainable
  form and is what is scored.)*
- **Expiry**: no trigger by 10:35 ET (60 min) → the candidate is an **entered-inclusive no-fill row**: it ranks,
  can win a slot, and books $0.
- **Fill** = the **next** bar's OPEN, and only if `next_open >= L`. A gap-through **below** `L` is **NO fill**
  (the limit protects us; a touch is never a fill). No-fill rows are entered-inclusive as above.
- **SSR (Reg SHO 201)**: flagged when, at the trigger bar's close, the name is **>= 10 % below prior close**.
  Under SSR a short executes only on an uptick above the NBB; modelled as the pre-committed proxy —
  **fill only if `next_open > next_bar.low`**. The book is reported a second time with **all SSR names
  excluded**, and the SSR share is reported.
- **Stop** = `range_high` (cover). `R = range_high − entry_fill` (per share).
- **Static lock, mirrored**: when a bar's `low <= entry − 1.75 R` the lock arms and the stop moves to
  `entry + 0.5 R`… i.e. to `entry − 0.5 R` **as a cover level locking +0.5 R** (`stop = min(stop, entry − 0.5R)`).
- **Stop fill**: a bar whose `high >= stop` covers at `max(stop, bar.open)` (gap-through against us is charged).
- **Force cover** 15:45 ET at that bar's close. No fixed target. Touchgo Rules M/D are **not** mirrored (they were
  fit on the long tape) — declared, not tested here.
- **Borrow**: assumed **ETB** (easy-to-borrow) for price >= $5 and ADV20 >= 500K; intraday only, so **no borrow
  fee and no overnight rebate is charged**. This is an assumption, stated as a caveat, not a measurement.

## 3. Cost — measured, never a band

Charged per share on BOTH legs as **half the full NBBO spread** at the leg's minute (short sells into the bid,
covers at the ask):

1. `research/mature_method/frames16/nbbo16.csv` where it covers the exact `(day, symbol, minute)`;
2. else **measured directly** (Alpaca SIP consolidated quotes, mean (ask−bid) over the one-minute window — the
   frames16 `nbbo.py` convention) for a **random 500-leg sample** of this population, stratified by split;
3. else **imputed** from a minute-of-day median half-spread table built **from this population's own measured
   sample** (frames14's table is a HOD/BF population and is not transferred).

**The imputed share is reported.** EQUS.MINI quote schemas are never used.

## 4. Splits

- **TRAIN** = 2025-01-01 … 2025-12-31 (H1 and H2 reported separately — same-signed is a pass-bar item)
- **VAL** = 2026-01-01 … 2026-05-31 (the pass bar is evaluated **here**)
- **TEST** = 2026-06-01 onward — **SEALED. Never read in this study.**

## 5. Stage A — is the RAW short breakdown anything?

Every triggered+filled short in the population, **no selection**, vs a **matched control** (frames13 F41
construction): for the same day, gap-down names in the same universe/band that did **NOT** trigger by 10:35,
entered short at the 09:36 open, same stop (`range_high`), same static lock, same 15:45 cover, same cost model.
Reported on TRAIN and VAL. The control isolates "what a gap-down name at 09:35 does under this exit spec" from
"what the break adds".

## 6. Stage B — selection

On the Stage-A signal set:

- **Composite** = mean of signed z-scores of the 7 shipped features (`gap_pct`, `range_total_volume`,
  `range_avg_bar_range_pct`, `range_size_pct`, `price_vs_20d_high_pct`, `prev_day_close_position`,
  `range_close_position`). **z mean/std refit on TRAIN gap-down candidates.**
- **Sign rule, TRAIN ONLY, pre-committed**: `sign(f) = +1` if the TRAIN mean net-R of the **top** tercile of `f`
  exceeds that of the **bottom** tercile, else `−1`. No other sign source; no per-feature tuning.
- **Quintile cutoffs** = the TRAIN quintiles of the composite. **Q1 filter**: drop the bottom quintile.
- **Vetoes, thresholds TRANSFERRED from the shipped long book (no refit — deliberately conservative)**:
  PDR veto `prev_day_range_pct <= 11.0` → veto; range-size veto `range_size_pct <= 2.221` → veto;
  G1 veto: keep iff `return_volatility_20d >= 7.106` **and** `prev_day_range_pct >= 9.226`, fail-open on
  NaN/missing rv20, `rv20 == 0.0` fails open (the shipped 9/13 form).
- All vetoes are **POST-ranking, slot consumed, NO REFILL**.
- **8 slots/day**, ranking Q4-preferred then composite DESC (the shipped order). Family/super-group and anchor
  dedup are **not** applied (declared).
- **Evaluated on VAL.**

## 7. Sizing

Risk-parity, mirror of the shipped B+ maths: `notional = min(risk / (stop_pct/100), account / N)`,
`shares = floor(notional / entry)`, `min_stop_pct = 1.0`. **$10K stage**: account 10,000, risk 375, N = 8 →
per-position cap $1,250. **$50K**: account 50,000, risk 1,875, N = 8 → cap $6,250. Adaptive quintile mults are
**not** applied (they are long-book fits; uniform 1.0, as B+ ships).

## 8. Pass bar — VAL, every line must hold

1. net **>= +0.15 R / trade**
2. **day-clustered t >= 2.0**
3. **ex-top-5 % net >= 0**
4. TRAIN H1 and H2 **same-signed**
5. **>= 3 fills / week**
6. green-week share **above** the count-matched null
7. **Stage B − Stage A control >= +0.10 R**

Anything short of all seven = **NO SHIP**. Power (MDE at 80 % on the realised day-clustered SE) is reported
beside every null. "No edge exists" is never the conclusion.

## 9. Freeze

`FREEZE.md` carries the git hash of every script before the first scoring run.

## Cell C — pre-registered before scoring

**Cell count: 1,273.** Registered before any cell-C number was computed; committed alone, ahead of the score.

**Motivation.** Under Stage A/B the entry was a mirrored no-chase stop-**limit** at
`L = range_low × (1 − 30 bps)` that fills only if the next bar's open is `>= L`. 62.9 % of triggers were
`gap_through` no-fills — the rule systematically **rejects the fast breakdowns and keeps the stalled ones**.
Cell C tests whether the book is different when the fast breakdowns are allowed in.

**The rule — everything identical to Stage B except the entry:**

- Trigger unchanged: first post-09:35 bar with `low <= L`, within the 60-min window (`m <= 635`).
- Entry = a **stop order**: fill at the **NEXT bar's OPEN** (obtainable — it is a price the market printed and
  reachable by a resting stop), **capped** at `cap = range_low × (1 − 100 bps)`. If the next open is **below**
  the cap (a gap of more than 1 % through the level), it is a **skip, not a chase** → no fill.
- **SSR names**: fill only on the uptick proxy as before — the fill bar must have `open > that bar's low`,
  else no fill.
- **Stop = `range_high`, unchanged.** R = `range_high − fill`, so **R grows when the fill is lower**; every
  R-denominated number is reported on the **actual fill**.
- Static lock (1.75 R arm → 0.5 R stop), 15:45 cover, universe, features, controls, Stage-B composite /
  sign rule / quintiles / vetoes / 8 slots / sizing: **all unchanged**.
- Cost as before: measured NBBO half-spread per leg where available, minute-of-day median imputation
  otherwise; the **imputed share is reported**.

**Pass bar — identical to Stage B, evaluated on VAL, every line must hold:**

1. VAL net **>= +0.15 R / trade**
2. VAL **day-clustered t >= 2.0**
3. VAL **ex-top-5 % net >= 0**
4. TRAIN H1 and TRAIN H2 **same-signed**
5. **>= 3 fills / week**
6. green-week share **above** the count-matched null
7. **Stage B − Stage A control >= +0.10 R**

Anything short of all seven = **NO SHIP**. TEST (>= 2026-06-01) stays **SEALED**.
