# hod_frames2 — PREREGISTRATION (written and committed BEFORE any cell was scored)

Pass 2 of the frame programme on HOD-break. Three frames, all declared here:
**F5 the RETEST book**, **F6 ABSORPTION at the level**, **F9 signal-minute cohort fields**.
`hod_frames/FRAMES.md` queued all three; `hod_frames/REPORT.md` §2.3 wrote the mandatory
downstream gate that F9 must pass.

Rails carried verbatim from `RUNBOOK.md` and the previous four PREREGs:
splits TRAIN = 2025-01-02..2025-12-31, VAL = 2026-01-01..2026-05-31, **TEST = 2026-06-01..2026-09-11
SEALED** (`FREEZE.md`; no script here reads a TEST-dated bar without `--test`), early closes removed,
test tickers removed, `daily_bars` membership enforced upstream, measured-NBBO cost with the declared
price-band × hour-band imputation, the engine's capped-limit fill with the obtainability test, book =
12/day × 4 concurrent at $100 risk, count-matched permutation null (2,000 draws) on green weeks,
day-clustered t beside the iid t, both TRAIN halves reported, MDE stated on every rejection.

---

## 0. Reproduction gate (run first, before any cell)

| id | what | reference |
|---|---|---|
| R1 | `B2` shipped book, TRAIN | 1,622 · 30.6/wk · −0.039 gross · −0.107 net · 32.1 % green · **−$17,346** |
| R2 | `B2` shipped book, VAL | 706 · 30.7/wk · +0.083 · +0.013 · 43.5 % green · **+$893** |
| R3 | **independent rebuild**: this pass's own bar walk (`walk2.py`, a NEW all-breaks loop) must reproduce `pop.csv`'s first-qualifying B2 rows on (entry minute, stop, rr) exactly — max \|Δrr\| = 0 |

A mismatch on R1/R2 stops the pass. R3 is the independent-rebuild requirement of
`feedback_independent_check_before_claims`.

## 1. The bar pass — what is emitted and what is causal

`walk2.py` re-walks every symbol-day that carries at least one qualifying break in TRAIN+VAL
(98,231 symbol-days over 344 sessions, from `hod_filter_stack/pop.csv`), and emits **one row per
QUALIFYING break** — not just the first. A break bar `i` qualifies exactly as `pass2.py` defines the
B2 combo (`tag='n'`, `band=0`): `h[i] >= hod[i-1]`, `level >= open × 1.05`, `level >= $1`, the
plain last-5-bar low is computable, `rv_profile >= 1`, and the next bar exists with
`m[i+1] <= 931`. Stop = the plain last-5-bar low (the B2 stop). Exit = the shipped walk
(priority eod → stop → target `entry + 2R`), starting at the bar AFTER the entry bar. **The exit is
not touched in this pass** (`hod_bleed`: exits are exhausted).

Every field below is computed from bars `0..i` — at or before the close of the break bar, which is
one minute before the decision is acted on at `m[i+1]`'s open. Availability (coverage, and
missingness on winners vs losers) is audited for each and a >5 pp win/loss missingness gap drops
the field.

**F5 fields** (per break, referring to the IMMEDIATELY PRECEDING qualifying break `q` on the same
symbol-day): `n_prior` (count of qualifying breaks before `i`), `prev_level`, `prev_why`,
`prev_exit_i`, and the two declared failure markers —
* **(i) `prev_stopped`** — the prior break's trade exited on `stop` **and** its exit bar index
  `<= i` (the stop is in the past at our decision instant; a still-open prior trade is not a failure
  and could not be known).
* **(ii) `prev_back_N`**, N ∈ {5, 15} — there exists a bar `k` with `q < k <= min(q+N, i)` whose
  CLOSE is below `prev_level` (price closed back below the level it broke, inside N bars).

**F6 fields** (the shelf — absorption at the level), from bars strictly before the break bar:
`shelf_vol` = Σ `v[j]`, `j < i`, over bars whose `[low, high]` range intersects
`[level × 0.995, level × 1.005]`; `shelf_bars` = the count of those bars;
`shelf_share` = `shelf_vol / adv20 × 100` (% of 20-day average daily SHARE volume).
`hod_age_bars` (already in `pop.csv`, bars since the running HOD last made a new high) is the age arm.

**F9 fields**:
* **(a) `dollar_frac`** = cumulative $ volume 09:30→break bar ÷ ADV$, where ADV$ = the mean of
  `close × volume` over the symbol's **20 strictly-prior sessions** in `research/bf_zero/universe.csv`.
* **(b) `exp5_n`** = the number of completed 5-minute blocks strictly before the break bar whose high
  exceeded the running maximum of all preceding blocks' highs (new 5-minute highs made).
* **(c) `add30_ratio`** = (session range at the break bar − session range 30 minutes earlier) ÷ the
  session range of the first 30 minutes, all in % of the open. NaN before 10:30.
* **(d) `rng_own`** = session range so far (% of open) ÷ the symbol's **20-prior-session median daily
  range %** (`(high − low) / open × 100` from `universe.csv`). **Declared deviation**: the frame text
  asked for the symbol's own median **first-hour** range; computing that needs 20 prior sessions of
  1-min bars per candidate symbol (~2M symbol-days of bars) which this node cannot afford in one
  pass. The daily-range normaliser is the same idea — a stock-specific scale for "how big is this
  move for THIS stock" — and the substitution is declared here, before scoring, not after.

Diagnostic labels (NOT inputs to any rule): `rng_day` = the RTH session range % of open,
`rng_sig` = the session range % of open at the break bar, `rng_after` = `rng_day − rng_sig`.

## 2. The mandatory downstream gate (F9, and reported for F6)

`hod_frames/REPORT.md` §2.3: the `>=10 %-range day` cohort is substantially the trade's own outcome —
inside it, days **already** wide at the mark read +0.118 R at 40.9 % WR and days whose range
**arrived after** the mark read +0.882 R at 82.0 % WR.

For each candidate field, BEFORE it is scored as a rule, on the B2 pre-book signal population:

| reported | meaning |
|---|---|
| ΔP(EOD range >= 10 %) | the cohort-membership rank: top rung minus the rest |
| Δ mean `rng_sig` | the **already-there** channel |
| Δ mean `rng_after` | the **arrived-after** channel |
| gross R inside the top rung, split already-wide vs arrived-after | §2.3's own split |

**Pre-committed exclusion rule**: a field is **EXCLUDED and listed as excluded** — not scored as a
rule — if its membership lift runs ONLY through the already-there channel, i.e.
`Δ mean rng_after <= 0` while `Δ mean rng_sig > 0`. That is `dist_open_pct` wearing a hat (pass 1:
ranks membership 40.5 % → 85.0 %, flat on gross) and the programme has already paid for it once.
Both ranks — membership AND gross — are reported for every candidate, and **gross is the target**.

## 3. THE CELLS — 26 declared, scored exactly as written

### F5 — the retest book (7 cells)
Base B2 (`tag='n'`, rv >= 1), shipped stop, shipped exit, shipped cost/price gates, 12/4 book.
Scan rule named: **KEEP-SCANNING** — the book admits the declared break, not the first one.

| id | admission |
|---|---|
| F5-0 | first break only (= B2, the reference row) |
| F5-a | the SECOND qualifying break (`n_prior == 1`) whose predecessor **stopped** (failure (i)) |
| F5-b | the SECOND qualifying break whose predecessor **closed back below the level within 5 bars** |
| F5-c | the SECOND qualifying break whose predecessor closed back below the level within **15 bars** |
| F5-d | the SECOND qualifying break under **(i) OR (ii,15)** |
| F5-e | ANY re-break (`n_prior >= 1`), no failure condition — the control |
| F5-f | the THIRD-or-later qualifying break (`n_prior >= 2`) — the control |

### F6 — absorption at the level (9 cells)
Base B2 first-break signal set (the shipped book), plus the declared admission filter.

| id | admission |
|---|---|
| F6-a…d | `shelf_share >= 2 / 5 / 10 / 20 %` of ADV |
| F6-e | `shelf_share >= 5 %` AND `hod_age_bars >= 20` |
| F6-f | `hod_age_bars >= 20` alone (the age arm, to separate absorption from age) |
| F6-g…i | `shelf_bars >= 5 / 10 / 20` (DURATION, independent of volume) |

### F9 — signal-minute cohort fields (10 cells)
Base B2 first-break signal set. Any field excluded by §2 is **not** scored and its cells are
reported as excluded.

| id | admission |
|---|---|
| F9-a1..a3 | `dollar_frac >= 10 / 25 / 50 %` |
| F9-b1..b2 | `exp5_n >= 3 / 6` |
| F9-c1..c2 | `add30_ratio >= 1.0 / 2.0` |
| F9-d1..d3 | `rng_own >= 0.5 / 1.0 / 1.5` |

## 4. What is reported per cell (no metric added after the fact)

n, trades/week, gross R, booked cost (= gross − net, measured per cell, never the retired 0.2151
constant — the booked-set cost is 0.061 TRAIN / 0.065 VAL), net R, % green weeks, longest red streak,
worst week $, total $ at $100 risk, MDD $, ex-top-5 % net, imputed-cost share, iid t, **day-clustered
t**, both TRAIN halves + VAL with the same-signed-positive flag, the count-matched null band, and —
for F5 — the downstream check `corr(gross R, rng_after)` exactly as pass 1 ran it.

## 5. The bars

* **Claim bar G1**: TRAIN net R > 0 with iid t >= 2.0 AND day-clustered t >= 2.0 at >= 10 trades/week.
  G2 (VAL sign + >= 55 % green weeks) is evaluated only for a cell that clears G1. TEST opens only
  behind a committed recommendation.
* **Live-exploration bar**: positive weekly $ AND >= 50 % green weeks on BOTH splits at >= 10
  trades/week, clustered t >= 2 on TRAIN, halves same-signed positive.
* A cell that clears the live-exploration bar → **SHIP-TO-DRY** with the exact `HodBreakParams` diff.
  F5 would be an admission/scan-rule change in `trading/hod_break.py::detect` (today's rule is
  first-break-only via `stale_break`); F6/F9 would be a new causal admission field on the same rule.
* If none clears → **STAY DRY**, the MDE, and the next three frames appended to `FRAMES.md` with
  mechanisms.

## 6. Cell count / multiplicity

Programme cumulative through `hod_frames`: **870**. This pass declares **26** decision cells
(7 + 9 + 10) → **896**. Diagnostics (the downstream table, availability, the reproduction rows, the
correlation checks) carry no decision and are not counted as cells; they are listed in the report.

## 7. Node rails

One python process, `nice -n 10`, `ulimit -v 3000000`, checkpointed per day. `cache.db`,
`bars_sip.db` opened **read-only**. No config, `orb.yaml`, systemd unit, cron, order or cache is
written. The dry run is not touched.
