# PREREG — hod_frames6 · F20 the PLACEBO · F19 era comparability · F21 survivorship

**Committed BEFORE any cell in this pass was scored.** Queue and specification: `hod_frames/FRAMES.md`
(the pass-5 queue). Run order is **F20 → F19 → F21**, because F20 decides what "gross ≈ 0" means and
every one of the 989 prior cells lacked its control.

These three frames test **the EVALUATION**, not a rule. No frozen trade-level cell is declared
(pre-refuted by `hod_frames3` §F12: trade-level frozen rules on this book are noise). Nothing here
can produce a SHIP verdict; the deliverables are three numbers and their interpretation.

---

## 0. Population, reproduction gate, rails

**Population object** — unchanged from passes 4 and 5: `hod_frames4.common4.load_breaks4()` over
`hod_frames2/breaks2.csv`, membership = minus early-close days, minus test tickers, intersected with
`select distinct symbol from daily_bars` (**this last intersection is exactly what F21 puts on
trial**), NBBO from `research/bf_zero/causal_filter/nbbo.csv` only (the pass-3/pass-5 dedicated
fetches are NOT merged — that is what reproduces `B2` exactly).

**Reference book `B2`** — `sigset5()` defaults (next_open >= $20, spread <= 100 bps, spread <= 15 % of
R, `r_pct_n >= 1`, `fill_capped == 1`, obtainable, entry_m <= 841) + `book_ranked(nday=12, nconc=4)`.

**R1/R2 reproduction gate (must MATCH before any cell is scored):**

| id | reference |
|---|---|
| R1 | `B2` TRAIN — 1,622 trades · 30.6/wk · gross −0.039 · net −0.107 · 32.1 % green · **−$17,346** |
| R2 | `B2` VAL — 706 · 30.7 · +0.083 · +0.013 · 43.5 % · **+$893** |

**R3 independent-rebuild rail (new object this pass).** The placebo simulator is a *second* exit walk.
It must reproduce the booked trades' own `rr` when handed the booked trades' own break minute and
stop: `max |Δrr| < 1e-9` over all 2,328 booked rows, asserted in code — the run aborts otherwise.
This is the only way a placebo number can be trusted: the same walk must price the real trade.

**Rails, every cell:** both TRAIN halves (H1 < 2025-07-01, H2 >= 2025-07-01) printed beside VAL;
day-clustered t beside iid t; count-matched permutation null (2,000 draws) wherever a *rule* is
scored; booked measured/imputed cost reported per cell; availability audit on every new field; MDE at
80 % power on every rejection; TEST never opened (`FREEZE.md`). Stores opened **read-only**. One
python process, `nice -n 10`, `ulimit -v 3000000`. No config / `orb.yaml` / order / service / cron is
touched.

**Cell count.** Programme stands at 989 after pass 5. This pass declares **18** cells (F20 6 + 2
diagnostics, F19 8, F21 4 minus overlap — the exact roster is below); the closing count is stated in
REPORT §0 and appended to `FRAMES.md`.

---

# F20 — THE PLACEBO  (6 scored cells + 2 diagnostics)

**Question.** The base book's gross is −0.002 / +0.015 R. Is that (i) a real break effect eaten by
cost, (ii) a detector that adds nothing to a population whose unconditional forward return is also
≈ 0, or (iii) a detector that is *anti-selected* — buying the top?

**Construction — the common bracket.** For a control mark at bar index `j` on a symbol-day:

* entry price `E = o[j+1]` (the open of the next minute — the engine's only obtainable fill);
* stop `S = E × (1 − r_pct/100)` where `r_pct` is the **matched booked trade's own** `r_pct_n`
  (the SAME stop distance as a % of price — the declared way to hold R geometry fixed);
* target `E + 2 × (E − S)`;
* exit walk = `walk2.vwalk` semantics, started at bar `j+2`, priority EOD(m >= 955) → stop → target,
  stop filled at `min(S, o[k]) × (1 − 0.001)`, target filled at the target, EOD at that bar's open;
* gross `R = (exit_px − E) / (E − S)`.

Cost is **not** applied: the frame's estimand is gross (net would confound the comparison with the
control's own spread, which is a different instrument's).

**The three controls, one per booked trade of `B2` (2,328 trades):**

| id | control | draw |
|---|---|---|
| **F20-a** | a random minute in 09:37–14:00 on the **same symbol-day** at which the symbol did **NOT** make a new high of day (`h[j] < max(h[0..j-1])`), and which is not the trade's own break minute | 200 |
| **F20-b** | the **same minute** as the trade, on a **matched symbol that produced no qualifying break that day** — matched on the same session, prior close (log distance), ADV20 (log distance) and wrapper-vs-common (`orb_asset_class`); pool = the 25 nearest matches | 200 |
| **F20-c** | the **same symbol**, mark placed **15 minutes before** the trade's own break bar (`j = break_m − 15`) | deterministic |

Each draw builds one complete control book (one control per booked trade) and its mean gross R is
recorded; the 200 means are the **control distribution**, and the break book's own mean gross R is
located in it as a percentile, per split. F20-c is deterministic, so its distribution is a 200-draw
**day-clustered bootstrap** of the control book's own mean (resample trading days with replacement) —
declared here so the asymmetry is on the record.

Inference beside the percentile: the **paired** difference (break gross − control gross, same trade)
with a day-clustered SE and t, per split and per TRAIN half.

**Coverage rule (availability rail).** A booked trade that cannot be matched in an arm (no eligible
non-break minute; no matched symbol with bars; `break_m − 15` before the first tradeable bar) is
dropped from **that arm only** and the drop share is reported. If an arm loses > 20 % of the booked
trades it is reported as a **diagnostic, not a cell**.

**Diagnostics (reported, never scored as a rule):**
* **F20-d** — *declared substitution, written before any cell was scored*: the set this diagnostic
  named ("days the book never traded") is **EMPTY** — `hod_frames4` F14 established 0 no-trade days
  over all 344 TRAIN+VAL sessions. The diagnostic's PURPOSE (bound the universe's unconditional
  intraday drift under the same bracket geometry) is served instead by pricing the **matched
  non-signal symbols at RANDOM eligible non-break minutes** — no break anywhere in the construction,
  neither in the symbol nor in the minute. 5 minutes per (trade, control) pair, seed 20260920.
* **F20-e** — the win-rate and the exit-reason mix of each arm beside the break's, so a difference in
  *mean* can be attributed to frequency of stops vs targets rather than asserted.

**THE PRE-COMMITTED READING RULE (fixed before any number is seen).** Let `p` be the break book's
percentile in the control distribution and `t_c` the day-clustered paired t.

* **BETTER** — `p >= 95` in **both** splits **and** `t_c > +2` in at least one. Reading: a real
  break effect that cost eats. *Consequence: the cost frames matter after all and the ledger's
  "no cost frame can rescue this book" line must be re-opened.*
* **WORSE** — `p <= 5` in **both** splits **and** `t_c < −2` in at least one. Reading: the break is
  anti-selected; the detector buys the top. *Consequence: the sign of the whole programme is wrong
  and the short frame F2 must be re-read on the placebo-adjusted baseline.*
* **NO DIFFERENT** — anything else. Reading: **a dead detector on a live universe** — the universe
  carries whatever there is and the break adds nothing. *Consequence: this is the single most
  decision-relevant number in the programme and it is stated at the top of `REPORT.md`.*

The reading is taken on **F20-a** (the tightest control: same symbol, same day, same R geometry).
F20-b and F20-c are corroboration; a disagreement between arms is reported as a disagreement and the
frame's verdict is then **NO DIFFERENT** by construction (no arm may be selected after the fact).

---

# F19 — ERA COMPARABILITY  (8 cells)

**Question.** Every object in 989 cells dies on H1-2025. 38.6 % of this pre-book set is leveraged
wrappers, and wrappers were still being listed through 2025–26. Is "H1-negative" **the edge failing**
or **the universe differing**?

**Point-in-time listing source.** `research/scripts/pit_listings.PitListings` over
`data/research/databento/pit_definition/def_YYYYMM.parquet` (coverage 2024-07 → 2026-09, so the whole
TRAIN+VAL window is inside the bought window; nothing in this pass needs the XNAS.ITCH 2018–2024
fallback, and that is stated rather than used). A symbol is **listed in an era** iff it carries a
definition record in **every** month the era spans that the symbol's own signals fall in — operational
rule: listed in era `e` iff `was_listed(symbol, d)` is true for the first trading day of each month of
`e`. Symbols outside the bought window, or absent from the definition feed entirely, are reported as
**UNCHECKABLE** and counted, never silently dropped.

**Sets.** `L_H1`, `L_H2`, `L_VAL` = symbols listed throughout each era. `INTERSECT = L_H1 ∩ L_H2 ∩ L_VAL`.
`H1_ONLY` = names in the H1 signal stream that are **not** in `INTERSECT`.

| id | cell |
|---|---|
| **F19-0** | *structural report, not a cell*: per era — distinct listed wrappers and commons in the candidate stream, their ADV$ and price distributions, share of the book supplied, and the UNCHECKABLE count |
| **F19-b1** | base book `B2` restricted to `INTERSECT`, three-era table |
| **F19-b2** | base book `B2` on `H1_ONLY` (H1 only, by construction) — the complement |
| **F19-o1** | wrappers-only, on `INTERSECT` |
| **F19-o2** | `spy_r5_pct > 0` (the SPY 09:35 day gate), on `INTERSECT` |
| **F19-o3** | `dollar_frac >= p80` (TRAIN-fit rung, unchanged from F10), on `INTERSECT` |
| **F19-o4** | price floor $30 (`next_open >= 30`), on `INTERSECT` |
| **F19-x1** | the H1 book on the **H2-listed** universe only (`L_H2`) — the frame's own "is the H1 rail rejecting a listing artefact" cell |

**THE PRE-COMMITTED RE-OPEN RULE.** An object is **RE-OPENED (with the listing caveat on the ledger)**
iff, on `INTERSECT`: (a) its H1 gross R is **>= 0**, (b) H1, H2 and VAL gross are **same-signed
positive**, and (c) it books **>= 10 trades/week** on both splits. Anything short of all three leaves
the object dead and the era rail **sound as written**. Re-opening is not a ship and not a claim: it
puts the object back on the queue with the caveat attached.

Frequency floor and MDE are reported for every cell; a restriction that drops an object below 10 tr/wk
is a **frequency** finding, not an era finding, and is labelled as such.

---

# F21 — SURVIVORSHIP  (4 cells)

**Question.** Every cell intersects the break stream with **today's** `daily_bars`. How much is that
worth, and in which direction, for a long-only intraday book that is flat by 15:55?

**PIT HOD-universe.** From the Databento PIT daily panel
(`data/research/databento/equs_daily_2025_2026.parquet`, plus `equs_daily_2024H2.parquet` for the
20-session lookback into 2024): a symbol-day is in the PIT HOD-universe iff prior close >= $17 and
ADV20 (20-session mean volume, strictly prior, min 15 observations) >= 100,000 — the engine's own
universe rule (`hod_break_engine`, `load_adv20_from_daily_bars`). Test tickers removed.

| id | cell |
|---|---|
| **F21-1** | share of PIT HOD-universe **symbol-days** absent from today's `daily_bars` symbol list, **per year** (2025, 2026) and per month |
| **F21-2** | the same share split **wrapper vs common** (`orb_asset_class`) and by price/ADV$ band |
| **F21-3** | **direction**: for the absent names, the distribution of their own daily open→close return and high/low range on the absent symbol-days, against the present names' — the mechanical proxy for whether a long-only intraday book would have done better or worse on them |
| **F21-4** | **materiality**: the base book's gross re-read on the subset of absent symbol-days for which 1-minute bars exist in the study's own bar stores (`cache.db.intraday_bars_1min` ∪ `research/bf_zero/bars_sip.db`); if none exist, that is stated as the bound and F21-4 is reported as NOT COMPUTABLE with its coverage number |

**THE PRE-COMMITTED MATERIALITY RULE.** Survivorship is **NOT load-bearing** — and that fact goes on
the ledger permanently — iff the absent share is < 5 % of PIT HOD-universe symbol-days in both years
**or** F21-3 shows the absent names' intraday return distribution inside the present names' (mean
difference under the book's own MDE). Otherwise every era-consistency verdict in eleven passes is
labelled **computed on a survivor set** and the base book is re-read on the PIT population.

Expectation stated in advance (so that it can be wrong): delisting risk is an *overnight* risk and
this book is flat by 15:55, so the bias is expected to be SMALL. It is measured, not assumed.

---

## Cells declared: **18** (F20 6 + F19 8 + F21 4), plus 4 reported diagnostics
(F20-d, F20-e, F19-0, and the F21 structural split). Programme count after this pass: **989 + 18 = 1,007.**
