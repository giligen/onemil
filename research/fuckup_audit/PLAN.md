# Where the research is wrong — the audit-and-re-search plan

Written 2026-09-16 for the model that will EXECUTE it. Owner's brief (9/16): "find out where your research sucks —
something is wrong/bad, or you are missing opportunities, or you take things for granted (e.g. −R is at the support
level where everyone is sitting). Money can be made with algo trading on features." Plus: "you haven't looked at the
losing days vs winning days — maybe they can be identified as down vs up days."

This is not a strategy. It is a list of the specific places the last month's "nothing works" verdict can be wrong, each
with a pre-registered test that settles it, in the order they should be run. Read §1 (the rules) and §2 (what is
already settled) before touching anything. The probe files under this directory (`probe_*.md`) are the evidence the
plan was written from; read them first, they are short.

---

## 0. The verdict being audited, in one paragraph

Over 9/13→9/16 the program ran: a clean-sheet family scan (`research/bf_zero`, look-ahead population, void), a second
scan on the whole point-in-time universe with the consolidated tape (`research/bf_zero2`), a literature-driven queue of
~20 published effects (`research/lit_review_2026/RESULTS.md`), four adversarial audits, and a cost study
(`cost_curve.md`, `cost_by_outcome.md`). Final state (`score4_tables.md`): 26 entry families × 2 exits, honest fill
(next bar's open under a 0.6% cap), per-outcome costs, 12/day 4-concurrent book → **0 of 52 cells positive on TRAIN**;
best −0.106R/trade. Conclusion drawn: "intraday long breakout/pullback books on ≥5%-range US equities do not pay."

That conclusion rests on five design choices that were never varied: (1) the stop sits at the obvious level (the
consolidation/range low); (2) the fill is the engine's reaction fill (next bar's open), not the best obtainable fill
(a resting stop-limit at the level, which the ORB engine already uses); (3) no day-level market context; (4) raw
families scored with NO selection, although every book that ever made money here made it on selection (bull flag:
conviction/MACD/regime; ORB: news × premarket $ × prev-day range × range-size); (5) a universe that forces late entries
(the ≥5%-range-day population plus its causal floor). Each is a hypothesis below.

---

## 1. Rules of engagement (non-negotiable)

**Node.** 2 CPUs, 7.8 GB RAM, ~19 GB disk free, a LIVE trader (`onemil-trader`) running on it. It froze twice under
parallel jobs. ONE heavy python process at a time, always `nice -n 10`, always `ulimit -v` (≤ 4,500,000 KB for a
pass-1 rebuild, ≤ 1,500,000 for analysis). Detached jobs:
`setsid nohup bash -c "cd /home/ec2-user/onemil; ulimit -v 4500000; nice -n 10 python3 X > LOG 2>&1; echo EXIT=\$? >> LOG" >/dev/null 2>&1 </dev/null &`
Never pipe a long job through `| tail/head/grep`. Never load `research/bf_zero2/candidates3.csv` (678 MB) whole —
`usecols` + `chunksize`. Every CSV read in this tree uses `keep_default_na=False, na_values=['']` (the ticker `NA`).

**Never touch.** `data/cache.db`, `data/trades.db` (read via `file:...?mode=ro`), `config.yaml`, `orb.yaml`, any
systemd unit, any order. The owner trades manually on the shared account; his positions are untouchable. No cache
file is overwritten or deleted without the owner's word. Everything this plan writes goes under
`research/fuckup_audit/<stage>/`.

**Every number that reaches the owner passes CLAUDE.md "No research claim ships without an independent check"**:
independent reimplementation from prose (catches code errors, NOT spec errors), obtainability of every fill
(low ≤ fill ≤ high of the filling bar AND reachable by an order the engine would have had resting), causality trace
of every field incl. universe membership, price-scale check, fill realism, tail dependence (top 1%/5% removed, winners
capped), and the count of every cell looked at. Phrasing rule: never "no edge exists" — always "no edge detectable in
THIS universe/horizon/book/window/cost, and the smallest effect the test could see was X".

**Splits are fixed**: TRAIN 2025-01-02..2025-12-31 · VAL 2026-01-01..2026-05-31 · TEST 2026-06-01..2026-09-11. TEST is
read ONCE per stage, after the stage's selection is frozen in writing. Early-close days excluded.

**Conventions**: minutes are ET minute-of-day (570 = 09:30, 585 = 09:45, 600 = 10:00, 841 = 14:01, 955 = 15:55).
Net R per trade (the score4 contract): `half = 0.5·spread_pct / max(r_pct, 0.05)`;
`net = rr − half − half·{stop: 0.875, eod: 0.412, target: 0.0}[why]`. Book = `trading.hod_break.run_book(rows, 12, 4)`.
Scoring population: price ≥ 5, entry ≤ 14:01, R ≥ 1% of price, and for F5–F10 `range_so_far_pct ≥ 5` on bars strictly
before the signal (unless a hypothesis says otherwise — H6).

**The gate that replaces "+10R/week" (H10, adopted here, pre-registered):**
- G1 TRAIN: mean net R > 0 with t ≥ 2.0 on the booked trades, ≥ 5 trades/week.
- G2 VAL: mean net R > 0, t ≥ 1.0, ≥ 55% of weeks green. Bar raised by 1 SE of weekly R per 10 cells that passed G1.
- G3 TEST: read once, reported whatever it says, week by week.
- Economic bar, reported not gated: expected weekly R at 4 slots (≥ 3R/week ≈ $300/week at $100 risk is the minimum
  worth a dry run; the owner's 10R/week is the scale target, not the detection threshold).
- Every candidate that clears G2 also gets: the permutation search-adjusted p over all cells of that stage, tail
  removal (top 1%/5%), winner cap at +3R, and a per-month table.

**Availability audit (standing rule from D1, 2026-09-17).** Every feature with < 100% coverage gets a missingness
table per split and per time-of-day band BEFORE it is used; coverage built from another stage's key set is a look-ahead
even when the value is causal (D1: premarket dollars backfilled only for symbol-days that signalled later → +0.415R on
VAL at t 5.1, pure artefact; the tail test cannot catch a cohort leak, only the missingness audit did).

**Pre-registered from D1 (2026-09-17), to be tested ONLY on Stage E's causal key set (100% news and premarket
coverage):** bucket `news_only` = pre-09:30 news present AND premarket $ ≤ $5.82M, vs the rest, hold exit, per
family and per time band. Adopt only under the H3 sign-agreement rule (TRAIN ≥ +0.05R, t ≥ 2, VAL sign agrees); the
opposite of the shipped ORB gate's logic, so it must be explained before it is believed.

**Reporting.** One `REPORT.md` per stage: pre-registration block (what will be run, the decision rule, written BEFORE
the run), then the tables, then "what it means", then the cell count. Failures are reported at the same length as
successes.

---

## 2. Settled — do not re-litigate, build on it

| fact | where |
|---|---|
| The pass-1 touch fill (`level×1.003` whenever a bar's HIGH reached it) was impossible on 41% of trades and those were 101% of the profit. Any fill must be inside the filling bar and reachable by a resting or reacting order. | `RESULTS.md` third audit; `bf_zero2/audit_data/` |
| A universe defined by end-of-day range needs a causal membership guarantee at the signal bar (`range_so_far_pct ≥ 5` on bars strictly before the signal). | `RESULTS.md` bugs 1 and audit 3 |
| Costs are per OUTCOME: entry half-spread always; target exit on a resting limit pays nothing; stop 0.875×, close 0.412× of the entry half-spread. Quoted spreads by band (first five minutes, this population): 1.90 / 1.20 / 0.80 / 0.60 / 0.50 % at $5–10 / 10–20 / 20–50 / 50–100 / 100+. | `cost_by_outcome.md`, `build_candidates3.py` |
| Tight stops are penalised twice (cost 0.19R at 1–2% vs 0.05R at 5–8%; stop rate 51% vs 17%). | `RESULTS.md` stage 2b |
| The ETF simulator reproduces Zarattini's SPY noise-area result in-sample (SR 1.03) — that simulator is calibrated; the single-stock simulator has NO in-sample calibration anchor (H11). | `RESULTS.md` row 1 |
| Gappers fade: 0 of 27 gap×$-volume cells have +50 bps open→close; prior-day attention names lose 20–60 bps in the first hour (a veto, and a short-side lead). | `daily_queue.md`, `open_fade.md` |
| The earlier books' edge lived in SELECTION, not in the raw setup (BF raw detector −0.01/−0.07R; ORB raw breakout −0.18/−0.04R). | `research/bf_consistency/README.md` §6, `orb_veto_study/REPORT.md` |
| Live HOD dry run 9/16: 7 signals, all obtainable, 4 stopped, 0 targets, −2.6R before costs. Consistent with the scan. | journal `[HOD DRY]` |

---

## 3. The suspects — ranked hypotheses, each with its decisive test

Ranking = (expected effect if true) × (cheapness) × (how much of the verdict it would overturn). Effort is for one
model working alone on this node.

**Ranking AFTER the five probes (9/16 evening) — read this before the hypotheses:**

| rank | hypothesis | measured size | status |
|---|---|---|---|
| 1 | **H7 cost model** | the spread charge in R is **−0.47R** of F5's −0.535R net; the band table is 3.8× too wide for names we trade, the entry crossing is double-counted on a next-open fill, and the repo's own two spread files disagree 3–5× | the null and the cost constant are the same number — A0 first |
| 2 | **H4 selection / features** | F8 N=5 (the ORB entry) raw −0.20R here, −0.18R in the ORB study, positive live after selection; F8/F6 are gross-positive (+0.018 / +0.017) but tail-dependent at population level | the only lever that has ever produced a book here — Stage D |
| 3 | **H3 + H5 same-day market state** | day direction separates the book (t 4.6 non-causal, all periods); the only causal carrier is intraday (IWM open→10:00, t 3.9) | a feature for H4 and a reason for later entries, not a day filter |
| 4 | **H2 resting fill** | +0.022R and +30% trades; 100% obtainable | fix the convention in every pass 1; not a book by itself |
| 5 | **H1 stop placement** (owner) | signature real (34% pure wicks) but the best variant is +0.023R; widening the stop 1% is worth +0.2R net via H7's mechanism | keep the wider-stop shape; drop "stop-run dodging" as a lever |
| 6 | H6 universe bias | untested; feasible in parquet (2 GB/yr) | Stage E, after C |
| 7 | H9 new families, H13 odd lots, H11 calibration, H12 short | untested | queued |

**And the one hard verdict the probes give:** F5, the HOD-break, is gross-negative at ZERO cost (−0.068R, t −7.9,
25,615 trades). It is not rescued by any stop, fill, or day split. It is dropped as a base family. F8 (opening-range
break) and F6 (red-to-green) are the base families going forward: roughly zero gross at population level, i.e. exactly
where a selection stack has something to select from.

### H1 — The stop is at the level where everyone's stop is (owner's hypothesis)  [effort: 0.5 day, data on disk]
**Taken for granted:** every family stops at the consolidation low / range low / lowest low before entry, on a TOUCH.
**Mechanism:** clustered stops at an obvious low invite a sweep (Osler-type cascade); the sweep fills our stop at the
worst print, then the move resumes. Under this mechanism the 46–57% stop rate is inflated by wick pierces, and the
"tight stop = expensive" finding is partly a "wrong stop" finding.
**Literature status (`probe_literature.md`):** the sign is published (Osler 2003, stop clusters at round numbers and
recent extremes; stops cascade more than take-profits), the MAGNITUDE of the post-sweep reversal in equities is not —
"stop hunting" / "liquidity sweep" have zero indexed finance hits. The equity anchor is Kavajecz & Odders-White (2004
RFS): support/resistance levels coincide with peaks in limit-order-book DEPTH, i.e. the consolidation low is where the
book is thick, which cuts both ways (it holds more often, and when it breaks the sweep is violent). Kaminski & Lo
(2014): stop rules add value only when returns are persistent at the stop's horizon — which is what the recovery
numbers below measure. So this hypothesis is settled by OUR data, not by a citation.
**Diagnostic (`probe_stops.md`, F5 K5/X4, all 25,615 filled trades, 420 days):** the signature is exactly as the
owner described — **34% of stops are pure wicks** (the stop bar closes back above the level), 74% pierce by < 0.5%,
38% trade back above the entry within 60 min, 18% later reach the original +2R. But the counterfactual walks also pay
for the trades that keep falling: gross mean R moves from −0.068 (touch) to −0.045 (two consecutive closes below),
−0.049 (close below), −0.070 (level −1%); the stopped cohort still averages −1.01..−1.10R under every variant.
Breakeven-at-+1R is NEGATIVE; the ORB static lock is flat and its only gross-positive form is −0.36R without the top
5%. **Prize of the best stop variant: +0.023R.** Real, ten times too small to matter on its own. The largest single
improvement the probe found is NOT a stop-run fix: widening the stop by 1% at constant $ risk takes F5 net from −0.535
to −0.334R, because it raises R relative to the spread (the H7 mechanism).
**Test (Stage A):** on the score4 population for F5(K5,X4), F8(N30), F6, F1(P12), re-walk exits with stop variants:
S0 touch (current) · S1 bar CLOSE below the level, exit at the NEXT bar's open (the engine reacts after the close) ·
S2 two consecutive closes below · S3 level−0.5% · S4 level−1.0% · S5 level − 0.5×ATR(5-min, last 30 min) ·
S6 fixed 3% / 5% below entry; crossed with exits E1 +2R on close · E4 hold to 15:55 · E5 ORB static lock (arm at
+1.75R → stop to +0.5R) · E6 breakeven after +1R then hold. R is re-defined per variant (risk = entry − stop_variant),
so compare in net R at fixed $ risk AND in $ per trade at $100 risk (identical by construction) and weekly R on the book.
Stop fills stay `min(stop, open)×0.999`.
**Decision rule (pre-registered):** a stop variant is adopted for later stages only if the paired difference vs S0 in
mean net R is ≥ +0.05R with t ≥ 2 on TRAIN and the sign agrees on VAL, AND the stop rate falls, AND the fill of the new
stop is obtainable. Report every cell (7 stops × 4 exits × 4 families = 112).

### H2 — The fill convention is the engine's, not the best obtainable  [effort: 1 day compute, 0.5 day code]
**Taken for granted:** the honest fill is "next bar's open under a cap" because `hod_break_engine.py` reacts after the
bar closes. But for F5/F6/F7/F8 the level is known BEFORE the break (consolidation done, prior close, PM high, range
high) and the ORB engine already pre-places a resting stop-limit buy at the level
(`data_sources/alpaca_client.py` ~L1819 `submit_stop_limit_bracket`, `orb_engine.py` ~L2877–2964).
**Mechanism:** the next-open convention deletes 40–43% of signals — the fastest continuations (open above the cap) —
and audit 1 measured the price one minute after the trigger at a median +48 bps (mean +69–79, p90 +209–255).
Note for the cost side: the repo's own measurement (`bf_zero/REPORT.md` §8) puts the last ask of the signal minute
+5.9 bps above the next bar's open, i.e. a next-open fill is already an ask-side price — one more reason A0 charges no
second crossing on it.
A resting order captures that; on a 2% stop it is ~0.25R per trade, larger than the whole cost model. And the
next-open cap systematically REJECTS the bursts (open above cap = no fill), which is adverse selection against the
best signals.
**Diagnostic (`probe_stops.md` §3 — F5 re-detected from the tape on 45 random days over the whole PIT universe,
parity 2,787/2,787 vs `candidates3`):** `candidates3.csv` CANNOT answer this, because it only contains signals whose
next-bar open came back under the cap (conditioned on cheap fills). Re-detected: the resting model books **3,616 vs
2,787 trades (+30%)**, 100% obtainable inside the signal bar; gross −0.029 → −0.007, net −0.503 → −0.447. On the 2,624
signals BOTH models fill, resting is worse (paired −0.039R, t −3.3) — the whole gain is the 992 bursts the next-open
convention threw away. **Prize: ≈ +0.022R and +30% trades.** The convention is genuinely too pessimistic and every
future pass 1 must carry both fills; it does not by itself close a 0.5R hole.
**Test (Stage B):** rebuild pass 1 as `candidates4.csv` carrying BOTH fills (`entry_next`, `entry_rest`), the signal
bar's o/h/l/c/v, and the level's pre-signal history (number of prior touches, minutes since the consolidation began).
Score both under H1's best stop and the E1/E4/E5 exits.
**Obtainability for the resting model:** fill inside the signal bar (low ≤ fill ≤ high) — true by construction for
max(open, level) — AND a queue check: the signal bar's volume ≥ 5× the order's shares at $100 risk (report the share that
fails). Touch-only breaks (high = level, no trade through) become fills at the level: keep them, they are real.
**Live implication if it wins:** the HOD engine gets a `entry_mode: resting` that arms `submit_stop_limit_bracket` at
the level once the consolidation criterion is met, cancels on a lower low beyond X% or at 14:00, and caps ARMED orders
at 4 (buying power is reserved per armed order at Alpaca — this makes "which 4 to arm" a selection problem, H4). Dry-run
items: trigger on odd-lot prints (H13), partial fills, cancel/replace latency.
**Decision rule:** the resting fill replaces next-open as the reference fill only if ≥ 95% of its fills pass
obtainability and the queue check, and the improvement holds on TRAIN and VAL.

### H3 — Winning days vs losing days (owner's second hypothesis)  [effort: 0.5 day, data on disk]
**Taken for granted:** no split by day context was ever run on the family books.
**Diagnostic (`probe_days.md`, score4's book reproduced to 3 decimals, then split; `day_features.csv` and
`etf_1min.db` SPY/IWM/QQQ minute bars already on disk):**
- **The days DO separate, by the day's own direction.** F8 on SPY-up days +0.00 / −0.01 / −0.43 R/day vs −1.08 / −1.04
  / −0.89 on SPY-down days (Welch t 4.6 / 3.0 / 1.1); on IWM close-vs-open t 4.8 / 3.9 / 2.2. Top-tercile breadth days
  (share of the family's own candidates closing above their open) are positive in ALL three periods for F8 (+0.19 /
  +0.16 / +0.52 R/day) and F6 (+0.97 / +0.80 / +0.11). The owner's intuition is right about the structure.
- **No 09:30-knowable split reproduces it.** 126 causal cells: largest TRAIN |t| 1.87 vs 4.8 non-causal; TRAIN→VAL sign
  agreement is a coin flip. Overnight gap / prior-day return / trend / vol / day-of-week / regime all imply at most
  0.2–0.5 R/day through the direction channel, BELOW the detection floor (1.54 / 0.51 / 0.74 R/day at 80% power for
  F5 / F8 / F6) — those nulls are uninformative, not evidence of absence.
- **The one causal link that is statistically real is intraday**: F5 daily R vs the IWM 09:30→10:00 return, corr 0.19,
  t 3.9 on 420 days. Its top tercile still leaves F5 negative (−0.45 / −0.33 / −0.39 R/trade) — under score4's costs
  (H7 says those are ~0.3R too harsh, so re-read this after A0).
- Loss is broad, not a tail (removing the best and worst 10% of days leaves 80% of it); daily R has NO autocorrelation
  (|t| ≤ 0.85) — "stand down after a bad day" buys nothing; heavy-signal days are NOT the losing days (corr +0.1..+0.2).
- Multiplicity on record: 126 causal + 30 non-causal cells, 33 filter rules, all selected maxima.
**What it means for the plan:** the tradable version of "up day vs down day" is a SAME-DAY, INTRADAY market feature
read at the entry minute (IWM/SPY open→now return, breadth-so-far = share of the universe's names above their open at
the signal minute), which only has content for entries after ~10:00 — this ties H3 to H5 (later entries) and makes the
intraday market state a first-class feature for H4, not a day filter. Overnight/regime day filters are dead for this
purpose and should not be re-run.
**Test (Stage A):** re-run the IWM-open→now and breadth-so-far splits under the A0 cost contract, restricted to entries
≥ 10:00 (the population where the feature is known), with the sign-agreement rule and the cell count. Path rules
(stop after −2R / two stops) are dropped — no autocorrelation to exploit.
**Decision rule:** a day filter is adopted only if the excluded bucket is negative on TRAIN AND VAL, the kept bucket's
mean net R improves by ≥ 0.05R, and the bucket is defined by data available at 09:30 (or at the entry minute for
intraday index returns). Non-causal splits are reported for understanding only, never traded.

### H4 — Raw family vs selection: the feature program ("money is made on features")  [effort: 2 days]
**Taken for granted:** families were scored raw. Every profitable book here was raw-negative and selection-positive.
**The calibration point already on disk (`probe_design.md`):** F8 N=5 IS the ORB entry; score4 scores it raw at
−0.200R, and the ORB veto study published the same raw −0.18R — next to a live ORB book that is positive on its
selection stack (news × premarket $ × prev-day range × range-size × catalyst cohort, weekly-refit quintiles). So the
scan's −0.11..−0.49R raw cells are exactly what a working selection book looks like BEFORE selection. "Raw family is
negative" was already known twice over and does not support "no book exists".
The family scans never computed the features the ORB rulebook validated (`research/orb_machine_rules.md`): premarket
news presence, premarket dollar volume, prev-day range %, 5-min range-size %, asset class / catalyst cohort, gap %.
`candidates3.csv` carries only rv_adv, gap_pct, adv20, dist_open_pct, range_so_far_pct, spread band.
**Test (Stage D), pre-registered:**
- Candidate table: all filled signals of the 3 best-mechanism families after Stages A–C (fill per H2, stop per H1).
- Features, ALL computable at the signal minute in the live engine (each with its live source named):
  tape: rv_profile, cum $ volume so far, signal-bar volume / consolidation mean volume, consolidation tightness (X
  realised), bars in consolidation, number of prior tests of the level, distance from VWAP, distance from 09:30 open,
  range-so-far %, minutes since open, spread proxy (band), price band, ADV20, 20-day daily vol;
  day context (H3): index gap, prior-day index return, trend, vol regime, IWM open→signal return;
  overnight/premarket: gap %, premarket $ volume (bars_sip.db carries 04:00–09:30 where they exist), prev-day range
  %, prev-day close-vs-high, prior-day attention rank (the open-fade veto), news presence (Alpaca news, prev 15:00 →
  signal; the ORB nightly CSV `data/research/orb_news_catalyst_nightly.csv` is the backfill pattern), wrapper flag,
  sibling-cohort count.
- Target: net R under the honest fill and the adopted exit. Secondary: P(target before stop).
- Model: HistGradientBoosting (sklearn, no new dependency) regression + classification, walk-forward: train on the
  trailing 9 months, predict the next month, monthly refit from 2025-10; hyper-parameters fixed in advance (depth 4,
  200 trees, lr 0.05, min_leaf 200) — ONE config, no tuning on VAL. Plus the transparent baseline: a rank on the top-3
  features by TRAIN univariate monotonicity (the thing that would actually be shipped).
- Selection: top-12 predicted per day → `run_book(12, 4)`; also "predicted net R > 0 only".
- Evaluation: mean net R of selected vs the rest, monthly; decile calibration (predicted vs realised, must be
  monotone on VAL); feature-importance stability across refits; the H10 gates; permutation p; tail test.
- Multiplicity: the model configs (2), selection rules (2), families (3) = 12 cells; declare them before running.
- **Mandatory gate from the literature (`probe_literature.md`): the reversed-tape test.** Nagel (2025, NBER w34104)
  shows that "complex" return predictors trained on short windows collapse into momentum-with-vol-timing and keep
  building the same strategy on synthetic data made to REVERSE. So: train the identical pipeline on the candidate
  table with the sign of every target flipped; if the selected book is still "profitable" on the flipped tape the
  model is fitting the population's momentum, not a feature edge — fail it. Also a shuffled-target run for the null.
- The cheapest conditioning variable the literature offers and this program never used: market STATE — Cooper,
  Gutierrez & Hameed (2004 JF): momentum pays after positive market returns and loses after negative ones (this is
  the same structure `probe_days.md` found at the day level); Avramov–Cheng–Hameed (2016 JFQA): aggregate LIQUIDITY
  state predicts it better than volatility state. Both enter as features (trailing 1–3 month index return; aggregate
  spread/turnover state) with a pre-registered monotonicity direction.
**Decision rule:** ship-candidate only if the transparent baseline ALSO clears G1/G2 (a black box that the rule
cannot approximate is not deployable in this engine), the decile calibration is monotone on VAL, and the reversed-tape
run FAILS to profit.

### H5 — Time of day: the first 30 minutes is the worst-cost window and where every family entered  [effort: hours]
**Literature (`probe_literature.md`):** two 2024–26 firm-level results (Iwanaga & Sakemoto 2026 NAJEF; Zhang, Zhang &
Xue 2024) find the overnight return NEGATIVELY predicts the first half-hour return — the published sign is AGAINST every
pre-10:00 long entry on a gapper. Mazza & Petitjean (2019): spreads are wider and depth thinner exactly at
technical-trading moments, so the cost at the signal minute is an event-time quantity (H7's per-trade NBBO pull, not a
band).
**Diagnostic (from `probe_stops.md` item 5):** mean R and stop rate by entry band.
**Test (Stage A):** score each family with entries restricted to ≥ 10:00 and ≥ 10:30 (fewer trades, cheaper fills,
smaller drift), and separately the resting-order fill (H2) for the 09:30–10:00 window where the drift is largest.
**Decision rule:** the sign-agreement rule; report trades/week (a window that leaves < 5/week fails G1 by construction).

### H6 — The universe forces late entries  [effort: 1 day fetch + 0.5 day scan; disk budget 8 GB]
**Taken for granted:** the population is "days that ended with a ≥5% range" and the causal floor makes F5–F10 wait
until a 5% move has ALREADY happened. Entries at 2–3% above the open, before the crowd, were excluded by the design,
not by the strategy. `REPORT.md` admits it ("what it cannot say").
**Test (Stage E):** define a CAUSAL universe known at 09:30: (a) gap ≥ +3% at the open (Databento daily open vs prev
close, `data/research/databento/` parquet, delisted included), (b) premarket $ volume ≥ $500K, (c) prior-day range ≥ 8%
(the ORB PDR mechanism, day-2 continuation), and (d) the unbiased liquid slice: EVERY symbol-day of names with ADV20 ≥
$5M (no range or gap gate at all). Count the symbol-days missing from `bars_sip.db`; fetch from Alpaca SIP with
`refetch_thin_tape.py`'s fetch path into a NEW store — **as partitioned parquet, not SQLite** (`probe_design.md`: the
"~100 GB, beyond this node" verdict in `bf_zero2/REPORT.md` is a storage artefact; bars_sip.db spends 84 KB per
symbol-day where compressed parquet takes 2–3 KB, so the liquid slice is ~1.5–2 GB per year against 19 GB free).
Re-run F5/F8 (and H9 families) on the causal universes WITHOUT the range floor (membership is causal by construction).
**Decision rule:** compare the same family on the two universes; the universe is changed for later stages only if the
causal one is not worse on TRAIN and VAL — the point is to remove a bias, not to find a number.

### H7 — Quoted spread vs what the account actually pays  [effort: hours, data on disk]
**Taken for granted:** half the QUOTED NBBO spread on entry. Retail flow routed to wholesalers gets price improvement;
resting orders pay less; the research population's spreads (a 09:31–09:35 gap-down set) were applied everywhere.
**Diagnostic (`probe_costs.md`, 191 entries / 107 exits / 101 round trips from `trades.db`, re-derived in pure SQL
14/14):**
- Against the quote at the MOMENT of the fill, our entries pay a median **0.0 bps** (46% at or below the mid, 64% at or
  below the ask). Against the quote at SUBMIT time they "pay" 60.9 bps — and that is price DRIFT during the order's life
  (bull flag median 71 s to fill), not execution. Books that fill on arrival (macd_wave) pay exactly the quoted half
  spread (ratio 1.00×, n=21).
- **score4 double-charges the entry leg**: its fill is the NEXT bar's open (the drift is already inside that price) and
  it then adds half the quoted spread on top. Our fills do not pay that second charge.
- **The band table is 3.8× too wide for names we trade**: live names quote a median full spread of 0.36% at the fill
  ($5–10: 0.39% real vs 1.90% charged; median band charge 1.20%). `cost_curve.md` itself measured 22–77 bps in the
  signal minute on the HOD population, while `build_candidates3.py` charges 190 bps at $5–10 — the two research files
  contradict each other and score4 used the wide one (it came from the 09:31 gap-down F6 population, a different set).
  At spread 1.2% and R 2%, the phantom entry charge alone is ≈ 0.3R per trade — larger than any family's net deficit.
- Exits: stops pay 0.97× the quoted half spread (the model's 0.875 is close); `stop_loss_market_fallback` is the tail
  at 6.3× (n=9). **Targets are free only when the engine RESTS a take-profit leg** (the HOD bracket does; ORB's
  lock-stops cross like stops — 13 of 14 "target" rows in the DB are lock-stops).
- Measured all-in round trip: 0.389R on the submit-quote basis (drift included), 0.045R against the contemporaneous
  market; the banded model would have charged the same trades 0.480R.
- Caveat that is load-bearing: live names passed liquidity gates (spread ≤ 300 bps, prev volume ≥ 500K, spread ≤ 15% of
  R); the research universe did not. The supported claim is "the banded quoted-spread model is not the cost of our
  fills in our names"; the untraded remainder of the universe may be wider.
**Test (Stage A, item A0 — first thing, 30 minutes):** re-score `candidates3.csv` with the corrected cost contract:
(1) entry execution cost = 0.25 × the quoted half spread for a next-bar-open fill (conservative: the measured median is
0, the quartile is 1.38× on 18 bps) and the FULL half spread for a resting fill (H2) executed on arrival; (2) spread per
candidate from the `cost_curve.md` band × time-of-day table (signal-minute NBBO of THIS population), not the 1.90/1.20
table; (3) target exit = 0 only for the resting-TP engine model, 0.875× otherwise; (4) stop = 0.875×, close = 0.412×
unchanged; (5) plus a liquidity gate mirroring live (quoted spread ≤ 15% of R) as a separate row. Report the score4
table under both contracts side by side, with the gross column.
**Then (Stage B):** pull the real NBBO in the signal minute for every BOOKED trade of the surviving cells (~12/day →
≈ 5K quote pulls per cell family, Alpaca quotes API) so the cost is per trade, not per band. Add the per-symbol-day
spread as a candidate feature (H4) and as the live gate.
**Decision rule:** the corrected contract replaces score4's for every later stage; costs are never lowered below the
measured table without the fill evidence, and the "spread ≤ 15% of R" gate is applied on both sides (BT and live).

### H8 — Exit shape was never searched under the honest fill  [effort: inside H1's cross]
Only E1 (+2R close) and E4 (hold) were scored. The E5 lock and E6 breakeven-then-hold variants are in H1's cross; add
E7 partial 50% at +1R then hold with the stop at breakeven, E8 time stop (flat at +30/+60 min if < +0.5R). The engine's
target leg is a resting limit and fills at the touch (the sim's close-fill is conservative by ~0.06R — report both).

### H9 — Missing families: confirmation, retest, sweep-and-reclaim  [effort: 0.5 day code inside Stage B]
- F11 close-confirmation: signal only if the break bar CLOSES above the level (touch-only wicks were −1R live).
- F12 retest: after a break, price returns to within 0.3% of the level within 30 min and the next bar's low holds it;
  enter at the following bar's open; stop = retest low.
- F13 sweep-and-reclaim (the owner's mechanism as an entry): the consolidation low is pierced by ≤ 1% and reclaimed
  (a close back above it) within 5 bars; enter at the next open; stop = the sweep low.
- F14 second break: the first break stopped out; a second break of the same level the same day.
All under the honest fills (H2 both models), H1's stops, the causal floor. Literature: none of F12–F14 has an indexed
empirical test (`probe_literature.md`); the nearest, Marshall–Sun–Young (2009), finds range-breakout rules paid mainly in
small illiquid US stocks — our band — before costs. Run F13 on the point-in-time universe, never on the ≥5%-range cache
alone (that would re-create the 9/15 end-of-day-selection defect: a sweep-and-reclaim that is followed by a 5% range is
selected by the outcome).

### H10 — The gate was not a statistical bar  [adopted in §1]
"TRAIN ≥ +10R/week at 4 slots" ≈ +0.43R/trade at ~23 trades/week, four times the best honest gross edge ever measured
and ~3.5× the design's own minimum detectable effect (`probe_design.md`).
It could not have passed anything and was not designed to discriminate. §1's gate replaces it; re-score
`score4_results.csv` under the new gate FIRST (5 minutes) so the reader sees what the old gate hid.

### H11 — The single-stock simulator has no in-sample calibration anchor  [effort: 1 day; ~$20–50 of Databento, owner OK needed]
The ETF sim reproduces a published result in-sample. The single-stock sim has only ever been run out of sample. Replicate
Zarattini–Barbon–Aziz (2024, "A profitable day trading strategy for the U.S. equity market", ORB on stocks in play,
2016–2023) IN ITS OWN PERIOD: top-20 relative-volume names per day from a Databento daily panel (the one on disk,
`data/research/databento/equs_daily_2025_2026.parquet`, covers 2025→ only — the 2016–2023 EQUS.SUMMARY daily panel must be
bought too, a few dollars), then 1-min bars for those symbol-days (~40K, Databento EQUS 1-min ≈ $0.0004 each), their exact
rules and costs. If the sim reproduces the paper's
Sharpe (they report ~2.8 gross before 2023's decay), the simulator is right and the 2025–26 nulls are decay; if it does
not, the simulator has a bug and every null in this program is suspect.

### H12 — The short side  [queued: after a long book exists, per the owner]
The gap table and the attention fade both point short. Same machinery, borrow availability and locate cost added,
`sell_short` semantics checked against the owner's manual shorts.

### H13 — Odd-lot prints and the bar high  [effort: hours]
Whether Alpaca minute-bar highs include odd-lot prints decides whether "high ≥ level" is a tradable trigger and
whether a stop can be hit by a print that never touched the NBBO. Test: 200 random signal minutes → Alpaca trades API
for that minute → is there a ROUND-LOT trade ≥ level? Same for 200 stop minutes ≤ stop. Report the share of phantom
triggers/stops. If material, both the signal and the stop move to "close" or "round-lot" definitions (this interacts
with H1 and H2).

---

## 4. Execution order, with stop rules

| stage | runs | inputs | output | stop rule |
|---|---|---|---|---|
| A (day 1) | **A0 = H7 corrected cost contract re-score of score4 (gross column shown)**; H10 re-gate; H1 stop×exit cross on F8/F6/F1 (F5 dropped — gross-negative at zero cost); H3/H5 intraday-market and time-band splits for entries ≥ 10:00; H13 odd lots | `candidates3.csv`, `bars_sip.db`, `trades.db`, `cache.db` daily_bars, `etf_1min.db`, the `probe_*.py` scripts | `A/REPORT.md`, `day_features.csv`, adopted cost contract + stop/exit rules (or none) | none — A is diagnostic and always completes |
| B (day 2) | pass-1 rebuild `candidates4.csv`: both fills, signal-bar OHLCV, level history, MAE/MFE, stop×exit matrix computed in the walk (long format), F11–F14 | `build_candidates3.py` as the base | `candidates4.csv` + header stating the scorer contract | if H2's resting fill fails obtainability (< 95%) it is dropped, not softened |
| C (day 3) | `score5.py`: family × fill × stop × exit under §1's gate; cell count; permutation p for anything clearing G2 | B | `C/REPORT.md` | if nothing clears G1: write the closest miss per family with its power, and go to D anyway (selection can carry a raw-negative family — that is the ORB/BF history) |
| D (days 4–5) | H4 feature table + walk-forward model + transparent baseline | C's best 3 mechanisms | `D/REPORT.md`, feature importances, monthly selected-vs-rest | ship-candidate only if the transparent baseline clears G1/G2 |
| E (day 6, if A/C show the late-entry bias matters) | H6 causal universe fetch + re-scan | Databento daily, Alpaca SIP | `bars_causal.db`, `E/REPORT.md` | disk ≤ 8 GB; if the fetch cannot serve ≥ 90% of the symbol-days, report the survivorship residual and stop |
| F (day 7) | the independent check on the surviving candidate (§1), the owner memo, the engine spec delta (`entry_mode`, stop rule, day filter, selection rule), dry-run plan | D/E | `F/OWNER_MEMO.md` | no memo without every item of the CLAUDE.md check |
| G (queued) | H11 simulator calibration (owner's $ approval), H12 short side | — | — | — |

Daily, at the end of each stage, append three lines to `research/fuckup_audit/LOG.md`: what ran, what it says, what is
next. If the owner asks "status", answer from LOG.md.

---

## 5. What "done" looks like

Either (a) a candidate book with: honest fills (both models reported), the adopted stop/exit/day/selection rules each
with its own before/after table, G1/G2 passed, TEST read once, tail and cap tests, permutation p, the cell count of the
whole program, the engine delta, and a dry-run plan — or (b) the same document with the closest miss per hypothesis and
the smallest effect each test could have detected. Both are complete deliverables. A book that "works" only under
one of the two fill models, or only before the tail test, is reported as (b).

---

## 6. File map

| path | what |
|---|---|
| `research/fuckup_audit/probe_stops.md` | H1/H2/H5/H9 diagnostics (agent, 9/16) + scripts |
| `research/fuckup_audit/probe_days.md` | H3 diagnostics + `day_features.csv` builder |
| `research/fuckup_audit/probe_costs.md` | H7 measurement on `trades.db` + script |
| `research/fuckup_audit/probe_design.md` | the blind-spot review that ranked these hypotheses |
| `research/fuckup_audit/probe_literature.md` | sources for H1/H4/H7/H9/H12 |
| `research/bf_zero2/build_candidates3.py`, `score4.py` | the base pass 1 and scorer (contracts in their headers) |
| `research/bf_zero/build_candidates.py` | the family definitions F1–F10 (the touch-fill defect lives here; do not reuse its fill) |
| `research/bf_zero/bars_sip.db` | 1-min SIP bars, `bars(symbol, day, t, o, h, l, c, v)`, 199K symbol-days |
| `research/bf_zero/universe.csv` | the point-in-time ≥5%-range universe (symbol, bar_date, open, high, low, close, volume, adv20, prev_vol) |
| `data/research/databento/` | point-in-time daily panel incl. delisted (for H6, H11) |
| `data/cache.db` daily_bars | SPY/IWM/QQQ daily (for H3); read-only |
| `data/trades.db` | 371 live trades with quote telemetry (for H7); read-only |
| `trading/hod_break.py` | the live spec: `entry_fill`, `run_book`, `HodBreakParams` |
| `trading/hod_break_engine.py` | the engine (capped limit after the bar closes; the H2 delta goes here) |
| `data_sources/alpaca_client.py` ~L1819 | `submit_stop_limit_bracket` — the resting entry ORB uses |
| `research/orb_machine_rules.md` | the validated ORB selection features for H4 |
| `research/lit_review_2026/RESULTS.md` | everything that was tried and how it died |

