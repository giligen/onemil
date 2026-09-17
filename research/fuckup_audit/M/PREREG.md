# Stage M — pre-registration (written BEFORE any P&L was computed)

Written 2026-09-17. Owner's book, one sentence: *"select at 09:30 with the features
that work, buy with a resting order, cut the stalls fast, hold the runners."*
Stages D1_orb (selection dose-response) and `orb_timestop_validation` (the 10-minute
time stop on the honest fills) settled the first two clauses and the population.
Stage M settles the last two: **which exit shape**, at the slot count that is now live.

Nothing below was run before this file was written. No threshold, z-param, quintile
cutoff, veto level or adaptive mult is re-fitted anywhere in this stage; every
selection rule is read from `orb.yaml` as it stands.

---

## 1. Population and selection — frozen, not a variable

* Features: `analysis_results/orb_features_20260916_2053.csv` — the entered-inclusive
  honest rebuild (13,033 candidates over 427 trading days, 2025-01-02 → 2026-09-16;
  7,402 fills + 5,631 modeled non-fills that consume a slot and book $0).
* Selection: the **shipped B+ stack**, unchanged — frozen z-params + quintile cutoffs
  from `orb.yaml`, composite threshold 0.012081536791, Q1 filter ON, family +
  super-group dedup, Q4-preferred ranking then composite DESC, adaptive mults
  (uniform 1.0), PM/news mult OFF, then the four POST-ranking no-refill vetoes
  (PDR ≥ 11.0, G1 fingerprint + short-history, range-size > 2.221, catalyst required).
* Entry: unchanged — the pre-placed stop-limit at `range_high × 1.003`, market
  breakout bar from the shared `find_breakout_bar_ts`, 60-minute order life.
* Sizing: unchanged — `_rp_position = min(risk / (max(range_size_pct, 1) / 100),
  account / N)`, `risk = $375`. The per-position cap binds on 100% of picks (D1), so
  the account budget is scaled with N to hold the per-position cap at the live
  **$3,333.33**: `ORB_BT_ACCOUNT = 3333.333333 × N`. Every trade is sized identically
  in every cell; only the slot count and the exit shape move.
* Splits, fixed, as in every stage of this program:
  **TRAIN = 2025 (12 mo) · VAL = 2026-01..05 (5 mo) · TEST = 2026-06+ (4 mo, partial
  September; ORB was paused 9/14-9/17).**

## 2. The six exit shapes (declared; nested ladder)

All six keep the touch-and-go filter (Rule M at the breakout bar's close, Rule D at
the first post-entry bar's close) and the 15:45 ET force close. All six keep the two
shipped winner-stack flags that are ON in `orb.yaml` right now — the ATR14 stop floor
(k = 0.25, applied to the protective stop only) and the 40% scale-out at +3R — so
each shape is a **single-lever** change from the shipped exit, not a re-design.

| id | shape |
|---|---|
| **X0** | **shipped = the reference.** Static lock: stop starts at the (floored) range low; when a bar's high reaches entry + 1.75R the stop ratchets once to entry + 0.5R and stays. Touchgo, scale-out, 15:45 flat. |
| **X1** | X0 **+ 10-minute time stop**: at the first bar at/after fill + 10 min, if `(that bar's open − entry) / R < +0.25`, exit at that open. |
| **X2** | X0 **+ 5-minute time stop** at the same rule with threshold `< 0.0`. |
| **X3** | **No lock.** Initial (floored) stop only, held to 15:45. Touchgo + scale-out unchanged. |
| **X4** | X3 + the 10-minute time stop of X1. |
| **X5** | X4 + **breakeven stop**: once a bar's high reaches entry + 1.0R, the stop moves to the entry price and stays. |

`R` is the 5-minute opening range (`range_high − range_low`) — the same R the shipped
lock and Rule D use. Exit-price conventions are the shipped ones: stop-type exits fill
at the stop level less 10 bps; the time-stop exit fills at the bar's **open** less
10 bps (a price inside the bar that fills it, reachable by a market order sent at the
bar boundary — the engine's own convention); the 15:45 exit fills at that bar's close
less 10 bps.

**Inherited deviations, declared now, not fixed here** (they are the shipped
simulator's and affect all six shapes equally, so they cancel in the X0-vs-Xn diff):
a stop that gaps through fills at the level rather than at the gap price; the entry
fill is the recorded `entry_price`; the 10 bps exit slip is a constant.

## 3. The grid — 18 cells, declared in full

6 exit shapes × N ∈ {3, 8, 12} slots.

* **N = 8** is the live config (`orb.yaml sizing.max_concurrent: 8`,
  `account_budget_usd: 26667`, set 2026-09-17 from D1's dose-response). **The decision
  is made at N = 8.**
* N = 3 (the pre-9/17 shipped slots) and N = 12 are reported as the robustness flanks —
  a shape that only wins at one slot count is not adopted.

Per cell, per split and per month: picks, fills, fill %, P&L $, WR on fills,
mean $/fill, mean R/fill, $/week, weeks green, max drawdown on the daily cumulative
curve, worst month, the exit-reason mix, and the two tail tests
(P&L with the top 5% of fills removed; P&L with every fill's return capped at +3R).

`R` per fill for the R/fill column = `_rp_position × max(range_size_pct, 1) / 100`,
the dollar risk the sizer actually took (D1: ~$149 at the $10K stage, not the
configured $375, because the per-position cap binds).

## 4. Decision rule — pre-committed

> **A shape replaces X0 only if, at N = 8, it improves BOTH the total P&L AND the
> maximum drawdown on TRAIN AND on VAL.** (Four inequalities, all strict.)

* TEST is read **once**, after the TRAIN/VAL verdict is written, and reported whatever
  it says — for the shapes that pass and for X0. TEST cannot promote a shape that
  failed TRAIN/VAL and cannot demote one that passed; it is reported as evidence, and
  a TEST result that contradicts the verdict is stated as such in the report.
* Ties and near-ties (within $50 or within $25 of MDD) count as **not improved**.
* If more than one shape passes, the one with the better VAL drawdown wins; if that
  ties, the simpler shape (fewer levers away from X0) wins.
* If no shape passes, X0 stands and the report says so at the same length.
* A shape that passes at N = 8 but reverses sign at N = 3 **and** N = 12 is reported
  as regime-fragile and is not recommended for the engine.

## 5. The money line — declared method

For the shape that wins (or X0 if none does), at N = 8:

* **$/month at stage sizing** = total P&L ÷ 21 months, at the live per-position cap
  $3,333.33.
* **$/month at 3×** = the same book with `account_budget_usd = 80,000`,
  `per-position = $10,000`. Because the cap binds on 100% of picks, the position and
  therefore the P&L scale exactly 3× **before market impact** — the 3× row is a
  linear extrapolation and is labelled as such.
* **Participation** = `position $ ÷ (range_total_volume × entry_price)` = the pick's
  share of its own 5-minute opening-range dollar volume, median and p90, at 1× and 3×,
  plus the count of picks above 5% of that tape.
* **Worst month at 3×** = 3 × the worst month at stage sizing, same caveat.

## 6. Verification, per CLAUDE.md "No research claim ships without an independent check"

1. **Reimplementation / parity.** The X0 walk in this stage is the *shipped*
   `simulate_winner_stack`, called through the shipped pipeline. The parametrised
   walker that produces X1–X5 is a separate function; with its three new knobs at
   their neutral values it must reproduce the shipped walker's exit price and reason
   on **all 7,402 fills, exactly** — asserted in code, and the run aborts if not.
   The X0 candidate dump must additionally equal D1's `candidates_dump.csv` P&L
   column to 1e-9 on all 13,033 rows, and the X0/N=8 book must reproduce D1's
   `$14,429 / 215 picks / 162 fills`.
2. **Obtainability.** Every new exit price is inside the bar that produces it: the
   time stop fills at that bar's open; the breakeven stop at the entry level, which
   the bar's low crossed. The share of trades whose exit differs from X0 is reported
   per shape.
3. **Causality.** The new levers read only the entry price, R (fixed at 09:35), and
   bars at or after the fill. No field from after the decision bar enters any rule.
4. **Price scale.** All prices in the walk come from one source (`cache.db`
   `intraday_bars_1min`, Alpaca raw); no daily-file price is compared to an intraday
   bar anywhere in this stage.
5. **Tails.** Every cell is reported ex-top-5%-of-fills and with winners capped at +3R.
6. **Multiplicity.** 18 pipeline cells + 2 sizing rows = **20 decision cells**; the
   descriptive cells are counted in the report's final section. The 9-cell time-stop
   grid (5/10/15 min × 0/0.25/0.5R) was already searched and reported in
   `orb_timestop_validation.md`; Stage M takes its **pre-declared primary cell
   (10 min / +0.25R)** and one flank (5 min / 0.0R) and searches nothing further.

## 7. Phrasing

No sentence in the report will say "no edge exists". Where a shape does not clear the
gate the finding is stated as "not detectable in this universe, at this book size,
over this window, at this cost", with the size of the smallest effect the test could
have seen.

## 8. Resources / safety

One `nice -n 10 python3` process at a time, `ulimit -v 1500000`, bars loaded month by
month so the walk never holds the whole tape. `data/cache.db` opened through the
existing read path only. Nothing outside `research/fuckup_audit/M/` is written — no
config, no service, no order, no cron, no production artefact. `orb.yaml` and
`config.yaml` are read, never written.
