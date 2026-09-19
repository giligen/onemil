# Entry-leg cost audit — does the `0.25 x half-spread` error touch Monday's two configs?

2026-09-19. Bounded arithmetic + provenance audit. **No book was re-run end to end.** Read-only on
`config.yaml`, `orb.yaml`, orders, the service and the crons; everything written is under this
directory plus two research cost helpers and one test file (§5).

Trigger: `research/mature_method/red_to_green/REPORT.md` §3 measured the entry leg of the research
cost contract at **1.00 x half-spread, four times the 0.25 x the contract charges**, and warned that
"every prior 'net' number on a small-cap book in this repo is too kind by ~0.04 R/trade."

---

## 0. The answer

| Monday decision | cost model it actually used | is it the 0.25 contract? | verdict |
|---|---|---|---|
| **ORB catalyst veto OFF** (`orb.yaml filter.catalyst_veto.enabled: false`) | Stage Q's **measured per-trade SIP NBBO** fill model | **No** | **unaffected** |
| **BF ADV gate off** (`config.yaml scanner.min_daily_volume: 0`) | shipped Stage-2 **flat 50 bps** entry slip | **No — but it is too kind anyway** | **affected: the delta's sign does not survive** |

Neither decision used the 0.25 contract, so the red-to-green finding does not transfer to them
directly. It *does* transfer as a method: price the entry at a quote the market actually showed. Doing
that on the 103 bull-flag entries behind decision 2 removes the entire case for it.

---

## 1. Decision 1 — ORB catalyst veto OFF. Provenance: MEASURED. Unaffected.

Verified in code, not assumed:

- `research/orb_gates2/run_grid.py:24` — `DUMPS['meas'] = research/fuckup_audit/Q_fill/dump_measured.csv`,
  labelled `# PRIMARY`; `research/orb_gates2/REPORT.md` §1 names it "Stage Q's measured capped-limit
  arm ... the walked **per-trade SIP NBBO** ask ... **Never the band constant.**"
- `research/fuckup_audit/Q_fill/rescore_q.py` builds that dump by walking the real quote path of
  every elected order. The entry price is the **stop-limit cap** `range_high x 1.003` — a posted,
  obtainable price — and **`ask > cap => no fill, $0, slot spent`**. There is no `x half-spread` term
  anywhere in the ORB path; `study_orb_pipeline_static_lock.py` carries only `EXIT_SLIP_BPS` (10 bps,
  exit side).
- The cap is an **over**-charge, measured against the live account: Q_fill §0 — 166 live ORB orders,
  every live fill at or below the cap, **median 13.8 bps below**; 14.4% of simulated fills had the ask
  above the cap and are booked as no-fills.

So the ORB book charges an entry price 30 bps above the trigger level, empirically worse than what the
account paid. **No re-pricing is possible in the direction the error runs**, and the delta needs none.

Reproduced from the books themselves (`book_G0_meas.csv` vs `book_G3_meas.csv`, 2026-04-01..09-30):

| cell | picks | P&L (book sizing) | weeks | green weeks |
|---|---|---|---|---|
| G0 shipped (veto ON) | 76 | +15,796 | 22 | **23%** |
| G3 veto OFF | 240 | +44,490 | 25 | **56%** |

The 23% -> 56% green-week shift quoted for Monday reproduces exactly, under the measured fill model.
(The as-is bracket agrees in sign: +26,221 / 36% vs +74,623 / 68%.) Nothing here moves.

---

## 2. Decision 2 — BF ADV gate off. Provenance: shipped Stage-2, flat 50 bps. Affected.

### 2a. What the Stage-2 path charges

`research/bf_frequency/run_grid.py` runs the shipped `batch_backtest.py` Stage-2. Its entry model is
`backtest.py:2622`:

```python
raw_fill   = max(bar_open, pending_order.breakout_level)
fill_price = raw_fill * (1 + self.entry_slippage_pct)      # trading.entry_slippage_pct = 0.005
```

A **flat 50 bps**, spread-blind, identical for a $900K-ADV name and an $80K-ADV name. Not the 0.25
contract — and `research/bf_frequency/separation.py:32` computes R straight from that cache's `pnl`,
so the -0.033 pooled separation inherits the same model.

### 2b. The delta the decision rests on

`runs/VOL_OFF.csv` minus `runs/P1.csv` = **48 added trades, zero displaced**; the ADV gate off is
purely additive. Booked: **+$11,855 over 20 months** at the $2K normalization, mean R +0.1235.

Arithmetic tolerance, before any measurement: median `r_pct` is 0.034, so **every extra 41 bps of
entry cost erases the whole delta** (`(sum R) / (sum 1/r_pct)`). The book already charges 50; the
delta survives a total entry cost of ~91 bps and no more.

### 2c. Measured (Alpaca SIP NBBO, `price_bf_entries.py`)

For every trade in both books: the first TRADE print in the entry minute (`t_fill`), the last NBBO at
or before it (Stage P/Q convention), and `excess = ask_fill - modelled entry_price`. 103 of 104 priced.

| | n | ADV20 med | full spread med | **half-spread med** | ask > modelled fill | booked | strict* | signed** |
|---|---|---|---|---|---|---|---|---|
| **delta** (ADV gate would block) | 47 | 83K | **250 bps** | **125 bps** | **51%** | +$8,195 (+0.087R, t 0.80) | **-$9,440 (-0.100R)** | +$1,162 (+0.012R, t 0.09) |
| P1 baseline (shipped) | 56 | 851K | 145 bps | 72 bps | 21% | +$139,114 (+1.242R) | +$123,555 (+1.103R) | +$147,367 |

\* strict = charge the ask whenever the ask was above the modelled fill, keep the modelled fill
otherwise (a resting buy cannot be improved by an ask it never saw).
\*\* signed = fill at the ask always, crediting the trades the flat 50 bps over-charged.

**The flat 50 bps is 2.5x too cheap on the delta (125 bps of real half-spread) and 1.4x too cheap on
the shipped book (72 bps).** The error red-to-green named is present here in a different dress: not a
wrong coefficient, a spread-blind constant applied to names whose spreads are 250 bps.

### 2d. Corrected weekly table and green weeks — the delta only

35 weeks carry at least one delta trade (`delta_weekly.csv`, $ at the $2K normalization; the L0 live
path is x0.075):

| | green weeks | red weeks | sum |
|---|---|---|---|
| booked (50 bps) | **63%** | 37% | +$8,195 |
| signed obtainable | 49% | 51% | +$1,162 |
| strict obtainable | **43%** | 57% | -$9,440 |

Two weeks carry the whole booked number (2025-02-17 +$6,711 and 2026-05-04 +$5,183); remove them and
the delta is negative under every model. **Under the owner's own metric — green weeks — the delta goes
from a majority-green add to a majority-red add.** Its t-stat is 0.80 as booked and 0.09 corrected.

**Conclusion for decision 2: the ADV-gate-off case does not survive an obtainable entry price.** The
booked +$11.9K is between -$9.4K and +$1.2K once the entry pays a price the market showed, on 47
measurable trades with a t of 0.09 at best.

### 2e. Direction of the separation number

`separation.csv` rates the ADV20 >= 200K gate at **-0.033 pooled ("wrong side in both years")** — the
number that justified turning it off. That statistic is computed on the same spread-blind 50 bps, and
the rejected side is systematically the wide-spread side: the measured half-spread differential
between the two matched samples above is **125 - 72 = 53 bps**, which at `r_pct` 0.034 is
**~0.16 R against the rejected side**. Corrected, the gate's separation is of order **+0.12 R, i.e.
the right side, not the wrong side.** This is an estimate from the two book-level samples, not a
re-measurement of the 630-trade L2 population — but it is the sign, and the sign is what the decision
turned on.

---

## 3. Scope — every other live-relevant number, and whether it changes a decision or a level

| artifact | cost path | correction changes |
|---|---|---|
| **`trading/ramp_bt_band.py` — BF band (`runs/VOL_OFF.csv`)** | Stage-2 flat 50 bps | **A DECISION.** n=8 band p10 **-0.29 -> -0.50**, p5 -0.48 -> -0.73 (2,000 draws, seed 20260919). The band sits ~0.2R too HIGH, so Gate-2 item 4 reads a live book as BELOW-p10 (no advance) or BELOW-p5 (**demote after >= 8 trades**) when it is performing to spec. The bias is toward false demotion, not toward passing a book it should not. |
| **`trading/ramp_bt_band.py` — ORB bands (`Q_fill/book_measured_n8.csv`, `orb_gates2/book_G3_meas.csv`)** | measured NBBO | nothing. Both are measured-fill books (§1). |
| `docs/scaling_plan_2026.md` Gate-2 item 4 | delegates to `ramp_bt_band.py` | same as the two rows above — the ORB half is clean, the BF half inherits the bias. |
| CLAUDE.md BF honest references ($107,351 as-is; $131K P1; $95,363 partial) | Stage-2 flat 50 bps | **a level, not a decision.** P1 measures 21% unobtainable at ~11% of P&L; every P1-vs-alternative comparison is a same-cost diff and holds. |
| CLAUDE.md ORB honest reference ($6,085 / 21 mo, B+ book) | pipeline cap entry + 10 bps exit | **nothing** — the cap is an over-charge (§1). |
| `research/fuckup_audit/A` (HOD-break Stage A) and `mature_method/hod_break` verdict | 0.25 contract | **a level.** Verdict was STAY DRY; a 4x entry leg only makes it more negative. |
| `research/fuckup_audit/{B,D,D1,G,H,J,L,O_halt}` | 0.25 contract | **levels.** All are dead or negative books; the correction pushes every one further negative. None is shipped. |
| `research/fuckup_audit/P_cost` `band` arm | 0.25 contract | **a level**, and only of the comparison arm — P_cost's primary is the measured arm and is unaffected. |
| `research/bf_zero`, `research/bf_zero2` | full half-spread entry already | **nothing.** They were already charging the corrected leg. |
| `research/mature_method/red_to_green` | both cells declared (`M2meas` legacy, `M3entry` corrected) | already superseded by its own §3. |

---

## 4. What this audit does NOT establish

- The delta correction is measured at the **entry bar's opening print**. When `bar_open` is below the
  breakout level the live stop elects mid-bar, so the true election instant is later than the quote
  used; 51%/21% unobtainable are estimates at one instant, not a full order-life walk (the Stage-Q
  treatment). A full walk would be a study, not this audit.
- One delta symbol-day (NTCL 2025-04-04) returned no SIP quotes and is excluded from all measured
  rows; it is in the booked columns.
- No claim is made that the ADV gate should be back ON. The claim is that the **evidence offered for
  turning it OFF does not survive an obtainable entry price**, and that is an owner decision.

---

## 5. Code changed, and the reports thereby superseded

- `research/fuckup_audit/A/acore.py`: `ENTRY_COEF_C 0.25 -> 1.00`, `LEGACY_ENTRY_COEF_C = 0.25` kept
  for reproducing committed reports; contract docstring rewritten with the measurement.
- `research/fuckup_audit/P_cost/rescore.py`: `ENTRY_COEF 0.25 -> 1.00` (band arm only), legacy kept.
- `tests/test_entry_cost_contract.py` (4 tests, all pass): pins both coefficients, pins the arithmetic
  (the entry leg quadruples, the exit leg is untouched), and **fails if a new `0.25 * half` site
  appears** in `research/` — the current 19 sites are frozen as a superseded allow-list that may only
  shrink.
- **Superseded on re-run** (not re-scored here — re-scoring them is a study): Stage A's `a0/a1/a2/a3`
  tables, `D/d0_features`, `D1/d1_features`, `B`, `G`, `H` (F5, F6_sizing, F6_rebuild, F6_reconcile,
  F14_F8_F11), `J`, `L`, `O_halt` (incl. PASSIVE), and `P_cost`'s band arm. Every one of those books
  was already dead or negative; the correction moves them further in that direction.
- **Not changed, deliberately**: `config.yaml trading.entry_slippage_pct: 0.005`. It is the shipped
  BF Stage-2 entry model and a production config value — an owner call, flagged in §2, never edited
  by this audit.

## 6. Reproduce

```bash
python3 research/mature_method/entry_cost_audit/fetch_bf_delta_nbbo.py     # scoping pull
python3 research/mature_method/entry_cost_audit/price_bf_entries.py \
        research/mature_method/entry_cost_audit/delta_trades.csv  .../delta_fill.csv
python3 research/mature_method/entry_cost_audit/price_bf_entries.py \
        research/bf_frequency/runs/P1.csv                         .../p1_fill.csv
python3 -m pytest tests/test_entry_cost_contract.py -q
```

Artifacts: `delta_trades.csv` (the 48), `delta_nbbo.csv`, `delta_fill{,_priced}.csv`,
`p1_fill{,_priced}.csv`, `delta_weekly.csv`.
