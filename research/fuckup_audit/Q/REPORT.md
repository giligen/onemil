# Q — obtainability and fill-realism audit of the QQQ noise-area sleeve

Stage Q of `research/fuckup_audit/PLAN.md`. Subject: `research/lit_review_2026/RESULTS.md` rows 1 / 1b, the
Zarattini-type intraday noise-area rule, the one thing in this program that was still positive out of sample.
Run 2026-09-17. Everything here is RESEARCH: no config, no service, no order was touched.

Scripts: `Q/zsim.py` (re-implementation), `Q/step1_rederive.py`, `Q/step2_live.py`, `Q/step5_extras.py`,
`Q/loader_check.py`. Data: `research/lit_review_2026/etf_1min.db` (Alpaca SIP 1-min, SPY/QQQ/TQQQ,
2016-01-04 -> 2026-09-15), read-only. Tables: `Q/step*.csv`, `Q/step1_rederive.md`, `Q/step5_extras.md`.

---

## 0. One page

**Does the OOS number survive the live-convention fill? Yes — the fill was never the problem.**
Replacing the paper's fill (the close of the check bar, which is a <= 60-second look-ahead) with the live
convention (decide on the closed bar, market order, fill at the NEXT bar's open) moves QQQ OOS from
**5.84 -> 5.83 bps per traded day** (1x). The next-bar open differs from the check-bar close on 82% of legs but
the difference is unbiased: **median 0.0 bp, mean +0.003 bp adverse, p90 0.53 bp**. Every fill in both
conventions is inside the bar that fills it. At a 30-minute decision cadence, execution timing does not matter.

**What the number does not survive is everything else.**

| test | result |
|---|---|
| cost | at 0.5 bp/leg OOS = 4.57 bps/traded day (t 1.27); at 1.0 bp/leg = 3.06 (t 0.85); **breakeven = 2.01 bp/leg** |
| power | OOS daily sd 53.5 bps over 678 days -> the smallest effect detectable at 80% power is **5.75 bps/day**; the observed effect is **2.60 bps/calendar day**. The OOS window cannot see an effect of the size it reports |
| tail | the **top 5 days are 102% of the whole OOS return**; removing the top 1% of days -> -0.53 bps/day (t -0.35); capping a day at +1% -> +0.03 bps/day |
| concentration | one month, **2025-04, is 51% of the 33-month total**; one day, 2025-04-09, is 45% |
| band multiplier | VM 0.8 -> OOS 0.80 bps (t 0.23); VM 1.0 -> 4.57 (t 1.27); VM 1.2 -> 6.52 (t 1.67). The chosen 1.0 is not a peak, but 0.8 is dead |
| by year (OOS, 1x, live fill + 0.5 bp) | 2024 +6.94 bps (t 1.45), 2025 +2.89 (t 0.37), 2026 YTD +3.45 (t 0.64) — all three positive, none significant |

**The honest $/month at $60K, OOS 2024-01 -> 2026-09, live-convention fills at 0.5 bp/leg:**

| | QQQ 1x | QQQ 3x notional | TQQQ 1x notional |
|---|---|---|---|
| mean $/month | **+$321** | +$962 | +$1,092 |
| median $/month | +$241 | +$723 | +$1,017 |
| **worst month** | **-$2,243** (2026-01) | -$6,729 | -$7,036 |
| best month | +$5,348 (2025-04) | +$16,044 | +$16,216 |
| green months | 22 / 33 | 22 / 33 | 22 / 33 |
| max drawdown | $6,369 (10.6%) | $17,573 (29.3%) | $15,845 (26.4%) |
| worst day | -$971 (2024-03-14) | -$2,913 | -$2,800 (2024-08-06) |

So: **~$300/month at $60K unlevered, against a -$2,200 worst month and a -$6,400 drawdown, with half of the
whole result in one month.** At 3x notional it is ~$960/month against a -$6,700 worst month. That is the
honest shape. It is not "no edge" — all three OOS years are positive and the in-sample anchor is real
(t 3.3, SR 1.18) — it is an effect this window cannot resolve, carried by a handful of crash days.

**Leverage finding.** TQQQ 1x beats QQQ 3x on both axes OOS (21.2%/yr, MDD 26.4% vs 18.0%/yr, MDD 29.3%). The
daily-reset decay that ruins buy-and-hold TQQQ does not bite a book that is flat at every close, and the 3x
QQQ position costs zero margin interest for the same reason (intraday only = day-trading buying power, not an
overnight debit). The three-way comparison is therefore about drawdown tolerance, not financing.

**Recommendation: do not enable anything.** If the owner wants it in front of him, the next step is a
**dry run against `Q/DRYRUN_SPEC.md`** whose only purpose is to measure execution (slippage vs the next-bar
open, decision parity, latency). A dry run cannot validate the edge: over 20 sessions the standard error is
12 bps/day against an effect of 2.6.

---

## 1. Re-derivation (step 1) — reproduces exactly

`Q/zsim.py` was written from the prose spec and then diffed against `test_zarattini_spy.py`
**trade by trade**: SPY 2,443 trades, QQQ 2,362 trades, `max|diff| = 0` on (day, side, entry, exit). The two
loaders were separately checksum-compared field by field (`Q/loader_check.py`): identical on C, O, VW, dopen,
prevclose, sig14, sigma, first, last.

| | RESULTS.md row 1b | this run |
|---|---|---|
| QQQ IS 2016-2023, paper costs | 12.8 bps/day dyn, t 3.7 | 12.77, t 3.73 |
| QQQ OOS 2024->, paper costs | 10.1 bps/day dyn, t 1.6, SR 0.99 (1x), 8.4%/yr | 10.14, t 1.59, SR 0.987, 8.35%/yr |
| SPY IS / OOS | 10.9 / 0.6 bps | 10.89 / 0.58 |

Caveat on the independent-check rule: this is a re-implementation by the same model that had read the original
code, not a blind rebuild from prose by a second agent. It catches transcription errors (and the trade-level
equality says there are none); it does **not** discharge CLAUDE.md "independent reimplementation". What
follows below — obtainability, causality, price scale, tails, power — is the part that catches spec errors,
and that is where this rule is weak.

## 2. The rule, as prose, with every field's construction time

Times are ET; `k` = minute index from 09:30 (k = 30 is the 10:00 bar).

1. **Day open** `O0` = the open of today's 09:30 bar. Known 09:31.
2. **Prev close** `Cp` = the previous session's 15:59 regular-hours bar close. Known before the open.
3. **Time-of-day noise** `sigma[k]` = mean over the **previous 14 sessions** (>= 10 present) of
   `abs(close(k)/open_0930 - 1)` for that same clock minute. Uses only sessions strictly before today
   (`rolling(14).mean().shift(1)`). Known 09:29.
4. **Bands** `UB[k] = max(O0, Cp)*(1 + VM*sigma[k])`, `LB[k] = min(O0, Cp)*(1 - VM*sigma[k])`, VM = 1. Known
   09:31, constant thereafter for the whole day (sigma[k] varies by clock minute, not by today's tape).
5. **VWAP[k]** = cumulative sum(vw*v)/sum(v) over regular-hours bars 09:30..k. Causal by construction.
6. **Decisions only at k in {30, 60, ..., 360}** = 10:00, 10:30 ... 15:30. The signal is the **close of that
   bar**. Flat -> long if close > UB[k], short if close < LB[k].
7. **Stop, checked only at the same 12 minutes**: long exits if close < max(UB[k], VWAP[k]); short exits if
   close > min(LB[k], VWAP[k]). On a stop the opposite side may be entered at the same check; the side just
   stopped may not.
8. **Flat at the close** (paper: the 15:59 bar's close).
9. **Sizing**: 1x = equity/O0 shares. The paper's dynamic sizing is `min(4, 2%/sig14)` where `sig14` is the
   std of the 14 daily close-to-close returns ending yesterday. Known 09:29.

**Causality trace: clean.** Every field above is computable before the decision it feeds. Two residual
non-causal details, both immaterial and both named here rather than buried:
(a) the loader drops days with < 150 bars (2 of 2,688 for QQQ) using the whole day's bar count — an end-of-day
universe fact; (b) the simulator reads the day's last bar index to stop the check loop, which live comes from
the market calendar (the HOD engine already does this; the spec requires it, fail-closed).

## 3. Obtainability / fill realism (step 2)

**Where the paper and the script fill: at the CLOSE of the crossing bar — not at the band level.** Since the
signal is "the close is beyond the band", that fill is already *worse* than the band by the overshoot
(QQQ entries: median 12.9 bp beyond the band OOS, mean 18.9, p90 44). So the paper's convention is not
optimistic about *price*; it is optimistic about *time*, by up to 60 seconds — the close of a bar is not
knowable until the bar has closed.

**Live convention re-simulated:** decide on the closed bar k, market order, fill at the open of bar k+1 with a
0.5 bp (and 1.0 bp) half-spread charged on each leg; exits the same way; the 15:59 flat modelled both as the
MOC/close print and as the 15:59 bar's open.

| symbol | scenario | period | bps/traded day 1x | t | ann % 1x | SR 1x | MDD % 1x | trades/day |
|---|---|---|---|---|---|---|---|---|
| QQQ | A paper faithful | IS | 7.38 | 3.79 | 11.36 | 1.34 | 9.8 | 0.89 |
| QQQ | A paper faithful | OOS | 5.84 | 1.62 | 8.35 | 0.99 | 9.1 | 0.86 |
| QQQ | B live fill, paper $/share | IS | 7.29 | 3.75 | 11.20 | 1.33 | 9.6 | 0.89 |
| QQQ | B live fill, paper $/share | OOS | 5.83 | 1.62 | 8.34 | 0.99 | 9.3 | 0.86 |
| QQQ | **C live fill, 0.5 bp/leg** | IS | 6.48 | 3.32 | 9.85 | 1.18 | 9.9 | 0.89 |
| QQQ | **C live fill, 0.5 bp/leg** | **OOS** | **4.57** | **1.27** | **6.40** | **0.77** | **10.6** | 0.86 |
| QQQ | D live fill, 1.0 bp/leg | IS | 4.99 | 2.55 | 7.42 | 0.91 | 11.0 | 0.89 |
| QQQ | D live fill, 1.0 bp/leg | OOS | 3.06 | 0.85 | 4.11 | 0.52 | 12.2 | 0.86 |
| QQQ | E = C, flat at the 15:59 OPEN | OOS | 4.62 | 1.29 | 6.48 | 0.79 | 9.9 | 0.86 |
| QQQ | F = D, flat at the 15:59 OPEN | OOS | 3.11 | 0.86 | 4.19 | 0.53 | 11.5 | 0.86 |
| SPY | C live fill, 0.5 bp/leg | IS | 3.60 | 2.23 | 5.35 | 0.79 | 13.0 | 0.90 |
| SPY | C live fill, 0.5 bp/leg | OOS | -1.86 | -0.81 | -2.99 | -0.50 | 14.1 | 0.94 |

Full grid incl. the dynamic-sizing columns: `Q/step2_scenarios.csv`.

**Fill-difference diagnostics (`Q/step2_fillgap.csv`)**

| symbol | period | legs | % where next-open != close | mean adverse bp | median | p90 |
|---|---|---|---|---|---|---|
| QQQ | IS | 2,956 | 73.8 | +0.037 | 0.000 | 0.73 |
| QQQ | OOS | 962 | 81.9 | +0.003 | 0.000 | 0.53 |
| SPY | OOS | 1,078 | 78.8 | +0.011 | 0.000 | 0.33 |

**Obtainability statement.** Both conventions fill inside the bar that fills them (verified: 2,362/2,362 QQQ
entries have `low <= fill <= high`). The next-open fill is reachable by an order the engine would actually
have sent (market order immediately after the bar closes). The close fill is **not** reachable — but it costs
the book nothing, because the two prices are the same to within a median of 0 bp.

**A resting limit at the band is NOT the better fill here** (unlike H2 in the main plan, where it is worth
+0.022R). The overshoot is 12.9 bp per entry, which at 0.86 trades/day would be ~11 bps/day — larger than the
entire edge — but it is unharvestable: a limit resting at the band fills whenever price crosses, i.e. at all
the minutes the rule deliberately does not trade, which turns the rule into the every-minute variant that is
already known dead (SR 0.10 IS, `zarattini_spy.md` ablation C). The semi-hourly cadence and the cheap fill are
the same choice; you cannot have both.

**P&L by clock window (`Q/step2_attribution.csv`, scenario C, share of total return)**

| symbol | period | 09:30-10:00 | 10:00-10:30 | 10:30-15:30 | 15:30-16:00 |
|---|---|---|---|---|---|
| QQQ | IS | **0.0%** | -2.4% | 92.7% | +9.8% |
| QQQ | OOS | **0.0%** | +1.6% | 118.9% | **-20.5%** |
| SPY | OOS | 0.0% | +27.1% | +8.3% | -64.6% |

The first 30 minutes contribute exactly nothing **by construction** — the rule's first decision is at 10:00, so
the opening window is structurally untradeable for it. The last 30 minutes were +10% of IS P&L and are
**-20% of OOS P&L**: holding into the close has been a drag since 2024. Flattening at 15:30 instead was tested
(report-only, one more cell): OOS 4.49 bps vs 4.57 — no help, because the loss is in the *position*, not the
exit minute.

## 4. Leverage (step 3)

All rows: live-convention fills, 0.5 bp/leg (TQQQ also shown at 1.0 bp/leg — its quoted spread in bp is wider),
1x of the stated notional, OOS 2024-01 -> 2026-09-15 (678 sessions, 386 traded).

| book | bps/traded day | t | ann % | SR | MDD % | MDD $ at $60K |
|---|---|---|---|---|---|---|
| QQQ 1x notional | 4.57 | 1.27 | 6.40 | 0.77 | 10.6 | $6,369 |
| QQQ 3x notional (intraday, no interest) | 13.71 | 1.27 | 18.05 | 0.77 | 29.3 | $17,573 |
| TQQQ 1x notional (0.5 bp/leg) | 15.56 | 1.43 | 21.18 | 0.87 | 26.4 | $15,845 |
| TQQQ 1x notional (1.0 bp/leg) | 14.07 | 1.29 | 18.61 | 0.79 | 27.7 | $16,611 |

IS for the same rows: QQQ 1x 6.48 bps / MDD 9.9%; QQQ 3x 19.44 / 28.0%; TQQQ 1x 19.32 / **42.9%**.

- **TQQQ's daily-reset drag does not apply.** The book is flat at every close, so there is no multi-day
  compounding path for the reset to erode; the 3x is realised within the session. TQQQ 1x OOS beats QQQ 3x
  on return (21.2 vs 18.0%/yr) and on drawdown (26.4 vs 29.3%) — its own bands adapt to its own volatility.
  In-sample TQQQ is the worse vehicle (MDD 42.9% vs 28.0%), which is the honest caveat.
- **Margin interest is zero either way.** A QQQ 3x position opened and closed the same session uses
  day-trading buying power (4x equity for a PDT account >= $25K; $60K -> $240K), not an overnight debit. No
  interest accrues. TQQQ 1x needs no margin at all ($60K position on $60K equity).
- **Charge TQQQ its own spread.** TQQQ quotes ~1 cent on ~$50-100, i.e. ~1-2 bp full / 0.5-1 bp half; both are
  shown. QQQ quotes ~1 cent on ~$600 = 0.17 bp full, so the 0.5 bp/leg charged here is ~6x its quoted half
  spread — deliberately conservative.

**Monthly $ at $60K equity, OOS (33 months, `Q/step3_months.csv`)**

| book | mean | median | worst | best | green |
|---|---|---|---|---|---|
| QQQ 1x | +$321 | +$241 | -$2,243 (2026-01) | +$5,348 (2025-04) | 22/33 |
| QQQ 3x notional | +$962 | +$723 | -$6,729 | +$16,044 | 22/33 |
| TQQQ 1x (0.5 bp) | +$1,092 | +$1,017 | -$7,036 | +$16,216 | 22/33 |

2025-04 alone is 51% of the QQQ 1x total; 2025-04-09 alone (+7.97% in one session, the tariff-pause reversal)
is 45%.

**Caveat on the TQQQ series.** `etf_1min.db` holds RAW bars and TQQQ split five times inside the sample
(2017-01-12 2:1, 2018-05-24 3:1, 2021-01-21 2:1, 2022-01-13 2:1, **2025-11-20 2:1 — inside the OOS window**).
Raw, the prev-close band anchor is 2-3x wrong on those days and the daily-vol series carries a phantom -50%
return for 14 sessions. `zsim.load_symbol` now back-adjusts them; the factors were **derived, not assumed**:
`factor = (TQQQ open/prev_close) / (1 + 3 * QQQ gap)` lands on 0.4996-0.5000 and 0.3333 for the five, and on
1.0225 for 2020-03-16 — which is therefore the COVID crash gap, not a split, and is left alone. Un-adjusted,
the TQQQ OOS row read 16.86 bps instead of 15.56.

## 5. Tail, robustness, power (step 4 + step 5)

**Cost curve (QQQ, live fill; `Q/step5_costcurve.csv`)**

| cost/leg (bp) | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|
| IS bps/traded day | 7.97 | 7.22 | 6.48 | 5.73 | 4.99 | 3.50 | 2.01 | -0.97 |
| OOS bps/traded day | 6.08 | 5.33 | 4.57 | 3.81 | 3.06 | 1.54 | 0.03 | -3.00 |

**Breakeven cost: 2.67 bp/leg IS, 2.01 bp/leg OOS.** QQQ's quoted half spread is ~0.09 bp, so the sleeve has
~20x headroom on spread — the risk is not the spread, it is market-order slippage on a $180K (3x) order at
10:00, which the dry run exists to measure.

**Tail (QQQ, scenario C, 1x; `Q/step5_tails.csv`)**

| period | variant | days | bps/calendar day | t | $/month at $60K |
|---|---|---|---|---|---|
| IS | full | 2,000 | 3.86 | 3.32 | +$487 |
| IS | top 1% of days removed | 1,980 | 1.29 | 1.27 | +$162 |
| IS | top 5% of days removed | 1,900 | -4.20 | -4.99 | -$529 |
| IS | daily return capped at +1% | 2,000 | 1.01 | 1.07 | +$128 |
| OOS | full | 678 | 2.60 | 1.27 | +$328 |
| OOS | **top 1% of days removed** | 671 | **-0.53** | -0.35 | -$66 |
| OOS | top 5% of days removed | 644 | -5.13 | -4.08 | -$647 |
| OOS | bottom 5% of days removed | 644 | +7.40 | 3.74 | +$933 |
| OOS | both 5% tails removed | 610 | -0.50 | -0.49 | -$63 |
| OOS | **daily return capped at +1%** | 678 | **+0.03** | 0.02 | +$3 |
| OOS | daily return capped at +0.5% | 678 | -3.11 | -2.53 | -$391 |

Concentration: OOS **top 5 days = 102%** of the total return, top 1% of traded days = 75%, top 5% = 206%.
IS is less extreme (top 5 days = 23%, top 1% = 42%) — i.e. the in-sample edge was broad and the out-of-sample
"edge" is five days. By the PLAN section 1 tail rule this is the lottery-ticket shape the owner has already
rejected once; it is reported, not argued away. Symmetrically, the book is genuinely long the crash tail: it
is a long/short breakout rule and 2025-04-09 was a +8% day for it.

**By year (OOS, scenario C, 1x; `Q/step4_years.csv`)**

| year | days | traded | bps/traded day | t | ann % | SR | MDD % |
|---|---|---|---|---|---|---|---|
| 2024 | 252 | 146 | 6.94 | 1.45 | 10.39 | 1.44 | 4.2 |
| 2025 | 250 | 139 | 2.89 | 0.37 | 3.55 | 0.38 | 8.1 |
| 2026 YTD | 176 | 101 | 3.45 | 0.64 | 4.90 | 0.77 | 3.5 |

**Band multiplier (`Q/step4_vm.csv`, live convention)**

| VM | IS bps | IS t | OOS bps | OOS t | OOS ann % | OOS MDD % | trades/day |
|---|---|---|---|---|---|---|---|
| 0.8 | 5.83 | 3.04 | 0.80 | 0.23 | 0.89 | 14.9 | 1.01 |
| **1.0 (paper)** | 6.48 | 3.32 | 4.57 | 1.27 | 6.40 | 10.6 | 0.86 |
| 1.2 | 5.56 | 2.80 | 6.52 | 1.67 | 7.87 | 7.0 | 0.72 |

IS is flat across the three (5.6-6.5), so VM = 1 is not an in-sample peak — good. OOS is monotone increasing
in VM, which is the same direction as the paper's own FAQ "optimum" of 1.5 and as SPY ablation G. Read it as:
fewer, wider-confirmed signals have done better since 2024. Not adopted — it is 2 extra cells on a window that
cannot resolve 5.75 bps/day.

**Power.**

| period | days | daily sd | MDE @80% power | observed |
|---|---|---|---|---|
| IS 2016-2023 | 2,000 | 52.1 bps | 3.26 bps/day | 3.86 bps/day |
| **OOS 2024-2026** | 678 | 53.5 bps | **5.75 bps/day** | **2.60 bps/day** |

The in-sample result is above its detection floor; the out-of-sample result is less than half of it. The
supported statement is: **no effect was detectable in QQQ, at a 30-minute decision cadence, at 1x notional,
over 2024-01 -> 2026-09, at 0.5 bp/leg; the smallest effect this window could have seen is 5.75 bps/day
(~$725/month at $60K), and the point estimate is 2.60 bps/day (~$328/month).** "No edge exists" is not
supported and is not claimed.

## 6. What this audit did NOT establish

- **No independent blind rebuild.** Section 1's diff is a re-implementation by a model that had read the
  original. A true second-agent rebuild from section 2's prose is still owed before any go-live number.
- **No quote data.** Costs are a bp assumption, not measured NBBO. QQQ's quoted spread is ~0.09 bp half, so
  the assumption is conservative, but market-impact/slippage on the 10:00 and 15:58 market orders is the real
  unknown and only the dry run measures it.
- **No survivorship question.** SPY/QQQ/TQQQ existed throughout; there is no universe selection at all. This
  is the one book in the program with zero survivorship exposure.
- **One data vendor.** All bars are Alpaca SIP from one table; there is no daily-vs-intraday price-scale
  crossing anywhere in the rule (section 2), and the only raw-price hazard, TQQQ's splits, is handled in
  section 4.
- **2026-09 is a partial month** (10 sessions) and is included in the monthly table as such.

## 7. Cell count

| where | cells |
|---|---|
| prior program (RESULTS rows 1/1b): 7 SPY ablations + the QQQ re-run + the 3 cost models | 9 |
| this audit: 6 execution scenarios x 3 symbols | 18 |
| cost curve, 8 levels on QQQ (0.5 and 1.0 already counted) | +6 |
| band multiplier 0.8 / 1.2 (1.0 already counted) | +2 |
| report-only: flat at 15:30 | +1 |
| **total distinct rule variants looked at for this sleeve** | **36** |

Reported but not counted as selection cells (they are slices of one rule, not variants of it): the 7 tail
treatments x 2 periods, the 3 OOS years, the 33 months, the 4 clock-window attribution buckets, the 12
entry-minute buckets, the 3 leverage books (2 are linear rescalings, 1 is a symbol already counted).

At 36 cells with the best OOS t = 1.67 (VM 1.2), a multiplicity-aware reading gives nothing: the largest |t|
expected from 36 independent null cells is ~2.4.

## 8. Verdict

The sleeve is real in-sample (t 3.3, SR 1.18, broad — top 1% of days is only 42% of it) and unresolvable
out-of-sample (t 1.27, five days = the whole thing, 51% in one month). **The live fill convention is not what
threatens it**; cost above 2 bp/leg and the tail are. At $60K it is ~$300/month unlevered with a -$2,200
worst month, or ~$1,000/month at 3x notional with a -$6,700 worst month — against a program target of
30-50%/yr delivered monthly, the levered version is in range and the consistency is not.

Next step, if the owner wants one: the dry run in `Q/DRYRUN_SPEC.md` — to measure slippage, decision parity
and latency, explicitly NOT to confirm the edge (20 sessions have a standard error of 12 bps/day against an
effect of 2.6).
