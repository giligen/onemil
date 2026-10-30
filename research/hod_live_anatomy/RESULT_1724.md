# RESULT cell 1,724 - time-aware exits on the live HOD paper book

Script `cell_1724.py`; PREREG_1724.md incl. Amendment 1. n = 231 fills, 23 trade dates, Alpaca 1-min SIP bars (fill minute -> 16:00 ET). Mean R realized: -0.026; live replay: +0.018.

## Coverage

* symbol-days 230; with < 90 % of expected minutes: 96
* minutes present / expected (fill+1 .. 15:55): 82.5%
* missing-minute share, winners (realized R>0, n 89) 13.2% vs losers (n 142) 19.4%; gap -6.25 pp
* gate (cov >= 80 % and |gap| <= 5 pp): **VOID**

**All variants VOID per the PREREG coverage gate.**

## Live-rule replay vs ledger

* exit-reason match: 95.7%; |replay R - realized R| <= 0.15: 81.4% (trusted bar 80 %: PASS)
* crosstab ledger (rows) x replay (cols):

```
live_kind    eod  stop  target
ledger_kind                   
eod           48     0       5
stop           1   119       4
target         0     0      54
```

## Variants (paired Δ = variant R - live-replay R, per fill)

| variant | n | touched | Δ mean | t | Δ ex-top5% | Δ capped(+2R) | A | B | 95% CI | MDE(2se) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 time stop 14:00 | 231 | 67 | -0.009 | -0.34 | -0.071 | -0.009 | -0.058 (n 121) | +0.044 (n 110) | [-0.063, +0.044] | 0.055 | VOID |
| V2 take-profit 14:00 | 231 | 52 | -0.006 | -0.23 | -0.065 | -0.006 | -0.042 (n 121) | +0.033 (n 110) | [-0.055, +0.043] | 0.050 | VOID |
| V3 protect BE 14:00 | 231 | 10 | +0.001 | 0.09 | -0.012 | +0.001 | -0.008 (n 121) | +0.011 (n 110) | [-0.021, +0.023] | 0.023 | VOID |
| V4 1.5R target | 231 | 81 | -0.021 | -0.56 | -0.133 | -0.019 | -0.065 (n 121) | +0.029 (n 110) | [-0.093, +0.052] | 0.074 | VOID |
| V5 decay 12:00->0.5R | 231 | 62 | -0.027 | -1.35 | -0.069 | -0.025 | -0.063 (n 121) | +0.013 (n 110) | [-0.066, +0.012] | 0.040 | VOID |
| V6 decay 11:00->0.25R | 231 | 84 | -0.012 | -0.42 | -0.085 | -0.010 | -0.050 (n 121) | +0.030 (n 110) | [-0.067, +0.043] | 0.056 | VOID |

Positions open at 14:00 in the replay (entered before 14:00, not yet exited): 67 of 231; of those with open P&L >= 0 R: 52, >= +0.5 R: 32.
Pass bar: Δ >= +0.10 R in BOTH halves, pooled t >= 2, Δ ex-top-5 % > 0, coverage gate met. 'touched' counts fills whose variant R differs from the live replay (nonzero Δ).

**No variant passes.** Δ with CI and the MDE at this n are above; no parameter sweeps.

## Caveats on this implementation

* Minute bars cannot order events inside a bar: only bars STRICTLY AFTER the fill minute are walked (same-minute stops/targets are missed); within a bar the stop is checked before the target (conservative).
* Stop fill = min(bar open, stop) x (1 - 12 bps) (gap-through at the open is an addition to the frozen 'stop - 12 bps'); target fills at the target price even when the bar gaps above it.
* EOD exit = close of the 15:55 bar with no extra cost; the ledger's real EOD fills (incl. 4 `eod_ops_ah` after-hours exits) carry their own slippage, so replay-vs-ledger R differs there.
* Missing minute bars mean no SIP trade that minute (thin names), not necessarily missing data; a stop/target touch cannot be seen in a minute with no trade, and the 14:00 open-P&L uses the last close before 14:00.
* 14:00 decision = last closed bar is the 13:59 bar; exit at the open of the first bar at/after 14:00, minus 12 bps. 'Open at 14:00' is taken from the replayed live rule, not the ledger.
* V4/V5/V6 target level = entry_price (the intended level, same basis as the live target) + R x risk_per_share, so V5/V6 equal the live target while R_t = 2.0; the PREREG text says 'fill + target x R' - using the actual fill would add entry-slip noise to Δ. Decaying target uses the target at the bar's START minute (conservative, causal).
* R = (exit - fill_used)/risk_per_share, risk_per_share = risk_eff/filled_qty; fill_used = actual fill, else entry_price (3 rows). Entry minute for rows with no filled_at = created_at minute.
* One same-name-same-day duplicate shares one bar file fetched from the earliest fill minute.
* Paired t is iid over fills; fills on the same day are correlated, so t is optimistic. 6 variants scored, no multiplicity correction.
* No independent rebuild was run inside this cell's budget; replay agreement with the ledger is the only external check.
