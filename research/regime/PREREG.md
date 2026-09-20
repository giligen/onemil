# PREREG: ORB per-regime multipliers, rule-regime vs HMM-regime

Cumulative cell count on this research line: 1,310 (rule-regime path), 1,311 (HMM-regime path).

## Question
Does slicing the honest ORB B+ book (`analysis_results/orb_bplus_book.csv`, filled rows)
by market regime and applying a 2025-fit per-state size multiplier improve the VAL
(2026-01..05) book vs the flat 1.0x book, for either regime system?

## Data
- Trades: `analysis_results/orb_bplus_book.csv`, rows with `entered==1`. PnL basis:
  `_sized_pnl` column ($, stage-sized) if present, else `pnl/15`.
- Rule regime (A/B/C1/C2): `trading/regime_helpers.classify_regime` +
  `build_regime_lookup`, fed SPY daily closes from `data/cache.db::daily_bars`
  (symbol='SPY'). Look-ahead safe by construction (day T uses T-1 features).
- HMM regime (0=calm/1=mid/2=crisis): `research/regime/hmm_labels.csv`
  (`bar_date, hmm_state`), causal labels through 2026-05-31.
- Join key: trade `date` -> regime label effective that trading day.

## Splits
- TRAIN: 2025-01-01 .. 2025-12-31 (fit multipliers here only)
- VAL: 2026-01-01 .. 2026-05-31 (apply, report)
- TEST: >= 2026-06-01 — SEALED, not touched in this study.

## Fit rule (TRAIN 2025 only, per regime system, per state)
State multiplier in {0, 0.5, 1.0, 1.5}, chosen from 2025 net R sign +
both-halves (H1 Jan-Jun / H2 Jul-Dec) agreement:
- 0 if net R negative in BOTH halves (no trade)
- 1.5 if net R positive in BOTH halves AND t >= 1.5
- 0.5 if halves disagree (mixed) AND pooled 2025 net R negative
- else 1.0 (default, includes mixed-but-pooled-positive, and low-t positive)

## Stats reported per state (TRAIN 2025)
n, net R, t-stat (iid, trade-level) and t-stat (day-clustered), split by
H1/H2 to check the both-halves-agreement rule inputs.

## Pass bar (pre-committed, applies independently to each regime system)
VAL per-regime book total $ > VAL flat-1.0 book total $, AND
VAL per-regime MDD not worse than flat, AND
at least one non-1.0 state actually exercised in VAL with n >= 10 trades
at that state.
Known: HMM state 2 (crisis) sample review — if it has 0 VAL days, say so
explicitly rather than silently reporting an empty row.

## Secondary
`scripts/cadence_bar.py --split VAL` (C1-C5) on each per-regime book
(columns date, pnl_R, symbol) — R denominator fixed to the book's own
per-trade risk-parity dollar amount (documented in REPORT.md once
inspected; if constant across trades, that constant is 1R).

## What this study cannot show
This is a TRAIN-fit-on-2025 / apply-to-VAL relative comparison on one
honest book, n<=~200 filled trades total split across up to 6-7 states.
It is not powered to detect small effects per state; a null per state is
a statement about this test's power, not proof the state carries no
information. TEST remains sealed for a future confirmation pass.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PKSsd3LjBkXYzPECLNYR5W
