# PREREG — cell 1,700i: "is now the time" gates and weak-year repairs for the momentum book (FROZEN 2026-10-02 12:50 UTC)

Owner 10/2: "how can we know if now is the right time to continue with the strat? is there a way to improve the
losing/weak years?" Book = the risk-adjusted top 20 (cell 1,700g V2: U2 universe, rank by 12-1 return ÷ 252-day
vol, weekly) — the best construction so far (CAGR 27.4 %, DD −42 %, 5/10 years) — and the plain A1 as the control.
Weak years are of two kinds: reversal years (2021: −40 pts vs SPY) and steady mega-cap years (2017, 2023: the book
rotates into high-vol names while a few mega-caps carry the index).

## Gates — "is it the time" (evaluated weekly on data through the prior Friday; OFF = cash)
* G1 the strategy's own momentum: ON when the book's trailing 126-day return > SPY's trailing 126-day return.
* G2 cross-sectional dispersion: ON when the trailing 63-day cross-sectional standard deviation of U2 monthly
  returns is above its trailing 3-year median (momentum pays in dispersed markets — Stivers & Sun).
* G3 Daniel–Moskowitz crash guard: OFF for 3 months after SPY's 252-day return turns negative AND its 63-day
  realised vol is above its 3-year median (the crash window), else ON.

## Repairs — weak years
* R1 per-name stop: a name exits to cash on a 20 % drawdown from its own entry price, re-entry only at a later
  rebalance when it re-qualifies (converts a reversal into cash).
* R2 index blend: 50 % book + 50 % SPY, rebalanced weekly (the mega-cap-year hedge; expected to halve both the
  excess and the drawdown, hit rate unchanged — stated as the expectation).
* R3 R1 + G1 (the owner's "when to continue" + the reversal repair together).

## Reads (2017-01..2026-09 from $50K; H1 2017–2021, H2 2022–2026)
Per cell (2 books × {none, G1, G2, G3, R1, R2, R3} = 14 cells): by-year table with $ vs SPY, years beating SPY /10,
excess by half, alpha t, Sharpe, max DD (book, SPY), worst year, weeks in cash, switches, the hit-rate null (300
draws), and the trailing-12-month shortfall tripwire read: how many times the book trailed SPY by > 25 points over
a trailing 12 months (2021's signature) and what pausing for 6 months after each would have done.

## Pass bar
As 1,700g: ≥ 7 of 10 years beating SPY + excess > 0 both halves + max DD ≤ 1.25 × SPY + hit-rate null ≥ 95 %.
Secondary read (reported, not a pass): rolling 5-year windows beating SPY (share of the 66 monthly-start windows).
Multiplicity 14 cells, stated; nothing tuned after seeing numbers.

## Output
`1700i_gates.py` (reuse 1700g_vol.py / 1700e_regime.py), `1700i_cells.csv`, `1700i_by_year.csv`,
`RESULT_1700i.md` (≤ 90 lines). The agent returns ≤ 150 words.
