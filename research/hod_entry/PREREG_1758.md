# PREREG 1,758 — two reads from disk: F3 leveraged-ETF rebalance flow into the close, I2 calendar effects on the HOD book

FROZEN 2026-11-13 19:30 UTC before any number. Programme cell count: 1,757 → 1,758. Rules as PREREG_1757 (write only
`research/hod_entry/1758_*`, `RESULT_1758.md`; data on disk; `ulimit -v 2500000; nice -n 19`; budget ≤ 40 calls;
TEST dropped at load; VAL once per read).

## F3 — leveraged-ETF rebalance (parent: survey — Cheng–Madhavan; 2x/3x funds buy winners / sell losers at the close)
SPY minute path from `paths.parquet.spy_close` (first arm's path per day gives SPY 09:3x → 15:55; days without an
arm are missing — report coverage per year; if < 80 % of trading days, VOID). Signal at 15:00 ET: r = SPY_15:00 /
SPY_open − 1 (open = the first available spy_close of the day; say the minute). Trade: if |r| ≥ k (k ∈ {0.5 %,
1.0 %, 1.5 %}) go WITH the sign from 15:00 to 15:55; cost 1 bp per leg. Numbers: n days/yr, mean bp, t, hit rate,
by sign (up vs down days), TRAIN halves (2016–18 / 2019–20 as the splits give), VAL. Comparator: the same on
days with |r| < k (should be ≈ 0). Kill: VAL mean ≤ +5 bp or t < 2 at the pre-picked k = 1.0 %; halves opposite;
the |r| < k comparator not below the trade.

## I2 — calendar (parent: the owner's "100 options"; one cheap read, multiplicity declared: 5 cuts)
On the live rule's net R per arm (`1735_per_signal.csv` B0): day of week; month-end (last 2 trading days);
opex Friday (3rd Friday); FOMC days (hard-coded list 2016–2023 from the Fed's public calendar — write the list
into the script and say so); first vs second half of the month. Numbers: n, mean net R, day-clustered t per cut,
TRAIN halves then VAL. Kill: a cut is "structure" only if the same sign on both TRAIN halves AND VAL with |t| ≥ 2
on VAL; else noise. No trade is built from this cell; a surviving cut becomes a child with its own PREREG.

## Output
`1758_score.py`, `1758_f3.csv`, `1758_i2.csv`, `RESULT_1758.md` ≤ 350 words ending with "CHILDREN:" (≥ 2).
Return ≤ 120 words: F3 coverage, VAL mean/t at k 1 %, by sign, KILL/SURVIVE; I2 any cut that passes, else "noise".
