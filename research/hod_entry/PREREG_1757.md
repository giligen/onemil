# PREREG 1,757 — three FLOW reads from data on disk (book F1 late-day MOC momentum, F2 overnight drift, F4 breadth → index)

FROZEN 2026-11-13 19:25 UTC before any number. Programme cell count: 1,756 → 1,757 (three reads, each its own kill).
Rules: write only `research/hod_entry/1757_*`, `RESULT_1757.md`; no git/service/orders/network; data on disk only;
`ulimit -v 2500000; nice -n 19`, peak RSS < 1 GB; budget ≤ 50 calls. TEST rows dropped at load (count). VAL once
per read after the TRAIN table is written.

## Data
`research/hod_exit_lab/paths.parquet` (per arm 1-min bars from the break to the close, with spy_close),
`signals.parquet`, `b0_trades.csv`, `research/hod_entry/1735_per_signal.csv`; daily bars on disk
(`research/signal_search/1748_cache` / `1746_cache`, loader as 1,754 — RAW prices for the overnight read).

## F1 — late-day MOC momentum (parent: survey; attention names drift into the close)
Among the arms (one per symbol-day), state at 15:30 ET: within 1 % of the day's high (from the path) and the day's
volume share ≥ its path median (proxy for RVOL — say so). Trade: buy at the 15:30 bar close + half_spread + 2 bp,
sell at the 15:59 close (MOC proxy, no spread). Compare with arms NOT near the high at 15:30. Numbers: n/wk, mean
bp, day-clustered t, ex-top-5 %, year signs, TRAIN halves, VAL. Kill: VAL net ≤ +20 bp or t < 2; halves opposite;
ex-top-5 % ≤ 0; < 3/wk.

## F2 — overnight drift (parent: Lou–Polk–Skouras tug of war, retail attention names)
Among the arms that closed in the top third of the day's range (winner-like), buy the 15:59 close (MOC, no
spread), sell the next day's OPEN (MOO, no spread); cost 2 bp per leg. Comparator: the same for arms in the bottom
third; and the universe overnight mean on the same days. Numbers as F1. Kill: VAL excess ≤ +20 bp or t < 2;
halves opposite; ex-top-5 % ≤ 0; 2022 negative; < 3/wk.

## F4 — breadth of micro-breaks → the index (parent F7: 93 arms/day)
Per day: number of arms by 10:30 ET (from entry_m), standardised by its trailing 60-day mean. Outcome: SPY
10:30 → close return (spy_close in paths; where missing, the SPY daily bars on disk — say which). Reads: quintiles
of breadth vs SPY mean return and sign; a long-SPY-on-top-quintile book (cost 1 bp). Numbers: n days, mean bp per
quintile, t, TRAIN halves, VAL, monotonicity. Kill: top − bottom quintile ≤ +10 bp on VAL or t < 2; halves
opposite; not monotone on TRAIN.

## Output
`1757_score.py`, `1757_f1.csv`, `1757_f2.csv`, `1757_f4.csv`, `RESULT_1757.md` ≤ 350 words ending with "CHILDREN:"
(≥ 2, from whichever read showed structure). Return ≤ 120 words: per read the VAL headline (n/wk, mean, t,
ex-top-5 %), KILL/SURVIVE, children.
