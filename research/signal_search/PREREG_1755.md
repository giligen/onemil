# PREREG 1,755 — SHORT the gap at the open (child 1 of cell 1,754): MOO short, MOC cover, borrow and squeeze tail charged

FROZEN 2026-11-13 19:40 UTC before any number. Programme cell count: 1,754 → 1,755. Rules as PREREG_1754 (write only
`research/signal_search/1755_*`, `RESULT_1755.md`; no git/service/orders/network; `ulimit -v 2500000; nice -n 19`,
peak RSS < 1 GB; daily bars on disk; budget ≤ 50 calls). TEST 2024-01 → sealed. Parent: 1,754 found the long
gapper book −185 bp open-to-close on VAL (t −4.5), every year negative. Prior (survey rule): the "gap fade" in small
caps is a published retail/academic effect; the open question is whether it survives borrow, the squeeze tail and
a hard stop.

## Signal (daily bars only; the same events as 1,754, sign flipped)
Universe at t−1: ranks 1–2,000 by 20-day dollar volume (borrowability proxy — say so), price ≥ $5 (shortable
proxy), gap = open_t / close_{t−1} − 1 ≥ g, g ∈ {10 %, 15 %, 20 %}; attention yesterday as 1,754. Entry = SHORT at
open_t (MOO), cover = close_t (MOC). Hard stop: the day's high ≥ open × (1 + s), s ∈ {10 %, 15 %, 25 %} → covered at
open × (1 + s) + 50 bp slip (daily bars cannot time the stop; charge it at the level and say so; a pass re-reads
on minute bars). Reads: by g × s; by price tier ($5–20 / > $20); by rank tier (1–1,000 / 1,001–2,000); the
NO-STOP book beside it (what the stop costs/saves).

## Cost (charged, every read)
2 bp per leg + 0 spread (auction) AND the half-spread sensitivity (1748_cost.csv); borrow sensitivity 0 / 25 / 50 /
100 bp per day (hot small caps are HTB — the 100 bp row is the honest one for g ≥ 15 %); the stop slip above.
Squeeze tail: report the worst 1 % and 5 % days, ex-top-5 % of the WINNERS, and the mean of the 5 % worst (tail
mean) — a short book lives or dies on the tail.

## Numbers (TRAIN 2016–20 halves, VAL 2021–23 once)
n/wk, mean short return (bp, gross and net at each borrow row), day-clustered t, ex-top-5 %, tail mean, winner-
capped +20 %, monthly green share, year signs, excess vs shorting the universe on the same days, MDE.

## Kill (any one; primary = g 15 %, $5–20, s 15 %, borrow 50 bp)
VAL net ≤ +50 bp or day-t < 2; TRAIN halves opposite sign; VAL ex-top-5 % (of winners) ≤ 0; 2022 negative;
< 3 names/wk; the 5 %-worst-days tail mean worse than −8 % (uninsurable); net at borrow 100 bp ≤ 0 for g ≥ 15 %.
Survive → TEST once on PIT → independent rebuild → PAPER sleeve spec (short at the open auction on the HOD paper
key, ETB names only, hard stop, 4 names max, $150 R).

## Output
`1755_score.py`, `1755_summary.csv`, `RESULT_1755.md` ≤ 350 words ending with "CHILDREN:" (≥ 2). Return ≤ 120 words.
