# PREREG 1,744 — no-stop hold to the close on the CHEAP names (synthesis D1: the close-hold gross vs the round trip)

FROZEN 2026-11-13 08:05 UTC before any number. Programme count on the HOD line: 1,743 → 1,744. Rules: no git /
service / orders; read-only everywhere; no DB, no network (disk files only); write only `research/hod_entry/1744_*`
and `RESULT_1744.md`; `nice -n 19`; scorer as a script file; never Read a file > 300 lines in full; budget ≤ 40
calls. This task IS the owner's request ("deduce new strats" → `SYNTHESIS_20261113.md` D1); do not pivot. TEST
(split == 'TEST') is never read.

## Why
1,743: the post-break path mean-reverts under a stop but drifts to +0.17 / +0.14 R gross by 15:55. 1,735 V1 (no stop,
to the close) netted −0.09 R on the whole book: the round trip (chase + 15:55 spread) ≈ 0.2 R on the median name.
1,741/1,742: the cheapest-spread tercile pays about half of that and optionable-with-print names lose least. The
untested combination is cost targeting on a geometry that keeps the runners.

## Data (all on disk)
`1735_per_signal.csv` (day, symbol, split, half, b0_net_R, v1_net_R = no-stop HALF-size net R, v2_net_R = no-stop +
target half-size) × `research/hod_exit_lab/signals.parquet` (half_spread, r_pct, price, R, wk) × `1741_flow.csv`
(covered, n_prints). Join on (day, symbol); report join loss. Full-size no-stop net R = `v1_net_R × 2` (V1F); the
half-size scaling was a variance choice, the per-signal R is what the pass bar reads. Same for V2F = `v2_net_R × 2`.

## Reads (TRAIN H1, H2, TRAIN, VAL; cuts from TRAIN)
Primary (pre-named): V1F on the cheapest half_spread tercile (half_spread in bps of price, tercile cuts from TRAIN).
Exploratory (4 more reads, noted): V1F cheapest tercile ∩ covered-with-print (n_prints > 0); V1F cheapest tercile ∩
r_pct ≥ 2 %; V2F cheapest tercile; V1F middle tercile (the dose-response check — if cheap ≈ middle the "cost" story is
wrong). For every read: n, mean net R, day-clustered t, iid t, ex-top-5 %, ex-top-1 %, winner-capped at +3 R, stop
share (none by construction — report the share of V1F < −1 R and < −2 R, the uncapped downside), green-week share,
weekly P10, max drawdown in R, fills/week, and the capped book (8/day, 4 concurrent, first-come, by entry minute).
Beside each: B0 net R of the same rows (so the lift vs the live rule is visible) and the count-matched green-week
null (300 draws of the same n from the whole split).

## Pass bar (pre-committed, primary only)
V1F cheapest tercile: mean net R ≥ +0.10 with day-t ≥ 2.0 on TRAIN AND VAL; both TRAIN halves same sign; ex-top-5 %
≥ 0 on VAL; weekly P10 ≥ −2 R; ≥ 3 fills/week; capped book ≥ +0.05 R on VAL. Pass → TEST read ONCE with the identical
code, then an independent rebuild from this prose before the owner sees the number, then a PAPER mode spec (no-stop
half-size on the cheap tercile) — never live from a cell. Fail → stop; the HOD line is closed at the engine level
(synthesis "order and rules"), with the MDE stated.

## Output
`1744_score.py`, `1744_rows.csv` (one row per signal with the features and V1F/V2F/B0), `1744_summary.csv`,
`RESULT_1744.md` ≤ 350 words. Return ≤ 120 words: tercile cuts, primary mean R / day-t / n on TRAIN and VAL,
ex-top-5 %, capped book, the middle-tercile dose check, PASS/FAIL, TEST if read.
