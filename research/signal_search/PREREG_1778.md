# PREREG 1,778 — LONG-side gapper frequency frontier: "gap-and-go survivors" at 10:30 (owner 2026-11-14: "increase the frequency of gappers, goal ≥ 3–5 a week")

## Why this cell (facts on disk, no new number read)
* Every long entry tested on the gapper population (families A–F, 1,752–1,771) loses at measured cost; the one
  passing instrument is the SHORT fade (1776R, +145 bp net, 20–25 names/wk). The long gapper fades −185 bp open→close
  (1,754), 62–68 % of it by 10:00 and 82–92 % by 10:30 (1,767). Untested: the gappers that are STILL ABOVE THE OPEN at
  10:30 — the fade has had its window and did not happen. Mechanism: by 10:30 the opening sellers are done; a name
  holding its gap has real demand (the "gap and go" the owner trades by hand). Live frequency target ≥ 3–5/wk.
* Data: `research/signal_search/1774_events.parquet` (2,269 symbol-days = cell-1,428 population, gap ≥ 5 %, $3–30,
  prior-day vol ≥ 500 K, top-10/day by gap, TRAIN 2025-07..12 / VAL 2026-01..05; columns open/high/low/close, p1030,
  measured bid/ask at 10:00/10:30/11:00 `ask1030`, `hs1030`, `nq1030`, L1 = pre-market $vol from
  `research/hod_entry/1762_cache/premarket/<day>.parquet` 04:00–09:25 as in `1776R_build.py::load_universe`),
  minute bars for the stop variant from `research/hod_entry/1762_cache/bars/` and `research/bf_zero/bars_sip.db`
  (`?mode=ro`; Saturday, no blackout). Nothing is fetched from Alpaca.

## Frozen definitions
* Event: a 1,774 row with `ask1030` present and `nq1030 ≥ 1`, `gap_pct ≤ 200`, `open > 0`.
* SURVIVOR (primary signal): `p1030 ≥ open` (price at 10:30 at or above the 09:30 open). Non-survivors are the
  in-day placebo pool.
* Entry: BUY at 10:30 at `ask1030` + 10 bp slip (the engine's chase), fees 4 bp round trip. Exit primary: MOC at
  `close` (auction, no spread). Net bp = (close / (ask1030·1.001) − 1)·1e4 − 4.
* Secondary exit (stop variant, minute bars): stop at the 10:30 price − 5 % (sell at the first minute whose low
  ≤ stop, at stop − 35 bp measured slip, cell 1,443), else MOC. Report both; the primary is the MOC leg.
* Cuts (pre-committed, all reported, none chosen post hoc): ALL survivors; survivors × L1 tercile (TRAIN cutoffs);
  survivors × `p1030 ≥ open·1.05` (holding ≥ 5 % above the open); survivors × tier ($3–5 / 5–20 / 20–30);
  survivors × above VWAP at 10:30 (VWAP from minute bars 09:30–10:30; skip the cut if coverage < 80 %).
* Frequency: names/day median and mean, names/week p10/median, share of weeks ≥ 3 and ≥ 5 — per cut, per split.

## Reads (one VAL read per cut, TRAIN halves first)
n, mean net bp, median, win rate, day-clustered t, ex-top-5 %, worst-5 % mean, weekly P10 in R (R = 1 unit =
NOTIONAL × 5 %), green-week share, TRAIN H1/H2 same sign.

## Placebos (binding 11/13 rule — every named cut carries its count-matched placebo p95)
(i) In-day: the NON-survivors on the same days, same entry/exit pricing, 200 count-matched draws → mean, p95.
(ii) Same-name: the same symbol on its other population days within ±5 days (any survivor status), same pricing
    at its own measured `ask1030` (only rows that have it), 200 draws → p95.

## Pass bar (primary = ALL survivors, MOC exit, VAL)
net ≥ +50 bp, day-t ≥ 2, ex-top-5 % > 0, worst-5 % ≥ −12 %, weekly P10 ≥ −2 R, both TRAIN halves same sign,
beats BOTH placebo p95s, ≥ 3 names/week median at the cut. A sub-cut may pass only if the ALL line is ≥ 0 and the
cut was listed above. PASS → independent prose-only rebuild before the owner sees the number; FAIL → the frequency
table still goes to the owner (that is the deliverable he asked for).

## Output
`1778_score.py`, `1778_events.parquet` (one row per event with every field used), `1778_summary.csv` (cut × split ×
read), `1778_placebo.csv`, `RESULT_1778.md` ≤ 120 lines: frequency table FIRST (names/wk by cut), then the reads,
placebos, caveats (quote coverage, bars coverage, the top-10/day construction = the quote-fetched subset, not the
whole population), then "CHILDREN:" ≥ 2 hypotheses from the winners-vs-losers anatomy.

## Rules for the agent
Sonnet, ≤ 40 calls, docstrings on every function, verbose progress, writes only `research/signal_search/1778_*`
and `RESULT_1778.md`. Read-only everywhere else (`?mode=ro` on every sqlite). No git, no services, no orders, no
Alpaca calls. Return ≤ 150 words: frequency line, primary VAL read with both placebo p95s, pass/fail, paths.
