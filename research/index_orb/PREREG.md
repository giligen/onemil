# PREREG + SPEC — Index opening-range breakout (QQQ / SPY), long and short. Cells 1,329–1,346

Owner 2026-09-21: small-cap ORB tops out at ~$2–5K/mo (fills × edge × thin-name cap); the north star needs a
line with near-zero cost, unlimited capacity and a fill nearly every day. Reference: Zarattini & Aziz (2023),
5-minute ORB on QQQ, held to close. This file is the frozen spec; the implementing agent reads ONLY this file.

## Data
Alpaca historical 1-minute bars (`feed=sip`, raw, not adjusted) for QQQ and SPY, 2016-01-01 .. 2026-05-31,
regular session bars only (09:30–15:59 ET, timestamps converted to America/New_York). Cache as parquet under
`research/index_orb/cache/<SYM>_1min.parquet` (fetch once, in monthly requests, resume if partial). Count the
session days per year and report any day with fewer than 380 bars (excluded from the walk, listed).
Use the repo's Alpaca client (`config.Config` for keys, `data_sources.alpaca_client.AlpacaClient`) or the
`alpaca-py` SDK directly with the same keys; never the EQUS.MINI feed.

## Rules (all frozen; no other variant may be scored)
* Range window W ∈ {5, 15, 30} minutes from 09:30: range_high / range_low = max high / min low of those bars.
* Signal (after the window closes): the FIRST 1-minute bar whose CLOSE is above range_high → long; whose close is
  below range_low → short. One trade per instrument per day, first signal wins; no signal after 15:00 ET.
* Entry = the OPEN of the bar after the signal bar (obtainable; never the level itself).
* Stop = the opposite side of the range (long: range_low; short: range_high). Stop fills at the stop price on the
  first later bar whose low ≤ stop (long) / high ≥ stop (short); if that bar OPENS beyond the stop, fill at its open
  (gap-through). Exit otherwise at the OPEN of the 15:55 bar.
* R = |entry − stop|. Trade P&L in R = (exit − entry) / R for long, (entry − exit) / R for short.
* Cost: 1 bp of price per side (2 bp round trip), charged on top; report gross and net. Also report net at 3 bp
  per side as a sensitivity line.
* No filters, no targets, no trailing, no sizing rules in this pass.

## Cells and splits
Cells: {QQQ, SPY} × {5, 15, 30} × {long-only, short-only, both} = 18 (1,329–1,346).
TRAIN = 2016-01-01..2024-12-31, halves 2016–2019 and 2020–2024. VAL = 2025-01-01..2026-05-31. TEST ≥ 2026-06-01
is NOT fetched.

## Report (per cell, per split) — `research/index_orb/REPORT.md`, plus `trades_<SYM>_W<W>.csv`
n trades, trades/week, mean R gross and net (1 bp and 3 bp), sd, iid t and day-clustered t (days are the
clusters; with one trade/day they coincide — say so), win rate, ex-top-5 % net mean R, net mean R with winners
capped at +5 R, year-by-year net mean R and total R (TRAIN years each), TRAIN halves, weekly net R series →
`scripts/cadence_bar.py --trades <csv> --split TRAIN|VAL` block (columns date, pnl_R), max drawdown in R, longest
losing streak in weeks. Also the share of trades stopped, held to close, and gap-through stops.

## Pass bar (pre-committed; a cell must clear all)
1. VAL net mean R (1 bp) ≥ +0.08 with day-clustered t ≥ 2.0;  2. TRAIN net mean R ≥ +0.08 and both halves > 0;
3. net positive in ≥ 6 of the 9 TRAIN years;  4. ex-top-5 % and capped-at-5R net mean R > 0 on both splits;
5. cadence bar C3/C4/C5 pass on VAL (C5 is automatic at ~1 trade/day);  6. the 3 bp line still > 0 on VAL.
Any pass → independent rebuild (Haiku, prose spec) before anything is claimed; then a dry-run PREREG.
No pass → report the MDE per cell (sd / √n × 2) and the year table; a null here is a claim about this test.

## Multiplicity
18 cells here; programme count 1,346 after this pass. No cell may be added after seeing results.
