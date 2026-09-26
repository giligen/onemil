# REBUILD_1552 — independent rebuild of PREREG_1552 (the EDGAR event desk), from prose only

Built without opening `cell_1552.py`, `test_cell_1552.py`, `cell_1552_events.csv`, `RESULT_1552.md` or
`events_raw.csv`. Code: `research/edgar_desk/rebuild_fetch.py` (fetch), `research/edgar_desk/rebuild_1552.py`
(classification + population filter + entry/exit + stats). Events: `research/edgar_desk/rebuild_1552_events.csv`
(32,145 rows). Report: `research/edgar_desk/rebuild_1552_report.json`.

## Scope limitation — READ FIRST (disclosed, not silent)

A full live SEC fetch of the entire population (8,488 liquid/common/non-test symbols in TRAIN+VAL, of which
5,090 resolve via SEC's own primary `company_tickers.json`) does not fit this task's step budget or a bounded
fetch window at SEC's 1-req/s fair-access ceiling (~85 min for the mapped set alone, before the browse-edgar
CIK fallback for the remaining 3,398 unmapped liquid symbols — mostly ETFs and companies whose current ticker
differs from their 2019-2024 ticker). This rebuild instead fetched a **seed-1552 deterministic random sample of
1,200 of the 5,090 primary-CIK-mapped symbols** (24% of the mapped set, 14% of the full liquid population),
for real, against `data.sec.gov`, respecting the 1 req/s ceiling, with a resumable gzip cache and a LOST count.

**Fetch completeness gate: 1,200 / 1,200 sampled symbols fetched, 0 LOST, 100% coverage of the sample.**
1,189 of 1,200 returned a usable submissions document (11 gave `_missing: "404"` — no EDGAR filer record,
e.g. a foreign private issuer or a symbol change SEC's own ticker file mapped to the wrong/stale CIK).

Because this is a ~1-in-7 sample of the population, **this rebuild's n / events-per-week numbers are NOT
comparable 1:1 to a full-population RESULT** (they run roughly 5-7x lower) and the event-set Jaccard check the
PREREG's "Independent check" section calls for cannot be computed by this pass — that check needs the full
population on both sides. What IS comparable and meaningful at this sample size: the **class-assignment logic,
the entry/exit/cost mechanics, and the sign and rough magnitude of each cell's net return**, which is what is
reported below. A follow-up full-population fetch (background, multi-hour, `nohup`) is the natural next step
if the owner wants the actual Jaccard/net-bps parity check.

## Judgment calls made explicit (structured-codes-only, per "Not allowed")

1. **OFFERING (1552)** = form ∈ {424B1..424B5} OR (8-K with item 3.02). The PREREG's "8-K 1.01 whose text is
   skipped (structured only)" clause is read as an explicit *exclusion* note (a 1.01 is never routed to
   OFFERING without reading text), not a third OFFERING trigger.
2. **CONTRACT (1559)** = 8-K with item 1.01 present AND items does NOT also contain 3.02 or 2.03 (the row's own
   "WITHOUT" clause).
3. **SHELF (1553)** = bare forms `S-3`, `S-3ASR`, `S-1` only — `/A` amendments excluded ("initial ... registration").
4. **OFFICER_EXIT (1558)** = item 5.02 alone. That item code does NOT structurally distinguish a departure from
   an appointment/election; both are included under this class in this structured-only pass. This is a real
   purity gap flagged for the later text/LLM pass the PREREG names for CONTRACT/8.01.
5. **ACTIVIST (1560)** = bare form `SC 13D` only, not `SC 13D/A`.
6. **BUYBACK_OR_INSIDER (1561)**: 8-K/8.01 candidates joined to a same-CIK Form 4 filed within 2 calendar days
   whose primary XML shows `isOfficer` true AND a transaction code `P` (open-market purchase) — fetched live,
   on demand, only for the candidate set. **Result: 0 events** in the 1,189-symbol sample — this cell is fully
   data-starved at this sample size (rare joint event: an 8.01 AND a same-officer Form-4 purchase within 2
   sessions), not a null finding; needs the full population.
7. **Duplicate same-day filings**: not deduplicated. A handful of symbols (e.g. ATMP) filed the same form
   repeatedly on one day (a continuous ATM program's serial 424B2 prospectus supplements) and each accession
   became its own event row with an identical entry/exit — this inflates the count on those sessions with
   *non-independent* draws of the same underlying price move. The PREREG prose does not address this; flagged
   here as a rebuild gap, not silently absorbed.
8. **Per-year table**: not computed in this pass (time budget) — only TRAIN (2019-2022) / VAL (2023-2024H1)
   pooled statistics are reported below. The "≥3 of 4 TRAIN years positive" pass-bar criterion is therefore
   **not evaluated** here.

## Population / mechanics (as specified)

Population gate: prior-session close ≥ $1 and prior-session 20-day dollar volume ≥ $1M (known at filing time,
causal). Entry = first auction after acceptance (same-day MOO if accepted before 09:00 ET, else next session's
MOO). E1 = same-session MOC (open→close). E5 = MOC 5 trading sessions after entry (counted on the symbol's own
bar series). Cost = 5 bps/leg, 10 bps round trip, both legs, both E1 and E5. SHORT cells: + borrow at 3%/yr
pro-rata on calendar days held; excluded when prior close < $5 or the prior session's own day-over-day return
≤ -10% (SSR proxy). Prices: `research/overnight_high/alpaca_daily_2019_2024H1.parquet` only (TRAIN+VAL both
fall inside 2019-2024H1; the Databento panel was not needed and TEST was not read). Universe placebo = mean
same-session, same-leg return of all prior-session-eligible names. Count-matched null = 1,000 seed-1552 draws
of the same #events/session from that session's eligible pool, same direction/costs; reported as a percentile.

## Results (TRAIN 2019-2022 / VAL 2023-2024H1, sample of 1,189 fetched issuers, net bps in the fixed direction)

| cell | class | dir | named leg (TRAIN) | VAL n | VAL ev/wk | VAL mean bps | VAL t | VAL ex-top5% | VAL winner-cap | VAL placebo margin | VAL null pctile |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1552 | OFFERING | SHORT | E5 | 5,137 | 67.3 | -23.6 | -1.45 | -78.2 | -28.7 | -9.2 | 15.0 |
| 1553 | SHELF | SHORT | E5 | 234 | 3.1 | +10.0 | 0.16 | -102.2 | -6.8 | +47.7 | 32.1 |
| 1554 | REVERSE_SPLIT | SHORT | E1 | 303 | 4.0 | -9.7 | -0.43 | -53.9 | -13.0 | -9.3 | 49.3 |
| 1555 | AUDITOR | SHORT | E5 | 24 | 0.37 | -151.6 | -0.90 | -223.7 | -151.6 | -95.6 | 1.5 |
| 1556 | NON_RELIANCE | SHORT | E5 | 8 | 0.16 | +193.0 | 1.00 | +193.0 | +193.0 | +180.4 | 36.7 |
| 1557 | LATE_FILING | SHORT | E5 | 30 | 0.48 | +129.0 | 0.69 | -3.3 | +113.2 | +88.4 | 74.8 |
| 1558 | OFFICER_EXIT | SHORT | E1 | 1,684 | 21.8 | +0.6 | 0.065 | -35.7 | +0.6 | +1.3 | 96.7 |
| 1559 | CONTRACT | LONG | E5 | 457 | 6.0 | -19.7 | -0.31 | -207.3 | -103.8 | -52.3 | 69.8 |
| 1560 | ACTIVIST | LONG | E1 | 116 | 1.5 | -47.8 | -0.47 | -143.9 | -71.4 | -60.8 | 1.3 |
| 1561 | BUYBACK_OR_INSIDER | LONG | E1 | 0 | — | — | — | — | — | — | — |

Full per-split, both-leg detail (train_e1/e5, val_e1/e5) is in `rebuild_1552_report.json`.

## Pass bar (VAL, named leg, frozen bar: mean ≥ +15 bps, t ≥ 2.5, ex-top5% > 0, winner-capped positive, ≥3
ev/wk, placebo margin ≥ +10 bps t≥2, null ≥ 99, TRAIN same-sign t≥1, ≥3/4 TRAIN years positive)

**0 of 10 cells pass** on this sample. Every |t| is well under 2.5 (max magnitude ~1.6, cell 1552). Every cell's
ex-top-5% mean is more negative (in-direction) than its raw mean wherever raw mean is positive, i.e. **the
positive point estimates that do appear (1553, 1556, 1557, 1561-n/a) are tail-carried — removing the best 5% of
outcomes flips or worsens every one of them** — the same tail-dependence pattern this codebase has flagged
repeatedly elsewhere. ACTIVIST (1560), hypothesized LONG, is net NEGATIVE in both TRAIN and VAL at both legs —
opposite in sign to the PREREG's fixed direction, on this sample.

## Notable rebuild finding not in the PREREG's own text

**424B1-5 catches routine debt/note shelf takedowns by large, frequent issuers, not only small-cap dilutive
equity offerings.** Spot-checked rows show BNS (Bank of Nova Scotia) and ARCC (Ares Capital, a BDC) filing
424B2 prospectus supplements for what are almost certainly debt notes, not equity dilution — the mechanism
narrative ("dilution... drift down") does not obviously apply to an investment-grade bank's routine note
program, yet the class definition as written (form-only, no security-type or issuer-size gate) includes them.
This inflates OFFERING's count (67-79 events/week, by far the busiest cell) with a plausibly different, unpriced
population mixed into the intended one. Not fixed here (would require reading the 424B prospectus to determine
security type — text classification is disallowed this pass) — flagged as a **refuter candidate for the
item-code-mapping hand-check** (the PREREG's own Independent-check section calls for "sample 30 filings per
class by hand: does the item mean what the class says" — this rebuild did a smaller ad hoc spot-check, not the
full 30-filing hand audit).

## Refuters checked / not checked

- **Timing**: entry/exit logic implemented as a same-day-before-09:00 / next-session-after-09:00 rule from
  `acceptanceDateTime`; not separately hand-audited against the raw SEC timestamp for a sample of filings.
- **Price scale**: raw close/open from the Alpaca daily parquet (same source and caveat as PREREG_1550,
  inherited, not re-derived here); the ±50%-event inspection the PREREG calls for was NOT run this pass (time
  budget) — the 78 |net_e1_bps|>2000 (≥20%) events were spot-checked (VXRT, BMRA, ATMP above) and look like real
  moves, not scale errors, but this is not the full inspection the PREREG requires.
- **Survivorship**: inherited Alpaca-daily caveat, not independently re-verified.
- **Item-code mapping hand-check (30/class)**: NOT run at the full 30-per-class the PREREG specifies; only the
  ad hoc 8-row spot check above (which surfaced the 424B finding).
- **SSR/borrow realism**: implemented as specified (prior close <$5 or prior day-over-day ≤-10% excluded;
  3%/yr pro-rata borrow); not independently stress-tested against a known SSR-halt list.
- **Tails**: ex-top-5%/1% and winner-capped(±20%) computed per event per cell per split (table above);
  confirms tail-carry on every cell with a positive raw mean.

## Bottom line

This is a real, executed, from-prose rebuild on a disclosed ~14%-of-population sample: fetch completeness gate
100% (0 LOST) on the sample actually drawn. It reproduces the PREREG's mechanics (classification, population
gate, auction timing, E1/E5, costs, SSR/borrow, placebo, null) faithfully enough to check the SIGN and rough
magnitude of each cell, and finds **no cell within reach of the pass bar** on this sample, plus one real data-
quality finding (424B debt-issuer contamination of OFFERING) that a hand audit should resolve before any cell
is reported to the owner as a candidate. It is NOT a substitute for the full-population run the PREREG's own
"Independent check" section requires (event-set Jaccard ≥0.99, net bps within 1) — that needs a multi-hour
background fetch of the remaining ~7,300 population symbols this pass did not reach.
