# Compare — cell 1,626 builder vs. independent rebuild

Task: PREREG_1626.md "Independent check" step 1, restricted to cell 1,626. Compares
`research/lev_decay/cell_1626_days.csv` (builder, `cell_1626.py`) against
`research/lev_decay/rebuild_1626_days.csv` (rebuild, `rebuild_1626.py`, built from PREREG prose only,
per its own header never opened the builder's code). All numbers below were recomputed directly from
the two days-CSVs and the two pairs-CSVs by a script that has not read either producing script's logic
beyond the grep cited under "Dominant cause"; script: `/tmp/.../scratchpad/compare_1626.py` (session-local).

## 1. Pair-set Jaccard

PREREG defines a "pair" by "the underlying ticker in the name and the leverage factor." At that
granularity, using each side's own matched-pairs list (`pairs.csv` vs `rebuild_1626_pairs.csv`):

| Granularity | builder n | rebuild n | intersection | union | Jaccard |
|---|---|---|---|---|---|
| **(underlying, factor)** — PREREG's own unit | 34 | 35 | 32 | 37 | **0.8649** |
| exact (long_symbol, short_symbol), matched-pairs lists | 34 | 93 | 33 | 94 | 0.3511 |
| exact (underlying, long, short), from the days files actually scored | 34 | 81 | 27 | 88 | 0.3068 |
| underlying-only, from the days files actually scored | 34 | 30 | 29 | 35 | 0.8286 |

**Every granularity fails the frozen bar (≥ 0.95).** Even the most lenient cut (underlying+factor,
0.86) misses by a wide margin; the exact-instrument cut the strategy actually trades on (0.31–0.35)
is far worse.

Set differences at the PREREG's own (underlying, factor) unit:
- builder-only: `AI` (2.0x), `SK` (2.0x) — not matched by the rebuild's parser.
- rebuild-only: `BETA` (3.0x, **HIBL/HIBS — confirmed non-single-stock**, see below), `DATA` (2.0x,
  the AIBU/AIBD pair the builder calls `AI`), `DRAM` (2.0x, AIBU/AIBD look-alike naming issue aside —
  the long names "T-REX 2X Long DRAM Daily Target ETF" / "Defiance Daily Target 2X Long DRAM ETF"
  read as a memory-chip sector theme, not a single company ticker; not independently confirmed against
  the builder's `pairs_excluded_non_single_stock.csv` within this task's step budget — flagged, not closed).

## 2. VAL mean bps/day, cell 1,626, 15%/yr borrow rail

| | n pair-days | mean bps/day |
|---|---|---|
| builder (`net_bps_15pct`, split=VAL) | 6,153 | 5.1856 |
| rebuild (`net_bps_1626_rail15`, split=VAL) | 15,356 | 7.0374 |
| **diff (rebuild − builder)** | | **+1.8518 bps/day** |

Fails the frozen "bps within 0.5" bar by ~3.7×. For context (not asked, but relevant): TRAIN flips
sign outright — builder −2.72 bps/day (n=371) vs rebuild +3.49 bps/day (n=3,242).

## 3. Passing cells

**Builder: `[]`** (RESULT_1626.md checklist, both cells 2/7: mean and %-positive pass, everything
else — t≥2.5, TRAIN same-sign t≥1, drag-vs-theory within 30%, worst month, ETB≥10 — fails or is unmet).

**Rebuild: `[]`** as well, on its own reported numbers, despite the higher point estimates:
- Cell 1,626: t=1.26 (< 2.5, fail); TRAIN rail15 t=0.59 (same sign but < 1, fail); top-vol-tercile
  realised/theory = 0.51 (outside the builder's stated 0.7–1.3 "within 30%" band, fail); worst-month
  and ETB≥10 are not reported by the rebuild for this cell at all (unverified → treated as not-passed).
- Cell 1,627: VAL mean = +2.34 bps/day, already below the +4 bps/day bar on its own — fails outright.

**`same_passing = true`**: the bottom-line verdict ("cells 1,626/1,627 do not clear the pass bar") is
unchanged by the rebuild. The two implementations disagree sharply on the underlying numbers but agree
on the conclusion — every hard threshold (t≥2.5 in particular) is missed by both, by a wide enough
margin that the ~1.85 bps/day and Jaccard 0.31–0.86 disagreement would not flip it.

## 4. Dominant cause of the differences

Ranked by evidence gathered this task (grep of both `cell_1626.py` and `rebuild_1626.py`, plus the
gross/net rail-scaling check on `rebuild_1626_days.csv`):

**1. Pair matching — dominant, confirmed.**
- The rebuild does not deduplicate multi-issuer legs per underlying: 30/35 of its underlyings have
  >1 matched pair (all long×short issuer combinations cross-joined), giving 81 scored pairs vs the
  builder's 34 — the rebuild's own doc flags these as sharing a leg and "NOT independent draws." The
  builder deliberately picks one issuer pair per (underlying, factor) to avoid this.
- **Confirmed contamination**: the rebuild's cell-1626 (2x) population includes `BETA` / HIBL / HIBS
  ("Direxion Daily S&P 500 High Beta Bull/Bear 3X ETF") — a 3x, **non-single-stock, broad-index basket
  product**. The builder's own RESULT.md names "HIBL/HIBS=S&P500 high-beta" verbatim as one of the
  excluded non-single-stock names caught by its blocklist. This is an unambiguous population-definition
  bug in the rebuild, not a rounding difference.
- Underlying-name disagreement: AIBU/AIBD (Direxion "AI and Big Data" 2x) is assigned to underlying
  `AI` by the builder and `DATA` by the rebuild — plausibly neither is a true single-stock match (the
  product tracks a thematic basket, not a company called "AI" or "DATA"), a shared weak spot neither
  side caught, but it still means the two sides are scoring a different, disagreeing definition of
  what that instrument's underlying even is.
- This directly explains most of the Jaccard shortfall and is the most likely primary driver of the
  return-level gap too: comparing a 34-pair vs an 81-pair, structurally different (and internally
  correlated) population will shift the sample mean and t-stat before any per-pair-day calculation
  difference is even considered.

**2. Bar adjustment / corporate-action handling — secondary, confirmed asymmetry.**
Both sides use Alpaca's `adjustment=split` bars (matching setting) but handle *residual* extreme-return
days differently: the rebuild explicitly neutralises 4 symbol-days with |daily return| > 90% to a flat
0% (logged examples: IONX 2026-03-19 r=+1.90, RGTX 2026-03-19 r=+2.85, STSM 2026-03-23 r=+1.89, STSM
2026-09-14 r=+0.95 — post-split-adjustment residual jumps, per its price-scale-check section).
`cell_1626.py` has no equivalent neutralisation step (grep found only a comment about raw-vs-adjusted
bars, no active clip/neutralise call in `simulate_pair`). A short-the-long-leg position left exposed to
an uncorrected +190%/+285% one-day print is a large loss on that pair-day; leaving those days in would
pull the builder's mean down relative to the rebuild's — directionally consistent with the observed gap,
and with the TRAIN sign flip (n=4 pairs; one or two such days are enough to flip a small-n mean).
**Not independently re-verified against the builder's raw bars within this task's budget** — flagged as
the second most likely driver, not closed.

**3. Rebalance convention — ruled out as a dominant cause.**
`grep` confirms both codebases independently arrived at the *same* mechanism from the same PREREG prose:
builder `REBALANCE_EVERY = 5` (sessions), cost charged on rebalance days only; rebuild `REBAL_SESSIONS = 5`,
`LEG_COST_BPS = 5.0`, `cost_dollar = 2 * (LEG_COST_BPS/10000) * 1.0` applied `if session_count >= REBAL_SESSIONS`.
Same 5-session interval, same 5 bps/leg trading cost, same $1/leg-target convention. Not a source of the gap.

**4. Borrow accrual — ruled out as a dominant cause.**
In `rebuild_1626_days.csv`, `gross_bps_1626 − net_bps_1626_rail{5,15,30}` scales ~1:3:6 across the three
rails on every row sampled (e.g. −2.03/−6.09/−12.19; −1.94/−5.83/−11.66), exactly as a linear %/yr rail
model should. The gap's median (1.98 bps/day) matches a naive single-$1-leg daily charge at 5%/yr
(0.05/252×10000 = 1.984 bps) almost exactly, consistent with the PREREG's `(D_long_prev+D_short_prev)/2
× rail/252` formula that the builder also documents using. No evidence of a formula discrepancy here.

## Verdict

`agreement_ok = false`. The PREREG's own frozen independent-check bar (pair-set Jaccard ≥ 0.95, bps
within 0.5) is missed on every reasonable definition of "pair," and by a wide margin on the bps check.
Per CLAUDE.md's research-claim gate, cell 1,626/1,627 numbers are **not yet reportable to the owner as
calibrated figures** — the pair-matching logic (issuer dedup rule + non-single-stock exclusion list)
needs to be reconciled and the extreme-return handling made consistent before the magnitudes can be
trusted. The qualitative, decision-relevant conclusion is unaffected by this gap: **both independent
implementations agree cells 1,626 and 1,627 fail the pass bar** (`same_passing = true`), primarily on
the day-clustered t-stat (1.26 and 0.47 vs the 2.5 bar) and the vol-tercile drag calibration — margins
large enough that closing the Jaccard/bps gap is very unlikely to flip the PASS/FAIL call for this
programme.
