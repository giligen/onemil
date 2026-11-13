# PREREG 1,754 — AUCTION-TO-AUCTION gapper day trade (book A1 + A2): MOO in, MOC out, zero quoted spread

FROZEN 2026-11-13 19:25 UTC before any number. Programme cell count: 1,753 → 1,754. Rules as PREREG_1746 (write
only `research/signal_search/1754_*`, `RESULT_1754.md`; no git/service/orders/network; `ulimit -v 2500000; nice
-n 19`, peak RSS < 1 GB — daily bars only, on disk in `1748_cache` / `1746_cache` (ranks 1–3,000, 2015-12 →
2026-11; loader pattern in 1748_score.py); budget ≤ 50 calls. TEST 2024-01 → sealed.

## Claim (parent F3/F6: cost kills the +0.1 R gross; the cost model charges NO spread on auction fills)
A gapper bought at the opening auction and sold at the closing auction pays no quoted spread (auction prints are
single-price). If the gap-day open-to-close drift is positive for the gappers our scanner would show, the book is
the only one that keeps its gross. Survivorship caveat as 1,746 (rejection-only; a pass re-reads on PIT).

## Signal (daily bars only, all known at 09:30 except the open price itself, which IS the MOO fill)
Universe at t−1: ranks 1–3,000 by 20-day dollar volume, price ≥ $1 (small caps are the point). Gap = open_t /
close_{t−1} − 1 ≥ g, g ∈ {5 %, 10 %, 20 %}; prior-day volume ≥ 1.5× its 20-day mean OR prior-day range ≥ 5 %
(attention yesterday — the only causal volume proxy without pre-market bars; say so). Entry = open_t (MOO),
exit = close_t (MOC). Reads: by g; by price tier ($1–5 / $5–20 / > $20); by rank tier; long only. Cost: 2 bp per
leg + 0 spread (auction) — AND the same book with the measured half-spread table as the sensitivity (if the
auction assumption is wrong, this is the number).

## Numbers (TRAIN 2016–20 halves, VAL 2021–23 once)
n/wk, mean open-to-close return (bp), day-clustered t, ex-top-5 %, winner-capped +20 %, monthly green share,
year signs, equal-weight vs the universe open-to-close on the same days (excess), MDE at 80 % power.

## Kill (any one, primary = g 10 %, $1–20, excess open-to-close)
VAL excess ≤ +30 bp or day-t < 2; TRAIN halves opposite sign; VAL ex-top-5 % ≤ 0; 2022 negative; < 3 names/wk;
net under the half-spread sensitivity ≤ 0 (then the book lives only if auction fills are real → paper test).
Survive → TEST once on PIT → independent rebuild → paper sleeve (MOO/MOC orders on the HOD paper key).

## Output
`1754_score.py`, `1754_summary.csv`, `RESULT_1754.md` ≤ 350 words ending with "CHILDREN:" (≥ 2 hypotheses from
the winners-vs-losers table). Return ≤ 120 words.
