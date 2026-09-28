# PREREG — cells 1,630–1,632: CLOSING-AUCTION IMBALANCE — providing the other side into the close

FROZEN 2026-09-28 18:35 UTC before any number. Programme count: 1,629 → 1,632. Idea 2 of `research/IDEAS_20260928.md`.

## Mechanism (documented: Bogousslavsky & Muravyev 2023 "Who trades at the close?")
The exchanges publish the closing-auction order imbalance from 15:50 (NYSE) / 15:55 (Nasdaq). A large buy imbalance
moves the closing price up relative to the 15:50 mid and the move partly reverses at the next open; a sell imbalance
the reverse. Providing the other side (sell into a buy imbalance with an MOC or a limit at 15:55, buy back at the next
open — or the mirror) earns the imbalance premium minus the auction cost. Retail-feasible on liquid names; capacity is
the imbalance size itself.

## Data
Databento `XNAS.ITCH` and `XNYS.PILLAR` schema `imbalance` (verified 9/28: available 2018-05 → today, ≈ $0.02 per
symbol-month) for the 300 most liquid names by dollar volume in each of 2024, 2025, 2026 (the panel's ADV; state the
list), plus `mbp-1` or `tbbo` at 15:49:55–15:50:05 and 15:54:55–15:55:05 for the reference mids (the pre-imbalance mid),
and the official closing print and next open from the Databento daily bars / Alpaca. Sample: 2024-01 → 2026-09; TRAIN
= 2024-01..2025-06, VAL = 2025-07..2026-09. Spend cap $60 total, estimate first, log every purchase.

## Signal and trades
I(t) = the published net imbalance at the first publication (15:50 NYSE / 15:55 Nasdaq) scaled by the name's ADV20
(shares). Buckets by |I| in TRAIN deciles; the trade fires in the top decile (pre-declared).
* 1,630 FADE-TO-OPEN: at the publication + 30 s, take the OTHER side with a limit at the mid (fill assumed at the mid
  only if the mid is touched in the next 60 s per the quotes — the through-touch rule; else no trade), exit at the
  next session's opening auction (MOO). Shorts: shortable flag, SSR excluded, borrow 3 %/yr overnight.
* 1,631 FADE-INTO-CLOSE: the same entry, exit AT the close via an MOC submitted before 15:55 (Nasdaq names only, where
  the cutoff allows) — the pure auction-premium leg, no overnight.
* 1,632 FOLLOW (report-only): the same side as the imbalance into the close (the momentum reading).
Costs: 5 bps per auction leg; the limit entry at the mid carries the through-touch rule; SEC fees on shorts.
Report per cell and split: n events, events/week, mean net bps, day-clustered t, ex-top-5 % / ex-top-1 %, the decile
table of the closing move vs |I| (the mechanism: the close moves with I) and of the next-open reversal, the share of
NYSE vs Nasdaq, the worst day, the P&L by ADV bucket (capacity).

## Pass bar (frozen; VAL, per cell)
Mean net ≥ +8 bps per event, day-clustered t ≥ 2.5, ex-top-5 % > 0, ≥ 10 events/week at the top decile, TRAIN same
sign t ≥ 1, the decile table monotone in |I| on both halves (the mechanism), worst day ≥ −1 % of the deployed notional.

## Independent check and consequences
Rebuild from the prose (event set Jaccard ≥ 0.98, bps within 1); refuters: the publication timestamp vs the entry
(nothing before publication), the reference mid, the through-touch rule, the MOC cutoff (15:50 NYSE — a NYSE name
cannot get a 15:55-decided MOC; those go to 1,630 only), halts/LULD at the close, survivorship of the liquid list
(chosen per year, point-in-time by that year's ADV), tails. PASS → the engine gets an auction desk (imbalance feed
needed live: Databento Live or the exchange feeds — a cost line for the owner) on the paper account first.
FAIL → closed with the decile table on record.

## Not allowed
Changing the decile, the +30 s, the 60 s window or the exits after a number.

## Amendment 1 (2026-09-28 19:20 UTC, before any number) — sample start, cap, touch quotes
The price panel (`panel_2024_2026.parquet`) starts 2024-07-01, so sessions before it cannot be scored; the measured
cost is ≈ $0.16 per session for the 314-name list. Sample = 2024-07-01..2026-09-04 (the panel's end); TRAIN =
2024-07-01..2025-06-30, VAL = 2025-07-01..2026-09-04; spend cap $90 (≈ 540 sessions × $0.16); the through-touch fill
check uses the bbo-1m snapshots at the next two minute marks after publication (a sell at the mid is filled if the
bid at +1 or +2 min ≥ the limit; a buy if the ask ≤ the limit) — a minute-granularity proxy for the 60-s rule,
disclosed. Venue from the point-in-time listing feed (the ITCH placeholder rows found by the fetch are excluded).
The first run (2024-01-02 onward, ≈ $5) was stopped at 19:37 UTC; its cached rows before 2024-07-01 stay unused.

## Amendment 2 (2026-09-28 19:50 UTC, before any number) — cap $100
The amended scope prices at ≈ $95 total (548 sessions at ≈ $0.15 plus the $12.7 already spent), so the $90 run halts
about six weeks before 2026-09-04. Cap raised to $100 so the VAL window is complete; the run resumes from the cache
under the new constant after the first run halts. Nothing else changes.

## Amendment 3 (2026-09-28 21:15 UTC) — pull STOPPED at $20.96 on the owner's spend decision; cells SUSPENDED
The owner ("too much $$$ is planned to go out … your call") → the closing-auction pull was stopped at 2024-09-18
(56 sessions, 548 purchases, $20.96); the options v3 pull continues. Cells 1,630–1,632 and the sister 1,637–1,639 are
SUSPENDED, not judged: no money number is read from 56 sessions. One free, pre-committed read is allowed on what is
on disk (all TRAIN): the MECHANISM table only — the decile table of the closing move (official close vs the pre-
publication mid) against |I| and of the next-open reversal, with the day-clustered t of the top-minus-bottom decile.
That read is the evidence for asking the owner to buy the rest (a monotone table with |t| ≥ 3 on both halves of the
56 sessions), or the reason not to. Nothing else changes.
