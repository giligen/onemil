# PREREG — cells 1,637–1,639: OPENING-AUCTION IMBALANCE — providing the other side at the open

FROZEN 2026-09-28 20:00 UTC before any number. Programme count: 1,636 → 1,639. Rank #1 of `research/ideas_web/
RANKED_20260928.md` (academic scan #5: retail market-on-open flow creates the imbalance; the open overshoots and
reverts). Sister of 1,630–1,632 (closing auction): same universe, same scorer, same pass bar, the opening window.

## Mechanism
Nasdaq disseminates the opening-cross net order imbalance from 09:28 ET, NYSE from its first publication before
09:30 (the data states the first timestamp per venue; nothing before it is used). A large buy imbalance lifts the
opening print above the pre-open mid and the excess reverts during the first 30 minutes; a sell imbalance the mirror.
Providing the other side in the auction (a sell MOO into a buy imbalance) or just after it earns the premium minus the
auction cost. Retail-feasible on liquid names; capacity = the imbalance size.

## Data
Databento `XNAS.ITCH` / `XNYS.PILLAR` schema `imbalance`, window 09:15–09:30:05 ET, the same 314-name point-in-time
liquid list and sessions 2024-07-01..2026-09-04 as 1,630 (estimate first; cap $100, its own spend ledger; the pull starts
only after the closing pull has written its sentinel — one Databento writer at a time). `bbo-1m` 09:29–10:02 for the
pre-open reference mid (the last quote before 09:30), the 10:00 exit mid and the through-touch checks. Opening print
and daily fields from `research/overnight_high/panel_2024_2026.parquet`. TRAIN 2024-07..2025-06, VAL 2025-07..2026-09.

## Signal and trades
I = the net imbalance at the LAST publication ≤ 09:29:30 ET, scaled by ADV20 (shares); top TRAIN decile of |I| fires.
* 1,637 FADE-IN-AUCTION: a MOO on the other side, submitted by 09:29:45 (decision at 09:29:30 — refuter checks the
  cutoff: Nasdaq MOO entry closes 09:28, so a Nasdaq name uses the 09:27:55 publication; NYSE MOO closes 09:29:30);
  fill at the official opening print; exit at 10:00 with a limit at the mid (through-touch at +1/+2 min, else 10:05
  marketable at the far touch).
* 1,638 FADE-AFTER-PRINT: at 09:30:30 take the other side with a limit at the 09:30 mid (through-touch rule), exit as 1,637.
* 1,639 FOLLOW (report-only): the same side as the imbalance, the same timings (the momentum reading).
Shorts: shortable flag, SSR names excluded (Rule 201 applies to the cross), borrow ignored (intraday).
Costs: 5 bps on the auction leg; the limit legs carry the through-touch rule; the marketable fallback pays the touch.
Report per cell and split: n, events/week, mean net bps, day-clustered t, ex-top-5 % / ex-top-1 %, the decile table of
the open-vs-pre-open-mid move against |I| (the mechanism) and of the 09:30→10:00 reversal, NYSE vs Nasdaq, worst day,
P&L by ADV bucket, the share filled on the fallback.

## Pass bar (frozen; VAL, per cell) — identical to 1,630
Mean net ≥ +8 bps per event, day-clustered t ≥ 2.5, ex-top-5 % > 0, ≥ 10 events/week, TRAIN same sign t ≥ 1, the
decile table monotone in |I| on both halves, worst day ≥ −1 % of deployed notional.

## Independent check and consequences
Rebuild from the prose (event set Jaccard ≥ 0.98, bps within 1); refuters: the publication-vs-cutoff timing, the
pre-open reference mid (a stale or crossed quote), the opening print as the fill (partial fills in the cross are not
modelled — disclosed), halts at the open, SSR, survivorship of the list, tails. PASS → paper auction desk first
(the live imbalance feed is a cost line). FAIL → closed with the decile table; nothing re-run on this window.

## Not allowed
Changing the decile, the decision times, the exit time or the fallback after a number.
