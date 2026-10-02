# PREREG — cell 1,700t: a data-hygiene guard on the sleeve's signal (FROZEN 2026-10-02 18:45 UTC, before any number)

Found while judging 1,700r: the panel holds 1,617 one-day moves above +200 % or below −75 % and 72 rows that follow a
listing gap > 30 days (recycled tickers: PCLN, ULTI, SPLS; share re-issues: WOLF 2025-09-29 $1.21 → $22.10 at its
bankruptcy exit, not a return anyone earned). 33 of the reference's 10,060 held name-weeks have such an event inside
the lookback (ABVX, AMC, AMRN, OCGN real moves; WOLF a fake one, held 2026-06-29, −16 % that week). The live sleeve
reads the same kind of bars, so it can buy a fake jump.

## Guard (one rule, shared by BT and live through trading/momentum_sleeve.py)
A name is INELIGIBLE on a rebalance date if, inside its 273-trading-day lookback, it has (a) a one-day close-to-close
move > +200 % or < −75 %, or (b) a gap > 10 calendar days between consecutive bars.

## Read
REF vs REF + guard on the daily engine: CAGR, max DD, end $, by year, name-weeks removed, the names removed.

## Rule (hygiene, not an edge claim)
Adopt the guard in the live sleeve and re-baseline REF if it moves CAGR by < 1.0 pt and max DD by < 2.0 pts either
way. If it moves more, the reference depended on unverifiable bars: report which names, adopt the guard anyway, and
restate every 1,700 headline on the guarded reference. 1 cell.
