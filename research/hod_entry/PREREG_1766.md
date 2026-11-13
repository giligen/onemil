# PREREG 1,766 — F5 SECTOR SYMPATHY: when several names in one sector break early, buy the laggard with the same setup

FROZEN 2026-11-13 21:00 UTC before any number. Programme cell count: 1,765 → 1,766. Rules: write only
`research/hod_entry/1766_*`, `RESULT_1766.md`; no git/service/orders/network; data on disk; `ulimit -v 2500000;
nice -n 19`, peak RSS < 1 GB; budget ≤ 50 calls. TEST sealed. Filters carry their count-matched placebo p95.

## Claim (sympathy flow: a sector move is information for the names that have not moved yet)
On the honest gapper population (`cell_1428_population.csv`, bars `1762_cache/bars/`), with a sector map from
`data/research/orb_asset_class_map_20260711.csv` (inspect columns; report the share of symbol-days mapped — ≥ 80 %
or VOID; wrappers/ETFs excluded from "sector"), define at each minute m the sector heat = number of population
names in the same sector that have broken their HOD by m (regular rule). LEADER = a break when heat ≥ 2 already;
LAGGARD = a break with heat ≥ 2 from OTHER names and the name's own break comes ≥ 10 minutes after the second
sector break; SOLO = heat 0. Outcome = the regular HOD-break net R (from `1763_score.py`'s comparator, import).

## Reads (TRAIN halves, TRAIN, VAL once)
Net R by class (SOLO / LEADER / LAGGARD) with n, day-clustered t of LAGGARD − SOLO, ex-top-5 %, fast-loser share;
heat as a count (0 / 1 / 2 / ≥ 3) monotonicity; the "trade only LAGGARD" filter with ΔR/wk, fills/wk and the
placebo p95; the same-day-other-sector placebo (heat from a random other sector).

## Kill (structure only if ALL hold)
LAGGARD − SOLO same sign on both TRAIN halves and VAL, day-t ≥ 2 on VAL; heat monotone on TRAIN; filter ΔR/wk
above placebo p95; ≥ 3 LAGGARD fills/wk; the other-sector placebo shows no effect. Survivor → TEST once →
independent rebuild → dry-run spec (engine already tracks per-symbol breaks; a sector counter is one dict).

## Output
`1766_score.py`, `1766_classes.csv`, `RESULT_1766.md` ≤ 350 words ending with "CHILDREN:" (≥ 2). Return ≤ 120
words: mapped share, class counts/wk, VAL LAGGARD − SOLO / t / heat monotone / filter ΔR vs placebo, other-sector
placebo, structure YES/NO, children.
