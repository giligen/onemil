# The Ignition Program — design + first research (2026-07-18)

Owner mandate: "the market has monster movers every week, our ORB must
find them." The bottom-up census proved him right and located the pool.

## The supply (measured, 19 months)

- Market monster-days (open→high ≥30%, $2–50, ≥$5M day-$vol): **2,744
  = ~34/week, every week** (81/81 weeks).
- ORB's universe admits 16%; captured 34 of them = **97% of the whole
  book** ($279K of $288K). Every selection gate re-validated under
  live-parity — the gapper architecture is at its information ceiling.
- **The untouched pool: 1,380 flat-open monsters (~17/wk), median run
  +42%, median day-$vol $24M** — excluded by the gap≥5% screen alone.

## The anatomy (322-sample, bar-level, 2026-07-18)

- **Ignition timing: 50% ignite in the 9:xx hour, 30% in 10:xx — 73%
  inside the existing ORB session window.** These are OPEN-HOUR movers
  that didn't gap; not a midday regime.
- **The entry moment exists with room**: after first crossing +10% from
  open, median further continuation **+28%**; 97% continue ≥15% more;
  peak ≥30min after ignition in 86% of cases.
- Character at ignition: median 30-min pre-range 9.3% (moving, not
  coiled), ignition-minute volume ≈2.3× the prior half hour.
- ⚠ SURVIVORSHIP: these stats are monster-conditioned. The junk river
  (all +10%-crossers that die) is unmeasured — the harness's whole job.

## Design (deliberate; maximum reuse)

**Trigger**: first cross of +10% intraday change from open (the scanner
ALREADY computes this every minute for the full 5,628-symbol universe —
the INTRADAY QUALIFIED stream currently feeds only BF, which cannot use
it post-10:45). New consumer, zero new scanning infrastructure.

**Entry**: on trigger cross (marketable, ORB-style stop-limit semantics
+ spread gate + buy-stop guard), entry window 9:35–~13:00 (tunable in
harness; 80% of ignitions are before 11:00).

**Risk structure**: stop = pre-ignition consolidation low (trailing
30-min low), risk-parity sizing with the ORB caps; exits = the
battle-tested static-lock family + touchgo-style failure cut + 15:45
flat. The move shape (fast ignition, deep tail) is exactly what these
exits were built for.

**Filter candidates (from the validated taxonomy, to be tested, not
assumed)**: catalyst (news / complex-confirmation), RelVol at trigger,
float band, price band, PDR-style prior-day context, one-per-family.

**What it is NOT**: not a BF flag (shape refuted), not a gap play (ORB
owns that), not day-rank (refuted), not curated sectors (era-flip).

## The harness (the honest EV answer — next sessions)

1. Universe: EVERY symbol-day crossing +10% from open (est. 10–20K over
   19mo) — full junk river included.
2. Bars backfill for those symbol-days (background, ~news-backfill
   scale ×3; all persisted to cache).
3. Bar-accurate entry/stop/exit sim; era splits (25H1/25H2/2026),
   leave-out-top-K, negMo/MDD/worst-month; live-parity discipline from
   day one (no fitted constants without orb.yaml pinning; decision-
   parity hooks designed in, not bolted on).
4. Ship bar: owner rule + era-consistency + leave-out, same as ORB.

## Why this can change the machine's shape

ORB's 4 monster months = fair capture of 16% of monster supply. The
ignition pool is ~3× larger and weekly. Even ORB-like capture rates on
a fraction of it converts "4 monster months" toward "monsters monthly,"
which is the owner's stated goal. All downstream machinery (sizing,
exits, StopMonitor, attribution, green-check) is reusable.

Artifacts: /tmp/market_monsters.csv, /tmp/flat_open_monsters.csv,
/tmp/ignition_bars.csv, /tmp/ignition_anatomy.csv,
/tmp/fetch_ignition_bars.py.


## Checkpoint 1 — full junk-river v0 numbers (2026-07-19, cached 86% sample)

**15,466 trades** (every +10% cross, gap<5, 9:35–13:00, no filters):
- **Mean +$132/trade (+0.062R), WR 46% — POSITIVE IN ALL THREE ERAS**
  (25H1 +$35 / 25H2 +$159 / 2026 +$184). The naked trigger carries edge
  on the full river — ORB's raw river is negative-mean by comparison.
- Naive 4-slot FCFS book: $654K/19mo model — NOT bookable as-is
  (worstMo −$51K, MDD −$55K; FCFS is the dumbest possible selector).

**Honest caveats (the program's next gates):**
1. LOTTERY CONCENTRATION: median trade −$174; top ~300 of 15,466 carry
   everything (drop-top-300 → negative). Familiar texture, magnified.
2. **EXECUTION IS THE EDGE'S BUDGET**: at 0.6% entry slip 25H1 goes
   negative; at 1.0% the river flips negative overall. Fill realism
   (participation-scaled slippage, minute-volume capacity) is gate #2 —
   nothing ships on 30bps assumptions for fast movers.
3. Era-consistent structure found (filters for v1): trigger ≤10:30
   (+$199 vs +$60 later), consolidation R ≥5% (+$174 vs +$27, 2026
   negative below) — both all-era-positive. vol-surge mild.
4. Sample bias note: cached-86% skews toward machine-relevant names;
   complete-river rerun (incl. 3,994 least-traded) in flight.

**Program gates to ship**: (G2) participation-based fill model;
(G3) v1 selector (early-window + min-R + quality rank, slot-capped) with
era + leave-out + monthly texture; (G4) catalyst/complex/news filters;
(G5) ORB-overlap and account-level interaction; (G6) walk-forward halves
+ live-parity constants pinned in yaml from day one; (G7) paper-shadow
period before real size.
