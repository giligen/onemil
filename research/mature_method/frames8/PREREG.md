# frames8 — PREREG (written and committed BEFORE any cell was scored)

Pass 8 of the frame programme. Queue set by `hod_frames/FRAMES.md` "THE QUEUE AFTER PASS 7":
**F25 the geometry transplant · F26 the ORB minute priced as a live decision · F27 the power floor.**

Node rails for the whole pass: one python process at a time, `nice -n 10`, `ulimit -v 3000000`,
checkpointed walks. `cache.db`, `bars_sip.db`, the Databento stores, `daily_bars` and `trades.db`
opened **read-only**. Nothing outside `research/mature_method/frames8/` is written. `config.yaml`,
`orb.yaml`, the systemd unit, the crons and every order are untouched. **TEST is sealed**
(`FREEZE.md`): TRAIN = 2025, VAL = 2026-01..05, TEST = 2026-06+ and is never opened.

---

# F25 — THE GEOMETRY TRANSPLANT

**This is the first frame in 1,050 cells that predicts a SIGN from a mechanism. The prediction and
the falsifier are written here, before the walk was run.**

## §1 The mechanism, the prediction, the falsifier

*Mechanism (from pass 7).* On all three books the signal minute buys a **worse stop**: HOD-break
52 % vs 30 % for a random minute, ORB +11 pp over its matched control, BF +15 pp. On ORB and BF that
worse stop is **paid for by an uncapped right tail** — ORB's signal minute reaches its +1.75 R lock
4.3× as often as its control (23.5 % vs 5.4 %), BF arms its R-trail 3.8× as often (48.5 % vs 12.7 %).
HOD-break's exit **caps every winner at +2 R while the stop stays 1 R**, so the same higher stop rate
has nothing to pay for it. F22 measured the consequence directly: the bare bracket is negative in
every entry-minute, stop-distance, ADV$ and class band (population −0.0693 R) — while the SAME
universe, same names, same clocks, under ORB's static lock reads **+0.070 / +0.024 R** and under BF's
R-trail **+0.053 / +0.010 R**. *The negative instrument is the capped +2 R target, not the market.*

**PREDICTION (pre-registered, one sentence).** Under ORB's static lock or BF's R-trail, transplanted
onto HOD-break's own booked trades and its own controls: (i) the **control population** (arm d, the
universe bound) turns from −0.058 / −0.042 R to **≥ 0 in all three eras**; and (ii) the HOD **signal
minute** turns from ≈ 0 against its own later-minute control (−0.063 / +0.026 R) to **> 0 on both
splits**, so that the name-day selection pass 6 measured (+0.123 / +0.240 R) now sits on a baseline
that does not eat it.

**FALSIFIER (pre-registered).** If, under **every** transplanted exit, the HOD signal minute is still
**≤ its own later-minute control (arm a′) on both splits**, the higher stop rate on HOD is **not the
same animal** as ORB's and BF's and **the frame is dead** — the verdict is written that way and the
`+2 R cap` explanation for seven passes of zeros is withdrawn, not rescued.

Secondary, declared in advance so it cannot be chosen afterwards: the frame's SHIP question is decided
only by the **booked book** cells (§1.3) against the live-exploration bar; the decomposition (§1.4) is
the mechanism reading and never a ship reason on its own.

## §2 Populations (fixed before scoring, no re-selection)

| id | object | n | source |
|---|---|---|---|
| **P_book** | `B2` — the reference HOD book, 12/day, 4 concurrent | **2,328** (1,622 TRAIN / 706 VAL) | `hod_frames6/book6.csv`, gate below |
| **P_sig** | the full admitted signal set B2 books from | 7,027 | `common6.base_book()` |
| **P_b** | arm b — matched non-signal name, the signal's own minute | **51,051** | `hod_frames6/pb6.csv` keys |
| **P_d** | arm d — matched non-signal name, random eligible minutes (the universe bound) | **288,174** | `hod_frames6/pd6.csv` keys |
| **P_a′** | arm a′ — same name-day, a random LATER minute (causal) | 10 per booked trade | `hod_frames6/pa6.csv` rows with `ctrl_entry_m > entry_m`, seeded sample |

Every control's stop is the booked trade's own `r_pct` applied to the control bar's open — pass 6's
construction, unchanged, so the geometry is the only thing that moves.

**Declared approximation, stated before scoring.** The booked SET is held fixed at B2's, so
**trades/wk is unchanged by construction in every geometry cell** and the comparison is exit-only. An
uncapped exit holds a slot longer and in a 4-concurrent book would take FEWER trades; cell **RB**
re-runs `trading.hod_break.run_book` on `P_sig` with each geometry's own `exit_m` to measure that,
and it is reported beside the fixed-set number, never instead of it.

## §3 The declared geometries (each on P_book, both splits, both TRAIN halves)

| id | geometry | spec |
|---|---|---|
| **X0** | the shipped +2 R cap | reproduction of `walk_from`; must equal `book6.rr` to 1e-12 or the run aborts |
| **G1** | **ORB's static lock, exactly** | stop = E − R; a bar high ≥ E + **1.75 R** moves the stop to E + **0.5 R** forever; **no target**; flat at **15:55**; stop fills at `min(stop, bar open) × (1 − 10 bps)` |
| **G1a** | G1 + ORB's **ATR stop floor** (SZ1, k = 0.25) | booked-only sub-arm (controls have no ATR) |
| **G1b** | G1 + ORB's **40 % @ +3.0 R scale-out** | frozen composition: same-bar stop+scale → the scale fills; runner keeps the initial stop |
| **G2** | **BF's unified R-trail, exactly** | hard stop E − R; the trail **arms at +2 R** and rides **1 R** below the running **CLOSED-bar** high; check-then-ratchet; **entry bar excluded**; flat 15:55 |
| **G2p** | G2 + the shipped **50 % @ +2 R partial**, stop → true breakeven | one partial per trade; remainder keeps the trail |
| **G2v** | G2 + the **prev-bar volume guard** (`min_vol_ratio` 1.0 vs the 5 pre-entry bars' mean) | booked-only sub-arm; the guard applies to the TRAIL stop only, never the hard stop |
| **G2plan** | G2 on **plan-R** (`r_basis: plan`): baseline = the break `level`, unit = `level − stop` | booked-only sub-arm (a control name has no level) |
| **G3** | **the uncapped null** | bare stop, **no target**, hold to 15:55 |
| **G5 / G6 / G7** | the target ladder UP | resting limits at **+3 R / +4 R / +6 R**, close-through fill AT the level (the programme's convention) |
| **RB** | the re-booked honesty check | `run_book(P_sig, 12, 4)` under each geometry's own `exit_m` |

## §4 What is reported for every cell (no cell may be reported partially)

gross R · **booked cost re-measured for that cell's own exit mix** · net R · the exit mix (target /
lock / trail / stop / flat-eod rates) · green weeks % · longest red streak · worst week $ · weekly $
at $100 risk · trades/wk (**unchanged by construction — stated, not hidden**) · both TRAIN halves ·
VAL · day-clustered t · count-matched permutation null on green weeks (2,000 draws, pick count fixed)
· **ex-top-5 % beside the headline** (this frame is ABOUT the tail: the trimmed number is reported
next to the untrimmed one and the owner's standing rule — monsters are fine if green weeks dominate —
decides, per `RUNBOOK` step 7) · MDE at 80 % power.

**Cost, re-measured per cell — the 0.065 R is NOT carried.** The model is the programme's:
`net = rr − half − half × ratio`, `half = 0.5 × sp_pct / r_pct`, with the exit-leg ratio by exit
reason. Declared ratios, fixed here: **target / scale-out leg = 0.0** (a resting limit pays no
spread), **stop / lock / trail_stop = 0.875** (marketable), **flat / eod = 0.412** (the 15:55 market
exit), **profit-partial leg = 0.875** (the live `execute_partial_exit` is marketable). Blended exits
use the share-weighted ratio. Removing the cap converts free target fills into paid stop/eod fills,
so the cost per cell MUST rise; a cell that does not show it is a bug.

## §5 Rails (a cell that fails one is demoted before its number is read)

1. **Reproduction gates**: `B2` = 1,622 / −$17,346 (TRAIN) and 706 / +$893 (VAL), asserted in code;
   **X0 must reproduce `book6.rr` to 1e-12**, asserted; **G1 must match `study_orb_pipeline_static_lock.simulate_winner_stack`** on a spot-check of **50** real ORB trades; **G2 must satisfy `trading/bf_trail.py`'s own contract**, checked by running `tests/test_bf_trail.py` and by asserting `arm_and_ratchet` reproduces the walker's stop path bar-for-bar on a sample.
2. **Era consistency applied to the CONTROL population FIRST**: the new baseline (arm d) must be ≥ 0
   in H1-25, H2-25 and VAL before any detector is put on top. Reported whatever it says.
3. **Both TRAIN halves** beside VAL on every cell.
4. **Day-clustered SE** everywhere; iid t not quoted where the cluster is the right unit.
5. **Count-matched permutation null** on green weeks.
6. **Tail**: rank-based trimming (`rr` has a point mass at exactly the target in X0/G5/G6/G7).
7. **Availability**: coverage reported per arm; an arm below **80 %** is a diagnostic, not a cell.
8. **Multiplicity**: 3 geometry families × the existing cell grid counted and printed.
9. **TEST never opened.**

## §6 The bars

* **Claim bar** (RUNBOOK step 10): clustered t ≥ 2 on TRAIN, VAL same-signed with ≥ 55 % green weeks.
* **Live-exploration bar**: positive weekly $ **and** green weeks ≥ 50 % on **both** splits at
  **≥ 10 tr/wk**, clustered t ≥ 2, both TRAIN halves same-signed.
* A cell clearing the live-exploration bar → **SHIP-TO-DRY** with the exact `HodBreakParams` / engine
  diff written out (an uncapped exit on HOD is an engine change: the +2 R bracket TP leg goes and a
  lock or a trail arrives; the BF and ORB code paths already exist and are named).
* No cell clears → **STAY-DRY**, print the MDE, and write the next three frames on `FRAMES.md`.

---

# F26 — THE ORB MINUTE PRICED AS A LIVE DECISION

## §7 Object and cells

Stage Q's **1,040 walked NBBO order lives** (`research/fuckup_audit/Q_fill/`) — the ORB picks whose
NBBO ask at the trigger instant was **above** the pick's own stop-limit cap. The cap ladder is
`orb.yaml::entry.stop_limit_buffer_bps` ∈ **{30 (shipped until 9/18), 50 (what ships Monday), 100,
150}**. Re-scoring uses `Q_fill/spreads.parquet`'s already-fetched entry asks and `cache.db` bars —
**no new API calls, no Databento spend, `orb.yaml` is not touched.**

Per rung, scored on **the CONVERTED subset alone** (orders that fail at the lower cap and fill at the
higher one), both splits and both TRAIN halves:

| cell | question |
|---|---|
| **Q30** | the baseline: how many orders the 30-bps cap leaves resting, and what they were worth under the measured (delayed / never) treatment |
| **Q50** | picks converted 30 → 50 bps: n, mean R, total $, and the R they had under the measured treatment |
| **Q100** | picks converted 50 → 100 bps: same |
| **Q150** | picks converted 100 → 150 bps: same |

Also reported, because the frame asks for it: **the price paid on picks that would have filled
anyway** — by construction the cap binds only when the ask exceeds it, so a wider cap cannot change
an already-marketable fill; this is asserted in code rather than assumed, and the count of
already-marketable picks is printed.

**The discriminator (RUNBOOK step 4, memory `project_passive_entry_adverse_selection`).** For each
rung: is a conversion *a better fill on the same setup* (a chase guard being relaxed — ORB's own
Stage-Q reading) or *the setup breaking, which is why it filled* (the halt-resume dip-buy, −2.03 R on
fills vs +0.63 R on non-fills)? Measured, not argued: the converted subset's own R against (a) the
same picks' measured treatment at the tighter cap and (b) the already-marketable picks of the same
book.

**Pre-committed kill rule**: a **negative converted-subset R at any rung kills every wider rung above
it**, whatever the wider rung's aggregate book says. The live-exploration bar is applied to the
converted subset alone.

---

# F27 — THE POWER FLOOR (infrastructure, not a money frame)

No cells, no book, no rule. One page, two deliverables:

1. **The arithmetic.** For ORB (5–8 tr/wk shipped) and BF-P1 (2.8 tr/mo): the trade count needed to
   detect **that book's own measured effect** at 80 % power at the **live per-trade SE**, and the
   calendar time it implies at the shipped frequency. Stated for the raw mean and for the
   control-differenced estimator.
2. **The estimator, SPECIFIED not built.** A control-differenced statistic — each live trade minus
   the mean of its own matched non-signal controls, computable at EOD from the universe the engine
   already streams — for `scripts/*_eod_check.py` and the ramp checkers' BT-band gate. The
   deliverable names the exact files, the fields, the join, the failure modes and what it would cost;
   **no file outside `frames8/` is modified in this pass.**

---

## §8 Cell count

Programme total before this pass: **1,050**. Declared here: **F25 14** (12 geometry cells + RB + the
4-arm decomposition counted as one) + **F26 4** + **F27 0** = **18**. Running total **1,068**, printed
in the report with whatever supplementary readings the report names.
