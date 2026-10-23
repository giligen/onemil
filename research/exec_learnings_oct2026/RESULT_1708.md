# RESULT 1,708 — cell 1,426 re-scored with a true share count (10/23)
Files: `replay_1708.py` (copy of replay.py; ONLY change: shares = `_rp_position / entry_price`, l.~398), `fills_1708.csv` (per fill x delay),
`table_1708.csv` (all numbers), `analyze_1708.py`, `run_1708.log`. Inputs read-only: orb_latency_bt/population.csv + raw/ ticks (no downloads).
**Share definition:** `_rp_position` (sized $ notional, $3,333 at the $375 stage) / BT entry_price: mean 456, median 367 shares. The old
`_sized_pnl/pnl` was 0.0667 (ratio). R = pnl/$375, mean over ALL replay-eligible fills (unfilled = 0), as in REPORT.md. 617 usable fills (21 missing ticks). **Harness check:** old formula reproduces REPORT.md exactly (every old_R below equals REPORT.md).

Mean R per fill [95% day-clustered CI] (new) vs old; $ = sum over the period; ex5 = ex-top-5 % mean R (new)
| period | d | old R | new R [CI] | old $ | new $ | new ex5 | new t |
|---|---|---|---|---|---|---|---|
| OOS 2023-24+25H2-26-09 (n 537) | 0 | .028 | .037 [-.005,.080] | 5,578 | 7,538 | -.045 | 1.74 |
| | 10 | .035 | .043 [.002,.085] | 6,995 | 8,732 | -.039 | 2.05 |
| | 20 | .030 | .039 [-.000,.078] | 5,984 | 7,844 | -.037 | 1.96 |
| | 30 | .011 | .021 [-.014,.056] | 2,258 | 4,268 | -.045 | 1.18 |
| | 60 | .007 | .019 [-.015,.052] | 1,386 | 3,747 | -.039 | 1.10 |
| | 90 | .005 | .021 [-.008,.051] | 910 | 4,262 | -.031 | 1.40 |
| May-Sep26 (n 153) | 0 | .013 | .022 [-.045,.089] | 729 | 1,250 | -.036 | 0.64 |
| | 20 | .018 | .025 [-.038,.088] | 1,019 | 1,427 | -.032 | 0.77 |
| | 30 | -.002 | .007 [-.043,.057] | -109 | 417 | -.040 | 0.28 |
| | 60 | -.018 | -.007 [-.041,.026] | -1,053 | -422 | -.037 | -0.43 |
| 2023-24 (n 163) | 0 | -.003 | .009 [-.054,.072] | -159 | 559 | -.070 | 0.29 |
| | 20 | .014 | .025 [-.037,.086] | 841 | 1,512 | -.054 | 0.79 |
| | 60 | -.003 | .009 [-.048,.066] | -166 | 533 | -.052 | 0.30 |
| 2025H1 (n 80) | 0 | .122 | .133 [.014,.253] | 3,675 | 3,997 | .039 | 2.18 |
| | 30 | .063 | .072 [-.029,.173] | 1,877 | 2,168 | -.016 | 1.40 |
| 2025H2-26-09 (n 374) | 0 | .041 | .050 [-.004,.104] | 5,737 | 6,979 | -.035 | 1.81 |
| | 20 | .037 | .045 [-.004,.094] | 5,143 | 6,332 | -.031 | 1.80 |
| | 30 | .018 | .027 [-.017,.072] | 2,532 | 3,836 | -.040 | 1.20 |
Full 12-delay grid x 5 periods (+ALL): `table_1708.csv`.

**Read:** the share fix adds ~+0.009 R/fill at every delay and period (+$7.6/fill at d=0; total +$2.0K OOS at d=0, +$3.4K at d=90). The
9/25 statement "+0.028 R at zero latency, ≈ 0 at >= 30 s" is REPLACED by: OOS +0.037 R (CI -0.005..0.080) at 0 s, +0.021 at 30 s, +0.019 at 60 s,
all CIs include 0 except d=10 (+0.002..0.085). Shape unchanged (hump at 10-20 s, decay after 30 s); latency cost 0 -> 60 s = 0.018 R (old 0.021).
May-Sep26 is still flat (+0.022 at 0 s, t 0.6). Ex-top-5 % stays NEGATIVE (-0.03..-0.05) in OOS, so the edge remains tail-carried.
**30 bps match:** 21.3 % of 301 d=0 fills agree with the BT entry within 1c; replay fills are 22.8 bps BELOW BT entry on average (median -29.9 bps = fill at
trigger), 0 % above (by construction: the chase limit = BT entry, higher asks are skipped_guard).

Caveats (adversarial):
1. The price term is only the gain from a cheaper entry; it is a gift from the BT's 30 bps assumption being conservative vs the tape ask at trigger. It does NOT
   model the live ask-to-fill drift (+72 bps measured live) or queue/partial fills - zero-latency d=0 is an upper bound, not a forecast.
2. Exit is unchanged: BT exit price/stop level kept; a cheaper entry is not re-risked (R stays $375). Shares sized on BT entry, not on fill.
3. Selection: fills above the limit are dropped (guard), so conditional on filling, price is always <= BT entry; the replay's mean-R includes unfilled = 0.
4. Day-clustered SE with ~100-300 days per period; MDE ~ 0.04 R (OOS) - every delay-to-delay difference is far inside the CI. Multiplicity: the 12-delay grid
   was read as a curve, not selected on.
5. NOT independently rebuilt yet (required before the owner sees a number); same population, same XNAS-only trigger bias (94 % of t* > 2 s after 09:35).
6. A larger latency effect does not reopen ORB; the untouched out-of-regime edge (2023-24 +0.009 R at 0 s, CI -.054..+.072) is a separate test and is still ~0.
