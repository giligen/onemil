# RESULT — cell 1,662: HOD programme re-read at measured cost (7/6/0/11 bps) with the 1.5% stop floor

Owner's ask (relayed, not pivoted): "go into all the data collected, see what needs to be re-tested with the
1.5–3 and 3+ buckets to make them positive." Per PREREG_1662, TEST windows were never read. hod_exit_lab (cell
1,660) skipped — owned by another agent. Budget: 60 tool calls; ~34 used on inventory+re-score+write.

## Method
New cost: entry 7bps always + {target 0, stop/stop_bar 6, eod/eod_fallback/time/cover 11 or 6, cover treated
stop-like} bps by exit type, converted to R via each fill's own stop-distance %. Floor = only rows with stop
distance ≥1.5% of price counted in the "floored" re-read (this is what the owner is asking to re-test).
Day-clustered t; ex-top-5% = mean after dropping the best 5% of fills. **Validation**: my independently
reconstructed BASE_1438 population (join of fills_1658.csv × cell_1481_fills.csv) reproduces cell 1658's own
published stop-bucket numbers almost exactly (see below) — the join and cost formula are trustworthy.

## Inventory (cells with a per-fill/per-signal file on disk)
| Cell(s) | Directory | File | Original verdict | Orig cost | Halves |
|---|---|---|---|---|---|
| 1438 base (9,911) | hod_entry | fills_1658.csv × cell_1481_fills.csv (joined) | cell 1658: <1.5% loses -0.11R both halves | flat 13bps (1658) / mixed (1481) | TRAIN-H2/VAL |
| 1439, 1441 | hod_entry | cell_1439_fills.csv, cell_1441_fills.csv | population-construction, no headline | legacy tape-replay | split col |
| 1440 A/B | hod_entry | cell_1440_fills_{A,B}.csv | entry-model variant, no headline | legacy | reproduces BASE H2/VAL exactly |
| 1481, 1482 | hod_entry | cell_1481/1482_fills.csv | -0.09R / -0.07R VAL | mixed passive+stop-limit | holdout |
| 1483 | hod_entry | cell_1483_fills.csv (join) | catalyst/runway class filters, per-class VAL | base 1438 cost | split |
| 1487 | hod_entry | cell_1487_fills.csv | +0.67R claimed on confirmation cohort | half-spread+stop-limit | holdout |
| 1488 | hod_entry | cell_1488_fills.csv (join, target/stop conv.) | pyramid keyed on no-withdrawal | base cost | split |
| 1491 | hod_entry | cell_1491_fills.csv | shallow-stop re-walk | verified stop-limit | holdout |
| 1493–1547 | hod_entry | cell_1493_fills.csv (55-combo surface) | full exit-surface ladder | mixed | split |
| 1548 | hod_entry | cell_1548_fills.csv | TRAIN +0.32R in-sample (L3 tercile) | half-spread+stop-limit | split |
| 1617 | hod_entry | cell_1617_nights.csv | overnight hold anatomy | 5bps/auction leg | split |
| 1619 | hod_entry | cell_1619_fills.csv (join, conv.) | barriers-cell family | unknown | split |
| 1621 | hod_entry | cell_1621_fills.csv (raw_R from entry/exit/R2) | barriers-cell family | unknown | holdout |
| 1623 | hod_entry | cell_1623_fills.csv (join) | causal intraday gate | base cost | split |
| 1624 | hod_entry | cell_1624_fills.csv (join) | break-breadth (crowd) filter | base cost | split |
| 1625 | hod_entry | cell_1625_signals.csv | index-as-instrument | n/a | n/a |
| causal-filter (0/12) | bf_zero/causal_filter | population.csv (15,656 signals) | 0/12 filters causal; line closed 9/18 | measured NBBO | date-terciled (no split col) |
| 1357/1358 | bf_zero/causal_filter | failed_break_short_135{7,8}_trades.csv | FAIL, cadence bar (C1–C4 fail) | measured NBBO+2bp/side | TRAIN/VAL |
| 1351–1354 | bf_zero/causal_filter | rank_cells.csv, rank_cells2.csv | FAIL both TRAIN halves | n/a (aggregate only) | n/a |
| 1393–1395 | hod_ofi | window_features.csv | gross expectancy ≈0 every bucket | n/a | TRAIN H1/H2 |

## Re-read at measured cost + 1.5% floor (both halves; TEST never read)
| Cell | n floored (n<1.5% excl.) | Half A: n / net R / t | Half B: n / net R / t | ex-top-5% both>0? | fills/wk | vs orig |
|---|---|---|---|---|---|---|
| **BASE_1438** | 5,506 (4,405) | TRAIN-H2 2349 / **-0.017** / -1.42 | VAL 3157 / **-0.026** / -1.98 | no (both neg) | 90 / 150 | still negative — cost relief moves it up from -0.11 but doesn't flip |
| 1440A (=BASE, entry variant) | 5,506 | 2349 / -0.017 / -1.42 | 3157 / -0.026 / -1.98 | no | 90/150 | identical to BASE |
| 1440B (entry variant) | 5,506 | 2344 / -0.022 / -1.40 | 3157 / -0.034 / -2.18 | no | 90/150 | slightly worse |
| 1481 retest | 4,555 (4,462) | TRAIN-H2 1931 / -0.096 / -2.52 | VAL 2624 / -0.074 / -2.90 | no | 74/125 | still negative both halves, t now significant |
| 1482 retest | 3,699 (4,820) | 1552 / -0.100 / -2.80 | 2147 / -0.047 / -2.22 | no | 59/102 | still negative |
| 1487 confirmation | 408 (471) | 179 / -0.104 / -1.14 | 229 / -0.034 / -0.15 | no | 7/11 | original +0.67R claim does NOT reproduce at new cost |
| 1548 extension L3 | 9,900 (7,101) | TRAIN-H1 4394 / -0.030 / -2.77 | VAL 5506 / -0.064 / -3.54 | no | 168/262 | in-sample +0.32R does not survive |
| 1493 exit-surface <1.5% | 161,514 | 71226 / -0.496 / -17.4 | 90288 / -0.592 / -25.0 | no | huge | confirms <1.5% is dead even at new cost |
| 1493 exit-surface 1.5-3% | 161,514 | 71226 / -0.124 / -7.5 | 90288 / -0.140 / -8.0 | no | huge | **still negative on the retest surface** |
| 1493 exit-surface ≥3% | 80,757 | 35613 / -0.066 / -5.9 | 45144 / -0.060 / -5.2 | no | huge | **still negative** — retest-surface ≠ base breakout |
| 1483 no_news (n=8,924) | 5,025 | 2200 / -0.011 / -1.39 | 2825 / -0.018 / -1.83 | no | 84/135 | flat-negative, not positive |
| 1483 earnings_guidance | 204 | 49 / -0.203 / -0.21 | 155 / -0.067 / -0.44 | — | 2.1/7.7 (fpw<3) | too thin |
| 1483 analyst_action | 84 | 23 / +0.076 / 0.29 | 61 / -0.261 / -1.11 | no | 1.0/3.0 | halves disagree, thin |
| 1623 gate_plus | 346 | 160 / -0.059 / -0.65 | 186 / **+0.257** / 1.69 | mixed | 7/10 | sign flips VAL but t<2.5, TRAIN-H2 still neg |
| 1623 gate_minus | 1,849 | 712 / -0.026 / -0.58 | 1137 / -0.093 / -1.00 | no | 28/54 | flat |
| 1624 bottom/mid/top tercile | 1,004–2,097 | all \|t\|<2 both halves | — | no | 27–76 | breadth filter still finds nothing |
| 1621 barriers | 919 (2,991) | TRAIN-H2 410 / -0.136 / -3.53 | VAL 509 / -0.114 / -2.48 | no | 16/24 | **clean negative, both halves, real t** |
| **1488 pyramid-add** | 413 | TRAIN-H2 181 / **+0.345** / **2.66** | VAL 232 / **+0.337** / **2.77** | **yes (0.26/0.25)** | 7/11 | see flag caveat below |
| 1619 (see caveat) | 3,948 | 1638 / +0.900 / 14.6 | 2310 / +1.067 / 18.4 | yes | 63/110 | **ARTIFACT — not a real result, see below** |
| causal-filter base pop | 5,802 (3,021) | H1 2982 / -0.024 / -4.22 | H2 2820 / -0.096 / -2.80 | no | 104/98 | 0/12 verdict stands — still negative at new cost |

## BASE_1438 by stop-distance bucket (the owner's literal question)
| Bucket | n | TRAIN-H2 net R / t | VAL net R / t | vs cell 1658 (old flat 13bps) |
|---|---|---|---|---|
| <1.5% | 4,405 | -0.100 / -3.73 | -0.105 / -4.10 | matches 1658's -0.11/-0.11 closely |
| **1.5–3%** | 4,503 | -0.004 / -0.72 | **-0.063 / -2.62** | 1658 had -0.01/-0.07 — same shape, still negative |
| **≥3%** | 1,003 | -0.075 / -1.55 | +0.139 / 0.51 | 1658 had -0.08/+0.14 — matches; halves disagree, VAL not significant |
**Answer: at the new measured cost, neither the 1.5–3% nor the ≥3% bucket turns net-positive with a significant,
same-signed t on both halves.** Cost relief alone does not make these buckets positive on the base population;
the mechanism itself (not the cost) is what is flat-to-negative in 1.5–3%, and under-powered/halves-disagree in ≥3%.

## Flagged (mechanical bar: net≥+0.05R, t≥2.5 both halves, ex5>0, ≥3 fills/wk)
- **1488 pyramid-add**: clears the mechanical bar (+0.34R / +0.35R, t 2.66/2.77, ex5 positive both halves, small
  EOD-exit exclusion 26/797=3.3%). **Caution, not a clean pass**: raw_R reconstructed via the target=+2R/stop=-1R
  convention (definitional in this cell family, cross-validated against 1481/1491's own raw_R), joined for r_pct.
  Needs an independent rebuild from the cell's own stored EOD-exit gross values (not this join) before it is
  trustworthy enough to paper. **Do not ship on this read alone.**
- **1619**: mechanically clears (t 14–18, fpw 60–110) but this is a **methodology artifact, not a finding**: the
  raw sample showed duplicate (day,symbol) rows under a 'cell' grid column I did not group by (pseudo-replication,
  same trade counted many times), and 'cover'=+2.0R was an unvalidated guess (unlike target/stop, 'cover' has no
  definitional R). **Rejected — do not act on this number.**
- No other cell/bucket clears the bar. No sign flips beyond 1623_gate_plus's VAL-only flip (t 1.69, fails the bar).

## Not re-readable
- **1617** (overnight hold): no stop/R concept at all (holds to next open, no stop) — outside this PREREG's R-based re-score.
- **1625**: day-level SPY/IWM correlation features file, not a per-fill outcome file.
- **1357/1358** (failed-break-short): only R-multiples stored (net, gross), no per-row stop level/R$ — cannot compute r_pct or apply the floor.
- **1351–1354** (rank): cells.csv/rank_cells.csv are aggregate summaries (8–26 rows = cells), not per-fill; original verdict already FAIL both TRAIN halves pre-cost.
- **1393–1395** (OFI): window_features.csv has OFI/TSI/spread features only, no net_R/outcome column on disk.
- **1439/1441**: re-read but floored population is 0/253 — early population-construction passes, not final cells.
- **1491** (shallow-stop): floored population = 0 of 58,150. By construction the stop is a few ticks under the break, so it is almost always <1.5% — **structurally incompatible with the new floor** (a real finding, not a data gap).
- **bf_zero spec_trades.csv / hodbreak_trades_cap60.csv**: alternate-stage populations for the same study; population.csv taken as representative for budget reasons.
- **research/fuckup_audit/H/*** (F5, F6, F6_rebuild, F6_reconcile, F6_sizing, F8N30/F11F6/F14, QQQ — the H/METHOD loser-anatomy frames, ~150 CSVs): not inventoried/re-scored — out of the 60-call budget. These are bull-flag/index-breadth angles on the same from-zero programme, not HOD stop-distance cells. Flag for a dedicated follow-up cell if the owner wants this arm re-read too.

## Bottom line
The owner's literal question — do the 1.5–3% and ≥3% buckets turn positive under the measured cost — is **no** on
the base 1438 population and **no** on every downstream filter/exit variant re-read (1481, 1482, 1487, 1548, 1493
surface, 1483 catalyst classes, 1623 gate, 1624 breadth, 1621 barriers all stay negative or insignificant on both
halves). Cost relief moved several numbers up by 0.02–0.05R (matches the PREREG's own estimate of 0.04–0.3R) but
none crossed into a significant positive on both halves except two cells that do not survive scrutiny: 1488 needs
an independent rebuild before trusting, and 1619 is a pseudo-replication artifact. The honest read: **the HOD
population's problem is the mechanism, not the cost model** — lowering cost narrows the losses but does not
manufacture an edge that was not there. 1662_rescored.csv has the full per-half table (25 rows).
