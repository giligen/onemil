# HOD-break through the mature method — REPORT (2026-09-19)

Candidate #1 of `research/mature_method/RUNBOOK.md`, all ten steps, cells as declared in
`PREREG.md` (committed `0948b65` **before any cell was scored**). Artifacts: `pass_breaks.py`
→ `breaks.csv` (2,228,304 break rows), `score.py` → `score.log`, `supp.py` → `supp.log`.
One python process at a time, `nice -n 10`, `ulimit -v 3000000`; `bars_sip.db`, `data/cache.db`,
`data/trades.db` opened **read-only**. No config, `orb.yaml`, systemd unit, cron, order or cache
was written. The dry run was not touched.

---

## VERDICT — **STAY DRY AS INSTRUMENT**

*In THIS universe — the whole point-in-time market on the Alpaca SIP tape, 22,841 first-qualifying
HOD-break signals, live config — at THIS horizon (consolidation-low stop, +2R close-fill target,
15:55 flat), at THIS book size (12/day, 4 concurrent), over 2025-01-02 → 2026-05-31, at THIS cost
(per-trade measured NBBO), **the HOD-break book is gross-flat and pays 0.215 R a trade to trade**.
The smallest per-trade effect the test could have seen is 0.043 R on the population and 0.080 R on
the booked TRAIN book — **2.7x to 5x smaller than the +0.215 R the book would need** — so this is a
powered rejection of the effect the book requires, not an underpowered null.*

Not STAY DEAD: `dry_run: true` places zero orders, the engine is parity-clean, and it is the only
source of forward data in the programme that cannot be a look-ahead. Not SHIP-TO-LIVE-SMALL: no
cell clears either bar, and the one cell with positive dollars anywhere (`T4`) is positive on VAL
and −$13,674 on TRAIN.

---

## 1. Reproduction gate — EXACT

`run_book(spec_trades.csv, 12, 4)` at the live knobs (price ≥ $20, entry ≤ 14:00), vs REPORT §6a:

| split | reproduced | §6a reference |
|---|---|---|
| TRAIN | 44.1/wk · −0.030 R · −1.3 R/wk · green **24/53** · worst −17.5 | 44.1 · −0.030 · −1.3 · 24/53 · −17.5 |
| VAL | 45.0/wk · −0.006 R · −0.3 R/wk · green **9/23** · worst −16.2 | 45.0 · −0.006 · −0.3 · 9/23 · −16.2 |
| TEST | 43.7/wk · +0.016 R · +0.7 R/wk · green **9/15** · worst −17.5 | 43.7 · +0.016 · +0.7 · 9/15 · −17.5 |

(These TEST figures are §6a's own published numbers being reproduced, not a cell being scored.)

**One defect found in the reference.** §6a's prose says it "(c) excludes the early-close days
2025-07-03, 11-28, 12-24". Its code did not: removing them gives TRAIN 43.5/wk and −0.0294 R, not
the printed 44.1 / −0.030. The membership cut is applied throughout this study; the discrepancy is
30 trades and changes nothing, but the prose was wrong.

**Independent rebuild.** `pass_breaks.py` re-derives every signal from the bars with no knowledge
of `spec_trades.csv`: 58,173 shared symbol-days, **99.77 % identical entry minute, max |Δ rr| =
0.00e+00** on those. The 0.23 % are bar-gap edges at the last-entry boundary. The pass emits breaks
at **every price** because `detect()` has no price floor and the engine applies `min_price` *after*
the break and then RETIRES the symbol-day (`hod_break_engine._try_enter`) — a sub-floor first break
kills the day, and a pass seeded above the floor would silently promote a later break the engine
would never take. That is the one modelling decision this study had to make, and it is the engine's.

## 2. Gross before net — the book is flat, and flat is not enough

No cost at all, after the membership cuts (test tickers, non-`daily_bars` names, early closes):

| population | split | n | /wk | **gross R** | ± SE | t | **MDE80** | WR |
|---|---|---|---|---|---|---|---|---|
| every live-config signal | TRAIN | 7,388 | 139 | **+0.0003** | 0.0155 | +0.02 | 0.043 R | 40.0 % |
| every live-config signal | VAL | 4,745 | 206 | **−0.0451** | 0.0194 | −2.32 | 0.054 R | 37.8 % |
| the shipped 12/4 book | TRAIN | 2,295 | 43.3 | **−0.0413** | 0.0284 | −1.45 | 0.080 R | 36.4 % |
| the shipped 12/4 book | VAL | 1,034 | 45.0 | **−0.0020** | 0.0421 | −0.05 | 0.118 R | 39.2 % |

The measured cost is **+0.2151 R per trade** (§3). The book therefore needs a gross edge of
+0.215 R and has 0.000 / −0.045. The runbook's step-2 stop condition — *gross ≤ 0 with the MDE
below the effect the book would need* — is met with room to spare: **0.043–0.118 R of resolution
against a 0.215 R requirement.** Everything below completes the map for the record, as instructed,
and nothing in it changes this line.

## 3. Measured cost — the band table DID over-charge, and it does not matter

Per-trade Alpaca SIP NBBO at the signal minute (`causal_filter/nbbo.csv`, re-used, **99.0 %
coverage** of the B0 signal set, 99.52 % non-null on the pre-book set):

| | median | mean | p90 |
|---|---|---|---|
| **measured** NBBO on these signals | **34 bps** | **51 bps** | 110 bps |
| §8 band constant on the SAME signals | 68 bps | 69 bps | — |

The band **over-charges this population by 1.35x on the mean and 1.96x on the median** — exactly
what a $20+, ADV20 ≥ 100 K population should do to a band fitted on thinner names. Charged through
the score4 contract: **measured +0.2151 R/trade vs band +0.3070 R/trade.** Correcting the cost
moves the book from −0.31 R to −0.22 R a trade. It is still a loss, because the gross is zero.

## 4. The engine's real fill model, and the unfilled counterfactual

Capped limit at `level x 1.006`, filled at the next bar's open iff that open is at or under the cap.

- **82.0 %** of first-qualifying signals fill (15,653 of 19,082); 18.0 % do not.
- Unfilled overshoot above the level: median **+91 bps**, p90 +172 bps.

| population | TRAIN gross | VAL gross | WR |
|---|---|---|---|
| FILLED (paid ≤ cap) | +0.000 | −0.045 | 40.0 / 37.8 % |
| UNFILLED counterfactual (paid the open anyway) | **+0.026** | **−0.027** | 45.9 / 40.7 % |

**Classification: NEITHER a chase guard nor a dip-buy — the cap is neutral-to-mildly-adverse.**
The runbook's ORB case (a chase guard) would show the unfilled population *worse*; the halt-resume
case (a dip-buy) would show the fills catastrophically worse. Here the unfilled are +0.026 R and
+0.018 R *better* in both splits, i.e. the no-chase rule declines trades that were marginally the
better ones — but both gaps sit inside their ±0.05–0.07 R standard error, so the honest reading is
that the cap neither helps nor hurts. It is a risk control, not an edge. (The counterfactual is
itself optimistic: it assumes the gapped-through open was obtainable.)

**Obtainability is the one place the model flatters itself.** On rows the bar-open model says
filled, the *quoted SIP ask at the decision instant* was above the cap for **15.3 %** of them — one
fill in six would not have happened. Those rows are dropped from every cell here (`Gk`), and the
direction of the remaining error makes the reported net R too kind, never too harsh.

## 5. Gate-separation map — leave-one-out on the whole shipped stack

For each gate the population is the entire live stack **minus that gate**, and the gate splits it.
GROSS and NET are both shown because three of these gates act on the cost term and their net
separation is partly definitional. `notional` = median position size at the live $100 risk (there
is no sizer to hide a gate behind — HOD risks a flat $100 — so notional is the analogue).

| gate | grossΔ | netΔ | t | n kept | n rej | TRAIN netΔ | VAL netΔ | notl kept | notl rej |
|---|---|---|---|---|---|---|---|---|---|
| Ga dist ≥ 7 % (ladder) | +0.086 | +0.078 | +1.46 | 741 | 5,552 | +0.177 | +0.123 | 3,911 | 5,436 |
| Ga dist ≥ 10 % | −0.012 | −0.023 | −0.26 | 263 | 6,030 | +0.172 | −0.107 | 3,818 | 5,294 |
| Ga dist ≥ 15 % | +0.084 | +0.073 | +0.42 | 71 | 6,222 | +0.420 | −0.316 | 3,790 | 5,254 |
| **Gb rv in [1,5)** | +0.017 | **+0.048** | +1.54 | 5,319 | 2,439 | **−0.009** | **+0.087** | 5,188 | 3,262 |
| Gb  .. vs rv < 1 only | −0.096 | **−0.069** | −1.84 | 5,319 | 1,281 | −0.275 | +0.118 | 5,188 | 3,499 |
| Gb  .. vs rv ≥ 5 only | +0.142 | **+0.177** | +4.24 | 5,319 | 1,158 | +0.272 | +0.054 | 5,188 | 3,008 |
| **Gc consolidation K5 / 4 %** | −0.135 | **−0.123** | **−2.91** | 3,561 | 1,323 | **−0.203** | **−0.053** | 6,310 | 2,495 |
| Gd price ≥ $5 | +0.070 | +0.086 | +4.57 | 13,201 | 7,727 | +0.091 | +0.084 | 4,018 | 3,420 |
| Gd price ≥ $10 | +0.057 | +0.077 | +4.23 | 9,755 | 11,173 | +0.072 | +0.074 | 4,759 | 3,420 |
| **Gd price ≥ $20** | +0.061 | **+0.088** | **+4.41** | 6,293 | 14,635 | **+0.088** | **+0.091** | 5,225 | 3,480 |
| Gd price ≥ $50 | +0.041 | +0.060 | +2.14 | 2,555 | 18,373 | +0.062 | +0.082 | 5,343 | 3,617 |
| Ge stop ≥ 1 % (shipped stack) | — | — | — | **INERT — 0 rejects** | | | | | |
| Ge stop ≥ 1 % (15 %-cap OFF) | +0.181 | +0.589 | +25.5 | 12,278 | 10,563 | +0.609 | +0.599 | 5,820 | **16,194** |
| Gf spread ≤ 15 % of R | **+0.033** | +0.201 | +8.03 | 6,293 | 5,985 | +0.155 | +0.208 | 5,225 | 6,426 |
| Gg spread ≤ 100 bps (shipped) | — | — | — | **INERT — 0 rejects** | | | | | |
| Gg spread ≤ 100 bps (15 %-cap OFF) | +0.090 | +0.609 | +12.0 | 12,278 | 986 | +0.467 | +0.806 | 5,820 | 5,028 |
| Gh entry ≤ 14:00 | −0.003 | +0.025 | +0.27 | 6,293 | **85** | +0.206 | −0.198 | 5,225 | 3,279 |
| Gk obtainable (quoted ask ≤ cap) | +0.034 | +0.051 | +0.80 | 6,293 | 414 | +0.063 | +0.027 | 5,225 | 3,244 |
| **Gi 12/day** | −0.051 | **−0.051** | −1.47 | 3,938 | 2,355 | **−0.223** | +0.088 | 5,079 | 5,515 |
| **Gj 4 concurrent** | −0.002 | **−0.001** | −0.02 | 3,094 | 3,199 | −0.078 | +0.025 | 5,216 | 5,229 |
| Gi + Gj together | +0.001 | +0.002 | +0.05 | 3,035 | 3,258 | −0.072 | +0.025 | 5,190 | 5,263 |

What the map says, in order of how much it matters:

1. **The defining shape of the book is on the wrong side in BOTH years.** Requiring ≥ 5 bars within
   4 % of the HOD instead of the loose 3-bar / 8 % form scores **−0.123 R (t −2.91), −0.203 TRAIN
   and −0.053 VAL** — measured with the *same* loose stop on both sides, so it is the shape being
   judged, not the stop distance. The one rule the clean-sheet study kept from 84 candidates is a
   net-negative filter on its own population. This had never been measured.
2. **The relative-volume band's lower leg is backwards.** The band as a whole is +0.048 at t 1.54
   and flips sign between years (−0.009 / +0.087). Split it and the picture is clean: excluding
   rv ≥ 5 is real and era-consistent (**+0.177, t 4.24**), excluding rv < 1 **costs** −0.069
   (TRAIN −0.275). The study's own anatomy said this ("`rv_profile` is monotone the other way")
   and the shipped band still has the wrong lower edge.
3. **The price floor is the one selection gate that works.** ≥ $20 is +0.088 at t 4.41, positive in
   both years; the ladder is flat from $5 to $20 and *falls* at $50 — the money is in "not the
   cheap names", not in "the expensive ones".
4. **Two shipped knobs are literally dead.** At the shipped stack the 15 %-of-R cap already removes
   every sub-1 % stop and every > 100 bps signal, so `min_r_pct` and `max_spread_bps` reject
   **nothing**: cells `F-e` and `F-g` come out byte-identical to `B0`. Measured at the engine's own
   cascade position (before `spread_r`) `min_r_pct` rejects **46.2 %** of signals — which answers
   the EOD check's item-16 WATCH ("28 % live, expected a few %"): 46 % is the historical rate, the
   28 % live reading is *low*, and the gate is not an anomaly. But its separation is +0.589 **net**
   on only **+0.181 gross** — 69 % of it is the cost model, i.e. it is the same cost gate as `Gf`,
   not a second source of selection. Same for `Gg` (+0.609 net, +0.090 gross). The rejected side is
   also where the size is: median notional $16,194 vs $5,820 kept, so `max_notional_usd: 10,500`
   would have bound on those trades anyway.
5. **The spread cap's edge is mostly definitional.** `Gf` is +0.201 net but **+0.033 gross** —
   ~84 % of it is arithmetic (it removes the trades whose modelled cost is large). The residual
   +0.033 gross is real but tiny.
6. **The book rails select nothing.** 4-concurrent is −0.001 (t −0.02), 12/day is −0.051 and flips
   sign between years, and the pair together is +0.002. First-come ordering is not a filter.
7. **Nothing above the +5 % floor.** The ladder to 7/10/15 % is noise with signs that flip between
   splits; the downward direction is not superset-safe on this universe file and was declared out
   of scope (REPORT §7's unbiased sample already says 0–2 % is +0.008 and 5–10 % is +0.160).
8. `Gh` last-entry 14:00 rejects **85** signals in 76 weeks. Inert.

## 6 / 7. Frequency frontier, ranked on % GREEN WEEKS (dollars at the live $100 risk)

23 declared cells. A no-trade week counts FLAT and is in the denominator (flat weeks are 1.9 % of
TRAIN and 4.3 % of VAL — at 32–45 trades a week this book essentially always trades, so unlike ORB
there is no flat-week reservoir to convert).

| cell | split | n | /wk | gross R | net R | net(band) | t | **green %** | red streak | worst wk $ | **total $** | MDD $ | green mo % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0 shipped** | TRAIN | 1,688 | 31.8 | −0.027 | −0.088 | −0.089 | −2.64 | **41.5** | 7 | −2,718 | **−14,835** | −17,209 | 25.0 |
| **B0 shipped** | VAL | 820 | 35.7 | +0.016 | −0.050 | −0.051 | −1.06 | **43.5** | 6 | −2,076 | **−4,128** | −7,519 | 40.0 |
| T3 price ≥ $50 | TRAIN | 808 | 15.2 | −0.031 | −0.097 | −0.097 | −2.05 | **43.4** | 5 | −1,532 | −7,837 | −8,201 | 16.7 |
| T3 price ≥ $50 | VAL | 532 | 23.1 | +0.037 | −0.031 | −0.032 | −0.54 | **47.8** | 6 | −1,390 | −1,675 | −6,812 | 40.0 |
| T5 spread ≤ 8 % of R | TRAIN | 1,165 | 22.0 | −0.067 | −0.105 | −0.105 | −2.69 | 41.5 | 5 | −1,826 | −12,267 | −13,710 | 16.7 |
| T5 spread ≤ 8 % of R | VAL | 599 | 26.0 | +0.025 | −0.013 | −0.013 | −0.25 | **47.8** | 3 | −1,984 | −807 | −3,784 | 40.0 |
| T4 r_min ≥ 2 % | TRAIN | 916 | 17.3 | −0.088 | −0.149 | −0.151 | −3.47 | 35.8 | 5 | −1,927 | −13,674 | −14,568 | 8.3 |
| **T4 r_min ≥ 2 %** | VAL | 525 | 22.8 | +0.091 | **+0.025** | +0.023 | +0.43 | **56.5** | **2** | −1,170 | **+1,298** | −3,962 | **60.0** |
| F-b rv band OFF | TRAIN | 1,770 | 33.4 | −0.042 | −0.113 | −0.143 | −3.56 | 30.2 | 9 | −2,012 | −20,028 | −20,028 | 8.3 |
| F-b rv band OFF | VAL | 846 | 36.8 | +0.039 | −0.038 | −0.075 | −0.82 | 47.8 | 3 | −2,270 | −3,174 | −8,300 | 40.0 |
| F-c consolidation loose | TRAIN | 1,475 | 27.8 | −0.108 | −0.175 | −0.189 | −4.88 | 34.0 | 6 | −2,130 | −25,788 | −25,788 | 16.7 |
| F-c consolidation loose | VAL | 797 | 34.7 | +0.031 | −0.041 | −0.061 | −0.83 | 43.5 | 4 | −1,673 | −3,254 | −4,846 | 40.0 |
| F-d price floor $5 | TRAIN | 2,247 | 42.4 | −0.062 | −0.140 | −0.200 | −4.98 | 26.4 | 7 | −2,482 | −31,485 | −31,485 | 0.0 |
| F-d price floor $5 | VAL | 956 | 41.6 | +0.017 | −0.063 | −0.122 | −1.44 | 43.5 | 4 | −3,227 | −6,024 | −8,368 | 40.0 |
| F-e r_min OFF | both | — | — | — | — | — | — | **byte-identical to B0** | | | | | |
| F-g 100 bps OFF | both | — | — | — | — | — | — | **byte-identical to B0** | | | | | |
| F-f 15 %-of-R cap OFF | TRAIN | 2,218 | 41.8 | −0.038 | −0.176 | −0.178 | −5.91 | 32.1 | 7 | −3,086 | −39,141 | −39,462 | 0.0 |
| F-f 15 %-of-R cap OFF | VAL | 1,020 | 44.3 | +0.048 | −0.105 | −0.107 | −2.36 | 21.7 | 11 | −3,164 | −10,712 | −11,331 | 20.0 |
| F-h last entry 15:30 | TRAIN | 1,722 | 32.5 | −0.029 | −0.091 | −0.093 | −2.77 | 41.5 | 7 | −2,718 | −15,604 | −17,878 | 16.7 |
| F-h last entry 15:30 | VAL | 839 | 36.5 | +0.019 | −0.048 | −0.050 | −1.03 | 43.5 | 6 | −2,194 | −3,996 | −7,352 | 40.0 |
| F-i 20/day | TRAIN | 1,716 | 32.4 | −0.029 | −0.090 | −0.090 | −2.72 | 39.6 | 7 | −2,718 | −15,403 | −17,244 | 16.7 |
| F-i 20/day | VAL | 844 | 36.7 | +0.016 | −0.050 | −0.051 | −1.08 | 39.1 | 6 | −2,199 | −4,260 | −7,400 | 40.0 |
| F-j 8 concurrent | TRAIN | 2,082 | 39.3 | −0.062 | −0.123 | −0.124 | −4.15 | 34.0 | 6 | −3,073 | −25,693 | −25,714 | 8.3 |
| F-j 8 concurrent | VAL | 1,101 | 47.9 | +0.020 | −0.046 | −0.046 | −1.11 | 34.8 | 9 | −1,941 | −5,015 | −9,518 | 40.0 |
| T1 dist ≥ 10 % | TRAIN | 521 | 9.8 | −0.024 | −0.112 | −0.187 | −1.85 | 39.6 | 5 | −1,444 | −5,820 | −7,591 | 25.0 |
| T1 dist ≥ 10 % | VAL | 401 | 17.4 | +0.033 | −0.055 | −0.138 | −0.80 | 26.1 | 9 | −1,107 | −2,216 | −5,170 | 20.0 |
| T2 rv in [1,3) | TRAIN | 1,338 | 25.2 | −0.039 | −0.105 | −0.117 | −2.82 | 39.6 | 7 | −2,006 | −14,040 | −14,040 | 25.0 |
| T2 rv in [1,3) | VAL | 737 | 32.0 | −0.053 | −0.126 | −0.142 | −2.54 | 26.1 | 12 | −2,132 | −9,304 | −10,559 | 0.0 |
| C1 both spread gates OFF | TRAIN | 2,223 | 41.9 | −0.036 | −0.209 | −0.211 | −6.95 | 22.6 | **15** | −3,293 | −46,563 | −46,625 | 0.0 |
| C1 both spread gates OFF | VAL | 1,029 | 44.7 | +0.022 | −0.183 | −0.185 | −4.08 | 17.4 | 11 | −3,335 | −18,872 | −19,323 | 0.0 |
| C2 rv OFF + r_min OFF | TRAIN | 1,770 | 33.4 | −0.042 | −0.113 | −0.143 | −3.56 | 30.2 | 9 | −2,012 | −20,028 | −20,028 | 8.3 |
| C2 rv OFF + r_min OFF | VAL | 846 | 36.8 | +0.039 | −0.038 | −0.075 | −0.82 | 47.8 | 3 | −2,270 | −3,174 | −8,300 | 40.0 |
| C3 price $5 + rv OFF | TRAIN | 2,347 | 44.3 | −0.078 | −0.161 | −0.235 | −5.94 | 26.4 | 8 | −3,052 | −37,792 | −37,792 | 0.0 |
| C3 price $5 + rv OFF | VAL | 960 | 41.7 | −0.017 | −0.102 | −0.176 | −2.40 | 26.1 | 9 | −2,845 | −9,829 | −14,218 | 40.0 |
| C4 slots 20/8 + 15:30 | TRAIN | 2,354 | 44.4 | −0.048 | −0.110 | −0.112 | −3.95 | 37.7 | 6 | −3,073 | −25,945 | −25,945 | 8.3 |
| C4 slots 20/8 + 15:30 | VAL | 1,344 | 58.4 | +0.004 | −0.061 | −0.064 | −1.68 | 30.4 | 9 | −2,637 | −8,265 | −11,272 | 40.0 |
| **C5 structural ceiling** | TRAIN | 4,832 | **91.2** | **−0.140** | −0.400 | −0.583 | −12.8 | **3.8** | **31** | −10,759 | **−193,495** | −193,495 | 0.0 |
| **C5 structural ceiling** | VAL | 2,040 | **88.7** | **−0.122** | −0.400 | −0.592 | −8.66 | **4.3** | 21 | −8,688 | **−81,505** | −82,745 | 0.0 |
| C6 T1+T2 (tight) | TRAIN | 375 | 7.1 | −0.035 | −0.126 | −0.210 | −1.76 | 37.7 | 6 | −901 | −4,716 | −5,635 | 16.7 |
| C6 T1+T2 (tight) | VAL | 310 | 13.5 | +0.010 | −0.081 | −0.171 | −1.03 | 30.4 | 5 | −1,568 | −2,516 | −3,422 | 20.0 |
| **C7 B0 + D5 veto** | TRAIN | 1,631 | 30.8 | −0.025 | −0.085 | −0.085 | −2.51 | **39.6** | 7 | −2,602 | −13,840 | −15,437 | 25.0 |
| **C7 B0 + D5 veto** | VAL | 809 | 35.2 | +0.018 | −0.048 | −0.049 | −1.01 | **34.8** | 6 | −1,698 | −3,902 | −7,476 | 40.0 |

**The structural ceiling, as the runbook requires**: every optional gate off (loose consolidation,
no rv band, no r_min, $5 floor, both spread gates off, 15:30, 20/8) gives **91 trades a week — and
the raw detector's gross R there is −0.140 (TRAIN) / −0.122 (VAL)**, i.e. 9 of every 10 available
trades are worse than the ones the stack keeps, and the ceiling book loses $193,495 on TRAIN at
$100 risk with 31 consecutive red weeks. **The frontier does not bend; it falls monotonically as
frequency rises.** That is the opposite of BF's F7 result, and the gate map says why: BF's added
trades were screened by rules with positive separation, while here the only positively-separating
gates (the price floor and the two cost gates) are exactly the ones that cut volume.

**The causal-filter study's dropped lead, run as the ONE declared cell (`C7`).** The veto on the
TRAIN 5th decile of entry minute — **[615, 624) = 10:15–10:23 ET**, declared with its edges in
PREREG before scoring, a bucket worth −0.547 R on 784 TRAIN signals — **fails on its own primary
metric in both splits**: green weeks 41.5 → **39.6** (TRAIN) and 43.5 → **34.8** (VAL). It does
improve dollars slightly (−14,835 → −13,840 and −4,128 → −3,902) and net R from −0.088 to −0.085,
which a veto on a bad bucket must do arithmetically; it buys no week shape at all. The decile-level
veto the causal-filter study named as its lead and then forbade itself is now measured, and it is
not a lever. (On the live dry run the same window holds 4 of 31 trades at −0.50 R, so it would have
moved the forward book by +$50.)

## 8. Weekly dollars at the live $100 risk — VAL (2026-01 → 2026-05, the last two open quarters)

`B0 shipped`, 23 W-FRI weeks, `$ (trades)`:

```
+228(8) -422(44) -451(36) +149(31) -1081(42) -1087(52) +607(40) -2044(35) -383(28) -687(42)
-1030(41) -793(30) -293(32) +1517(32) -1007(39) +2375(33) -2076(48) +1155(34) +4(46) +495(51)
+126(38) +570(38) +0(0)
```
10 green / 23, total **−$4,128**, worst −$2,076, best +$2,375, 6 consecutive red weeks.

`C7 (D5 veto)` over the same weeks: 8 green / 23, total −$3,902 — the ratio falls while the dollars
rise, precisely the failure mode step 8 exists to catch, here in reverse.
`C5 (ceiling)`: 1 green / 23, **−$81,505** at 89 trades a week.

## 9. Count-matched permutation null (2,000 draws, per-week pick count held fixed)

| cell | split | observed green % | null mean | [p5, p95] | outside? |
|---|---|---|---|---|---|
| B0 shipped | TRAIN | 41.5 | 34.7 | [28.3, 41.5] | inside (at p95) |
| B0 shipped | VAL | 43.5 | 38.7 | [30.4, 47.8] | inside |
| T3 price ≥ $50 | TRAIN / VAL | 43.4 / 47.8 | 37.2 / 42.9 | [30.2,45.3] / [34.8,52.2] | inside / inside |
| T4 r_min ≥ 2 % | TRAIN / VAL | 35.8 / 56.5 | 30.9 / 50.6 | [24.5,37.7] / [39.1,60.9] | inside / inside |
| T5 spread ≤ 8 % | TRAIN / VAL | 41.5 / 47.8 | 34.4 / 45.1 | [28.3,41.5] / [34.8,56.5] | inside / inside |
| C7 D5 veto | TRAIN / VAL | 39.6 / 34.8 | 35.1 / 38.8 | [28.3,41.5] / [30.4,47.8] | inside / inside |
| C5 ceiling | TRAIN / VAL | 3.8 / 4.3 | 1.6 / 2.1 | [0.0,3.8] / [0.0,8.7] | inside / inside |

**Of 46 cell x split nulls, 3 sit above their TRAIN band (`F-c`, `F-f`, `C4`) and 1 below its VAL
band (`T1`) — and not one cell is favourably outside on BOTH splits.** The shipped book sits inside
its own band on both, and exactly at the TRAIN p95: its week shape is the top of what pure pick
count explains, and no further. This is the same answer `orb_gates2` got — **green weeks on this
book are bought with pick count, not with week-level timing skill** — reached independently.

## 10. Both bars, and the adequacy review answered in writing

**Claim bar — 0 of 23 cells pass G1** (TRAIN mean net R > 0 with t ≥ 2 and ≥ 5 trades/week). Every
TRAIN mean is negative and the best TRAIN t is **−1.76**. G2 was therefore never evaluated and, per
PREREG §3, **TEST was never opened** (`FREEZE.md` records this; the only TEST numbers anywhere above
are §6a's own published reproduction row).

**Live-exploration bar — not met.** It requires a positive point estimate on green weeks *and* on
dollars at live size. No cell is positive on dollars in both splits. `T4` (r_min ≥ 2 %) is the only
positive-dollar cell anywhere — VAL +$1,298 at 56.5 % green weeks, 60 % green months and a 2-week
worst red streak — and it is −$13,674 at 35.8 % green on TRAIN, sits inside its null band on both
splits, and its 2-percent-stop rule is the cost gate again in another costume. `T3` and `T5` are the
only cells top-ranked on green weeks in *both* splits (43.4/47.8 and 41.5/47.8 vs B0's 41.5/43.5)
and both still lose money in both splits.

**Adequacy review:**

1. *Did we test what the book actually IS?* Yes. `trading/hod_break.py` at the live `config.yaml
   hod_break` block, on the whole point-in-time market at the SIP tape; the §6a reference reproduces
   to the printed digit and an independent rebuild from the bars matches it at max |Δ rr| = 0.
   Standing caveats, unchanged: the universe file is the range ≥ 5 % daily screen, made
   superset-exact by the causal +5 % floor (which is why the dist ladder runs upward only); 1,140
   delisted symbol-days have no consolidated tape and stay out; touch-only breaks (2.1 % of live
   signals) are outside the population.
2. *Is the cost and fill model right for its venue?* Better than it has ever been. Per-trade SIP
   NBBO at the signal minute on 99.0 % of signals; the band table over-charged 1.35–1.96x and was
   replaced. The fill is the engine's own capped limit. The one residual optimism (the bar-open fill
   on rows whose quoted ask was above the cap, 15.3 %) makes the reported net **too kind**.
3. *Does any caveat in our own report explain the headline?* No. The known one — §6b's cache
   population — explains the *old* +0.42 R headline, not this one; this study runs on the population
   the engine actually streams. The caveats that remain (obtainability, the optimistic unfilled
   counterfactual, the band→measured cost correction) all push the same way: the book is at best as
   bad as stated.
4. *What is the MDE?* Per trade: **0.043 R** (population TRAIN), 0.054 R (population VAL),
   **0.080 R** (booked TRAIN), 0.118 R (booked VAL). On the green-week share: **±19.0 pp** over 53
   TRAIN weeks and **±28.9 pp** over 23 VAL weeks — which is why no cell in §6/§7 is resolvable
   against B0 on the primary metric, and is itself part of the answer.

**Multiplicity.** 23 declared decision cells x 2 splits = 46, plus 19 gate-map rows, 2 supplementary
rows and ~15 descriptive tables. Expected largest |t| under a pure null over ~46 cells ≈ 2.8–3.0.
Moot in the favourable direction: no cell has a positive TRAIN t at all.

**Ex-tail diagnostics** (reported, never a rejection reason): B0 TRAIN net −0.088 → ex-top-1 %
−0.109 → ex-top-5 % −0.198; VAL −0.050 → −0.073 → −0.158. The book is not a lottery ticket whose
edge lives in a tail — it has no edge to concentrate.

**Forward, the live dry run (not a cell).** 31 booked trades over 4 sessions 9/14–9/18: gross
−0.341 R, **net −0.454 R**, −$1,408 at $100 risk, WR 25.8 %, 1 green session of 4. Directionally
consistent with −0.088 R but far worse; at n = 31 the standard error is ~0.20 R, so it is not a
verdict in either direction — it is the instrument working.

---

## What would change this verdict

Nothing inside this gate set. Two things could:

1. **A causal filter with GROSS separation, not cost separation.** Every gate that scores well here
   (`Gf`, `Ge`, `Gg`) earns 70–85 % of its number from the cost term. The only gate with real gross
   separation is the price floor (+0.061 gross, era-consistent). A filter worth +0.25 R of GROSS
   would be needed, and the causal-filter study already searched 17 features and found its best
   tercile spread to be an H1-2025 effect.
2. **A re-shaped STOP.** `research/green_weeks` settled that the shipped +2R target is the best of
   12 exit cells for green weeks, but every cell there left the average loss at −1.03 R. The gross
   problem is a 40 % win rate against a 2:1 payoff the stop keeps not surviving — re-shaping the
   stop (not the target) is the only untested direction, and it is a new pre-registration.

**Recommended action: NONE.** `hod_break.enabled: true, dry_run: true` stays exactly as the owner
set it on 9/19. Going live still needs a new pre-registration and the owner's word, and this study
supplies no grounds for one.
