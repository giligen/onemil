# REPORT — FINAL multi-day stage: F5, A2, A3, A4, F1, F6 (11 cells) + the program-level close on 26

Run 2026-09-18 on the panel in `DATA.md`. Code: `data/fetch_final.py` (three new free sources),
`build_panel_final.py` (opens + the exact split factor), `run_final.py` (the 11 cells + every arm).
Artifacts: `out_final/` (primary) and `out_final_{adv10m,flat5,diss12}/` (arms),
`out_indep_final/` (the independent rebuild). Corrections from `REPORT_F2_A1.md` §9 and
`REPORT_F4_F3.md` are inherited wholesale and are not re-litigated.

**Verdict in one line:** *no edge was detectable in the US-common-stock universe above $5 and $1M
ADV20$, at a 6-month-held 52-week-high tilt, a semi-monthly short-interest sort, a monthly
net-share-issuance sort, a predicted-dividend-month long, a 60-session SUE drift, or a 12-1
overnight-vs-intraday cross-section, at a $66K / 20-slot book, over 2016-01 → 2023-12, at
close-to-close auction cost — and the smallest monthly effect the BOOK could have seen is
**$443–$2,307/month**, against a Chen–Velikov prior of **$25–65/month**. **0 of 11 cells clear G1**
(best positive TRAIN t = 0.10); **0 of 11 clear G2** on a positive point estimate. TEST was NOT
opened and no `FREEZE.md` was written.*

**Two of the eleven are not nulls, and both are negative results with enough power to mean something:**

1. **A2's published LONG-leg claim does not replicate — the TRAIN sign is significantly the wrong
   way.** Boehmer–Huszár–Jordan (JFE 96(1) 2010) put the significant abnormal return on the LOW
   short-interest (long) side. On TRAIN (2018-02 → 2021-12, the whole history FINRA gives us) the
   per-trade net mean rises **monotonically with short interest**: D1 +37.2 → D2 +61.0 → D3 +65.4 →
   D9 +69.7 → **D10 +88.2 bps**. The long-only cell is **−65.7 bps/month, t = −2.05** (Newey–West
   −2.12), block-bootstrap p = 0.037. It **strengthens** when the dissemination lag is made more
   conservative (12 business days: −77.3 bps, t = −2.38), so it is not a look-ahead artefact. On VAL
   the ordering flips to the published direction (D10 −30.7 bps, t −1.97) without the portfolio
   series reaching significance (+19.6 bps, t 0.36). **Answer to the pre-registered question: no,
   the published long-leg claim does not replicate here.**
2. **A4 is the one cell in the entire 26-cell program that is POWERED for its own published effect —
   and it refutes it at this universe.** The dividend-month premium (Hartzmark–Solomon, and the 2025
   AER 115(9) 3171-3213 follow-up) is a **same-firm, adjacent-month** contrast: predicted dividend
   payers against dividend payers in a non-dividend month. That makes its MDE tiny (**13.3 bps/month
   on TRAIN, 23.6 on VAL**) and makes it the least survivorship-exposed cell in the program, because
   both sides are drawn from the identical surviving population in adjacent months. HS's published
   premium is of order 40 bps/month. We measure **−0.6 bps (t −0.09) on TRAIN and −2.3 bps (t −0.19)
   on VAL** — i.e. an effect three times our MDE would have been seen and was not.

---

## 0. Pre-registration status — stated honestly, because it is not clean

What was fixed **before any return was computed**: the amended family grid and cell budget in
`PLAN.md` (F5 2 cells with January split out, A2 2 on the long side, A3 2, A4 2, F1 2 as a *declared
null-replication*, F6 1 measurement cell), the four mandatory columns, the gates, the splits, the
cost model, and the specific constructions named in the task brief — A4's **two-regime** ex-date
rule, A2's **dissemination-date** keying, F1's framing against Martineau (CFR 11(3-4) 2022). Those
are the scored cells and none of them moved.

What was added **after the first scoring run**, in response to the independent rebuild (§9), and is
therefore a diagnostic rather than a pre-registered statistic: the symmetric-tail column
(`sym_top_*`), the unreverted-reverse-split taint arm (`ex_taint_*`), the empty-exec-book-month count
and the held-months-only book alpha, the A2 per-decile table, and the universe-hygiene counts.
**None of them changes a verdict**; three of them make the reading of the *inherited* columns more
honest, which is why they are here. No pre-registration document was written for this stage before
the run — the prior two stages had one, this one did not, and that is a process regression to record.

---

## 1. What was run

Execution **close-to-close, `cls` both legs, one session between the signal close and the entry
close**. No open is traded anywhere (AMENDMENT 2(a), Goyal–Jegadeesh–Wu JFQA 2026). No quoted spread
on an auction cross. Costs: impact `10 bps × (order$ / 1% of ADV20$)` both sides, **order$ scaled to
the number of positions the portfolio actually holds on that trade's entry day** (F2/A1 defect 3 —
measured mean concurrent positions here: F5 1,521, A4 796, F6 451, A3 354, A2 353, F1 187, i.e. the
academic order is **$48–$441**, not $3,300), + 0.4 bps SEC/TAF on the sell, + 0.3%/yr borrow on any
short leg gated on `easy_to_borrow`. **Costs are charged to the book only; the benchmark is gross.**
Benchmarks exclude the book's own decile. Sealed splits; VAL reported both ways (the two never
differ by more than 36 bps and never change a sign except F1-LS, noted in §4).

**Reference population for every decile sort, stated explicitly:** the eligible universe *at that
rebalance* — `kind=='common'`, RAW close ≥ $5, **RAW-panel** ADV20$ ≥ $1M, adjusted close finite.
Mean cross-section ≈ 2,100–2,200 names. F1 is the exception and is stated separately: its decile is
cut against **every prior eligible SUE-computable event in the trailing 250 calendar days**, never
against the contemporaneous cross-section (which would be a look-ahead) and never against the full
sample.

**New data, all free, all point-in-time** (`data/fetch_final.py`, sizes in §8):
* **FINRA consolidated equity short interest** (A2) — `api.finra.org`, 2.62M rows, 144 settlement
  dates, **history begins 2018-02-15**; earlier dates return zero rows. A2's TRAIN is therefore
  **2018-02 → 2021-12 (48 months), not 72.**
* **Alpaca cash dividends** (A4) — 280,512 rows, 15,013 symbols, 2016-01-04 → 2025-02-14.
* **EDGAR `dei:EntityCommonStockSharesOutstanding`** (A3, and A2's denominator) — 154,434 cover-page
  facts on 3,913 CIKs, each with its `filed` date; `us-gaap:CommonStockSharesOutstanding` fallback.
* **The exact corporate-action factor** `(adj_t/adj_{t−1})/(raw_t/raw_{t−1})` — this is Alpaca's own
  factor, not a detector, and `REPORT_F4_F3.md` §6 established it as ground truth. 7,456 share-count
  events on 1,667 symbols; `shares(t)/splitcum(t)` is a share count on ONE basis across time, which
  is what A3's 12-month issuance ratio and A2's short-interest denominator need. F3's PIT re-run
  could not do this because it had only an unadjusted venue tape and had to *detect* splits at 38%
  precision; here we own both panels and the factor is exact.

**Declared construction deviations.** (i) F5 uses a 6-month overlapping hold at monthly rebalance
(George–Hwang's published form) and starts **2017-02** because it needs 252 sessions of history —
TRAIN is 59 months, not 72. (ii) F6 is scored at a **12-1-month formation** on cumulated overnight
(long) and intraday (short) returns; Lou–Polk–Skouras use other horizons, so §7's rejection applies
to *this* construction, not literally to theirs. (iii) F1's SUE is the Foster–Olsen–Shevlin seasonal
random walk with drift over the last 8 available seasonal differences (minimum 6), computed only from
facts **filed strictly before** the 8-K acceptance instant; computable on 75.9% of events, eligible on
73,272, ranked on 73,002.

---

## 2. The 11-cell table, per split

`bps` = monthly benchmark-adjusted excess. `t` ordinary; `NW t` Newey–West at lag = hold length in
months. `ex-1% / ex-5%` = the portfolio **rebuilt** without the top 1% / 5% of trade contributions
(mandatory, inherited). `sym-1% / sym-5%` = the same trim applied to **both** sides (§9 C4 — the
one-sided version drags a long-only excess negative by arithmetic). `MDE` = 2 × SE of the monthly
mean. VAL rows are the **sealed** convention.

| cell | split | n mo | n trades | **excess bps** | **t** | NW t | %mo+ | **ex-1%** | **ex-5%** | sym-1% | sym-5% | ex-Jan | Jan-only | **MDE** | tr/wk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F5-LO | TRAIN | 59 | 10,371 | **−4.1** | −0.13 | −0.16 | 49.2 | −21.7 | −53.8 | −6.4 | −5.8 | −5.7 | +17.2 | 61.5 | 0.58 |
| F5-LO | VAL | 24 | 4,234 | **+40.2** | 0.71 | 0.79 | 62.5 | +28.6 | −1.1 | +43.6 | +46.8 | +70.5 | −292.9 | 113.5 | 0.58 |
| F5-LS | TRAIN | 59 | 19,827 | −38.7 | −0.46 | −0.48 | 50.9 | −66.5 | −151.3 | −66.5 | −151.3 | −34.8 | −92.3 | 168.4 | 0.58 |
| F5-LS | VAL | 24 | 8,033 | +103.3 | 0.65 | 0.70 | 58.3 | +65.4 | −48.2 | +65.4 | −48.2 | +121.2 | −93.9 | 316.2 | 0.58 |
| **A2-LO** | TRAIN | 48 | 15,492 | **−65.7** | **−2.05** | **−2.12** | 37.5 | −126.6 | −235.4 | −59.6 | −36.1 | −62.7 | −98.5 | 64.1 | **6.01** |
| **A2-LO** | VAL | 24 | 9,085 | +19.6 | 0.36 | 0.35 | 58.3 | −35.7 | −136.6 | +31.0 | +61.9 | +24.6 | −35.4 | 109.8 | **9.03** |
| A2-LS | TRAIN | 48 | 29,505 | −129.2 | −1.70 | −1.72 | 39.6 | −269.8 | −573.7 | −269.8 | −573.7 | −98.4 | −468.9 | 152.2 | 6.01 |
| A2-LS | VAL | 24 | 17,480 | +46.7 | 0.37 | 0.35 | 58.3 | −101.0 | −427.6 | −101.0 | −427.6 | +67.8 | −186.2 | 250.2 | 9.03 |
| A3-LO | TRAIN | 59 | 8,996 | **+2.6** | **0.10** | 0.10 | 50.9 | −41.2 | −123.3 | +3.7 | +8.7 | −1.8 | +63.7 | 53.7 | 3.71 |
| A3-LO | VAL | 24 | 4,149 | +65.1 | 1.46 | 1.53 | 54.2 | +32.3 | −40.0 | +75.4 | +90.6 | +41.3 | +326.3 | 89.0 | 4.42 |
| A3-LS | TRAIN | 59 | 16,836 | +1.0 | 0.02 | 0.02 | 47.5 | −91.7 | −293.7 | −91.7 | −293.7 | +7.7 | −91.0 | 91.2 | 3.71 |
| A3-LS | VAL | 24 | 7,893 | +143.0 | 1.68 | 1.68 | 58.3 | +47.7 | −162.4 | +47.7 | −162.4 | +97.0 | +648.3 | 170.2 | 4.42 |
| **A4-LO** | TRAIN | 58 | 20,848 | **−0.6** | −0.09 | −0.11 | 46.6 | −29.3 | −90.8 | +1.1 | +4.5 | −2.2 | +20.9 | **13.3** | 3.64 |
| **A4-LO** | VAL | 24 | 9,329 | **−2.3** | −0.19 | −0.17 | 54.2 | −26.3 | −84.9 | −1.4 | −1.2 | +1.1 | −39.0 | **23.6** | 4.42 |
| A4-LS | TRAIN | 58 | 63,452 | −6.4 | −0.88 | −1.02 | 43.1 | −72.6 | −212.9 | −72.6 | −212.9 | −8.6 | +23.3 | 14.6 | 3.64 |
| A4-LS | VAL | 24 | 28,279 | −6.2 | −0.54 | −0.49 | 50.0 | −62.0 | −186.5 | −62.0 | −186.5 | −3.2 | −39.4 | 22.8 | 4.42 |
| F1-LO-60 | TRAIN | 71 | 3,697 | −17.6 | −1.03 | −1.08 | 54.9 | −36.9 | −77.0 | −15.7 | −13.9 | −22.3 | +44.3 | 34.1 | 1.47 |
| F1-LO-60 | VAL | 24 | 1,024 | +38.6 | 0.62 | 0.52 | 54.2 | +19.0 | −15.0 | +40.3 | +51.8 | +1.5 | +446.6 | 124.1 | 1.34 |
| F1-LS-60 | TRAIN | 71 | 6,881 | −22.3 | −0.87 | −0.94 | 52.1 | −56.7 | −137.1 | −56.7 | −137.1 | −29.3 | +71.3 | 51.0 | 1.47 |
| F1-LS-60 | VAL | 24 | 2,413 | −34.4 | −0.51 | −0.45 | 45.8 | −70.7 | −161.5 | −70.7 | −161.5 | −56.9 | +213.4 | 134.1 | 1.34 |
| **F6-LS** | TRAIN | 58 | 22,004 | −26.1 | −0.69 | −0.73 | 41.4 | −142.9 | −385.6 | −142.9 | −385.6 | −42.7 | +197.9 | 75.6 | 3.64 |
| **F6-LS** | VAL | 24 | 10,554 | **−129.9** | **−2.58** | **−3.01** | 33.3 | −222.4 | −436.0 | −222.4 | −436.0 | −105.9 | −394.7 | 100.8 | 4.42 |

**How to read the tail columns now.** The independent rebuild's C4 is right and it changes what the
inherited `ex-1%/ex-5%` column licenses: for a long-only cell benchmarked against a long-only
cross-section, trimming the book's winners and not the benchmark's drives the excess negative by
arithmetic. The `sym-` columns trim both sides at the same quantile. Under the symmetric trim
**F5-LO, A3-LO, A4-LO and F1-LO survive their own tails** (their point estimates barely move), and
**A2-LO and F6-LS stay negative**. What survives from the one-sided reading is the *relative* fact:
every L-S cell in this stage is heavily tail-dependent on both readings (−48 to −574 bps at 5%), and
no long-only cell has a point estimate large enough for the distinction to matter. It also means
the ex-top-1% conclusions in `REPORT_F2_A1.md` §9 defect 2 and `REPORT_F4_F3.md` §2 are *partly*
this arithmetic and should be re-read with that caveat; they are not withdrawn, because in those
stages the cells being killed had point estimates of +56 to +137 bps, not +2.6.

### The four mandatory columns, read out

1. **Long-leg share.** Defined only where the L−S spread is positive: **A2-LS VAL 51%**, **A3-LS VAL
   48%**, **F5-LS VAL 45%**. All three are ~half, matching Israel–Moskowitz's published ~50% and
   *worse* than F2's 87–98%. Undefined everywhere else (a share of a spread ≤ 0 is not a number).
   Structurally, half the prize of every family in this stage sits in a short leg this account cannot
   reach without margin and reliable borrow.
2. **Break-even cost.** The honest auction round trip is **0.41–0.50 bps** long-only (the whole
   charge is the 0.4 bps SEC/TAF; impact at a $48–$441 order against ≥ $1M ADV is 0.01–0.02 bps) and
   **1.05–7.57 bps** for an L-S cell (borrow, scaled by hold: F5-LS holds 126 sessions). Break-evens
   on TRAIN are **negative for 7 of 11 cells** — no cost reduction whatsoever makes them positive —
   and where positive they are small: A3-LO 3.0 bps/round trip (6.8× the honest cost), A3-LS 4.4 bps,
   everything else ≤ 0. Only on VAL do the F5/A3/A2 break-evens become large (21–589×), which is the
   sign flip, not a cost result.
3. **Ex-January.** Nothing here is a January effect, including the family the literature flags for it.
   George–Hwang's headline is 0.45%/mo raw vs 1.23% ex-January; **our F5-LO reads −4.1 all-months vs
   −5.7 ex-January on TRAIN** (January itself +17.2 bps over 5 Januaries) and +40.2 vs +70.5 on VAL
   (two Januaries, both bad, −292.9 bps mean). The January column has **n = 2 on VAL and 5–6 on
   TRAIN** and is not capable of carrying a conclusion — reported because it is mandatory, not
   because it resolves anything. No cell is rescued or killed by it.
4. **Trades/week at $50–66K, and additivity vs ORB.** **A2 is the only cell in this stage that clears
   the owner's ≥10/week column, at 6.0–9.0/week** — because 20 slots over a ~10-session semi-monthly
   hold is ~9 entries a week. A4/F6 run 3.6–4.6, A3 3.7–4.4, F1 1.3–1.5, and **F5 is the slowest cell
   in the whole program at 0.58/week** (20 slots over a 126-session hold). Additivity: §6.

---

## 3. The executable $66K / 20-slot book — the money object, and its power

Alpha basis = the 20-slot book's own equal-weighted daily series, net of cost, **minus the gross
benchmark**. Long-only in every row (this account has no margin and no reliable borrow, so the two
L-S rows share their long cousin's book by construction).

| cell | split | book trades | tr/wk | **$/month (alpha)** | $/mo held-months only | empty months | t | **MDE $/mo** | %mo+ | worst month | book MDD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F5-LO | TRAIN | 180 | 0.58 | +156 | +101 | **4 of 59** | 0.60 | **521** | 52.5 | −4,675 | −15,887 |
| F5-LO | VAL | 60 | 0.58 | −299 | −515 | **5 of 24** | −0.46 | **1,306** | 58.3 | −7,336 | −15,667 |
| A2-LO | TRAIN | 1,880 | 6.01 | −553 | −553 | 0 | −1.33 | **832** | 54.2 | −8,885 | −31,772 |
| A2-LO | VAL | 940 | 9.03 | +329 | +329 | 0 | 0.65 | **1,011** | 54.2 | −4,596 | −7,930 |
| A3-LO | TRAIN | 1,160 | 3.71 | −105 | −105 | 0 | −0.16 | **1,318** | 42.4 | −7,746 | −25,778 |
| A3-LO | VAL | 460 | 4.42 | +999 | +999 | 0 | 1.82 | **1,099** | 58.3 | −4,799 | −4,922 |
| A4-LO | TRAIN | 1,140 | 3.64 | +323 | +323 | 0 | 1.33 | **486** | 60.3 | −2,769 | −6,956 |
| A4-LO | VAL | 460 | 4.42 | −116 | −116 | 0 | −0.35 | **670** | 41.7 | −3,234 | −8,466 |
| F1-LO-60 | TRAIN | 461 | 1.47 | −209 | −209 | 0 | −0.94 | **443** | 40.9 | −10,254 | −18,160 |
| F1-LO-60 | VAL | 140 | 1.34 | −142 | +339 | 2 of 24 | −0.24 | **1,199** | 45.8 | −6,171 | −16,533 |
| F6-LS | TRAIN | 1,140 | 3.64 | +308 | +308 | 0 | 0.37 | **1,670** | 44.8 | −16,671 | −36,677 |
| F6-LS | VAL | 460 | 4.42 | **−2,558** | −2,558 | 0 | **−2.22** | **2,307** | 33.3 | −14,179 | −41,235 |

**This table is the report's second half.** Every |t| ≤ 1.9 except F6-LS VAL at −2.22 (negative).
The MDE column says the book could not have detected anything smaller than **$443–$2,307/month**,
while the pre-committed calibration prior (Chen & Velikov: 204 published anomalies net ~4 bps/month)
puts the expected effect at **$25–65/month on $66K**. **The book is 7–92× too blunt** — F2/A1
measured 10–120×, F4/F3 16–80×, and this is the third independent measurement of the same property
of $66K and 20 slots.

**The empty-month defect, found by the independent rebuild (§9 C2) and confirmed.** The alpha basis
scores a month in which the 20-slot book holds *nothing* as a **short of the benchmark**. It bites on
the two slowest cells: F5-LO has 4 empty TRAIN months and **5 of 24 on VAL**, and the held-months-only
alpha is **−$515 vs the −$299 reported** (F1-LO VAL: −$142 → **+$339**). Both F5 rows are therefore
worse than they look and F1's VAL row is better; neither changes a verdict, and both are disclosed
rather than quietly fixed. Cells with a continuous book (A2, A3, A4, F6) are unaffected.

**Bounded downside, the one thing well measured.** A $66K long-only auction sleeve at 20 positions,
no leverage, no short, no stop: worst month **−$2,769 to −$16,671**, book drawdown **−$4,922 to
−$41,235**, i.e. **7–62% of the sleeve**. F6-LS's −$41K book drawdown on a t = 0.37 TRAIN estimate is
the worst risk-to-evidence ratio in the whole 26-cell program.

---

## 4. Gates, tails, multiplicity, arms

**G1 (TRAIN t ≥ 2 on the benchmark-adjusted monthly series): 0 of 11.** The best *positive* TRAIN t
is **A3-LO at 0.10**. Two cells exceed |t| = 2 with the **wrong** sign (A2-LO −2.05; A2-LS −1.70, NW
−1.72, misses).
**G2 (VAL same sign AND ≥ 55% of months positive): 0 of 11 on a positive point estimate.** Exactly
one cell clears G2's *letter* — **A3-LS**, TRAIN +1.0 bps → VAL +143.0 bps, 58.3% of months positive
— on a TRAIN t of **0.02**, which is the gate ordering working as designed: G2 is only meaningful
behind G1 and G1 rejected it. A3-LO misses on the months-positive half (54.2% < 55%). A4-LO, A4-LS,
F1-LS and F6-LS keep their sign on VAL and it is **negative**, which is not a pass.
**TEST (2024-01 → 2026-09) stays sealed.** `SPLIT_ORDER` is `('TRAIN','VAL')`; no `FREEZE.md` exists;
no TEST return, series or statistic was computed anywhere in this stage. The only 2024+ data touched
is the dividend **date** table used for §5's reconciliation, which carries no price.

**Robustness arms — the verdict is identical in all four** (`out_final_*`):

| arm | A2-LO TRAIN | A4-LO TRAIN | A4-LO VAL | F6-LS VAL | F5-LO TRAIN | best positive t anywhere |
|---|---|---|---|---|---|---|
| primary ($1M ADV, auction cost, 8-bd diss lag) | −65.7 (t −2.05) | −0.6 | −2.3 | −129.9 (t −2.58) | −4.1 | 0.10 (A3-LO) |
| ADV20$ ≥ $10M | −34.4 (t −1.00) | +4.4 | −9.1 | −131.4 (t −2.49) | −3.2 | 0.50 (A4-LO) |
| flat 5 bps/side | −84.2 (t −2.64) | −10.0 | −11.5 | −148.0 (t −2.95) | −5.8 | — (none positive) |
| A2 dissemination lag 12 bd | −77.3 (t −2.38) | — | — | — | — | — |

Two things to take from the arms. (i) **A2's wrong-sign result is not a look-ahead**: making the
dissemination lag *more* conservative makes it *stronger*, and the monotone TRAIN decile ordering
survives at 12 bd (D1 +35.5 → D10 +102.5 bps). (ii) Raising the liquidity floor to $10M halves it
(t −1.00), so what little signal there is lives in the $1M–$10M band — the same "it is in the cheap
names" shape F4 had, with the opposite sign.

**Multiplicity.** 11 scored cells this stage × 4 arms = **44 distinct looks**. Block-bootstrap p
(block 3 months, 5,000 resamples) on the TRAIN excess, Šidák-adjusted across the 11 cells:

| cell | raw p | Šidák | | cell | raw p | Šidák |
|---|---|---|---|---|---|---|
| A2-LO | 0.037 | **0.34** | | F5-LS | 0.624 | >0.99 |
| A2-LS | 0.114 | 0.74 | | F5-LO | 0.875 | >0.99 |
| A4-LS | 0.281 | 0.97 | | A4-LO | 0.916 | >0.99 |
| F1-LO-60 | 0.320 | 0.99 | | A3-LO | 0.932 | >0.99 |
| F1-LS-60 | 0.367 | 0.99 | | A3-LS | 0.985 | >0.99 |
| F6-LS | 0.517 | >0.99 | | | | |

Nothing is close, before or after adjustment — including A2-LO, whose Šidák-adjusted p is 0.34.
**Cumulative multi-day scored-cell count: K 20 + N2 4 + R_daily 20 + F2/A1 8 + F4/F3 7 + 11 here = 70.**

**Survivorship — declared, with the per-family reason.** The panel is 100% survivors
(`REPORT_F2_A1.md` §9 defect 1; `delisted_names.parquet` has zero ticker overlap with
`universe.parquet` and those names were never priced). Per the task's item 3 the **GROSS / level
column is NOT quotable for any cell in this stage** and only the benchmark-differenced column is
scored. The −100% delisting haircut arm was built and is **inert on all 11 cells**: `ends_mid_hold
= 0` everywhere — on a survivor panel no position ever loses its price mid-hold, which is precisely
the point (the haircut cannot bite on names that were never there). A PIT re-run was **not possible**
for any family here and the reasons are specific, not a shrug: F5 needs a 252-session formation on a
delisting-inclusive tape, and the only such tape we own is unadjusted Nasdaq-only, where the split
detector validates at 38% precision (`REPORT_F4_F3.md` §6 — the same blocker that killed F3's
re-run); A2/A3 key on FINRA symbol codes and EDGAR CIKs that do not exist for the delisted cohort;
F1's events are 8-K filings whose issuers are gone. **A4 is the exception and it matters**: its
contrast is the *same firms* in a predicted vs a non-predicted month, weeks apart, so the survivorship
bias is differenced out almost exactly — which is why §0's second finding is stated as a refutation
rather than a null.

**Other rails.** Price scale: cleared at the data stage (0 of 200 keys fail at 0.01%). Causality:
every field is computable at or before its decision close — F5's 252-session window ends AT the
decision close, A2's short interest is keyed to a **derived dissemination date** (settlement + 8
business days; the 12-bd arm is the guard and strengthens the result), A3's share counts are the last
facts **filed** before the decision close, A4's dividend calendar uses only ex-dates strictly before
it, F1's EPS facts are filed strictly before the 8-K acceptance instant, and every entry is one
session *after* the signal. Obtainability: every fill is an official consolidated closing-auction
print decided one session in advance on both legs — no gap-through, no intrabar stop, no touch fill,
no double-counted slip.

---

## 5. A4's two-regime ex-date construction — the trap, and the count

US settlement moved T+2 → T+1 on **2024-05-28**, four months inside the sealed TEST window. Under
FINRA Rule 11140(b)(1) the ex-dividend date was **record − 1 business day** before that date and **is
the record date** from it. Reconciled against Alpaca's actual `ex_date` on the **177,746** cash
dividends that carry a record date (dates only — no TEST price was read):

| era | events with a record date | wrong under a single "ex = record − 1 bd" | wrong under a single "ex = record" | **wrong CALENDAR MONTH**, "−1 bd" | **wrong month**, "= record" | residual under the correct two-regime rule |
|---|---|---|---|---|---|---|
| pre (ex < 2024-05-28) | 152,893 | **12,277** (8.0%) | 152,408 (99.7%) | 586 | **8,076** | 12,277 |
| post (ex ≥ 2024-05-28) | 24,853 | **24,331 (97.9%)** | 590 (2.4%) | **3,362** | 125 | 590 |

Month by month across the boundary:

| window | n | wrong "−1 bd" | wrong "= record" |
|---|---|---|---|
| 2024-04 | 2,387 | 30 | 2,380 |
| 2024-05-01..27 | 2,411 | 109 | 2,409 |
| 2024-05-28..31 | 621 | **581** | 41 |
| 2024-06 | 4,558 | **4,451** | 112 |
| 2024-07 | 2,336 | **2,326** | 14 |

**A single-regime formula mis-dates 97.9% of post-boundary events** and puts **3,362 of them in the
wrong calendar month** — which for a monthly dividend-month strategy is the difference between being
in the trade and being in the opposite one. The switch is unambiguous in the data and lands exactly
on 2024-05-28. **Our A4 never uses either formula**: it dates every event by Alpaca's own `ex_date`
field, so the cell is immune by construction — the reconciliation exists to prove the immunity and to
price what a record-date-derived calendar would have cost. The independent rebuild reproduced every
count in this table **exactly**, and added one correction of its own: `pd.offsets.BDay` is
holiday-blind, and recomputing "one business day before" on the real session calendar cuts the pre-era
residual from **12,277 to 5,958** — i.e. roughly half the apparent pre-era mis-dating is the
*formula's* holiday handling, not the data's. Of what survives, 19% are `foreign == True` issues,
which legitimately break the record-date relation.

---

## 6. Additivity vs the live ORB book

**Resource overlap: none on ORB's binding constraint.** ORB's binding resources are 4 concurrent
intraday slots, 09:35 buying power and the 09:35 attention window; every ORB position is flat by
15:45. All six families here open in the **16:00 closing auction** and hold 10 to 126 sessions. They
never contend for an ORB slot, never need capital at 09:35 on the day they enter, and never touch the
ORB engine, StopMonitor or the intraday tape. **Structurally additive on all three.** The one real
interaction is unchanged: $66K of equity held overnight reduces the next morning's day-trading buying
power by that notional in a margin account; ORB's stage budget is $10K so the two fit today, but it
is an account-level check to run before any multi-day sleeve is armed. Stage I's "stacking hurts" is
**not** imported (that was same-day intraday books fighting the same slots).

**Combined trades/week.** ORB runs ~2.5/week. A2 at 6.0–9.0/week would take the combined book to
**8.5–11.5/week**, the first multi-day + ORB pairing that would reach the owner's ≥10/week resolution
bar. It is moot: A2-LO's TRAIN estimate is significantly *negative*. F5 at 0.58/week would add
nothing to resolution and nothing to money.

**Return correlation with ORB: NOT COMPUTED, and the reason is the seal.** The live ORB book runs
2025-01-07 → 2026-09-15, entirely inside sealed TEST. Computing the correlation means opening TEST,
and no cell earned that. There are zero overlapping months on TRAIN/VAL, so there is no partial
answer. This is the third stage in a row that has had to say so; the seal is worth more than the
correlation.

---

## 7. F6 — the measurement cell, which is the one thing in this stage that replicated

PLAN.md demoted F6 to a single **measurement** cell on Haghani–Ragulin–Dewey's arithmetic (1 bp
round-trip removes ~5 pts/yr from a 38%-gross overnight L-S) and our own `overnight_auction.md`
(+25/+15/**−12** bps) and M29 (+7.4/+6.7/**−13.4**). Both halves were run.

**(a) The decomposition replicates, strongly.** Cross-sectional mean daily return on the eligible
universe, split at the opening print (`data/panel_final.npz::open_adj`; the ADV and $5 gates apply):

| year | names | overnight bps/day | intraday bps/day | **overnight %/yr** | **intraday %/yr** |
|---|---|---|---|---|---|
| 2016 | 1,969 | +2.2 | +8.4 | +5.5 | +21.1 |
| 2017 | 2,104 | +6.8 | +1.8 | +17.1 | +4.5 |
| 2018 | 2,247 | +6.1 | −8.0 | +15.4 | **−20.1** |
| 2019 | 2,322 | +6.4 | +4.7 | +16.2 | +11.9 |
| 2020 | 2,627 | **+18.9** | −1.7 | **+47.8** | −4.3 |
| 2021 | 3,154 | +14.1 | −5.0 | +35.4 | **−12.7** |
| 2022 | 2,910 | −2.4 | −4.1 | −5.9 | −10.3 |
| 2023 | 2,811 | +6.5 | +2.6 | +16.3 | +6.5 |

**In 6 of 8 years essentially all of the equity return is earned overnight, and in 4 of 8 the
intraday session is negative.** That is the Lou–Polk–Skouras fact reproduced on our own panel, and it
is the *second* structural prediction in the whole multi-day program to replicate (the first was
Blitz–Huij–Martens' halved momentum crash).

**(b) The tradable cross-section of it is negative, and this cell had the power to say so.** The
12-1-formation L-S (long past-overnight winners, short past-intraday winners, monthly) reads
**−26.1 bps/month on TRAIN and −129.9 bps/month on VAL, t −2.58, NW −3.01**, 33% of months positive,
book −$2,558/month with a −$41K drawdown. The MDE is 75.6 / 100.8 bps against a published L-S of
order 300 bps/month — **this cell was powered by ~3–4× and it rejected the sign, not just the size.**
Caveat stated once and plainly: the 12-1 formation is *our* construction, not LPS's horizon, so the
rejection is of this construction. The honest reading of (a) + (b) together is the one the owner
already has from `overnight_auction.md`: **the overnight premium is real and is not reachable by a
cross-sectional long-short at monthly rebalance and auction cost.**

---

## 8. Data added this stage, and what it costs to keep

| source | rows | span | cost | gap it closes |
|---|---|---|---|---|
| FINRA consolidated short interest | 2,620,837 | 2018-02-15 → 2023-12-29, 144 settlement dates, 40,190 symbols | free, no key | A2's signal. **History starts 2018-02** — 2018-01-15 and earlier return zero rows, truncating A2's TRAIN to 48 months |
| Alpaca cash dividends | 280,512 | 2016-01-04 → 2025-02-14, 15,013 symbols | free (existing key) | A4's calendar and §5's reconciliation. `record_date` present on 63.4% of rows, essentially none before 2019-07 |
| EDGAR `dei:EntityCommonStockSharesOutstanding` | 154,434 facts, 3,913 CIKs | 2009 → 2026 | free | A3's issuance numerator and A2's RSI denominator, each with a point-in-time `filed` date |
| exact corporate-action factor (derived) | 7,456 events, 1,667 symbols | 2016 → 2026 | free | the split normalisation both of the above need. Largest cumulative multipliers are serial reverse-splitters (BINI 1.2e−18, JAGX 1.6e−11, TOPS 4.6e−11) — correct direction, sanity-checked |

`data/short_interest.parquet` 47 MB, `dividends.parquet` and `shares_facts.parquet` a few MB,
`panel_final.npz` (opens + split factor) 40 MB. All gitignored with the rest of the bulk panel.

---

## 9. Independent rebuild (CLAUDE.md "No research claim ships without an independent check")

A second implementation was written from a **prose specification only**, by an agent explicitly
forbidden to read `run_final.py`, `build_panel_final.py`, any prior stage's code, or any
`REPORT_*/PREREG_*/PLAN/DATA` page (`indep_final.py`, `out_indep_final/`). It rebuilt **F5-LO**,
**A4-LO** and the §5 reconciliation from the raw year-partitioned parquets: its own session calendar,
its own RAW-panel ADV20, its own deciles, its own overlapping portfolio, its own tail rebuilds. It
never loaded a 2024+ price file, so the seal was enforced at the I/O layer on its side.

**(a) The numbers agree.**

| cell | split | mine: trades / bps / t | independent: trades / bps / t | mine MDE | indep MDE | mine cost bps | indep cost bps |
|---|---|---|---|---|---|---|---|
| F5-LO | TRAIN | 10,371 / −4.1 / −0.13 | 10,379 / **−5.5** / **−0.18** | 61.5 | **62.1** | 0.41 | **0.412** |
| F5-LO | VAL | 4,234 / +40.2 / 0.71 | 4,234 / **+47.8** / **0.81** | 113.5 | **117.8** | 0.41 | **0.410** |
| A4-LO | TRAIN | 20,848 / −0.6 / −0.09 | 21,399 / **−2.7** / **−0.40** | 13.3 | **13.5** | 0.42 | **0.421** |
| A4-LO | VAL | 9,329 / −2.3 / −0.19 | 9,322 / **−1.8** / **−0.16** | 23.6 | **23.2** | 0.42 | **0.418** |

Trade counts agree to 0.08% (F5 VAL exact) and 2.6% (A4 TRAIN, from the 24-month history window's
treatment at the panel start); every MDE agrees within 4.3 bps; mean cost agrees to three decimals on
all four rows. **The §5 reconciliation reproduced exactly, every cell of the table.** A coding error
is ruled out for the calendar, the ADV gate, the eligibility screen, the decile, the 6-month
overlapping hold, the dividend-month prediction, the cost model and the tail rebuild. Per the standing
rail this cannot rule out a *specification* error — and it found four.

**(b) Specification defects it found, all adopted as columns in §2/§3 rather than argued with.**
C4, the one-sided tail trim (→ the `sym-` columns, §2). C2, empty-exec-book months scored as a short
of the benchmark (→ the `held-months only` and `empty months` columns, §3; F5-LO VAL is 5 of 24).
C1, the concurrency-scaled order size makes the academic column effectively gross of impact (0.010–
0.021 bps measured) — correct, and it is why §2 is labelled a return estimate and §3 the money
estimate. C3, the sealed-split rule interacting with a globally-simulated slot book drops ~35% of
F5's generated trades and ramps the cohort count at both split seams. C6, the January column has n=2
on VAL — stated in §2. C7, A4's 24-month history window is starved before 2018-01.

**(c) A panel defect the rebuild found, quantified here, and NOT load-bearing.** It reported that the
"adjusted" panel contains **reverse splits applied forward and never back-propagated**: DCTH
2020-04-30 $0.0999 → 2020-05-01 $7.66 (+7,568%, then flat), PFH +3,740%, CRC +1,072%. These are a
*different* population from `DATA.md` gap 10's Chapter-11 ticker-recycling jumps, and unlike those
they **clear the RAW $5 gate**. Measured on the full 2016 → 2026 panel: **335 unreverted >+200%
adjusted sessions on 284 symbols; 203 of them with a raw close ≥ $5**; a 252-session taint window
covers **0.56%** of symbol-sessions. The mechanism the rebuild named is real — after such a break the
trailing 252-session adjusted max IS the post-break level, so F5's nearness ≈ 1.0 and the name is a
guaranteed decile-10 member for a year. **Every cell was re-scored without the tainted trades**
(§2's `ex_taint` column in `cells.csv`): the tainted share is **0.0% (A4) to 2.2% (F6-LS TRAIN)** and
the largest move in any point estimate is F6-LS TRAIN −26.1 → −17.5 bps. F5-LO's own tainted share is
0.25% TRAIN / 0.50% VAL and its estimate moves −4.1 → −4.5 and +40.2 → +37.1. **Disclosed, measured,
and it is not what produces any number in this report — but it belongs in `DATA.md` as a second half
of gap 10, because a family that relaxes the $5 raw gate would be eating it.**

**(d) Universe hygiene it found.** `kind == 'common'` carries **39 dotted tickers**, including SPAC
units (`AIIA.U`, `DGAC.U`, `GLED.U`, `JACS.U`, …) and dual-class/ADR lines (`BRK.A/B`, `BF.A/B`,
`AKO.A/B`, `CRD.A/B`). SPAC units pinned near $10 have nearness → 1.0 with near-zero volatility and
are structural decile-10 residents for F5. 39 of 4,871 is 0.8% and cannot drive a result, but the
shared `_is_common_stock` name rule should learn the `.U` suffix.

---

## 10. The two bars

**Claim bar (G1 + G2): FAILED, 0 of 11.** Nothing from this stage may be stated as a positive
finding, and TEST stays sealed. The two *negative* findings in §0 are reported as such, with their
power stated.

**Live-exploration bar** (positive point estimate + mechanism + bounded downside + resolution inside
a quarter at its own trade frequency):

| cell | TRAIN / VAL | mechanism | bounded downside | resolves in a quarter | **MDE $/mo vs prize** | verdict |
|---|---|---|---|---|---|---|
| F5-LO | −4 / +40 bps | published (GH 2004), but GHL 2018 JFE 128(1) shows q-factors *explain* price-to-high | yes (−$16K) | **no, 0.58/wk — the slowest cell in the program** | $521–$1,306 vs ~$150 | **FAIL**: sign flip, no January effect, 5 empty book months of 24 |
| F5-LS | −39 / +103 | published | no (−56% L-S drawdown on TRAIN) | no | same | **FAIL** |
| **A2-LO** | **−66 (t −2.05)** / +20 | published (BHJ 2010) — and the **published direction is contradicted**: TRAIN returns rise monotonically with short interest | yes (−$32K) | **yes, 6–9/wk** — the only cell here that clears the frequency column | $832–$1,011 vs ~$200 | **FAIL, wrong sign, significantly** |
| A2-LS | −129 / +47 | published | no | yes | same | **FAIL** |
| A3-LO | +3 / +65 | published (PW 2008, FF 2008, Goto et al. on costs) | yes (−$26K) | no (3.7/wk) | $1,099–$1,318 vs ~$200 | **FAIL on noise** (TRAIN t 0.10) |
| A3-LS | +1 / +143 | published | no | no | same | **FAIL** (clears G2's letter at TRAIN t 0.02) |
| **A4-LO** | **−0.6 / −2.3** | published (HS 2013; AER 115(9) 2025) | yes (−$7–8K, the mildest in the program) | no (3.6–4.4/wk) | **academic MDE 13.3 bps vs a ~40 bps published effect — POWERED** | **FAIL — and this one is a refutation, not a null** |
| A4-LS | −6 / −6 | published | yes | no | same | **FAIL** |
| F1-LO-60 | −18 / +39 | Martineau 2022: no drift for all-but-microcap after 2006 | yes (−$18K) | no (1.3–1.5/wk) | $443–$1,199 vs ~$100 | **NULL-REPLICATION CONFIRMED** |
| F1-LS-60 | −22 / −34 | same | yes | no | same | **NULL-REPLICATION CONFIRMED** |
| **F6-LS** | −26 / **−130 (t −2.58)** | published (LPS 2019) — the **decomposition replicates**, the L-S does not | **no** (−$41K book drawdown, the worst in the program) | no (3.6–4.4/wk) | academic MDE 76–101 bps vs ~300 bps — **POWERED** | **FAIL, wrong sign, significantly** |

**F1 is a declared null-replication and it did its job.** It was never a candidate. Martineau (CFR
11(3-4), 2022) reports no significant PEAD for all-but-microcap stocks after 2006; our pipeline, on
73,002 causally-ranked SUE events at a 60-session hold, returns **−17.6 bps/month TRAIN (t −1.03)**
and **+38.6 VAL (t 0.62)** long-only, **−22.3 / −34.4** L-S. That is Martineau's prediction, and it
is the third prediction in the program to replicate. Its value is as a **positive control on the
pipeline**: a stack that produced a large PEAD here would have been evidence the machinery was
broken, and it did not.

---

## 11. PROGRAM-LEVEL CLOSE — all 26 cells across four family reports

### 11a. The ledger

| stage | cells | G1 pass | G2 pass | best TRAIN t | book MDE $/month | report |
|---|---|---|---|---|---|---|
| F2 announcement drift + A1 announcement premium | 8 | 1 (0 under the sealed convention) | 0 | 2.15 → 1.83 sealed | $670 – $3,034 | `REPORT_F2_A1.md` |
| F4 industry-adjusted reversal + F3 momentum | 7 | 0 | 0 | 1.45 | $1,012 – $1,996 | `REPORT_F4_F3.md` |
| F5 + A2 + A3 + A4 + F1 + F6 | 11 | 0 | 0 (one clears the letter at t = 0.02) | 0.10 | $443 – $2,307 | this page |
| **total** | **26** | **0 durable** | **0** | — | **$443 – $3,034** | — |

Cumulative multi-day scored-cell count including the earlier home-made families:
**K 20 + N2 4 + R_daily 20 + 26 here = 70.** Distinct looks in the three published-construction
stages: 8×4 + 7×3 + 11×4 = **97.**

### 11b. The MDE-vs-prize arithmetic, which is the program's actual answer

The pre-committed calibration prior is Chen & Velikov: 204 published anomalies earn **~4 bps/month
net** of realistic trading costs, the strongest ~10 bps before impact. On a $66K book that is
**$25–65/month**. Across all 26 cells the executable book's minimum detectable effect is
**$443–$3,034/month**. **The experiment is 7–121× too blunt for the thing it is looking for, in
every cell, in every family, in all three stages.** That is not a property of the anomalies; it is a
property of $66K and 20 slots and a 6-year TRAIN window, and no further searching on this panel
changes it.

Classify every cell as the task requires — (a) economically irrelevant at our size even if true,
(b) unmeasurable at our size, (c) both:

* **(c) both — 24 of 26 cells.** Every F2, A1, F3, F4, F5, A2, A3 and F1 cell. Their prizes at $66K
  net of the literature's own cost estimates are **$25–$200/month**, and their book MDEs are
  **$443–$3,034/month**. Even a perfectly harvested, undecayed version of the published effect would
  be a rounding error against this account's other books, and the test could never have confirmed it.
* **(a) economically irrelevant but MEASURABLE — 2 cells, A4 and F6.** Their *academic* columns are
  powered: A4-LO's MDE is **13.3 bps/month against a ~40 bps published effect**, F6-LS's is
  **76 bps against ~300**. Both were measured and both came back against the published claim (A4:
  −0.6 / −2.3 bps; F6: −26 / −130 bps). Their **executable books** are still unmeasurable
  ($486–$2,307/month MDE), so they belong in (a), not in "tradable".
* **(b) alone — none.** There is no cell in this program that would have been worth money if only we
  could have measured it.

**The separation is the finding.** "Does the anomaly exist" and "can this account harvest it" are two
different questions, and this program answered them separately, as PLAN §1 promised it would. Where
the construction happened to be a *low-variance, same-population contrast* — A4's predicted vs
non-predicted dividend month, F6's overnight vs intraday split — we had power and the answer was no
at this universe, this window and this cost. Where the construction was a *decile spread across the
whole cross-section* — everything else — the account is 7–121× too small to learn anything either
way. **A $66K, 20-slot, long-only, no-borrow book cannot resolve a published cross-sectional anomaly.
That is the program-level result, and it is a result about the account, not about the literature.**

### 11c. What replicated, in three stages of looking

Three predictions out of everything tested:
1. **Johnson–So's reversal of the announcement premium** (F2/A1): holding *through* the event gives
   the premium back — TRAIN excess 137 → 46 bps, VAL flips negative.
2. **Blitz–Huij–Martens' residual momentum halves the Daniel–Moskowitz crash** (F4/F3): 12-1 L-S MDD
   −37.3% → residual −20.5%; long-only −15.5% → −11.0%.
3. **Martineau's dead PEAD** and **the overnight/intraday decomposition** (this stage): F1 returns a
   null exactly where Martineau says it should, and 6 of 8 years earn essentially all of the equity
   return overnight with 4 of 8 intraday sessions negative.

Every one of the three is a *negative or structural* prediction. **Not one positive return premium
survived.** That pattern — the literature's warnings replicate and the literature's prizes do not —
is itself the most decision-relevant thing on this page.

### 11d. Verdict on the multi-day line as a whole, in PLAN §1 phrasing

*No edge was detectable in the US-common-stock universe above $5 and $1M ADV20$, across 26
pre-registered cells spanning six published families and four additions, at horizons from 4 sessions
to 6 months, at a $66K / 20-slot long-only auction book, over 2016-01 → 2023-12, at close-to-close
auction cost — and the power of the test is that its smallest detectable monthly effect is
**$443–$3,034 at book size against an expected effect of $25–65**, i.e. **7–121× too blunt**, in every
cell but two. In the two cells where the construction gave us power (A4's same-firm dividend-month
contrast at 13 bps/month, F6's overnight/intraday split at 76), the published effect was **not
present and the measured sign was against it**. The multi-day published-anomaly line is therefore
**CLOSED at this account size**: not because the anomalies are shown to be absent, but because this
book demonstrably cannot see, and could not profitably hold, an effect of the size the literature
reports.*

---

## 12. What I would do next, and what I would not

* **Do not re-propose any cell in this stage.** A2 is significantly the wrong way and gets *more* so
  under a more conservative lag. F5 is 0.58 trades/week with 5 empty book months in 24 and no January
  effect. A3 is TRAIN t 0.10. A4 is refuted where we had power. F1 was never a candidate. F6's
  tradable form has the worst risk-to-evidence ratio in the program.
* **Do not re-run this program at a larger book to "get power".** The MDE scales with the *variance*
  of the monthly series, not with the book, so more capital moves the $/month prize and the $/month
  MDE together. What moves the MDE is *months* and *cross-sectional breadth* — a 20-year panel and a
  200-name book, neither of which this account has.
* **The one purchase that would change any of it** is still a split-adjusted, delisting-inclusive US
  daily file (`DATA.md` §7 gap 8). It would make F3's and F5's level column quotable and let every
  survivor be PIT-re-run instead of argued about. Everything else in all three stages was free.
* **Two data fixes are owed to the shared code**, both found by independent rebuilds and both cheap:
  `_is_common_stock` should learn the `.U` SPAC-unit suffix (§9d), and `DATA.md` gap 10 should be
  extended with the 335 unreverted reverse-split sessions (§9c), because they clear the raw $5 gate
  and gap 10 as written says the gates exclude them.
* **Newest verified evidence age.** A4 — Hartzmark–Solomon's 2025 AER 115(9) 3171-3213 follow-up is
  the newest top-5 citation in the whole program and the only family with one; our refutation is at
  a $5 / $1M-ADV universe over 2016–2023 and says nothing about theirs. A2 — BHJ 2010, no verified
  post-2020 US re-test found either way. A3 — Pontiff–Woodgate 2008 with Goto et al. on costs; no
  post-2020 re-test found. F5 — George–Hwang–Li 2018 JFE 128(1) already re-labelled it a factor tilt
  rather than an anomaly, and that is what it behaves like here. F1 — Martineau 2022, which is the
  prediction that replicated. F6 — LPS 2019 plus our own `overnight_auction.md` and M29, all three
  agreeing that the premium is real and the round trip eats it.
