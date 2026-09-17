---

## 7. Verdict

**Nothing a trader can see at the fill separates this book's losers from its winners, and the two
loser classes that ARE decidable at the fill are the wrong ones to cut.** Of the eight classes, only
(e) late/extended and (d) spread/thin are computable before the trade exists. (d) is the biggest
single class — 418 of 1,127 losers, **−341.5 R, 39.3% of all loser R** — and it is also 41.5% of the
winners for **+454.7 R**: the thin cohort is **net +95.7 R on 808 trades**, so dropping it takes TRAIN
from **+0.062 to +0.028 R/trade** ($1,725 → $433 a month at $300 risk) and VAL from +0.207 to +0.194.
(e) is 85 losers for −51.5 R (5.9%) against 74 winners for +54.8 R, net **+3.3 R on 159 trades** —
a coin. Dropping both: TRAIN **+0.045**, VAL **+0.177**, TEST **−0.071** — worse than as-booked in
VAL and TEST. This is the H/F6 §5.2 inversion again, measured a third time: the cohort that owns the
losses owns the wins as well, and staring at losers picks the wrong side of it. The classes that DO
sort almost perfectly are **(a) level failure** (178 losers −168.1 R against 5 winners +1.2 R) and
**(g) slow bleed** (445 losers −302.2 R against 43 winners +3.5 R) — but both are statements about the
tape AFTER the fill, i.e. arguments for a time stop, not for an entry filter, and no time stop was run
here. On the entry facts themselves (§2) every quantity a trader would check — gap, prior-day range,
entry vs prior close, open→entry, stop distance, 5-min $ volume, range-so-far, price, ordinal, day of
week — moves less than 21% between winners and losers, and only the index at the entry minute clears
the declared ≥20%-same-direction bar in TRAIN and VAL; its absolute size is **1.4 bp of SPY** (winner
median +0.024%, loser +0.010%), so that test is degenerate on a fact whose median is ~0, not a finding.
The only structure that holds its sign in all three splits is the **entry ordinal**: 1st entry of the
day +0.164 / +0.681 / +0.148 R, decaying monotonically to +0.038 / +0.033 / −0.067 at the 4th, and it
is not time-of-day (inside the 09:xx hour alone: +0.142 / +0.706 / +0.148 for the 1st against
+0.002 / +0.065 / −0.073 for the 4th+). "First two entries only" books **+0.108 / +0.463 / +0.097
R/trade** and flips TEST from −15.4 R to +13.1 R — but it is a subset filter that never refills the
slot, it halves the trade count, and at $300 risk it *lowers* the TRAIN book from **$1,725 to $1,310
a month**. The stop-distance floor suggested by the 16 worse-than-1.5R prints (median stop 1.65% of
entry on a $35K 5-min tape, −34.5 R = 4.0% of all loser R) helps TRAIN (+0.078 at ≥3%) and **halves
VAL** (+0.101 vs +0.207) — not a rule. **The tail settles it (§6c): every cell, every rule, every
split is negative once the top 5% of trades is removed** — as booked −0.156 / −0.116 / −0.248, seq≤2
−0.196 / +0.002 / −0.174, seq≤2 & stop≥3% −0.158 / −0.057 / −0.133 — and capping winners at +3 R puts
even TRAIN as-booked at −0.005. This book is its top 5% of trades; it is the lottery ticket the owner
has already rejected once, and its weekly result is +0.36 correlated with the SPY intraday sum and
+0.40 with IWM (7 of the 10 worst weeks were negative-SPY weeks), so it is long beta on top of that.
**Book as it stands at $300 risk: TRAIN $1,725/mo, VAL $6,584/mo, TEST −$1,473/mo; with every
entry-avoidable loser class removed, $636 / $3,340 / −$1,468.** TEST was read once, at the end, for
`seq <= 2 AND stop >= 3%` only, because VAL agreed in sign with TRAIN on it: **+0.079 R/trade,
$880/month, 14/21 months green, worst month −$3,229 — and ex-top-5% −0.133.** Recommendation: do not
flip `red_to_green` out of `dry_run`. Note also the frame: this whole dive is the FIRST-BREAK book,
which is **not** the rule `trading/red_to_green.py` runs; the shipped rule's own book is −0.027 /
−0.012 / −0.102 R (`H/F6_reconcile` §3), so nothing here rehabilitates it.

**Phrasing.** No entry-decidable loser class was found in THIS book whose removal improves it in both
TRAIN and VAL, on THIS universe (the point-in-time ≥5%-range day list, PDR ≥ 8, price ≥ $5), at THIS
horizon (intraday, flat 15:55), at THIS book size (12/day, 4 concurrent, HOLD exit), over
2025-01-02..2026-09-04, at THIS cost (contract c). Power: per-trade SD is 1.386 R on TRAIN
(n=1,112, SE 0.042), 1.894 on VAL (n=531, SE 0.082), 1.266 on TEST (n=366, SE 0.066), so the smallest
mean effect resolvable at t=2 is **+0.083 R/trade on TRAIN, +0.164 on VAL, +0.132 on TEST**. A filter
worth 0.03–0.08 R/trade — the size of most of the differences tabulated above — cannot be resolved
here, and on the `seq<=2` subset the VAL MDE is +0.356 R/trade, wider than the effect it is being
asked to confirm.

---

## 8. Cells

Every cell inspected in this dive, counted:

| block | cells |
|---|---|
| §1a/1b day tape lines (25 worst + 25 best days) | 50 |
| §1c loser class × split | 24 |
| §1c raw flag × (winner\|loser) | 14 |
| §2 entry fact × split × (winner\|loser) | 72 |
| §2 categorical entry cells (day of week, price band) | 27 |
| §3 worst-week rows | 10 |
| §4 sequence ordinal × split | 12 |
| §4 entry hour × split | 18 |
| §4 declared sequence rule cells | 9 |
| §5 avoidance rule cells | 12 |
| §5 flag loss-share cells | 6 |
| §6a seq × hour, and seq × split inside 09:xx | 24 |
| §6b stop-distance bands, stop-floor rules, combined rules, tail, monthly | 39 |
| §6c tail dependence (4 rules × 3 splits × 3 statistics) | 36 |
| **total** | **353** |

No search was run for a new rule: §4's two rule cells, §6b's stop floors and the combinations were
each declared in the script before it ran, and TEST was read once, at the end, for the single
combination VAL agreed with TRAIN about. The 353 is nonetheless the honest program-wide count for
this file, and it sits on top of the ~10,000 cells the rest of `research/fuckup_audit/` has already
spent on this same population.

---

## 9. Files

`facts.py` → `trades_facts.csv` (2,009 booked trades × 40 tape/entry facts), `days_facts.csv`
(417 days) · `analyze.py` → `trades_classed.csv` (**the per-trade CSV with the class labels** — one
`cls` column plus the seven non-exclusive `f_*` flags), `body.md`, `cells.txt` ·
`extras.py` → `extras.md` · `tail.md` · `verdict.md` · `REPORT.md` (this file, assembled).

Stores were opened read-only: `research/bf_zero/bars_sip.db` (primary tape), `data/cache.db`
`intraday_bars_1min` (fallback, selected per trade by the book's own `src` column) and `daily_bars`,
`research/lit_review_2026/etf_1min.db` (SPY, IWM). Nothing outside
`research/fuckup_audit/D5_r2g/` was written; no config, service, cache or order was touched.
One `nice -n 10` process at a time under `ulimit -v 1300000`. `keep_default_na=False` on every read.
Test tickers (`^Z[A-Z]ZZT$`) dropped: 2 rows, +0.79 R, both TRAIN.
