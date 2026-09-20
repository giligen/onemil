# orb_inplay — REPORT

Pre-registered replication of **Zarattini, Barbon & Aziz (2024)** "stocks in play" ORB on our data,
our splits, our measured costs. PREREG `40b4e6e` (+ Amendment 1), FREEZE before scoring.
**TEST (>= 2026-06-01) was never fetched, ranked or scored.** Cells: **1,280–1,282** (long, short, combined).

## 1. Universe, coverage, availability rail

| | |
|---|---|
| days | 373 fetched (2024-12-02 .. 2026-05-29); 250 TRAIN, 102 VAL scored |
| universe / day | median **1,861** names (ADV20 >= 1M, prev close >= $5, whole market); **1,788** rankable (>= 10 of the prior 14 sessions present); open-tape coverage **99.8 %** mean / 99.2 % min |
| picks | 19.8 / day; 7,020 in TRAIN+VAL |
| status mix | ok 6,315 · Reg SHO excluded 461 · **no 1-min bars 151 (2.2 %)** · doji 93 |
| **availability rail** | **PASS** — 97.8 % of picks scoreable (bar >= 80 %) |
| missingness gap | the 151 missing name-days are unscoreable by construction, so no winner/loser split exists; proxy: their median RVOL 7.97 vs 7.41 and median prev close $22.23 vs $26.88 — slightly hotter and cheaper, i.e. if anything the missing names are the movers. Declared, not resolved. |
| Reg SHO 201 | 461 shorts dropped = **13.3 % of short candidates**, 6.6 % of all picks |
| wrappers | in the universe and flagged; not separated (they are a minority of ADV>=1M names) |

**Mid-run change #1 (declared in FREEZE before scoring).** `cache.db::intraday_bars_1min` holds only
**~10 %** of the ADV>=1M universe (173/1,724 on 2025-03-05; 233/2,237 on 2026-03-04) — it is the
gap-up/mover seed. Ranking the paper's universe off it would have rebuilt the exact look-ahead
population that killed `research/bf_zero`. The 09:30-09:34 tape for **every** universe name and the
09:35-15:55 tape for every pick were therefore fetched from **Alpaca SIP, adjustment=RAW**.
`daily_bars` remains the source of ADV20 / ATR14 / prev close (all causal, shifted).

## 2. Cost — the whole story

R = 0.10 x ATR14 is **median 0.378 % of price** (p10 0.163 %, p90 0.819 %). The measured NBBO
half-spread at the 09:35 entry minute is **0.124-0.168 % of price** (839 legs, Alpaca SIP,
`mean(ask-bid)/2`, 12 price-band x clock cells, >= 69 legs each; **measured share 6.6 % of the
12,630 legs, 93.4 % imputed from those cells**). So **one entry leg alone costs ~0.35-0.45 R** and
the round trip plus $0.0035/share costs **1.03-1.15 R per trade**.

| cost model | TRAIN net R (t_cl) | VAL net R (t_cl) |
|---|---|---|
| zero (gross) | **+0.120** (+2.29) | **+0.286** (+3.06) |
| paper's $0.0035/share only | −0.004 (−0.07) | +0.168 (+1.76) |
| **measured NBBO + $0.0035 (PREREG)** | **−1.018** (−7.6) | **−0.445** (−1.5) |
| frames14 BF minute table (upper bound) | −1.783 | −1.669 |

The BF table over-charges (thin bull-flag population vs our ADV >= 1M universe) and is a bound only.
All rows on the full 6,315-trade candidate set.

## 3. The book (PREREG sizing: 1 % risk, notional cap, RVOL rank order)

The cap binds hard: 1 % of $66K at a 0.38 %-of-price stop implies ~$174K notional **per trade**, so
at 1x only **1.8 fills/day** and at 4x **3.0-3.2 fills/day** of the 17.9 daily candidates are admitted.

### 1x ($66,000, unlevered)
| split | side | n | fills/d | net R (iid t / clust t) | MDE | WR | avgW/avgL | ex-top1% | ex-top5% | top5 share of gross wins | longest gap >=3R | green wks vs null | $ | $/wk | MDD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|TRAIN|comb|455|1.8|**−0.962** (−6.06/−6.15)|0.445|10.5%|+7.80/−1.99|−1.100|−1.534|46.0%|5 wk|24.5% vs 49.8% (p 1.00)|−58,747|−1,108|−59,304|
|TRAIN|long|270|1.4|−0.799 (−4.01/−3.97)|0.558|10.7%|+8.13/−1.87|−0.917|−1.355|65.9%|12 wk|20.8% vs 49.9%|−35,870|−677|−40,911|
|TRAIN|short|185|1.3|−1.200 (−4.61/−4.68)|0.729|10.3%|+7.29/−2.17|−1.367|−1.855|82.0%|8 wk|13.5% vs 50.0%|−22,878|−432|−23,039|
|VAL|comb|204|2.0|**−0.583** (−1.76/−1.63)|0.926|16.7%|+7.36/−2.17|−0.737|−1.161|50.1%|4 wk|40.9% vs 50.0% (p 0.85)|+11,375|+517|−14,620|
|VAL|long|115|1.5|−0.171 (−0.35/−0.33)|1.379|19.1%|+8.38/−2.19|−0.346|−0.711|60.7%|6 wk|40.9% vs 50.5%|+21,538|+979|−7,593|
|VAL|short|89|1.4|−1.114 (−2.73/−2.71)|1.144|13.5%|+5.49/−2.14|−1.239|−1.729|99.3%|8 wk|13.6% vs 50.0%|−10,163|−462|−19,251|

### 4x ($264,000 gross cap)
| split | side | n | fills/d | net R (clust t) | WR | $ | $/wk | MDD |
|---|---|---|---|---|---|---|---|---|
|TRAIN|comb|738|3.0|−1.013 (−7.60)|10.6%|−183,761|−3,467|−188,274|
|VAL|comb|330|3.2|−0.445 (−1.54)|18.2%|+8,422|+383|−43,572|
|VAL|long|194|2.0|−0.300 (−0.79)|19.6%|+34,133|+1,552|−25,029|
|VAL|short|136|1.7|−0.652 (−1.72)|16.2%|−25,711|−1,169|−45,553|

(4x TRAIN long −1.050 R / −$107,995; TRAIN short −1.326 R / −$75,765.)

TRAIN halves (1x, combined): H1 −1.043, H2 −0.885 — same-signed (both negative).
**$ and R disagree on VAL** (positive dollars, negative R): the notional cap truncates share counts,
so per-trade risk is *not* 1 % — only the widest-R (cheapest-per-R) names get real size, and on VAL
those were winners. **The R column is the honest read; the VAL dollar profit is a cap artefact.**

## 4. Pass bar, line by line (VAL, combined, 1x)

| # | bar | result |
|---|---|---|
|1|net >= +0.10 R/trade|**FAIL** — −0.583 R|
|2|day-clustered t >= 2.0|**FAIL** — −1.63|
|3a (Amd 1)|net positive ex-top-1 %, both splits|**FAIL** — −0.737 (VAL), −1.100 (TRAIN)|
|3b (Amd 1)|a >= +3R winner in every rolling 4-week window|**PASS on VAL combined** (longest gap 4 wk); FAIL long (6 wk), short (8 wk), TRAIN (5 wk)|
|3c (Amd 1)|top 5 trades < 50 % of split P&L|**FAIL** — 50.1 % of gross wins on VAL combined (long 60.7 %, short 99.3 %)|
|4|TRAIN halves same-signed|**PASS** (both negative)|
|5|green-week share > count-matched null|**FAIL** — 40.9 % vs 50.0 % null (p 0.85)|
|6|unlevered annualised return > 0 both splits|**FAIL** — TRAIN −89 %/yr, VAL +41 %/yr (and see the cap artefact above)|

**Verdict: the combined book FAILS. Long FAILS. Short FAILS** — and short fails on every
formulation including gross-with-paper-costs.

Diagnostics only per Amendment 1: ex-top-5 % VAL combined −1.161 R; capped at +3R, −1.42 R/trade.

**$/week at $66K** — 1x: TRAIN −$1,108, VAL +$517. 4x: TRAIN −$3,467, VAL +$383. Both VAL figures
are the cap artefact above, on a book whose per-trade net R is negative at t = −1.5.

## 5. The ONE caveat per side that alone could explain the headline

- **Both sides / the headline null: the entry half-spread charge.** We charge a full measured NBBO
  half-spread on a bar-**open** print that may itself already have executed at the bid or the offer.
  At R = 0.38 % of price and a 09:35 half-spread of 0.12-0.17 %, that single assumption is worth
  ~0.4 R per trade — larger than any effect in the table. Under the paper's own $0.0035/share the
  book is −0.004 R (TRAIN) / +0.168 R (VAL). **The disagreement with the published result is a cost
  disagreement, not a signal disagreement.** Resolving it needs per-trade trade-vs-quote matching at
  the 09:35 open (which side of the spread the open print traded on) — not done here.
- **Long side:** its entire VAL result rests on the right tail — top 5 trades = 60.7 % of gross wins
  and the longest gap between >= 3R winners is 6 weeks, so the point estimate is a handful of days.
- **Short side:** 13.3 % of short candidates are removed by Reg SHO 201 — precisely the hardest-down
  names a short-at-open book most wants. On the trades the cap admits the short book is *gross*
  negative (TRAIN −0.035 R, VAL −0.032 R) even though the full short candidate set is gross positive
  (+0.108 / +0.219 R): the cap selects wide-R shorts and those are the ones that fail. Borrow
  availability is assumed, not measured, and cannot repair that.

## 6. Power / phrasing

MDE (80 % power, two-sided 0.05): VAL combined **0.93 R**, long 1.38 R, short 1.14 R at 1x — the
capped book is small (204 VAL trades). On the **full 6,315-trade candidate set** the MDE is 0.13 R,
and there the gross edge is real and significant (TRAIN +0.120 R t_cl 2.29, VAL +0.286 R t_cl 3.06).

No edge was detectable **in this universe (US ADV >= 1M, price >= $5), at this horizon
(09:35 -> 15:55), at this book size ($66K at 1x / 4x with the 1 %-risk 0.1-ATR stop), over this window
(2025-01 .. 2026-05), at this cost model (measured Alpaca SIP NBBO half-spread both legs +
$0.0035/share)**. The same rule is **gross-positive and significant** on the same trades. The
programme's own standing finding applies in reverse: *the cost model is the null.* The next cell is
not another filter — it is the entry-leg cost measurement (trade-vs-quote at the 09:35 open) and a
stop wide enough (e.g. 0.5-1.0 ATR) that the spread is a few percent of R rather than ~100 %.

## 7. Mid-run changes
1. Open-tape and pick-bar source moved from `cache.db` to Alpaca SIP (FREEZE, §1), declared pre-scoring.
2. Amendment 1 (owner, pre-scoring) replaced the ex-top-5 % pass item with the winner-frequency rule.
3. Reported R-statistics are computed on the trades the notional cap actually admits (`shares > 0`);
   the full-candidate R-statistics are given separately in §2 and §7. No other change.
