# RESULT 1,770 (J8 cost audit, measured NBBO) -- KILL, final

All VAL 2021-23 primary trades of 1,769 (2,498) plus 1,000 seeded TRAIN; SIP quotes, no 2024+.

**Coverage.** VAL 96.5 %, TRAIN 97.2 %. Missing winners 4.5 % vs losers 3.0 % (gap +1.5 pp, passed).

**Half-spread (bp of mid, VAL median / mean).** Entry 09:35: 23 / 39. Cover 10:30: 13 / 22. Total 39 / 64 vs the 1,769 band 72 / 63. By tier (<10, 10-20, 20-50, 50+) measured total median 37 / 41 / 44 / 34 vs band 44 / 73 / 72 / 67. The band overstated the median; the mean is unchanged (wide tail).

**VAL net, measured costs** (sell at bid, cover at ask, 2 bp/leg, borrow 50 bp): -65.9 bp, day-t -5.0, stop-hit 48 %, worst-5 % day -798, ex-top-5 % -139, green weeks 34 % vs null 47 %. Years 2021 / 2022 / 2023: -17 / -113 / -104. At borrow 100 bp: -116. The band net on the same trades is -66.9 (paired delta +1.0 bp).

**Stop slip.** Mean 14.8 bp, median 6.1, stop level to first ask at or above it (bar model 4.5).

**TRAIN sample (n 972).** -75.0 bp, day-t -4.1, same sign as VAL.

**Kill checks failed:** net <= 50, day-t < 2, ex-top-5 <= 0, 2022 negative, borrow-100 <= 0. At zero spread VAL net is about +7 bp: no cost model rescues J8.

**Caveats.** (1) A cleaning rule outside the PREREG set 14 bad NBBO prints missing (half-spread > 1,000 bp, or mid outside 0.5-2x the bar price). (2) Quotes are the last in the minute after the instant the bar price is struck; the pre-instant variant covers 70 % of VAL with a 17 pp winner/loser gap, so void (-84 bp). (3) Covered-subset band net -66.9 vs -56.6 on all 2,498.

CHILDREN:
1. Gross-only anatomy of the early gapper fade by year, 2016-26, before any cost.
2. Tick-level anatomy of the 48 % stopped trades, for a resting stop-limit cover.
3. The same measured-NBBO audit on the 2025-26 names of 1,768 (TEST sealed).
