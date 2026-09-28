# Refuter: PREREG_1633 (cells 1,633–1,636, PEAD after the 8-K 2.02 reaction)

**Verdict: the FAIL stands. The result is not refuted.** I tried 7 sensitivity variants. Each one fixes one or more real
defects. None passes the frozen VAL bar, even with the decile-monotonicity criterion switched off. The defects are
real, but none of them changes the verdict. The closure has to be scoped: it holds for **survivors as of 2024-07, one
long or short leg at a time, 5 bps per leg, over 2019 to 2024H1**. It does not cover "PEAD" in general. TEST was not
touched: `assign_split` drops it and nothing here scores it.

Script: `review/1633_refuter_checks.py`. It reuses the builder's loaders read-only, and every check is recomputed.
Log: `review/1633_refuter_checks.log`. Per-variant table: `review/1633_refuter_summary.csv`. Every hold beyond ±50 %
is listed in `review/1633_refuter_big_holds.csv`.

## Variants (VAL unless marked; bps net; day-clustered t)

| variant | 1633 mean / t | 1634 mean / t / ex-top-5 % / SPY-adj | 1635 mean / t | any PASS |
|---|---|---|---|---|
| T0 builder, reproduced exactly | 9.1 / 0.19 | 85.3 / 1.28 / −112 / −153 | −34.1 / −0.57 | no |
| T1 drop suspect UTC hours | 21.3 / 0.45 | 99.4 / 1.47 / −100 / −136 | −52.9 / −0.85 | no |
| T1 remap suspect hours as ET | −4.5 / −0.09 | 72.1 / 1.06 / −127 / −164 | −50.0 / −0.78 | no |
| T2 drop split artifacts | 9.0 / 0.19 | 89.9 / 1.35 / −108 / −149 | −34.1 / −0.57 | no |
| T9 one event per name-quarter and per filing | 37.8 / 0.81 | 108.8 / 1.63 / −86 / −134 | −21.1 / −0.35 | no |
| T10 prior session = reaction−1 (the rebuild's reading) | 6.9 / 0.15 | 84.5 / 1.27 / −114 / −152 | −30.1 / −0.50 | no |
| T11 T9 + T2 stacked | 37.7 / 0.80 | 114.2 / 1.72 / −81 / −128 | −21.1 / −0.35 | no |

In every variant, the decile table is NOT monotone on either half for either hold. The best case is 1634 under T11. It
still misses t ≥ 2.5 by 0.8, ex-top-5 % > 0 by 81 bps and SPY-adjusted ≥ +30 by 158 bps.

## Lens by lens

1. **Acceptance time to reaction session (UTC vs ET).**
   - The bulk of the raw digits are **true UTC**. AAPL shows 20:30Z in summer and 21:30Z in winter, which is 16:30 ET
     in both cases. MSFT, AMZN, NVDA, GOOGL, META, NFLX and INTC show the same one-hour DST jump. JPM, WMT, PG and KO
     sit at 10:30–11:00Z in summer and 11:30–12:00Z in winter. No row has an hour of 03–05Z, when EDGAR is closed.
   - So the builder's zoneinfo conversion is correct. Cell 1,552's "the digits are ET" reading was the bug.
   - **However, some rows use a mixed convention.** 2,787 raw 2.02 rows fall at 06–09Z. Under UTC that is 02:00–05:59
     ET, when EDGAR is closed, so these rows must be ET digits with a spurious Z. Their mapping is unaffected because
     they are pre-market under either reading.
   - The 16–19Z group reads as 12:00–15:59 ET under UTC, so it is classed intraday. It holds 3,494 universe rows
     (5.9 %), and the share is stable at 5.5–6.3 % every year. This group fails a price news-day test: the largest
     move in sessions −2 to +2 falls on the session *after* the mapped reaction in 41.7 % of rows, and on the mapped
     session in only 21.1 %. For comparison, the after-close group (20–23Z) and the pre-market group (10–13Z) land
     on the mapped session in 51–54 % of rows.
   - So most of the 16–19Z rows are after-close releases mapped one session early. Their R0 is the day before the
     news, and the entry is on the news gap.
   - These rows are 119 of 6,249 in 1633 and 32 of 1,318 in 1635. Dropping them or remapping them leaves the verdict
     unchanged (T1).
2. **Raw price scale.**
   - The bars are RAW:
     - AAPL 499.23 → 129.04 on 2020-08-31.
     - NVDA 1,208.88 → 121.79 on 2024-06-10.
     - CMG 3,283 → 65.86.
     - AVGO 1,701 → 171.
   - The builder's split guard only removes one-day close ratios outside [0.4, 2.5]. That misses 2:1 and 3:2
     splits. The gated cells contain three genuine split artifacts:
     - SPSC 2:1 on 2019-08-23 (1634 TRAIN, −56 %).
     - CHDN 2:1 on 2023-05-22 (1634 VAL, −51 %).
     - PCAR 3:2 on 2023-02-08 (1634 VAL, −34 %).
   - The ratio screen also flagged BAND, PRFT and VERU. Those are real news days of +50 % or more, not splits.
   - The other holds beyond ±50 % are real moves: the Feb–Mar 2020 crash, the May and Nov 2020 rallies, meme names.
   - Separately, the guard removes 28, 35 and 24 real-tail rows from 1633, 1634 and 1635.
   - The panel seam has a small gap: file A ends on 2024-06-27 and file B starts on 2024-07-01, so the 2024-06-28
     session is missing for every symbol. This is negligible.
3. **Duplicate 8-Ks.** The builder does not deduplicate events. Collapsing to one event per name-quarter and one per
   filing drops **16 % of the TRAIN+VAL rows** (82,134 → 69,188). The duplicates come from three sources:
   - Renamed tickers that carry the same filing with an identical price path, e.g. MESA/RJET on 2021-02-10 and
     AXL/DCH on 2020-05-08.
   - Share classes, e.g. Z/ZG.
   - Several 2.02 filings in one quarter (81 extra rows in 1633).

   Amendments are already excluded because the form must be exactly `8-K`. The verdict is unchanged (T9).
4. **Delistings inside the hold.** No row in any gated cell has a missing exit. The builder's code would drop such
   rows, where the PREREG says to keep them at −100 %, but this never triggers because of item 5.
5. **Survivorship, 2019 to 2024H1: real, and built into the data.**
   - `events_raw` maps symbols to CIKs through SEC's *current* `company_tickers.json`, plus a fallback that
     explicitly excluded delisted names. Only 8,496 of 30,440 symbols were mapped.
   - Take the 4,194 names that traded in 2019 at ≥ $3 with a median dollar volume of ≥ $1M. The 561 that died
     before 2024-07 have **0.4 %** 2.02 coverage. The 3,633 still alive have 61.9 %.
   - So the event set is survivors only. The direction of the bias:
     - It flatters the long cells, so their FAIL is conservative.
     - It hurts the short cell 1635. But 1635 also fails on TRAIN t (−0.24), ex-top-5 % (−128) and monotonicity.
       Its SSR proxy (close < $5 or R0 ≤ −10 %) already removes most of the names that later died.
6. **Decile edges on TRAIN only.** Recomputing them gives exactly the logged edges. VAL is never re-cut.
7. **Tails and month concentration.**
   - 1634 VAL totals +178K bps. The top 5 % of events add up to +401K on their own. The top three months (2023-11
     +266K, 2023-05 +111K, 2023-10 +55K) are larger than the total. Without those months the mean is **−169 bps per
     event**.
   - In TRAIN, the top three months (2020-05, 2020-11, 2021-05) are also larger than the total. Without them the mean
     is −79 bps.
   - So the positive raw means come from small caps moving with the market in rally months.
8. **SPY-adjusted.**
   - The builder measures SPY from the reaction close, but the stock from the next open. That overstates SPY's
     return in VAL by 12 bps.
   - With matching windows, the SPY-adjusted net is −141 for 1634 and −107 for 1633. Both are still far below +30.
   - 1635 passes the SPY-adjusted criterion only because its long-SPY hedge collects the bull market's drift.
9. **Small cap / illiquidity (the ≤ $1B split).**
   - I confirmed there is no market-cap or shares-outstanding source on disk.
   - As a proxy I used the 20-day dollar volume at the session before the reaction:

     | 1634 by dollar volume | < $5M | $5–25M | > $25M |
     |---|---|---|---|
     | TRAIN (bps) | 168 (t 1.72) | 91 | 55 |
     | VAL (bps) | 111 (t 1.22) | −28 | 181 (t 2.36; ex-top-5 % −23, SPY-adj −53) |

   - The direction reverses between the two halves, so there is no stable illiquidity premium. The VAL > $25M
     bucket is post hoc, and it fails anyway.
10. **Prior-session convention** (why the compare's Jaccard is 0.70). Under the rebuild's reading, all three cells
    still fail (T10). The rebuild's 1635 Jaccard of 0.21 comes from the rebuild skipping the SSR filter, which the
    PREREG requires. It is not a builder defect.

Every t above clusters by entry day only. The 10- and 20-session holds overlap from one entry day to the next, so if
anything these t values are overstated.

## Adequacy (this is not a rescue)

- **The frozen bar needs a large effect.** The standard error of the 1634 VAL mean is about 66 bps. Reaching t ≥ 2.5
  on a single long leg needs about +165 bps per 20 sessions. Strict 10-step monotonicity is close to unreachable when
  each decile's error is 40–70 bps.
- **The spread has the same sign in both halves.** At +20 sessions, the top-minus-bottom decile spread is +176 bps in
  TRAIN and +153 in VAL (Spearman 0.75 and 0.68).
- **That spread cannot be used here:**
  - It was not a pre-registered cell.
  - Its short leg cannot be executed because of SSR and borrow.
  - It sits on an event set that is survivors-only and 16 % duplicated.
- **What a follow-up would need.** Any new test needs its own PREREG, after fixing items 1 (the mixed convention), 3
  (deduplication) and 5 (a point-in-time CIK map that includes delisted names).
