# RESULT 1,750 — opportunistic insider cluster buys on filing (PREREG_1750): KILL

Scorer `1750_score.py` rebuilt memory-lean (dictionary-encoded keys, column-at-a-time float32 matrices, block rank/median, bars cut at 2024-02-29, TEST never scored). Peak RSS 1.25 GB under `ulimit -v 2.5 GB`; 24 s. Log `1750_score.log`, tables `1750_summary.csv`, `1750_yearly.csv`, symbol-day events `1750_symdays.parquet`, owner-events `1750_events.parquet`.

**Coverage.** SEC Form 3/4/5 sets 2014Q1–2026Q1 (49 quarters; 2026Q2/Q3 not yet published, HTTP 404; both after the seal). 58,893 qualifying owner-events before 2024. Bars: 94.1 % mapped (LOST symbols 98 of 2,633, 5.9 % of events); 65 % of the mapped events are in the universe (rank ≤ 3,000, px ≥ $3) at the filing day, giving 27,827 symbol-day events. ROUTINE is only 3.3 % of owner-events (2-year history rule), so the placebo is thin (VAL n 294, MDE 126 bp).

**Primary (OPP, 5 d, all-universe, events/wk TRAIN 71 / VAL 61).**
- VAL gross +31.5 bp, day-clustered NW t 2.43 (passes +30 / |t| 2 by a hair); ex-top-5 % −51 bp (KILL); winner-capped +20 % +25 bp.
- VAL cost 77.9 bp round trip, net −41.9 bp (KILL). By rank bucket the only positive-net cell is none (r1–1000 VAL +25 bp gross, t 1.3, net −1.5).
- OPP vs ROUTINE 5 d: TRAIN −0.9 bp (KILL, Welch t −0.03), VAL +2.0 bp (t 0.06). Mechanism not shown.
- TRAIN halves: +21 / +49 bp, same sign. Years (OPP 5 d): 2016 +49, 2017 −17, 2018 +35, 2019 +31, 2020 +68, 2021 +27, 2022 +53 (2022 positive, passes), 2023 +13.
- 20 d: VAL OPP +57 bp (t 2.1), ROUTINE +133 bp, ex-top-5 % −125 bp.

**Cluster read (5 d).** TRAIN +92 bp (t 4.3, ex-top-5 % −94), TRAIN halves +84 / +103, but VAL +15 bp (t 0.6), 2021 negative: the cluster edge does not hold out of sample.

**Kills hit (any one kills):** VAL ex-top-5 % ≤ 0, VAL net ≤ 0, OPP not above ROUTINE on TRAIN. Not hit: VAL gross, t, TRAIN halves, 2022, events/wk.

**Caveats.** Universe benchmark uses cache names only; extras (2,004 symbols) are rank-only. 20-day ADV rank through filing day. The positive mean is tail-carried (ex-top-5 % negative on every read) and costs (20–133 bp by bucket) exceed it everywhere. 1,750th cell of the programme; TEST stays sealed. An adequacy gap: with ROUTINE 3 % the paper's zero is not resolvable at this n.
