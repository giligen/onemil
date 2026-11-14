# RESULT 1,773 — gap-fade short, MOO short / MOC cover, no stop (executed per PREREG_1773.md)

Population 7,487; open/close coverage 95.9 %; 274 rows dropped for gap > 200 % (incl. the 26,796 % artifact); used 6,908
(VAL 3,540 / TRAIN 3,368). Costs: fees 4 bp + locate 9 bp + borrow 100 %/360 (27.8 bp) = 41 bp. R unit: $300 x net / $150;
weekly R from <= 10 seeded names/day. Threshold-list proxy: only 2 symbols (PLYX, RPGL; 3 rows) are recoverable from the
on-disk CSV, so that haircut is ~void (n dropped 1).

PRIMARY VAL: net +79.5 bp/trade, day-t 1.61 (iid 3.5), ex-top-5 % **−81 bp**, ex-top-1 % +15, worst-5 % **−2,114 bp (−21 %)**,
worst-1 % −3,493, capped +61, weekly P10 −2.02 R, green weeks 59 % (null 23 %, p95 36 %), median 25 names/day.
TRAIN +247 bp (t 3.4), H1 +238 / H2 +253 (same sign). Cost fragility: VAL +50 bp at locate 38/borrow 100 %, −80 bp at 113/300 %.
Tiers VAL: $1–5 +204 (t 3.1, n 628, median 5 names/day), $5–20 +58 (t 1.05), >$20 +34 (t 0.7). Gap T3 +229, T1/T2 ~0.
Filter (c) VAL gross +196 vs count-matched placebo p95 +180 (net +155, t 2.2): beats it, but ex-top-5 % −77 and worst-5 % −28 %.
Placebo (i) random universe VAL p95 +5.7 gross: beaten. Placebo (ii) same name +5 d VAL p95 +126 vs primary gross +120: NOT beaten.
10:30 cover gross VAL +206 (t 8.6), TRAIN +253 (fact; no taker cost). Capacity (<= 2 % of 09:30 and 16:00 bar $vol): 51 % of VAL < 60 %;
bar-cache missingness by winner 34 % vs 50 % (gap 16 pp > 5 pp) so the capacity read is VOID per the rail; open/close itself 95.9 %.
SSR irrelevant to a same-day MOO short (applies next day only).

VERDICT: FAIL (t < 2, ex-top-5 % < 0, worst-5 % −21 % vs −12 % bar, capacity 51 %, placebo (ii) not beaten, P10 −2.02). The tail, not the mean, kills the no-stop book.
CHILDREN: (1) the $1–5 tier alone with <= 5 names/day and a hard per-name loss cap implemented as a resting buy-stop-limit; (2) 10:30 cover (gross t 8.6) re-priced with measured NBBO taker cost; (3) gap-T3 x filter (c) on the same-name-+5d placebo with the 1,762 bar cache restricted to complete-bar days.
