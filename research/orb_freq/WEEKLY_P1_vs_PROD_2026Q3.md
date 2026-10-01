# Weekly Comparison: Production vs Pool idea1 (gap 3–5%) — Q3 2026

| Week | Prod Fills | Prod sum R | Prod mean R | Prod $ | Prod ✓ | idea1 Fills | idea1 sum R | idea1 mean R | idea1 $ | idea1 ✓ | Union Fills | Union sum R | Union mean R | Union $ | Union ✓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W27 | — | — | — | — | — | 3 | −0.75 | −0.25 | −$281 | ✗ | 3 | −0.75 | −0.25 | −$281 | ✗ |
| W28 | 10 | −2.07 | −0.21 | −$776 | ✗ | 9 | −0.36 | −0.04 | −$135 | ✗ | 19 | −2.43 | −0.13 | −$911 | ✗ |
| W29 | 13 | +0.23 | +0.02 | +$85 | ✓ | 8 | −0.26 | −0.03 | −$98 | ✗ | 21 | −0.03 | −0.002 | −$13 | ✗ |
| W30 | 15 | +1.60 | +0.11 | +$599 | ✓ | 6 | +1.22 | +0.20 | +$458 | ✓ | 21 | +2.82 | +0.13 | +$1,057 | ✓ |
| W31 | 14 | +3.96 | +0.28 | +$1,483 | ✓ | 6 | −0.98 | −0.16 | −$367 | ✗ | 20 | +2.98 | +0.15 | +$1,116 | ✓ |
| W32 | 7 | −1.01 | −0.14 | −$380 | ✗ | 7 | −1.26 | −0.18 | −$474 | ✗ | 14 | −2.28 | −0.16 | −$854 | ✗ |
| W33 | 5 | −0.94 | −0.19 | −$354 | ✗ | 8 | +1.14 | +0.14 | +$426 | ✓ | 13 | +0.19 | +0.01 | +$72 | ✓ |
| W34 | — | — | — | — | — | 5 | +0.45 | +0.09 | +$170 | ✓ | 5 | +0.45 | +0.09 | +$170 | ✓ |
| W35 | 6 | +1.46 | +0.24 | +$547 | ✓ | 5 | −0.41 | −0.08 | −$153 | ✗ | 11 | +1.05 | +0.10 | +$394 | ✓ |
| W36 | 7 | +0.54 | +0.08 | +$203 | ✓ | 4 | +1.24 | +0.31 | +$466 | ✓ | 11 | +1.78 | +0.16 | +$669 | ✓ |
| W37 | 1 | −0.17 | −0.17 | −$65 | ✗ | 2 | −0.29 | −0.15 | −$109 | ✗ | 3 | −0.46 | −0.15 | −$174 | ✗ |
| W38 | 9 | +0.35 | +0.04 | +$131 | ✓ | 1 | −0.24 | −0.24 | −$89 | ✗ | 10 | +0.11 | +0.01 | +$42 | ✓ |
| **Q3 2026** | **87** | **+3.93** | **+0.045** | **+$1,473** | **6 wks** | **64** | **−0.49** | **−0.008** | **−$185** | **4 wks** | **151** | **+3.44** | **+0.023** | **+$1,288** | **7 wks** |

**Columns:** Fills = entered trades; sum R = cumulative excess return (units $375); mean R = per-trade; $ = sum R × $375 per fill; ✓/✗ = green (sum R > 0) flag.

**Definitions:** 
- **Production (prod):** ORB admission filter gap ≥ 5% & price $3–30 (per-fill books from cell 1,684 entry reconstruction, LIVE config).
- **idea1:** Gap 3–5% & range_high ≥ prev_close + 5% by 09:35 (same pipeline, per-pool selection chain, no overlap with prod by construction).
- **Union:** De-duplicated by (date, symbol), taking both if independent or the higher R if same trade; union fills = production + idea1 non-duplicates.
- **Q3 2026 window:** 2026-07-01 (W27 start) through 2026-09-30; data available through 2026-09-18 (W38 partial).

**Key findings:**
- Production: +$1,473 (6 green weeks, +0.045 R/fill). idea1 underperforms in-sample: −$185 (4 green weeks, −0.008 R/fill, ex-top-5% −0.024, cell 1,684 FAIL verdict stands).
- Union adds idea1 fills (+71% frequency) but lowers mean R to +0.023 and expands worst week from −$776 to −$911 (W28).
- Worst week either book: W28 prod −$776, idea1 −$135 → union −$911; worst week union alone: W32 −$854.

