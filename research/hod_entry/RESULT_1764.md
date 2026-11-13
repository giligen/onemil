# RESULT 1,764 - EDGAR events vs the HOD-break book (PREREG_1764)

**Both reads REJECT; the availability rail also VOIDs them (mapped 67.9 % of symbol-days < 80 %).** Numbers are
descriptive, mapped subset only. TEST untouched.

Data: 1,621 of 2,097 symbols mapped (77.3 %), 1,592 CIKs, 959,138 filings; 4 CIKs needed `filings.files` older
pages. Winner/loser mapped gap 3.9 pp (VAL). Mapped prevalence: dil_90 40.4 %, dil_30 21.8 %, earn_day 18.8 %.
Book: 1,763 HOD comparator trades, R floor 1 %, VAL baseline -0.19 R, so dropping any subset raises R/wk.

| Read, split | yes mean | no mean | diff | day-t | fast yes/no |
|---|---|---|---|---|---|
| E1 dil_90, H1 | -0.116 | -0.315 | +0.20 | 1.5 | 0.33/0.29 |
| E1 dil_90, H2 | -0.227 | -0.314 | +0.09 | 0.7 | 0.33/0.31 |
| E1 dil_90, VAL | -0.122 | -0.213 | +0.09 | 0.9 | 0.32/0.33 |
| E2 earn_day, H1 | -0.408 | -0.190 | -0.22 | -1.3 | 0.28/0.31 |
| E2 earn_day, H2 | -0.258 | -0.271 | +0.01 | 0.1 | 0.31/0.32 |
| E2 earn_day, VAL | -0.227 | -0.159 | -0.07 | -0.6 | 0.34/0.33 |

E1: flagged bucket is ABOVE unflagged on all splits, opposite to the claim. VAL filter +1.88 R/wk (t 1.3, 19.1
fills/wk kept); the random-50 %-of-CIKs placebo gives +2.55 R/wk, so the gain is the mechanical negative-book
effect. E2: earn_day is below the rest on H1 and VAL, flat on H2; 6.9 fills/wk. Fast-loser share (stop within
15 min) is 0.28-0.34 in every bucket: neither flag selects it.

Caveats: 8-K "purchase/ATM agreement" is proxied by item 3.02 (no document text); ticker maps to the CURRENT CIK,
so delisted names are unmapped and E1 is likely understated; no holiday calendar in the earnings window.

CHILDREN:
1. E1 with point-in-time ticker-to-CIK (delisted names) to lift mapping above 80 %.
2. Takedown-day event: 424B5/S-3 accepted t-1 after close or t pre-market vs the fast-loser tail.
3. E2 earnings-day gappers as a multi-day hold or gap-fade, not an HOD-break entry.
