# D5 — red-to-green (F6-PDR, FIRST-BREAK rule): the losers, by hand

Book: `research/fuckup_audit/H/F6_rebuild/trades_hold_ai.csv` — implementation B (the independent rebuild), HOLD exit, booked 12/day at 4 concurrent. 2,011 rows minus 2 `Z?ZZT` test-ticker rows = **2009 trades**, 417 trading days, 2025-01-02 .. 2026-09-04.

| split | n | total net R | R/trade | WR | losers | loser R | winner R |
|---|---|---|---|---|---|---|---|
| TRAIN | 1112 | +69.0 | +0.0620 | 43.7% | 626 | -470.0 | +539.0 |
| VAL | 531 | +109.7 | +0.2066 | 47.3% | 280 | -226.0 | +335.7 |
| TEST | 366 | -15.4 | -0.0420 | 39.6% | 221 | -172.4 | +157.0 |
| ALL | 2009 | +163.4 | +0.0813 | 43.9% | 1127 | -868.3 | +1031.7 |

Reference (`H/F6_reconcile/REPORT.md`): this is the rule the two studies ran, NOT the rule `trading/red_to_green.py` implements. The shipped engine keeps scanning past a floor-failing bar and its book is -0.027 / -0.012 / -0.102 R. Everything below is the anatomy of the FIRST-BREAK book.

---

## 1. Losing days as tape

### 1a. The 25 worst days

**2026-06-03** (TEST)  book **-8.15 R** · 8 trades (1 green) · candidates 25 · SPY o->c -0.52% / o->10:00 -0.21% · IWM o->c -0.81% / o->10:00 -0.70%
  - `ODD   ` 09:32 net **+0.60R** · entry 9.90 (+0.6% vs prior close, gap -5.5%, PDR 12%, stop 6.6% away, 5m$ $1,651,447) · held 383m to the close, MFE +1.06R · **other**
  - `MRLN  ` 09:35 net **-1.05R** · entry 7.66 (+0.7% vs prior close, gap -2.6%, PDR 12%, stop 6.1% away, 5m$ $1,159,524) · stopped 3m later after MFE -0.28R · **levelfail**
  - `SLMT  ` 09:35 net **-3.65R** · entry 5.92 (+0.3% vs prior close, gap -0.8%, PDR 15%, stop 1.1% away, 5m$ $9,073) · stopped 23m later after MFE -3.38R · **thin**
  - `NVRI  ` 09:37 net **-1.07R** · entry 18.90 (-1.6% vs prior close, gap -0.9%, PDR 23%, stop 4.8% away, 5m$ $1,198,268) · stopped 36m later after MFE +0.13R · **levelfail**
  - `CRMT  ` 09:41 net **-1.03R** · entry 8.28 (-0.7% vs prior close, gap -5.6%, PDR 29%, stop 9.2% away, 5m$ $111,774) · stopped 286m later after MFE +0.46R · **thin**
  - `TRAX  ` 10:00 net **-0.35R** · entry 15.98 (-1.0% vs prior close, gap -2.2%, PDR 15%, stop 7.9% away, 5m$ $48,782) · held 355m to the close, MFE +0.34R · **thin**
  - `QBTX  ` 10:14 net **-1.03R** · entry 25.77 (-0.3% vs prior close, gap -2.0%, PDR 16%, stop 13.4% away, 5m$ $3,399,700) · stopped 60m later after MFE -0.13R · **levelfail**
  - `HYLN  ` 11:31 net **-0.57R** · entry 6.79 (-0.1% vs prior close, gap -2.5%, PDR 16%, stop 7.9% away, 5m$ $600,473) · held 264m to the close, MFE +0.36R · **other**

**2025-01-08** (TRAIN)  book **-7.47 R** · 8 trades (1 green) · candidates 9 · SPY o->c 0.12% / o->10:00 -0.15% · IWM o->c 0.37% / o->10:00 -0.54%
  - `MSTU  ` 09:34 net **-1.05R** · entry 10.09 (-0.0% vs prior close, gap -4.0%, PDR 24%, stop 6.4% away, 5m$ $20,114,979) · stopped 149m later after MFE +0.19R · **dump**
  - `MSTX  ` 09:34 net **-1.06R** · entry 47.08 (+0.6% vs prior close, gap -3.5%, PDR 25%, stop 6.5% away, 5m$ $16,272,833) · stopped 149m later after MFE +0.14R · **dump**
  - `RCAT  ` 09:37 net **-1.06R** · entry 12.88 (-1.4% vs prior close, gap -6.1%, PDR 16%, stop 5.8% away, 5m$ $5,385,564) · stopped 13m later after MFE -0.05R · **levelfail**
  - `NVAWW ` 09:43 net **-3.18R** · entry 9.20 (-3.7% vs prior close, gap -5.2%, PDR 17%, stop 1.6% away, 5m$ $13,601) · stopped 13m later after MFE -3.00R · **thin**
  - `NIXX  ` 10:05 net **-1.10R** · entry 5.56 (-4.6% vs prior close, gap -1.4%, PDR 15%, stop 2.6% away, 5m$ $73,685) · stopped 64m later after MFE +0.28R · **thin**
  - `NVD   ` 10:15 net **-0.13R** · entry 27.14 (+0.6% vs prior close, gap -3.4%, PDR 18%, stop 5.9% away, 5m$ $1,723,367) · held 340m to the close, MFE +0.43R · **other**
  - `ANVS  ` 11:22 net **+0.42R** · entry 5.07 (+0.4% vs prior close, gap -0.8%, PDR 17%, stop 8.3% away, 5m$ $96,538) · held 218m to the close, MFE +0.43R · **thin**
  - `SGMT  ` 13:41 net **-0.31R** · entry 5.59 (+0.4% vs prior close, gap -0.5%, PDR 21%, stop 11.5% away, 5m$ $272,804) · held 79m to the close, MFE -0.01R · **bleed**

**2025-12-17** (TRAIN)  book **-7.25 R** · 7 trades (0 green) · candidates 10 · SPY o->c -1.26% / o->10:00 -0.15% · IWM o->c -1.27% / o->10:00 0.49%
  - `CRCD  ` 09:50 net **-1.23R** · entry 32.15 (-1.4% vs prior close, gap -1.4%, PDR 20%, stop 4.2% away, 5m$ $241,396) · stopped 8m later after MFE -0.04R · **dump**
  - `MST   ` 09:57 net **-1.16R** · entry 7.31 (+0.7% vs prior close, gap -0.7%, PDR 10%, stop 5.6% away, 5m$ $51,473) · stopped 78m later after MFE +0.30R · **thin**
  - `WKHS  ` 10:08 net **-1.11R** · entry 5.64 (-1.4% vs prior close, gap -0.2%, PDR 10%, stop 2.5% away, 5m$ $12,631) · stopped 82m later after MFE +0.14R · **thin**
  - `ARBK  ` 10:23 net **-1.04R** · entry 5.72 (-0.2% vs prior close, gap -5.6%, PDR 20%, stop 6.5% away, 5m$ $108,655) · stopped 46m later after MFE +0.11R · **thin**
  - `LUD   ` 10:34 net **-0.99R** · entry 10.44 (-0.7% vs prior close, gap -1.6%, PDR 8%, stop 15.9% away, 5m$ $28,487) · held 325m to the close, MFE +0.70R · **thin**
  - `GDXD  ` 11:14 net **-0.53R** · entry 9.40 (+0.2% vs prior close, gap -4.6%, PDR 9%, stop 6.6% away, 5m$ $126,026) · held 281m to the close, MFE +0.02R · **thin**
  - `GLTO  ` 11:36 net **-1.20R** · entry 30.14 (+0.5% vs prior close, gap -4.8%, PDR 16%, stop 7.1% away, 5m$ $174,931) · stopped 168m later after MFE +1.48R · **thin**

**2026-07-16** (TEST)  book **-7.22 R** · 10 trades (2 green) · candidates 25 · SPY o->c -0.26% / o->10:00 -0.04% · IWM o->c 0.33% / o->10:00 0.74%
  - `ELVA  ` 09:32 net **-1.07R** · entry 11.79 (+0.3% vs prior close, gap -3.7%, PDR 21%, stop 4.6% away, 5m$ $2,317,735) · stopped 1m later after MFE -0.69R · **news**
  - `CLSX  ` 09:33 net **-1.06R** · entry 18.38 (-1.2% vs prior close, gap -4.8%, PDR 27%, stop 6.0% away, 5m$ $200,282) · stopped 16m later after MFE -0.00R · **levelfail**
  - `CLRO  ` 09:34 net **-1.16R** · entry 7.69 (+0.0% vs prior close, gap -1.3%, PDR 49%, stop 1.8% away, 5m$ $147,975) · stopped 12m later after MFE +1.14R · **thin**
  - `DLLL  ` 09:34 net **-1.07R** · entry 22.06 (+0.1% vs prior close, gap -4.1%, PDR 43%, stop 5.5% away, 5m$ $1,370,534) · stopped 11m later after MFE -0.18R · **levelfail**
  - `MANE  ` 09:36 net **-1.07R** · entry 124.00 (+0.2% vs prior close, gap -0.5%, PDR 10%, stop 7.1% away, 5m$ $1,367,787) · stopped 101m later after MFE +0.07R · **bleed**
  - `NUCL  ` 09:46 net **-1.07R** · entry 5.99 (+0.6% vs prior close, gap -1.5%, PDR 16%, stop 3.9% away, 5m$ $148,021) · stopped 105m later after MFE +0.40R · **thin**
  - `AMA   ` 10:03 net **-1.05R** · entry 36.72 (+0.1% vs prior close, gap -6.3%, PDR 21%, stop 6.4% away, 5m$ $438,114) · stopped 227m later after MFE +0.62R · **news**
  - `AMAU  ` 10:03 net **-1.06R** · entry 22.13 (-0.4% vs prior close, gap -6.5%, PDR 20%, stop 6.1% away, 5m$ $171,712) · stopped 218m later after MFE +0.67R · **thin**
  - `IBX   ` 11:18 net **+1.03R** · entry 14.68 (+0.3% vs prior close, gap -1.9%, PDR 13%, stop 7.0% away, 5m$ $312,667) · held 277m to the close, MFE +1.08R · **other**
  - `ZSQR  ` 11:54 net **+0.38R** · entry 5.17 (+0.0% vs prior close, gap -3.1%, PDR 32%, stop 10.3% away, 5m$ $77,874) · held 241m to the close, MFE +0.47R · **thin**

**2026-08-24** (TEST)  book **-6.43 R** · 7 trades (0 green) · candidates 25 · SPY o->c -0.17% / o->10:00 -0.17% · IWM o->c -0.53% / o->10:00 -0.32%
  - `UUUG  ` 09:37 net **-1.72R** · entry 5.16 (+0.0% vs prior close, gap -1.2%, PDR 12%, stop 1.7% away, 5m$ $39,475) · stopped 7m later after MFE +0.83R · **thin**
  - `IPST  ` 09:38 net **-1.03R** · entry 9.55 (-1.0% vs prior close, gap -6.9%, PDR 32%, stop 9.5% away, 5m$ $243,486) · stopped 36m later after MFE +0.43R · **news**
  - `CEPL  ` 10:04 net **-1.06R** · entry 6.47 (-1.9% vs prior close, gap -2.6%, PDR 12%, stop 4.4% away, 5m$ $80,654) · stopped 160m later after MFE +0.82R · **thin**
  - `AEHL  ` 10:15 net **-0.61R** · entry 6.32 (+0.6% vs prior close, gap -6.2%, PDR 20%, stop 7.7% away, 5m$ $43,299) · held 339m to the close, MFE +0.14R · **thin**
  - `UI    ` 10:27 net **-0.38R** · entry 563.00 (+0.7% vs prior close, gap -3.4%, PDR 16%, stop 5.9% away, 5m$ $5,378,181) · held 328m to the close, MFE +0.44R · **other**
  - `GLXU  ` 10:33 net **-1.04R** · entry 5.70 (+0.5% vs prior close, gap -0.9%, PDR 16%, stop 6.3% away, 5m$ $18,953) · stopped 275m later after MFE +0.11R · **thin**
  - `FTK   ` 12:50 net **-0.58R** · entry 24.80 (+0.5% vs prior close, gap -1.1%, PDR 9%, stop 5.4% away, 5m$ $352,371) · held 185m to the close, MFE +0.05R · **bleed**

**2025-10-21** (TRAIN)  book **-6.38 R** · 6 trades (0 green) · candidates 25 · SPY o->c -0.03% / o->10:00 -0.18% · IWM o->c -0.18% / o->10:00 -0.66%
  - `BAIG  ` 09:37 net **-1.06R** · entry 22.59 (+0.6% vs prior close, gap -3.4%, PDR 13%, stop 5.7% away, 5m$ $431,790) · stopped 5m later after MFE -0.05R · **levelfail**
  - `RIOX  ` 09:39 net **-1.07R** · entry 42.05 (+0.0% vs prior close, gap -2.3%, PDR 17%, stop 4.9% away, 5m$ $600,220) · stopped 10m later after MFE -0.13R · **levelfail**
  - `STEX  ` 10:11 net **-1.05R** · entry 5.38 (+0.1% vs prior close, gap -0.7%, PDR 12%, stop 5.1% away, 5m$ $89,687) · stopped 192m later after MFE +0.64R · **thin**
  - `RGTI  ` 10:16 net **-1.06R** · entry 43.48 (+0.4% vs prior close, gap -1.8%, PDR 13%, stop 5.6% away, 5m$ $67,131,602) · stopped 216m later after MFE +0.00R · **bleed**
  - `RGTX  ` 10:16 net **-1.05R** · entry 266.65 (+0.5% vs prior close, gap -3.5%, PDR 29%, stop 10.7% away, 5m$ $4,050,439) · stopped 216m later after MFE +0.03R · **bleed**
  - `RGTIW ` 10:17 net **-1.08R** · entry 31.68 (-0.0% vs prior close, gap -2.3%, PDR 19%, stop 6.7% away, 5m$ $56,721) · stopped 216m later after MFE -0.00R · **thin**

**2026-05-13** (VAL)  book **-6.23 R** · 10 trades (2 green) · candidates 31 · SPY o->c 0.52% / o->10:00 -0.20% · IWM o->c -0.08% / o->10:00 -0.82%
  - `USGG  ` 09:33 net **-1.05R** · entry 19.75 (+0.6% vs prior close, gap -1.0%, PDR 20%, stop 6.3% away, 5m$ $290,072) · stopped 11m later after MFE -0.10R · **other**
  - `INBX  ` 09:34 net **-1.16R** · entry 108.31 (-0.3% vs prior close, gap -2.0%, PDR 26%, stop 3.0% away, 5m$ $1,853,019) · stopped 4m later after MFE +0.06R · **other**
  - `SMX   ` 09:39 net **-1.04R** · entry 9.21 (+0.7% vs prior close, gap -4.7%, PDR 73%, stop 7.2% away, 5m$ $118,829) · stopped 21m later after MFE +1.18R · **thin**
  - `SEAT  ` 09:41 net **-1.12R** · entry 8.75 (+0.8% vs prior close, gap -1.7%, PDR 9%, stop 2.5% away, 5m$ $17,714) · stopped 6m later after MFE +0.00R · **thin**
  - `ACHV  ` 09:42 net **-0.03R** · entry 6.00 (+0.2% vs prior close, gap -0.2%, PDR 15%, stop 6.2% away, 5m$ $238,123) · held 373m to the close, MFE +0.26R · **bleed**
  - `RXT   ` 09:57 net **-1.03R** · entry 6.36 (+0.5% vs prior close, gap -5.4%, PDR 33%, stop 8.8% away, 5m$ $6,852,744) · stopped 102m later after MFE +0.64R · **other**
  - `MMED  ` 10:09 net **-1.07R** · entry 11.88 (-0.1% vs prior close, gap -1.9%, PDR 8%, stop 4.0% away, 5m$ $247,428) · stopped 195m later after MFE +0.03R · **news**
  - `PACS  ` 10:09 net **-0.21R** · entry 41.20 (+0.5% vs prior close, gap -0.5%, PDR 11%, stop 5.5% away, 5m$ $770,562) · held 346m to the close, MFE +0.35R · **other**
  - `LQDA  ` 11:56 net **+0.09R** · entry 56.80 (+0.4% vs prior close, gap -1.1%, PDR 13%, stop 5.4% away, 5m$ $736,283) · held 239m to the close, MFE +0.21R · **bleed**
  - `BNAI  ` 13:32 net **+0.39R** · entry 23.99 (+0.8% vs prior close, gap -0.8%, PDR 17%, stop 5.7% away, 5m$ $74,168) · held 143m to the close, MFE +0.62R · **thin**

**2026-01-29** (VAL)  book **-5.49 R** · 7 trades (1 green) · candidates 8 · SPY o->c -0.34% / o->10:00 -0.82% · IWM o->c -0.27% / o->10:00 -0.55%
  - `QBTX  ` 09:39 net **-1.06R** · entry 28.44 (-0.1% vs prior close, gap -3.1%, PDR 9%, stop 6.2% away, 5m$ $829,574) · stopped 13m later after MFE -0.25R · **dump**
  - `SATL  ` 09:39 net **-1.04R** · entry 5.33 (-0.7% vs prior close, gap -1.5%, PDR 12%, stop 6.6% away, 5m$ $1,222,602) · stopped 354m later after MFE +1.71R · **dump**
  - `KZIA  ` 09:49 net **-1.09R** · entry 6.45 (-1.7% vs prior close, gap -0.6%, PDR 15%, stop 3.1% away, 5m$ $12,185) · stopped 60m later after MFE +0.21R · **thin**
  - `CRWG  ` 09:51 net **-1.03R** · entry 5.63 (+0.4% vs prior close, gap -4.6%, PDR 24%, stop 8.3% away, 5m$ $3,937,069) · stopped 10m later after MFE -0.19R · **dump**
  - `KOLD  ` 10:42 net **-1.04R** · entry 19.35 (+0.5% vs prior close, gap -3.3%, PDR 11%, stop 6.6% away, 5m$ $15,666,758) · stopped 203m later after MFE +0.04R · **bleed**
  - `NAVI  ` 11:43 net **+0.01R** · entry 9.82 (+0.1% vs prior close, gap -0.5%, PDR 22%, stop 4.8% away, 5m$ $228,692) · held 252m to the close, MFE +0.47R · **other**
  - `FBRX  ` 12:27 net **-0.23R** · entry 29.94 (+0.3% vs prior close, gap -0.3%, PDR 21%, stop 10.0% away, 5m$ $119,413) · held 208m to the close, MFE +0.07R · **thin**

**2026-07-29** (TEST)  book **-5.40 R** · 7 trades (1 green) · candidates 18 · SPY o->c -1.41% / o->10:00 -0.33% · IWM o->c -1.33% / o->10:00 -0.14%
  - `SNDQ  ` 09:32 net **-1.10R** · entry 53.21 (+0.0% vs prior close, gap -4.6%, PDR 19%, stop 4.8% away, 5m$ $17,744,806) · stopped 330m later after MFE +3.67R · **dump**
  - `CRDU  ` 09:36 net **-1.06R** · entry 10.96 (+0.5% vs prior close, gap -2.4%, PDR 17%, stop 5.3% away, 5m$ $164,926) · stopped 11m later after MFE -0.21R · **thin**
  - `SMU   ` 09:36 net **-1.05R** · entry 5.29 (+0.2% vs prior close, gap -4.4%, PDR 17%, stop 6.2% away, 5m$ $114,563) · stopped 68m later after MFE +0.45R · **thin**
  - `FGNX  ` 09:42 net **+0.05R** · entry 6.75 (-2.0% vs prior close, gap -3.4%, PDR 11%, stop 10.4% away, 5m$ $147,028) · held 377m to the close, MFE +0.63R · **thin**
  - `CHRN  ` 09:50 net **-1.05R** · entry 20.25 (+0.1% vs prior close, gap -7.7%, PDR 19%, stop 7.8% away, 5m$ $38,454) · stopped 346m later after MFE +0.40R · **late**
  - `NXTC  ` 11:10 net **-1.05R** · entry 5.33 (-1.8% vs prior close, gap -3.3%, PDR 18%, stop 4.8% away, 5m$ $69,065) · stopped 148m later after MFE +1.69R · **thin**
  - `REPL  ` 13:46 net **-0.14R** · entry 5.33 (-0.3% vs prior close, gap -1.9%, PDR 29%, stop 10.7% away, 5m$ $781,862) · held 129m to the close, MFE +0.41R · **other**

**2025-11-21** (TRAIN)  book **-5.32 R** · 8 trades (1 green) · candidates 40 · SPY o->c 0.62% / o->10:00 -0.06% · IWM o->c 2.46% / o->10:00 0.57%
  - `CIFR  ` 09:32 net **-1.10R** · entry 14.59 (+0.2% vs prior close, gap -2.3%, PDR 23%, stop 3.4% away, 5m$ $16,316,152) · stopped 30m later after MFE +0.55R · **other**
  - `IRE   ` 09:32 net **-1.11R** · entry 9.19 (-0.9% vs prior close, gap -3.5%, PDR 54%, stop 2.6% away, 5m$ $2,057,126) · stopped 30m later after MFE +1.67R · **other**
  - `KMTS  ` 09:32 net **+0.61R** · entry 24.97 (-0.1% vs prior close, gap -0.4%, PDR 8%, stop 7.2% away, 5m$ $160,762) · held 383m to the close, MFE +0.76R · **thin**
  - `MSTX  ` 09:32 net **-1.07R** · entry 5.56 (-1.4% vs prior close, gap -5.3%, PDR 28%, stop 4.1% away, 5m$ $5,406,002) · stopped 45m later after MFE +1.28R · **other**
  - `UVIX  ` 10:03 net **-1.04R** · entry 12.98 (+0.3% vs prior close, gap -5.2%, PDR 35%, stop 7.1% away, 5m$ $11,337,350) · stopped 112m later after MFE +0.77R · **other**
  - `HOOZ  ` 10:04 net **-1.06R** · entry 29.50 (+0.2% vs prior close, gap -3.1%, PDR 31%, stop 6.8% away, 5m$ $133,320) · stopped 257m later after MFE +1.06R · **thin**
  - `CONI  ` 10:18 net **-0.33R** · entry 66.74 (-0.3% vs prior close, gap -4.1%, PDR 22%, stop 6.1% away, 5m$ $454,781) · held 337m to the close, MFE +1.00R · **news**
  - `SMMT  ` 11:59 net **-0.22R** · entry 16.53 (+0.5% vs prior close, gap -0.3%, PDR 9%, stop 5.5% away, 5m$ $490,281) · held 236m to the close, MFE +0.03R · **bleed**

**2026-03-26** (VAL)  book **-5.14 R** · 6 trades (0 green) · candidates 19 · SPY o->c -1.07% / o->10:00 0.35% · IWM o->c -0.59% / o->10:00 0.87%
  - `RDWU  ` 09:50 net **-0.84R** · entry 9.98 (-0.4% vs prior close, gap -5.1%, PDR 22%, stop 7.9% away, 5m$ $81,994) · held 369m to the close, MFE +0.43R · **thin**
  - `AZ    ` 09:51 net **-1.08R** · entry 6.87 (-1.4% vs prior close, gap -0.9%, PDR 8%, stop 3.5% away, 5m$ $15,855) · stopped 67m later after MFE +0.71R · **thin**
  - `QNRX  ` 09:57 net **-1.03R** · entry 9.30 (+0.3% vs prior close, gap -3.2%, PDR 41%, stop 11.3% away, 5m$ $58,944) · stopped 237m later after MFE +0.02R · **thin**
  - `NUCL  ` 09:58 net **-1.05R** · entry 7.35 (+0.3% vs prior close, gap -0.8%, PDR 11%, stop 5.4% away, 5m$ $111,161) · stopped 40m later after MFE +0.15R · **thin**
  - `OPEX  ` 10:49 net **-1.09R** · entry 25.48 (-2.2% vs prior close, gap -5.5%, PDR 15%, stop 4.2% away, 5m$ $45,821) · stopped 151m later after MFE -0.16R · **thin**
  - `ARWR  ` 11:04 net **-0.05R** · entry 60.95 (+0.3% vs prior close, gap -6.7%, PDR 10%, stop 9.1% away, 5m$ $1,616,807) · held 291m to the close, MFE +0.13R · **bleed**

**2026-03-12** (VAL)  book **-5.02 R** · 6 trades (0 green) · candidates 11 · SPY o->c -0.76% / o->10:00 -0.19% · IWM o->c -0.60% / o->10:00 -0.12%
  - `PLYX  ` 09:43 net **-1.06R** · entry 5.76 (+0.1% vs prior close, gap -3.8%, PDR 23%, stop 4.9% away, 5m$ $237,955) · stopped 76m later after MFE +0.90R · **dump**
  - `SEAT  ` 10:01 net **-0.39R** · entry 6.06 (-0.7% vs prior close, gap -17.0%, PDR 14%, stop 16.5% away, 5m$ $20,520) · held 354m to the close, MFE +0.57R · **late**
  - `KTUP  ` 10:13 net **-1.15R** · entry 26.30 (+0.7% vs prior close, gap -1.1%, PDR 10%, stop 7.1% away, 5m$ $35,514) · stopped 60m later after MFE +0.03R · **thin**
  - `CWVX  ` 11:44 net **-0.69R** · entry 22.64 (+0.2% vs prior close, gap -5.3%, PDR 13%, stop 8.6% away, 5m$ $880,981) · held 251m to the close, MFE +0.08R · **bleed**
  - `INTT  ` 11:54 net **-1.07R** · entry 14.58 (-0.7% vs prior close, gap -0.3%, PDR 8%, stop 3.1% away, 5m$ $45,975) · stopped 169m later after MFE +0.67R · **thin**
  - `KSS   ` 12:16 net **-0.67R** · entry 13.85 (+0.4% vs prior close, gap -1.8%, PDR 10%, stop 6.0% away, 5m$ $2,764,694) · held 219m to the close, MFE +0.16R · **bleed**

**2025-03-12** (TRAIN)  book **-4.98 R** · 6 trades (1 green) · candidates 6 · SPY o->c -0.58% / o->10:00 -0.54% · IWM o->c -0.95% / o->10:00 -0.77%
  - `SKBL  ` 09:38 net **-1.05R** · entry 11.04 (+0.8% vs prior close, gap -5.5%, PDR 11%, stop 6.3% away, 5m$ $97,703) · stopped 125m later after MFE +0.52R · **thin**
  - `DNTH  ` 09:42 net **-1.09R** · entry 23.41 (+0.1% vs prior close, gap -6.5%, PDR 9%, stop 6.8% away, 5m$ $149,052) · stopped 66m later after MFE +0.03R · **thin**
  - `UVIX  ` 11:00 net **-1.03R** · entry 47.40 (+0.4% vs prior close, gap -7.4%, PDR 15%, stop 8.7% away, 5m$ $4,494,909) · stopped 220m later after MFE +0.21R · **late**
  - `UVXY  ` 11:00 net **-1.04R** · entry 27.74 (+0.5% vs prior close, gap -5.4%, PDR 12%, stop 6.6% away, 5m$ $9,340,644) · stopped 220m later after MFE +0.19R · **bleed**
  - `EHLDV ` 11:41 net **-1.02R** · entry 17.80 (+0.6% vs prior close, gap -2.6%, PDR 77%, stop 12.1% away, 5m$ $67,736) · stopped 180m later after MFE +9.20R · **thin**
  - `SRI   ` 12:25 net **+0.26R** · entry 5.26 (+0.6% vs prior close, gap -3.6%, PDR 8%, stop 4.2% away, 5m$ $22,415) · held 210m to the close, MFE +0.76R · **thin**

**2026-07-01** (TEST)  book **-4.93 R** · 7 trades (2 green) · candidates 33 · SPY o->c 0.10% / o->10:00 0.07% · IWM o->c -0.17% / o->10:00 0.48%
  - `AEVA  ` 09:32 net **-0.61R** · entry 28.16 (-1.9% vs prior close, gap -3.2%, PDR 17%, stop 2.4% away, 5m$ $2,924,923) · held 383m to the close, MFE +4.71R · **other**
  - `CRCD  ` 09:32 net **-1.17R** · entry 6.85 (-2.0% vs prior close, gap -3.7%, PDR 26%, stop 1.8% away, 5m$ $606,147) · stopped 2m later after MFE -0.08R · **other**
  - `HYLN  ` 09:32 net **-1.16R** · entry 5.17 (-0.9% vs prior close, gap -2.7%, PDR 14%, stop 1.8% away, 5m$ $508,039) · stopped 63m later after MFE +2.37R · **other**
  - `PLU   ` 09:33 net **-1.06R** · entry 22.32 (-0.9% vs prior close, gap -2.5%, PDR 26%, stop 6.3% away, 5m$ $103,814) · stopped 372m later after MFE +1.30R · **thin**
  - `RDWU  ` 09:36 net **-1.10R** · entry 9.81 (-1.2% vs prior close, gap -3.1%, PDR 13%, stop 3.1% away, 5m$ $218,907) · stopped 79m later after MFE +2.17R · **other**
  - `OKLS  ` 10:37 net **+0.02R** · entry 24.25 (-0.7% vs prior close, gap -5.5%, PDR 10%, stop 7.2% away, 5m$ $130,281) · held 320m to the close, MFE +0.53R · **thin**
  - `MPLT  ` 11:05 net **+0.16R** · entry 35.97 (+0.4% vs prior close, gap -1.0%, PDR 10%, stop 5.7% away, 5m$ $173,071) · held 290m to the close, MFE +0.42R · **thin**

**2026-08-05** (TEST)  book **-4.91 R** · 8 trades (2 green) · candidates 33 · SPY o->c -0.78% / o->10:00 0.03% · IWM o->c -0.88% / o->10:00 0.05%
  - `AAOX  ` 09:32 net **-0.77R** · entry 19.25 (-0.6% vs prior close, gap -5.4%, PDR 19%, stop 5.7% away, 5m$ $6,179,221) · held 383m to the close, MFE +2.37R · **news**
  - `CIFU  ` 09:35 net **-1.12R** · entry 17.89 (-0.2% vs prior close, gap -2.4%, PDR 25%, stop 6.0% away, 5m$ $288,499) · stopped 56m later after MFE +1.03R · **other**
  - `FCUV  ` 09:37 net **-1.07R** · entry 10.53 (-0.7% vs prior close, gap -4.5%, PDR 49%, stop 4.5% away, 5m$ $235,835) · stopped 7m later after MFE +0.03R · **other**
  - `JEM   ` 09:41 net **-1.04R** · entry 5.18 (-0.4% vs prior close, gap -1.0%, PDR 10%, stop 7.3% away, 5m$ $49,491) · stopped 75m later after MFE +0.09R · **thin**
  - `INTW  ` 09:46 net **-1.07R** · entry 24.11 (+0.4% vs prior close, gap -3.1%, PDR 14%, stop 4.9% away, 5m$ $2,649,885) · stopped 37m later after MFE +0.41R · **other**
  - `OSS   ` 10:36 net **-0.40R** · entry 14.32 (+0.4% vs prior close, gap -15.6%, PDR 9%, stop 17.6% away, 5m$ $516,519) · held 319m to the close, MFE -0.03R · **late**
  - `LGIH  ` 11:25 net **+0.12R** · entry 61.40 (-0.1% vs prior close, gap -2.7%, PDR 10%, stop 5.5% away, 5m$ $174,579) · held 270m to the close, MFE +0.28R · **thin**
  - `ADVB  ` 11:34 net **+0.45R** · entry 7.89 (-1.3% vs prior close, gap -1.9%, PDR 18%, stop 5.2% away, 5m$ $100,605) · held 264m to the close, MFE +3.06R · **thin**

**2025-02-20** (TRAIN)  book **-4.80 R** · 6 trades (2 green) · candidates 11 · SPY o->c -0.19% / o->10:00 -0.65% · IWM o->c -0.76% / o->10:00 -1.06%
  - `DUOT  ` 09:35 net **-3.00R** · entry 6.79 (+0.9% vs prior close, gap -0.4%, PDR 10%, stop 1.3% away, 5m$ $71,890) · stopped 1m later after MFE -0.44R · **thin**
  - `LUNRW ` 09:49 net **-1.07R** · entry 6.91 (-15.7% vs prior close, gap -1.2%, PDR 24%, stop 4.2% away, 5m$ $382,021) · stopped 3m later after MFE -0.07R · **dump**
  - `MED   ` 09:50 net **+0.30R** · entry 14.11 (-1.3% vs prior close, gap -0.9%, PDR 13%, stop 3.8% away, 5m$ $202,158) · held 365m to the close, MFE +1.17R · **news**
  - `MSTZ  ` 09:54 net **-0.67R** · entry 18.48 (+0.8% vs prior close, gap -3.0%, PDR 14%, stop 6.4% away, 5m$ $5,314,228) · held 361m to the close, MFE +0.27R · **bleed**
  - `CVI   ` 10:29 net **+0.21R** · entry 19.97 (+0.6% vs prior close, gap -1.0%, PDR 14%, stop 5.3% away, 5m$ $246,867) · held 326m to the close, MFE +0.38R · **other**
  - `UMAC  ` 10:43 net **-0.58R** · entry 11.54 (+0.6% vs prior close, gap -0.2%, PDR 11%, stop 7.3% away, 5m$ $174,815) · held 312m to the close, MFE +0.23R · **thin**

**2025-04-01** (TRAIN)  book **-4.56 R** · 7 trades (2 green) · candidates 16 · SPY o->c 0.64% / o->10:00 -0.09% · IWM o->c 0.27% / o->10:00 -0.55%
  - `TBH   ` 09:32 net **-1.10R** · entry 6.55 (-0.9% vs prior close, gap -6.1%, PDR 11%, stop 5.2% away, 5m$ $54,628) · stopped 31m later after MFE +0.50R · **thin**
  - `NCT   ` 09:33 net **-2.41R** · entry 5.70 (-0.3% vs prior close, gap -1.9%, PDR 25%, stop 1.6% away, 5m$ $8,300) · stopped 1m later after MFE -2.22R · **thin**
  - `MBX   ` 09:39 net **-1.49R** · entry 7.39 (+0.2% vs prior close, gap -0.9%, PDR 11%, stop 1.1% away, 5m$ $11,996) · stopped 11m later after MFE +0.12R · **thin**
  - `CRVO  ` 09:46 net **+2.03R** · entry 9.08 (-0.8% vs prior close, gap -1.6%, PDR 15%, stop 4.8% away, 5m$ $69,281) · held 369m to the close, MFE +3.91R · **thin**
  - `PHH   ` 10:07 net **-0.68R** · entry 13.21 (-0.1% vs prior close, gap -1.8%, PDR 27%, stop 12.9% away, 5m$ $43,142) · held 349m to the close, MFE +0.66R · **thin**
  - `MNRO  ` 10:38 net **+0.13R** · entry 14.51 (+0.3% vs prior close, gap -0.6%, PDR 11%, stop 4.7% away, 5m$ $120,071) · held 317m to the close, MFE +0.46R · **thin**
  - `KRRO  ` 10:45 net **-1.04R** · entry 17.47 (+0.4% vs prior close, gap -1.7%, PDR 20%, stop 5.9% away, 5m$ $76,260) · stopped 223m later after MFE +0.33R · **thin**

**2025-08-14** (TRAIN)  book **-4.51 R** · 6 trades (1 green) · candidates 27 · SPY o->c 0.34% / o->10:00 0.19% · IWM o->c -0.10% / o->10:00 -0.11%
  - `BBOT  ` 09:33 net **-2.02R** · entry 9.80 (+0.1% vs prior close, gap -1.4%, PDR 19%, stop 1.5% away, 5m$ $59,086) · stopped 12m later after MFE -0.40R · **thin**
  - `BTAI  ` 09:34 net **-1.04R** · entry 5.42 (-3.2% vs prior close, gap -10.5%, PDR 43%, stop 7.6% away, 5m$ $2,071,281) · stopped 271m later after MFE +1.98R · **late**
  - `UPB   ` 09:36 net **+0.02R** · entry 18.55 (-0.1% vs prior close, gap -0.7%, PDR 10%, stop 9.4% away, 5m$ $364,050) · held 379m to the close, MFE +0.23R · **news**
  - `BMNR  ` 09:40 net **-0.30R** · entry 61.84 (+0.3% vs prior close, gap -5.6%, PDR 19%, stop 7.5% away, 5m$ $235,651,285) · held 375m to the close, MFE +0.41R · **other**
  - `AEVA  ` 09:47 net **-1.06R** · entry 13.84 (+0.2% vs prior close, gap -3.8%, PDR 11%, stop 5.7% away, 5m$ $415,079) · stopped 137m later after MFE +0.52R · **other**
  - `CORZW ` 13:06 net **-0.12R** · entry 7.66 (+0.5% vs prior close, gap -3.5%, PDR 16%, stop 7.5% away, 5m$ $22,920) · held 170m to the close, MFE +0.53R · **thin**

**2026-07-13** (TEST)  book **-4.50 R** · 7 trades (1 green) · candidates 14 · SPY o->c -0.45% / o->10:00 0.17% · IWM o->c -0.54% / o->10:00 0.04%
  - `ZSQR  ` 09:33 net **-1.34R** · entry 8.45 (-1.3% vs prior close, gap -3.3%, PDR 12%, stop 2.0% away, 5m$ $25,302) · stopped 2m later after MFE -0.94R · **thin**
  - `WYFL  ` 09:35 net **-1.03R** · entry 27.70 (-2.6% vs prior close, gap -7.2%, PDR 25%, stop 10.1% away, 5m$ $223,029) · stopped 168m later after MFE +0.43R · **news**
  - `RTB   ` 09:40 net **-1.05R** · entry 16.60 (-1.8% vs prior close, gap -3.7%, PDR 19%, stop 6.7% away, 5m$ $95,730) · stopped 3m later after MFE -0.99R · **thin**
  - `BEX   ` 09:47 net **-0.66R** · entry 37.76 (+0.4% vs prior close, gap -11.3%, PDR 21%, stop 13.1% away, 5m$ $2,402,797) · held 368m to the close, MFE +0.07R · **late**
  - `BE    ` 09:48 net **-0.66R** · entry 245.27 (+0.3% vs prior close, gap -5.4%, PDR 9%, stop 6.9% away, 5m$ $82,441,917) · held 367m to the close, MFE +0.04R · **bleed**
  - `JBIO  ` 09:50 net **+0.41R** · entry 20.90 (-0.2% vs prior close, gap -2.3%, PDR 18%, stop 4.9% away, 5m$ $1,144,755) · held 365m to the close, MFE +0.62R · **other**
  - `INBX  ` 13:22 net **-0.17R** · entry 93.47 (+0.5% vs prior close, gap -2.3%, PDR 11%, stop 5.0% away, 5m$ $420,708) · held 153m to the close, MFE +0.19R · **news**

**2026-07-08** (TEST)  book **-4.49 R** · 7 trades (2 green) · candidates 25 · SPY o->c 0.29% / o->10:00 0.15% · IWM o->c -0.19% / o->10:00 0.28%
  - `AMDL  ` 09:33 net **-1.12R** · entry 63.54 (-1.0% vs prior close, gap -4.4%, PDR 10%, stop 3.9% away, 5m$ $4,720,218) · stopped 104m later after MFE +0.91R · **other**
  - `EOSU  ` 09:33 net **-1.06R** · entry 14.50 (+0.1% vs prior close, gap -1.0%, PDR 46%, stop 9.0% away, 5m$ $248,847) · stopped 99m later after MFE +0.48R · **news**
  - `QBTX  ` 09:33 net **-1.05R** · entry 11.57 (+0.7% vs prior close, gap -3.8%, PDR 19%, stop 6.6% away, 5m$ $496,526) · stopped 108m later after MFE +0.43R · **other**
  - `TECX  ` 09:34 net **-0.71R** · entry 36.83 (+0.8% vs prior close, gap -0.6%, PDR 11%, stop 5.9% away, 5m$ $179,981) · held 381m to the close, MFE +0.42R · **thin**
  - `KRUS  ` 11:28 net **+0.06R** · entry 53.00 (+0.2% vs prior close, gap -11.0%, PDR 10%, stop 11.2% away, 5m$ $387,104) · held 267m to the close, MFE +0.23R · **late**
  - `QNTU  ` 12:59 net **-0.79R** · entry 22.15 (+0.3% vs prior close, gap -0.4%, PDR 17%, stop 5.6% away, 5m$ $25,039) · held 178m to the close, MFE +0.00R · **thin**
  - `RIVN  ` 13:13 net **+0.18R** · entry 16.55 (+0.4% vs prior close, gap -5.0%, PDR 10%, stop 6.8% away, 5m$ $12,351,452) · held 162m to the close, MFE +0.21R · **bleed**

**2026-05-19** (VAL)  book **-4.47 R** · 8 trades (2 green) · candidates 64 · SPY o->c -0.13% / o->10:00 -0.01% · IWM o->c -0.29% / o->10:00 -0.64%
  - `MRAM  ` 09:33 net **-1.06R** · entry 33.03 (-1.0% vs prior close, gap -7.0%, PDR 23%, stop 6.2% away, 5m$ $11,870,705) · stopped 17m later after MFE +0.37R · **other**
  - `MVLL  ` 09:33 net **+0.89R** · entry 70.40 (+0.7% vs prior close, gap -6.2%, PDR 23%, stop 7.2% away, 5m$ $1,014,426) · held 382m to the close, MFE +1.94R · **other**
  - `ARMG  ` 09:34 net **-1.05R** · entry 17.51 (+0.3% vs prior close, gap -5.0%, PDR 13%, stop 6.9% away, 5m$ $585,760) · stopped 44m later after MFE +0.75R · **other**
  - `AMUU  ` 09:35 net **-1.09R** · entry 166.00 (-0.0% vs prior close, gap -4.3%, PDR 14%, stop 4.8% away, 5m$ $696,251) · stopped 21m later after MFE +0.69R · **other**
  - `MLEC  ` 09:51 net **-1.04R** · entry 7.98 (-4.1% vs prior close, gap -3.8%, PDR 12%, stop 7.5% away, 5m$ $45,901) · stopped 82m later after MFE -0.08R · **thin**
  - `MI    ` 09:58 net **+0.36R** · entry 19.00 (+0.5% vs prior close, gap -9.4%, PDR 33%, stop 11.9% away, 5m$ $29,242) · held 359m to the close, MFE +0.37R · **late**
  - `LABU  ` 11:16 net **-0.44R** · entry 160.97 (+0.2% vs prior close, gap -1.2%, PDR 12%, stop 6.3% away, 5m$ $2,079,457) · held 279m to the close, MFE +0.09R · **bleed**
  - `SLN   ` 11:16 net **-1.04R** · entry 6.20 (-0.5% vs prior close, gap -1.9%, PDR 10%, stop 6.1% away, 5m$ $112,427) · stopped 232m later after MFE +0.03R · **thin**

**2025-01-17** (TRAIN)  book **-4.33 R** · 5 trades (0 green) · candidates 5 · SPY o->c 0.10% / o->10:00 -0.04% · IWM o->c -0.65% / o->10:00 -0.25%
  - `ATRA  ` 09:33 net **-1.04R** · entry 7.86 (+0.4% vs prior close, gap -4.0%, PDR 37%, stop 7.5% away, 5m$ $383,164) · stopped 44m later after MFE -0.21R · **levelfail**
  - `QBTS  ` 09:34 net **-1.06R** · entry 5.76 (-0.8% vs prior close, gap -1.4%, PDR 15%, stop 4.6% away, 5m$ $20,210,440) · stopped 41m later after MFE +0.97R · **other**
  - `STFS  ` 09:47 net **-0.11R** · entry 14.94 (-3.6% vs prior close, gap -0.7%, PDR 18%, stop 10.0% away, 5m$ $1,100,617) · held 313m to the close, MFE +0.57R · **news**
  - `TRAW  ` 09:50 net **-1.05R** · entry 6.90 (+0.9% vs prior close, gap -0.7%, PDR 11%, stop 5.8% away, 5m$ $31,319) · stopped 109m later after MFE -0.18R · **thin**
  - `IONQ  ` 10:59 net **-1.06R** · entry 41.79 (+0.6% vs prior close, gap -3.7%, PDR 16%, stop 5.8% away, 5m$ $50,738,701) · stopped 91m later after MFE +0.23R · **bleed**

**2026-01-30** (VAL)  book **-4.29 R** · 8 trades (2 green) · candidates 14 · SPY o->c 0.01% / o->10:00 0.02% · IWM o->c -0.55% / o->10:00 -0.27%
  - `MTA   ` 09:33 net **-1.07R** · entry 8.13 (-0.4% vs prior close, gap -2.1%, PDR 10%, stop 4.6% away, 5m$ $1,029,623) · stopped 142m later after MFE +0.46R · **dump**
  - `RVYL  ` 09:34 net **-1.55R** · entry 6.36 (-0.5% vs prior close, gap -2.5%, PDR 14%, stop 2.0% away, 5m$ $4,589) · stopped 15m later after MFE -0.04R · **thin**
  - `EOSU  ` 09:36 net **-1.14R** · entry 21.94 (+0.1% vs prior close, gap -1.9%, PDR 25%, stop 5.4% away, 5m$ $58,117) · stopped 22m later after MFE -0.37R · **thin**
  - `SMU   ` 09:37 net **-1.06R** · entry 16.66 (+0.1% vs prior close, gap -3.8%, PDR 24%, stop 5.6% away, 5m$ $1,072,954) · stopped 27m later after MFE +0.42R · **dump**
  - `LABX  ` 10:45 net **-1.04R** · entry 13.68 (+0.6% vs prior close, gap -2.8%, PDR 18%, stop 5.8% away, 5m$ $252,799) · stopped 137m later after MFE -0.01R · **news**
  - `AHMA  ` 10:55 net **-0.42R** · entry 31.00 (+0.0% vs prior close, gap -5.4%, PDR 12%, stop 5.4% away, 5m$ $131,482) · held 304m to the close, MFE +0.65R · **thin**
  - `MSTU  ` 11:10 net **+1.93R** · entry 6.92 (-0.4% vs prior close, gap -4.6%, PDR 28%, stop 4.6% away, 5m$ $2,651,467) · held 285m to the close, MFE +2.42R · **other**
  - `FBRX  ` 12:09 net **+0.07R** · entry 29.00 (-0.7% vs prior close, gap -0.9%, PDR 12%, stop 8.1% away, 5m$ $114,193) · held 226m to the close, MFE +0.08R · **thin**

**2026-02-05** (VAL)  book **-4.25 R** · 5 trades (1 green) · candidates 43 · SPY o->c -0.51% / o->10:00 -0.19% · IWM o->c -1.03% / o->10:00 0.72%
  - `SYM   ` 09:32 net **+0.17R** · entry 53.46 (-0.0% vs prior close, gap -3.2%, PDR 9%, stop 4.8% away, 5m$ $13,367,486) · held 383m to the close, MFE +1.51R · **dump**
  - `OPEX  ` 09:33 net **-1.47R** · entry 9.27 (-2.4% vs prior close, gap -9.6%, PDR 19%, stop 7.3% away, 5m$ $189,049) · stopped 377m later after MFE +0.48R · **thin**
  - `RGC   ` 09:33 net **-1.68R** · entry 27.80 (-2.0% vs prior close, gap -2.6%, PDR 13%, stop 1.9% away, 5m$ $73,515) · stopped 1m later after MFE -1.48R · **thin**
  - `SNXX  ` 09:36 net **-0.23R** · entry 34.70 (-3.7% vs prior close, gap -7.9%, PDR 34%, stop 4.5% away, 5m$ $13,534,952) · held 379m to the close, MFE +4.03R · **news**
  - `TE    ` 09:38 net **-1.05R** · entry 7.77 (+0.5% vs prior close, gap -4.3%, PDR 29%, stop 6.0% away, 5m$ $5,054,270) · stopped 343m later after MFE +0.56R · **dump**

**2026-02-10** (VAL)  book **-4.24 R** · 7 trades (2 green) · candidates 19 · SPY o->c -0.41% / o->10:00 0.09% · IWM o->c -0.40% / o->10:00 -0.18%
  - `SNBR  ` 09:32 net **-1.12R** · entry 11.07 (-0.4% vs prior close, gap -1.1%, PDR 10%, stop 2.9% away, 5m$ $86,147) · stopped 19m later after MFE +0.61R · **thin**
  - `RIOX  ` 09:33 net **-0.24R** · entry 8.35 (-0.2% vs prior close, gap -5.0%, PDR 20%, stop 5.0% away, 5m$ $119,869) · held 382m to the close, MFE +0.87R · **thin**
  - `ONDL  ` 09:36 net **-1.05R** · entry 23.92 (+0.7% vs prior close, gap -4.6%, PDR 19%, stop 6.7% away, 5m$ $923,025) · stopped 9m later after MFE -0.34R · **levelfail**
  - `SMU   ` 09:37 net **-1.08R** · entry 13.29 (-2.1% vs prior close, gap -2.8%, PDR 14%, stop 4.2% away, 5m$ $1,360,641) · stopped 10m later after MFE -0.02R · **other**
  - `AGMB  ` 10:09 net **+0.15R** · entry 15.45 (-1.1% vs prior close, gap -3.9%, PDR 10%, stop 7.3% away, 5m$ $97,872) · held 291m to the close, MFE +0.54R · **thin**
  - `MAZE  ` 10:30 net **+0.13R** · entry 47.90 (+0.4% vs prior close, gap -0.2%, PDR 9%, stop 5.6% away, 5m$ $515,103) · held 325m to the close, MFE +0.23R · **news**
  - `MSTU  ` 10:34 net **-1.04R** · entry 5.80 (+0.1% vs prior close, gap -6.5%, PDR 25%, stop 7.1% away, 5m$ $5,971,453) · stopped 277m later after MFE +0.16R · **bleed**

### 1b. The 25 best days

**2025-08-12** (TRAIN)  book **+19.03 R** · 4 trades (3 green) · candidates 14 · SPY o->c 0.68% / o->10:00 -0.22% · IWM o->c 2.22% / o->10:00 0.08%
  - `ARCX  ` 09:32 net **+1.80R** · entry 15.25 (-0.1% vs prior close, gap -7.0%, PDR 9%, stop 7.8% away, 5m$ $328,984) · held 385m to the close, MFE +2.08R · **news**
  - `OKLL  ` 09:32 net **+15.64R** · entry 28.41 (-4.4% vs prior close, gap -5.4%, PDR 11%, stop 1.4% away, 5m$ $1,523,209) · held 383m to the close, MFE +16.10R · **news**
  - `SOUX  ` 09:32 net **-1.20R** · entry 47.77 (-0.2% vs prior close, gap -1.2%, PDR 33%, stop 2.0% away, 5m$ $1,064,694) · stopped 153m later after MFE +3.60R · **news**
  - `SDM   ` 09:41 net **+2.79R** · entry 10.25 (-0.8% vs prior close, gap -8.9%, PDR 22%, stop 8.2% away, 5m$ $213,637) · held 377m to the close, MFE +4.12R · **late**

**2026-03-03** (VAL)  book **+18.78 R** · 5 trades (4 green) · candidates 71 · SPY o->c 0.77% / o->10:00 -0.32% · IWM o->c 0.95% / o->10:00 -0.82%
  - `ASTX  ` 09:32 net **+18.42R** · entry 42.35 (-5.3% vs prior close, gap -1.0%, PDR 23%, stop 1.2% away, 5m$ $2,626,284) · held 383m to the close, MFE +27.04R · **news**
  - `PLTU  ` 09:33 net **-1.07R** · entry 45.70 (+0.7% vs prior close, gap -4.5%, PDR 9%, stop 5.4% away, 5m$ $5,624,905) · stopped 16m later after MFE +0.01R · **dump**
  - `POET  ` 09:38 net **+0.14R** · entry 6.93 (+0.3% vs prior close, gap -4.9%, PDR 17%, stop 5.5% away, 5m$ $3,863,188) · held 377m to the close, MFE +1.00R · **dump**
  - `VELO  ` 09:40 net **+0.00R** · entry 11.95 (+0.8% vs prior close, gap -7.2%, PDR 25%, stop 9.2% away, 5m$ $569,467) · held 320m to the close, MFE +0.23R · **late**
  - `NGNE  ` 10:29 net **+1.29R** · entry 20.56 (-0.9% vs prior close, gap -4.2%, PDR 19%, stop 5.5% away, 5m$ $54,898) · held 326m to the close, MFE +1.96R · **thin**

**2026-04-06** (VAL)  book **+18.27 R** · 5 trades (3 green) · candidates 6 · SPY o->c 0.46% / o->10:00 0.39% · IWM o->c 0.55% / o->10:00 0.40%
  - `SMX   ` 09:33 net **+11.22R** · entry 8.49 (+0.4% vs prior close, gap -4.5%, PDR 13%, stop 4.8% away, 5m$ $110,276) · held 382m to the close, MFE +27.29R · **thin**
  - `LUNL  ` 09:34 net **-1.06R** · entry 18.52 (-1.4% vs prior close, gap -1.3%, PDR 56%, stop 5.7% away, 5m$ $648,767) · stopped 32m later after MFE +0.42R · **other**
  - `SVRN  ` 09:36 net **+8.10R** · entry 7.93 (+0.3% vs prior close, gap -0.6%, PDR 23%, stop 3.3% away, 5m$ $29,139) · held 379m to the close, MFE +8.93R · **thin**
  - `PSNY  ` 09:59 net **-0.14R** · entry 20.00 (+0.4% vs prior close, gap -2.2%, PDR 11%, stop 7.6% away, 5m$ $35,557) · held 357m to the close, MFE +0.06R · **thin**
  - `BEZ   ` 10:19 net **+0.15R** · entry 17.88 (-0.4% vs prior close, gap -0.2%, PDR 21%, stop 6.4% away, 5m$ $110,755) · held 337m to the close, MFE +0.48R · **thin**

**2026-04-16** (VAL)  book **+15.19 R** · 6 trades (3 green) · candidates 14 · SPY o->c 0.07% / o->10:00 -0.28% · IWM o->c 0.19% / o->10:00 -0.23%
  - `AAOX  ` 09:36 net **+2.53R** · entry 47.16 (+0.1% vs prior close, gap -4.4%, PDR 18%, stop 8.2% away, 5m$ $1,696,086) · held 379m to the close, MFE +2.61R · **news**
  - `ARTV  ` 09:38 net **+13.91R** · entry 9.15 (+0.6% vs prior close, gap -1.3%, PDR 19%, stop 2.4% away, 5m$ $152,747) · held 377m to the close, MFE +14.73R · **thin**
  - `CORD  ` 09:38 net **-0.18R** · entry 5.82 (-0.2% vs prior close, gap -2.4%, PDR 13%, stop 5.5% away, 5m$ $1,135,602) · held 377m to the close, MFE +1.22R · **dump**
  - `NBIL  ` 09:44 net **-1.09R** · entry 25.97 (+0.3% vs prior close, gap -0.2%, PDR 14%, stop 4.9% away, 5m$ $1,180,039) · stopped 9m later after MFE +0.21R · **other**
  - `ASTN  ` 09:54 net **-1.04R** · entry 12.54 (+0.3% vs prior close, gap -1.6%, PDR 10%, stop 7.7% away, 5m$ $374,064) · stopped 102m later after MFE +0.04R · **bleed**
  - `MSTU  ` 11:38 net **+1.07R** · entry 5.50 (+0.4% vs prior close, gap -0.7%, PDR 10%, stop 6.4% away, 5m$ $2,820,920) · held 257m to the close, MFE +1.11R · **other**

**2026-02-09** (VAL)  book **+12.50 R** · 4 trades (4 green) · candidates 53 · SPY o->c 0.65% / o->10:00 0.15% · IWM o->c 0.76% / o->10:00 -0.31%
  - `IRE   ` 09:34 net **+6.38R** · entry 5.79 (-1.9% vs prior close, gap -4.6%, PDR 46%, stop 3.8% away, 5m$ $3,934,373) · held 381m to the close, MFE +7.45R · **other**
  - `NEBX  ` 09:35 net **+2.92R** · entry 26.66 (-0.4% vs prior close, gap -3.8%, PDR 23%, stop 5.4% away, 5m$ $993,324) · held 380m to the close, MFE +3.28R · **other**
  - `OKLL  ` 09:35 net **+2.20R** · entry 13.82 (-1.4% vs prior close, gap -4.7%, PDR 24%, stop 5.8% away, 5m$ $2,968,634) · held 380m to the close, MFE +2.39R · **other**
  - `RDWU  ` 09:35 net **+1.00R** · entry 14.00 (-2.0% vs prior close, gap -2.0%, PDR 29%, stop 3.4% away, 5m$ $56,949) · held 381m to the close, MFE +2.32R · **thin**

**2026-05-28** (VAL)  book **+12.24 R** · 5 trades (4 green) · candidates 43 · SPY o->c 0.59% / o->10:00 0.05% · IWM o->c 0.82% / o->10:00 -0.30%
  - `LIXT  ` 09:32 net **-1.30R** · entry 6.20 (-3.4% vs prior close, gap -6.2%, PDR 18%, stop 2.9% away, 5m$ $29,322) · stopped 22m later after MFE -0.28R · **thin**
  - `RDWU  ` 09:32 net **+1.40R** · entry 48.95 (+0.6% vs prior close, gap -9.4%, PDR 25%, stop 10.1% away, 5m$ $2,868,119) · held 383m to the close, MFE +2.09R · **late**
  - `QPUX  ` 09:35 net **+6.26R** · entry 45.00 (+0.9% vs prior close, gap -1.6%, PDR 19%, stop 2.4% away, 5m$ $726,509) · held 380m to the close, MFE +7.34R · **news**
  - `HUTG  ` 09:36 net **+2.96R** · entry 34.14 (-1.0% vs prior close, gap -0.6%, PDR 18%, stop 4.2% away, 5m$ $25,418) · held 379m to the close, MFE +3.53R · **thin**
  - `HODU  ` 09:56 net **+2.92R** · entry 7.77 (+0.6% vs prior close, gap -3.0%, PDR 8%, stop 7.3% away, 5m$ $115,097) · held 362m to the close, MFE +2.93R · **thin**

**2026-05-22** (VAL)  book **+11.54 R** · 5 trades (3 green) · candidates 12 · SPY o->c -0.09% / o->10:00 0.04% · IWM o->c 0.36% / o->10:00 0.40%
  - `FBRX  ` 09:34 net **-0.45R** · entry 22.17 (+0.8% vs prior close, gap -1.0%, PDR 14%, stop 6.7% away, 5m$ $153,695) · held 381m to the close, MFE +0.12R · **thin**
  - `PIII  ` 09:37 net **-1.05R** · entry 13.87 (+0.1% vs prior close, gap -2.3%, PDR 54%, stop 6.2% away, 5m$ $283,658) · stopped 78m later after MFE +0.63R · **other**
  - `QNTM  ` 09:41 net **+11.88R** · entry 7.37 (-5.7% vs prior close, gap -0.3%, PDR 11%, stop 2.1% away, 5m$ $155,798) · held 374m to the close, MFE +12.31R · **thin**
  - `ARMG  ` 09:48 net **+0.45R** · entry 32.45 (+0.5% vs prior close, gap -6.0%, PDR 23%, stop 7.4% away, 5m$ $1,343,092) · held 367m to the close, MFE +1.42R · **other**
  - `CIFU  ` 11:33 net **+0.71R** · entry 28.82 (+0.6% vs prior close, gap -1.0%, PDR 18%, stop 5.3% away, 5m$ $77,298) · held 264m to the close, MFE +1.28R · **thin**

**2026-03-30** (VAL)  book **+11.13 R** · 4 trades (4 green) · candidates 13 · SPY o->c -1.27% / o->10:00 -0.72% · IWM o->c -2.34% / o->10:00 -1.29%
  - `ASTN  ` 09:32 net **+8.76R** · entry 17.64 (-1.5% vs prior close, gap -3.4%, PDR 28%, stop 2.0% away, 5m$ $268,104) · held 360m to the close, MFE +9.80R · **news**
  - `UPB   ` 09:35 net **+0.05R** · entry 8.72 (+0.6% vs prior close, gap -1.6%, PDR 13%, stop 5.0% away, 5m$ $181,364) · held 380m to the close, MFE +0.10R · **thin**
  - `OKLS  ` 09:54 net **+2.24R** · entry 66.54 (+0.5% vs prior close, gap -2.7%, PDR 11%, stop 7.3% away, 5m$ $401,577) · held 362m to the close, MFE +2.68R · **news**
  - `CONI  ` 10:07 net **+0.09R** · entry 74.24 (-0.2% vs prior close, gap -3.6%, PDR 10%, stop 6.0% away, 5m$ $279,941) · held 349m to the close, MFE +0.56R · **news**

**2025-11-20** (TRAIN)  book **+11.05 R** · 4 trades (4 green) · candidates 10 · SPY o->c -3.03% / o->10:00 0.32% · IWM o->c -3.42% / o->10:00 0.52%
  - `CCHH  ` 09:35 net **+6.14R** · entry 5.77 (-5.1% vs prior close, gap -2.5%, PDR 31%, stop 6.4% away, 5m$ $577,790) · held 380m to the close, MFE +6.65R · **news**
  - `MSTZ  ` 10:06 net **+1.28R** · entry 12.63 (+0.3% vs prior close, gap -5.6%, PDR 20%, stop 7.5% away, 5m$ $4,803,025) · held 349m to the close, MFE +2.13R · **dump**
  - `SMST  ` 10:06 net **+1.22R** · entry 69.62 (+0.4% vs prior close, gap -6.7%, PDR 20%, stop 7.7% away, 5m$ $688,827) · held 349m to the close, MFE +2.00R · **dump**
  - `QBTZ  ` 10:49 net **+2.40R** · entry 21.80 (-0.0% vs prior close, gap -4.1%, PDR 15%, stop 10.6% away, 5m$ $758,111) · held 306m to the close, MFE +2.50R · **dump**

**2025-07-16** (TRAIN)  book **+9.67 R** · 2 trades (1 green) · candidates 2 · SPY o->c 0.07% / o->10:00 -0.24% · IWM o->c 0.36% / o->10:00 -0.33%
  - `SDM   ` 09:53 net **+9.89R** · entry 13.55 (-1.0% vs prior close, gap -5.3%, PDR 32%, stop 4.4% away, 5m$ $183,715) · held 362m to the close, MFE +10.93R · **thin**
  - `AREN  ` 12:34 net **-0.22R** · entry 5.38 (+0.3% vs prior close, gap -0.4%, PDR 9%, stop 5.1% away, 5m$ $18,845) · held 201m to the close, MFE +0.18R · **thin**

**2025-09-26** (TRAIN)  book **+9.64 R** · 4 trades (4 green) · candidates 5 · SPY o->c 0.34% / o->10:00 0.39% · IWM o->c 0.73% / o->10:00 0.76%
  - `PLTS  ` 09:33 net **+7.40R** · entry 11.75 (-0.8% vs prior close, gap -7.1%, PDR 30%, stop 6.3% away, 5m$ $69,476) · held 382m to the close, MFE +10.47R · **thin**
  - `QURE  ` 09:39 net **+0.29R** · entry 52.81 (+0.3% vs prior close, gap -3.1%, PDR 18%, stop 6.8% away, 5m$ $29,888,557) · held 376m to the close, MFE +0.64R · **other**
  - `CLPT  ` 09:48 net **+1.74R** · entry 19.81 (+0.4% vs prior close, gap -2.8%, PDR 12%, stop 4.9% away, 5m$ $599,079) · held 367m to the close, MFE +2.00R · **other**
  - `SHFS  ` 10:02 net **+0.20R** · entry 6.97 (+0.3% vs prior close, gap -9.5%, PDR 22%, stop 12.3% away, 5m$ $132,681) · held 353m to the close, MFE +0.34R · **late**

**2026-04-13** (VAL)  book **+9.36 R** · 5 trades (4 green) · candidates 24 · SPY o->c 1.27% / o->10:00 0.37% · IWM o->c 1.84% / o->10:00 0.84%
  - `AXTI  ` 09:36 net **-1.07R** · entry 64.45 (+0.4% vs prior close, gap -0.6%, PDR 13%, stop 6.4% away, 5m$ $36,343,878) · stopped 12m later after MFE +0.10R · **other**
  - `ALOY  ` 09:41 net **+1.54R** · entry 11.77 (-0.2% vs prior close, gap -0.8%, PDR 28%, stop 6.0% away, 5m$ $568,889) · held 374m to the close, MFE +1.73R · **other**
  - `IRE   ` 09:43 net **+3.40R** · entry 17.05 (+0.6% vs prior close, gap -4.4%, PDR 16%, stop 5.5% away, 5m$ $3,877,202) · held 372m to the close, MFE +3.75R · **other**
  - `IREX  ` 09:43 net **+4.04R** · entry 22.61 (-0.4% vs prior close, gap -5.2%, PDR 16%, stop 4.9% away, 5m$ $393,216) · held 372m to the close, MFE +4.39R · **news**
  - `ASTX  ` 09:51 net **+1.45R** · entry 46.98 (+0.4% vs prior close, gap -3.3%, PDR 14%, stop 5.5% away, 5m$ $1,666,452) · held 364m to the close, MFE +1.97R · **other**

**2026-05-21** (VAL)  book **+9.21 R** · 5 trades (4 green) · candidates 31 · SPY o->c 0.55% / o->10:00 0.16% · IWM o->c 1.38% / o->10:00 0.56%
  - `POEL  ` 09:32 net **-1.08R** · entry 55.81 (-4.9% vs prior close, gap -9.9%, PDR 18%, stop 5.9% away, 5m$ $1,613,080) · stopped 17m later after MFE +0.39R · **news**
  - `SNXX  ` 09:32 net **+6.72R** · entry 153.00 (+0.6% vs prior close, gap -1.7%, PDR 11%, stop 2.9% away, 5m$ $19,858,001) · held 383m to the close, MFE +7.33R · **other**
  - `NVTX  ` 09:37 net **+1.73R** · entry 102.61 (+0.1% vs prior close, gap -3.8%, PDR 33%, stop 6.7% away, 5m$ $1,073,904) · held 378m to the close, MFE +2.84R · **news**
  - `VELO  ` 09:42 net **+1.50R** · entry 18.63 (+0.6% vs prior close, gap -3.1%, PDR 19%, stop 5.7% away, 5m$ $952,233) · held 373m to the close, MFE +2.23R · **other**
  - `BLLN  ` 10:03 net **+0.32R** · entry 83.89 (+0.6% vs prior close, gap -2.6%, PDR 9%, stop 6.8% away, 5m$ $2,290,988) · held 352m to the close, MFE +0.54R · **news**

**2026-01-09** (VAL)  book **+8.77 R** · 6 trades (1 green) · candidates 11 · SPY o->c 0.48% / o->10:00 0.07% · IWM o->c 0.25% / o->10:00 0.34%
  - `THH   ` 09:47 net **+12.58R** · entry 20.68 (-4.4% vs prior close, gap -3.6%, PDR 15%, stop 3.3% away, 5m$ $519,253) · held 368m to the close, MFE +14.04R · **news**
  - `CORD  ` 09:54 net **-1.06R** · entry 32.90 (-0.4% vs prior close, gap -3.2%, PDR 11%, stop 5.8% away, 5m$ $284,761) · stopped 43m later after MFE +0.69R · **wick**
  - `XRPT  ` 10:28 net **-0.84R** · entry 6.52 (+0.6% vs prior close, gap -2.9%, PDR 11%, stop 5.4% away, 5m$ $1,078,147) · held 327m to the close, MFE +0.45R · **other**
  - `XXRP  ` 10:28 net **-1.05R** · entry 12.56 (+0.7% vs prior close, gap -3.0%, PDR 11%, stop 5.3% away, 5m$ $304,968) · stopped 179m later after MFE +0.45R · **other**
  - `ELAB  ` 10:40 net **-0.38R** · entry 5.53 (-0.7% vs prior close, gap -4.3%, PDR 22%, stop 7.6% away, 5m$ $145,762) · held 260m to the close, MFE +0.29R · **thin**
  - `KXIN  ` 13:31 net **-0.49R** · entry 7.76 (+0.6% vs prior close, gap -0.5%, PDR 10%, stop 8.4% away, 5m$ $21,002) · held 146m to the close, MFE -0.22R · **thin**

**2025-02-12** (TRAIN)  book **+8.40 R** · 5 trades (4 green) · candidates 22 · SPY o->c 0.69% / o->10:00 0.26% · IWM o->c 0.63% / o->10:00 0.17%
  - `CSAI  ` 09:35 net **+9.26R** · entry 12.67 (-1.8% vs prior close, gap -6.0%, PDR 80%, stop 4.2% away, 5m$ $101,557) · held 380m to the close, MFE +13.21R · **thin**
  - `SLQT  ` 09:37 net **-1.04R** · entry 5.74 (+0.3% vs prior close, gap -5.6%, PDR 31%, stop 7.7% away, 5m$ $581,680) · stopped 109m later after MFE +0.37R · **other**
  - `JSPR  ` 09:45 net **+0.07R** · entry 5.39 (+0.1% vs prior close, gap -0.9%, PDR 13%, stop 15.5% away, 5m$ $274,195) · held 370m to the close, MFE +0.21R · **news**
  - `TEO   ` 09:47 net **+0.02R** · entry 10.85 (+0.6% vs prior close, gap -2.5%, PDR 9%, stop 6.1% away, 5m$ $33,553) · held 369m to the close, MFE +0.65R · **thin**
  - `DUOT  ` 11:34 net **+0.09R** · entry 7.05 (+0.4% vs prior close, gap -0.3%, PDR 9%, stop 9.8% away, 5m$ $71,514) · held 192m to the close, MFE +0.30R · **thin**

**2025-10-20** (TRAIN)  book **+7.91 R** · 4 trades (4 green) · candidates 6 · SPY o->c 0.59% / o->10:00 0.25% · IWM o->c 0.60% / o->10:00 0.14%
  - `QBTZ  ` 09:35 net **+2.77R** · entry 12.81 (+0.3% vs prior close, gap -6.3%, PDR 16%, stop 7.6% away, 5m$ $1,057,228) · held 380m to the close, MFE +3.18R · **other**
  - `FLUX  ` 09:56 net **+2.01R** · entry 5.34 (-0.6% vs prior close, gap -1.5%, PDR 10%, stop 5.2% away, 5m$ $130,460) · held 359m to the close, MFE +2.32R · **thin**
  - `SLMT  ` 10:15 net **+2.09R** · entry 11.51 (-0.9% vs prior close, gap -2.8%, PDR 20%, stop 4.2% away, 5m$ $59,934) · held 340m to the close, MFE +3.77R · **thin**
  - `GWAV  ` 11:49 net **+1.04R** · entry 8.46 (-2.3% vs prior close, gap -4.4%, PDR 20%, stop 5.3% away, 5m$ $12,324) · held 246m to the close, MFE +1.64R · **thin**

**2026-04-02** (VAL)  book **+7.83 R** · 4 trades (4 green) · candidates 88 · SPY o->c 1.46% / o->10:00 0.50% · IWM o->c 2.58% / o->10:00 0.71%
  - `UMAC  ` 09:33 net **+2.81R** · entry 12.22 (-1.0% vs prior close, gap -3.2%, PDR 11%, stop 3.8% away, 5m$ $1,765,309) · held 382m to the close, MFE +3.96R · **other**
  - `DLLL  ` 09:40 net **+2.15R** · entry 36.16 (-3.2% vs prior close, gap -4.4%, PDR 8%, stop 4.0% away, 5m$ $301,743) · held 375m to the close, MFE +2.19R · **news**
  - `KRRO  ` 09:42 net **+1.44R** · entry 12.76 (+0.8% vs prior close, gap -2.4%, PDR 11%, stop 5.6% away, 5m$ $152,902) · held 373m to the close, MFE +2.25R · **thin**
  - `USAR  ` 09:42 net **+1.43R** · entry 14.84 (+0.3% vs prior close, gap -4.2%, PDR 9%, stop 5.2% away, 5m$ $5,642,302) · held 373m to the close, MFE +1.94R · **other**

**2025-05-22** (TRAIN)  book **+7.39 R** · 4 trades (4 green) · candidates 8 · SPY o->c 0.06% / o->10:00 0.30% · IWM o->c 0.46% / o->10:00 0.10%
  - `DFDV  ` 09:32 net **+4.15R** · entry 35.00 (-1.5% vs prior close, gap -4.5%, PDR 63%, stop 5.0% away, 5m$ $2,400,757) · held 383m to the close, MFE +10.79R · **news**
  - `BCAX  ` 09:36 net **+1.47R** · entry 14.40 (-1.3% vs prior close, gap -0.7%, PDR 10%, stop 4.9% away, 5m$ $265,754) · held 379m to the close, MFE +1.63R · **news**
  - `UPB   ` 10:00 net **+0.83R** · entry 8.93 (+0.0% vs prior close, gap -1.0%, PDR 9%, stop 5.1% away, 5m$ $79,331) · held 355m to the close, MFE +1.40R · **thin**
  - `SMR   ` 10:10 net **+0.94R** · entry 23.95 (+0.3% vs prior close, gap -2.4%, PDR 10%, stop 6.0% away, 5m$ $4,074,276) · held 345m to the close, MFE +1.11R · **other**

**2025-03-26** (TRAIN)  book **+7.17 R** · 4 trades (2 green) · candidates 5 · SPY o->c -1.12% / o->10:00 -0.24% · IWM o->c -1.15% / o->10:00 -0.13%
  - `LZMH  ` 09:46 net **+8.15R** · entry 7.36 (-3.9% vs prior close, gap -1.0%, PDR 26%, stop 3.8% away, 5m$ $118,178) · held 369m to the close, MFE +9.04R · **thin**
  - `CRVO  ` 09:47 net **-1.05R** · entry 8.82 (-0.4% vs prior close, gap -0.6%, PDR 22%, stop 5.6% away, 5m$ $188,949) · stopped 29m later after MFE +0.11R · **thin**
  - `ZYBT  ` 09:55 net **+1.11R** · entry 6.80 (-17.1% vs prior close, gap -18.0%, PDR 77%, stop 4.4% away, 5m$ $325,763) · held 360m to the close, MFE +10.67R · **news**
  - `ADTX  ` 10:02 net **-1.04R** · entry 7.00 (+0.5% vs prior close, gap -4.0%, PDR 12%, stop 6.4% away, 5m$ $62,014) · stopped 212m later after MFE +1.56R · **thin**

**2025-03-06** (TRAIN)  book **+6.87 R** · 4 trades (1 green) · candidates 33 · SPY o->c -0.47% / o->10:00 -0.09% · IWM o->c -0.25% / o->10:00 -0.16%
  - `LZMH  ` 09:33 net **+8.13R** · entry 7.37 (+0.5% vs prior close, gap -3.7%, PDR 14%, stop 4.2% away, 5m$ $120,894) · held 382m to the close, MFE +8.48R · **thin**
  - `MSTU  ` 09:52 net **-0.28R** · entry 7.02 (+0.3% vs prior close, gap -5.9%, PDR 29%, stop 10.1% away, 5m$ $19,102,706) · held 363m to the close, MFE +0.85R · **other**
  - `SITM  ` 09:56 net **-0.47R** · entry 183.51 (+0.7% vs prior close, gap -3.8%, PDR 17%, stop 6.2% away, 5m$ $1,100,455) · held 359m to the close, MFE +1.89R · **news**
  - `BYRN  ` 10:01 net **-0.51R** · entry 23.50 (+0.4% vs prior close, gap -3.6%, PDR 18%, stop 5.2% away, 5m$ $271,170) · held 354m to the close, MFE +0.52R · **other**

**2025-02-27** (TRAIN)  book **+6.84 R** · 5 trades (4 green) · candidates 7 · SPY o->c -1.98% / o->10:00 -0.95% · IWM o->c -1.50% / o->10:00 -0.93%
  - `SEZL  ` 09:36 net **-1.06R** · entry 308.33 (+0.7% vs prior close, gap -1.3%, PDR 20%, stop 8.1% away, 5m$ $6,279,519) · stopped 163m later after MFE +0.44R · **news**
  - `NVD   ` 09:38 net **+2.94R** · entry 26.09 (-0.1% vs prior close, gap -5.4%, PDR 9%, stop 5.3% away, 5m$ $8,672,271) · held 377m to the close, MFE +3.15R · **other**
  - `MSTZ  ` 09:48 net **+2.35R** · entry 25.14 (+0.8% vs prior close, gap -2.6%, PDR 18%, stop 6.5% away, 5m$ $10,553,657) · held 367m to the close, MFE +2.63R · **other**
  - `SMST  ` 09:49 net **+2.12R** · entry 7.39 (+0.8% vs prior close, gap -3.0%, PDR 18%, stop 7.0% away, 5m$ $3,624,441) · held 366m to the close, MFE +2.37R · **other**
  - `STAA  ` 12:52 net **+0.49R** · entry 17.13 (+0.2% vs prior close, gap -2.8%, PDR 8%, stop 6.6% away, 5m$ $861,495) · held 183m to the close, MFE +0.63R · **other**

**2026-05-11** (VAL)  book **+6.74 R** · 4 trades (4 green) · candidates 39 · SPY o->c 0.38% / o->10:00 0.28% · IWM o->c 0.16% / o->10:00 0.42%
  - `QUBX  ` 09:42 net **+1.79R** · entry 13.40 (-0.1% vs prior close, gap -2.6%, PDR 16%, stop 6.2% away, 5m$ $104,747) · held 373m to the close, MFE +2.33R · **thin**
  - `GLXU  ` 09:44 net **+1.25R** · entry 12.73 (+0.3% vs prior close, gap -2.1%, PDR 15%, stop 6.6% away, 5m$ $334,369) · held 372m to the close, MFE +1.92R · **news**
  - `QBTX  ` 09:44 net **+1.69R** · entry 16.87 (+0.5% vs prior close, gap -3.3%, PDR 12%, stop 7.0% away, 5m$ $282,582) · held 371m to the close, MFE +2.79R · **other**
  - `COIG  ` 09:46 net **+2.01R** · entry 8.68 (+0.2% vs prior close, gap -0.3%, PDR 20%, stop 6.7% away, 5m$ $95,745) · held 368m to the close, MFE +2.39R · **thin**

**2025-08-22** (TRAIN)  book **+6.46 R** · 4 trades (4 green) · candidates 7 · SPY o->c 1.18% / o->10:00 0.78% · IWM o->c 3.26% / o->10:00 1.28%
  - `TEMT  ` 09:34 net **+4.50R** · entry 29.34 (-0.4% vs prior close, gap -0.5%, PDR 19%, stop 2.3% away, 5m$ $1,100,323) · held 381m to the close, MFE +9.81R · **other**
  - `MB    ` 09:46 net **+1.14R** · entry 5.60 (+0.5% vs prior close, gap -7.4%, PDR 22%, stop 7.9% away, 5m$ $31,830) · held 373m to the close, MFE +1.82R · **late**
  - `HIMZ  ` 10:04 net **+0.19R** · entry 14.86 (+0.9% vs prior close, gap -4.8%, PDR 9%, stop 7.1% away, 5m$ $4,531,174) · held 351m to the close, MFE +0.95R · **other**
  - `TZUP  ` 10:04 net **+0.62R** · entry 5.83 (-1.4% vs prior close, gap -2.7%, PDR 24%, stop 5.7% away, 5m$ $341,071) · held 351m to the close, MFE +0.70R · **news**

**2025-11-14** (TRAIN)  book **+6.38 R** · 4 trades (4 green) · candidates 194 · SPY o->c 0.98% / o->10:00 0.42% · IWM o->c 1.78% / o->10:00 1.14%
  - `FDMT  ` 09:32 net **+2.85R** · entry 10.23 (-0.5% vs prior close, gap -1.7%, PDR 9%, stop 2.2% away, 5m$ $274,960) · held 383m to the close, MFE +4.07R · **other**
  - `PWP   ` 09:34 net **+0.61R** · entry 18.17 (-2.8% vs prior close, gap -1.0%, PDR 10%, stop 4.5% away, 5m$ $321,183) · held 381m to the close, MFE +0.83R · **news**
  - `NB    ` 09:36 net **+1.95R** · entry 5.45 (+0.4% vs prior close, gap -3.7%, PDR 16%, stop 4.4% away, 5m$ $1,076,306) · held 379m to the close, MFE +3.08R · **other**
  - `ANRO  ` 09:37 net **+0.97R** · entry 11.42 (+0.7% vs prior close, gap -3.3%, PDR 27%, stop 5.5% away, 5m$ $68,978) · held 378m to the close, MFE +2.88R · **thin**

**2025-12-04** (TRAIN)  book **+6.35 R** · 6 trades (4 green) · candidates 15 · SPY o->c -0.13% / o->10:00 -0.31% · IWM o->c 1.14% / o->10:00 -0.06%
  - `CAPR  ` 09:32 net **-1.06R** · entry 30.16 (+0.7% vs prior close, gap -5.7%, PDR 88%, stop 6.5% away, 5m$ $7,493,254) · stopped 1m later after MFE +0.12R · **news**
  - `MPLT  ` 09:36 net **+1.38R** · entry 16.91 (-0.1% vs prior close, gap -2.7%, PDR 22%, stop 5.4% away, 5m$ $276,464) · held 379m to the close, MFE +2.25R · **news**
  - `QSU   ` 09:38 net **+1.50R** · entry 9.91 (+0.5% vs prior close, gap -4.3%, PDR 12%, stop 5.2% away, 5m$ $313,461) · held 377m to the close, MFE +1.81R · **news**
  - `QUBX  ` 09:39 net **+3.92R** · entry 33.36 (+0.7% vs prior close, gap -4.4%, PDR 29%, stop 6.0% away, 5m$ $292,646) · held 376m to the close, MFE +4.04R · **other**
  - `MSTU  ` 09:45 net **-1.08R** · entry 13.41 (-0.6% vs prior close, gap -2.2%, PDR 13%, stop 4.3% away, 5m$ $5,526,185) · stopped 13m later after MFE +0.04R · **wick**
  - `IRE   ` 10:07 net **+1.70R** · entry 8.84 (+0.3% vs prior close, gap -4.0%, PDR 23%, stop 7.0% away, 5m$ $1,214,010) · held 348m to the close, MFE +1.77R · **other**

### 1c. Class counts across ALL losers

Precedence (declared before the run, entry-knowable first): e -> d -> c -> b -> a -> f -> g -> h. A trade gets exactly one class; the raw flag counts (a trade can carry several) are in the second table.

| class | TRAIN n | TRAIN R | VAL n | VAL R | TEST n | TEST R | ALL n | ALL R | % of loss |
|---|---|---|---|---|---|---|---|---|---|
| (e) late/extended  entry >8% above the 09:30 open | 55 | -32.8 | 14 | -8.2 | 16 | -10.5 | 85 | -51.5 | 5.9% |
| (d) spread/thin    5-min $ volume < $200K | 251 | -202.0 | 95 | -78.1 | 72 | -61.3 | 418 | -341.5 | 39.3% |
| (c) news/halt      bar gap >=5 min or a 1-min range >5% | 95 | -66.4 | 39 | -27.3 | 41 | -29.5 | 175 | -123.2 | 14.2% |
| (b) market dump    SPY or IWM -0.5% in the hour after entry | 46 | -35.5 | 32 | -31.8 | 17 | -16.2 | 95 | -83.5 | 9.6% |
| (a) level failure  never traded above the level after the fill | 29 | -28.0 | 10 | -8.9 | 12 | -11.3 | 51 | -48.2 | 5.5% |
| (f) wick stop      stop bar closed above the stop, back over entry <30m | 6 | -6.7 | 4 | -4.6 | 2 | -2.3 | 12 | -13.6 | 1.6% |
| (g) slow bleed     MFE <0.3R then >30 min to the stop | 50 | -33.2 | 32 | -22.2 | 16 | -10.8 | 98 | -66.2 | 7.6% |
| (h) other | 94 | -65.2 | 54 | -44.9 | 45 | -30.6 | 193 | -140.7 | 16.2% |
| **all losers** | 626 | -470.0 | 280 | -226.0 | 221 | -172.4 | 1127 | -868.3 | 100% |

Raw flags, NOT exclusive (share of all losers carrying the flag, and the same flag among winners for contrast):

| flag | losers n | losers % | loser R | winners n | winners % | winner R |
|---|---|---|---|---|---|---|
| (e) late/extended  entry >8% above the 09:30 open | 85 | 7.5% | -51.5 | 74 | 8.4% | +54.8 |
| (d) spread/thin    5-min $ volume < $200K | 442 | 39.2% | -359.0 | 366 | 41.5% | +454.7 |
| (c) news/halt      bar gap >=5 min or a 1-min range >5% | 549 | 48.7% | -403.5 | 510 | 57.8% | +674.7 |
| (b) market dump    SPY or IWM -0.5% in the hour after entry | 212 | 18.8% | -185.9 | 126 | 14.3% | +156.7 |
| (a) level failure  never traded above the level after the fill | 178 | 15.8% | -168.1 | 5 | 0.6% | +1.2 |
| (f) wick stop      stop bar closed above the stop, back over entry <30m | 41 | 3.6% | -47.4 | 0 | 0.0% | +0.0 |
| (g) slow bleed     MFE <0.3R then >30 min to the stop | 445 | 39.5% | -302.2 | 43 | 4.9% | +3.5 |

---

## 2. Winners at entry vs losers at entry

Median [Q1, Q3] per split. `gap` = the relative gap between the winner and loser medians, `(w-l)/|l|`.

**TRAIN** (n 486 winners / 626 losers)

| fact | winners med [Q1,Q3] | losers med [Q1,Q3] | gap |
|---|---|---|---|
| entry minute (ET min of day) | 596 [581, 644] | 591 [578, 627] | +1% |
| gap % (09:30 open vs prior close) | -3.05 [-5.01, -1.44] | -3.13 [-5.17, -1.22] | +2% |
| prior-day range % | 15.45 [10.72, 22.19] | 16.24 [11.39, 25.00] | -5% |
| entry vs prior close % | 0.16 [-0.55, 0.45] | 0.18 [-0.48, 0.50] | -8% |
| open -> entry % | 2.97 [1.20, 5.09] | 3.25 [1.20, 5.41] | -9% |
| stop distance % of entry | 6.19 [5.23, 7.94] | 6.10 [5.03, 7.99] | +1% |
| 5-min $ volume | 274,578 [83,133, 1,226,571] | 286,814 [85,577, 1,487,362] | -4% |
| range-so-far % at the signal | 7.68 [6.40, 10.15] | 7.62 [6.34, 9.90] | +1% |
| IWM open -> entry % | 0.077 [-0.291, 0.372] | 0.016 [-0.276, 0.291] | +379% |
| SPY open -> entry % | 0.024 [-0.135, 0.194] | 0.010 [-0.135, 0.164] | +137% |
| trades already taken that day (this trades ordinal) | 3.0 [2.0, 4.0] | 3.0 [2.0, 4.0] | +0% |
| entry price $ | 11.68 [7.36, 21.63] | 11.96 [7.38, 24.44] | -2% |

**VAL** (n 251 winners / 280 losers)

| fact | winners med [Q1,Q3] | losers med [Q1,Q3] | gap |
|---|---|---|---|
| entry minute (ET min of day) | 584 [576, 619] | 584 [576, 611] | +0% |
| gap % (09:30 open vs prior close) | -3.26 [-5.01, -1.54] | -2.97 [-4.93, -1.19] | -10% |
| prior-day range % | 15.52 [11.11, 22.56] | 14.66 [11.13, 20.76] | +6% |
| entry vs prior close % | 0.20 [-0.43, 0.52] | 0.10 [-0.59, 0.44] | +104% |
| open -> entry % | 3.31 [1.60, 5.14] | 3.03 [1.20, 4.98] | +9% |
| stop distance % of entry | 6.29 [5.09, 7.95] | 5.96 [4.80, 7.53] | +6% |
| 5-min $ volume | 387,253 [117,769, 1,804,364] | 489,120 [117,411, 1,507,879] | -21% |
| range-so-far % at the signal | 7.72 [6.42, 9.90] | 7.46 [6.31, 9.42] | +3% |
| IWM open -> entry % | 0.007 [-0.377, 0.228] | -0.030 [-0.352, 0.249] | +125% |
| SPY open -> entry % | 0.048 [-0.067, 0.168] | -0.016 [-0.141, 0.124] | +399% |
| trades already taken that day (this trades ordinal) | 3.0 [2.0, 4.5] | 3.0 [2.0, 4.0] | +0% |
| entry price $ | 15.40 [8.50, 33.53] | 13.90 [7.93, 28.11] | +11% |

**TEST** (n 145 winners / 221 losers)

| fact | winners med [Q1,Q3] | losers med [Q1,Q3] | gap |
|---|---|---|---|
| entry minute (ET min of day) | 584 [576, 608] | 580 [575, 603] | +1% |
| gap % (09:30 open vs prior close) | -3.12 [-5.49, -1.51] | -2.92 [-4.71, -1.31] | -7% |
| prior-day range % | 14.53 [11.38, 20.97] | 15.77 [11.02, 22.25] | -8% |
| entry vs prior close % | 0.18 [-0.50, 0.52] | 0.12 [-0.56, 0.48] | +47% |
| open -> entry % | 3.35 [1.43, 5.57] | 2.88 [1.25, 5.00] | +16% |
| stop distance % of entry | 6.48 [5.38, 8.19] | 6.00 [4.86, 7.72] | +8% |
| 5-min $ volume | 456,184 [139,242, 2,021,823] | 376,238 [142,772, 1,609,926] | +21% |
| range-so-far % at the signal | 8.06 [6.55, 10.01] | 7.46 [6.31, 9.37] | +8% |
| IWM open -> entry % | 0.017 [-0.232, 0.217] | 0.010 [-0.240, 0.219] | +72% |
| SPY open -> entry % | 0.039 [-0.121, 0.119] | -0.013 [-0.139, 0.096] | +398% |
| trades already taken that day (this trades ordinal) | 3.0 [2.0, 4.0] | 3.0 [2.0, 5.0] | +0% |
| entry price $ | 14.84 [9.45, 27.90] | 18.49 [8.40, 29.00] | -20% |

**Which facts separate by >=20% in the SAME direction in TRAIN and VAL** (the declared bar):

| fact | TRAIN gap | VAL gap | same sign & both >=20%? | TEST gap |
|---|---|---|---|---|
| entry minute (ET min of day) | +1% | +0% | no | +1% |
| gap % (09:30 open vs prior close) | +2% | -10% | no | -7% |
| prior-day range % | -5% | +6% | no | -8% |
| entry vs prior close % | -8% | +104% | no | +47% |
| open -> entry % | -9% | +9% | no | +16% |
| stop distance % of entry | +1% | +6% | no | +8% |
| 5-min $ volume | -4% | -21% | no | +21% |
| range-so-far % at the signal | +1% | +3% | no | +8% |
| IWM open -> entry % | +379% | +125% | **YES** | +72% |
| SPY open -> entry % | +137% | +399% | **YES** | +398% |
| trades already taken that day (this trades ordinal) | +0% | +0% | no | +0% |
| entry price $ | -2% | +11% | no | -20% |

Passing: IWM open -> entry %, SPY open -> entry %.

**Day of week and price band** (net R per trade):

| day of week | TRAIN n | TRAIN R/tr | VAL n | VAL R/tr | TEST n | TEST R/tr |
|---|---|---|---|---|---|---|
| Mon | 225 | -0.008 | 95 | +0.677 | 74 | -0.078 |
| Tue | 231 | +0.024 | 109 | +0.182 | 76 | +0.096 |
| Wed | 233 | +0.042 | 106 | -0.116 | 85 | -0.365 |
| Thu | 211 | +0.057 | 117 | +0.152 | 70 | +0.177 |
| Fri | 212 | +0.206 | 104 | +0.193 | 61 | +0.030 |

| price band | TRAIN n | TRAIN R/tr | VAL n | VAL R/tr | TEST n | TEST R/tr |
|---|---|---|---|---|---|---|
| $5-10 | 468 | +0.026 | 183 | +0.311 | 114 | -0.028 |
| $10-20 | 325 | +0.115 | 148 | +0.047 | 91 | +0.133 |
| $20-50 | 250 | +0.092 | 136 | +0.282 | 127 | -0.108 |
| >$50 | 69 | -0.049 | 64 | +0.118 | 34 | -0.311 |

---

## 3. The 10 worst weeks

SPY/IWM weekly return = sum of the daily open->close moves in the week (the book is flat overnight, so the intraday sum is the relevant market). "realised vol" = stdev of the SPY daily open->close in that week.

| week | split | book R | n | SPY wk % | IWM wk % | SPY daily sd | worst loser class (share of the week's loss) |
|---|---|---|---|---|---|---|---|
| 2026-03-23 | VAL | **-12.85** | 28 | -2.79 | -0.29 | 0.64 | thin (62% of -17.2R) |
| 2026-02-02 | VAL | **-11.32** | 29 | +0.11 | +0.93 | 1.02 | thin (48% of -17.1R) |
| 2026-07-06 | TEST | **-11.28** | 29 | +1.25 | -0.71 | 0.35 | thin (44% of -14.3R) |
| 2025-03-10 | TRAIN | **-9.96** | 19 | -2.42 | -2.73 | 1.00 | thin (44% of -12.2R) |
| 2025-12-15 | TRAIN | **-9.34** | 24 | -1.62 | -2.84 | 0.70 | thin (70% of -13.8R) |
| 2025-01-13 | TRAIN | **-8.63** | 24 | +0.74 | +0.44 | 0.59 | other (27% of -11.9R) |
| 2026-01-26 | VAL | **-8.49** | 30 | -0.05 | -1.98 | 0.28 | dump (34% of -18.7R) |
| 2025-10-27 | TRAIN | **-8.04** | 25 | -0.88 | -1.28 | 0.37 | thin (46% of -12.5R) |
| 2025-03-31 | TRAIN | **-7.70** | 29 | -0.78 | +1.69 | 2.24 | thin (66% of -15.5R) |
| 2025-12-29 | TRAIN | **-7.11** | 15 | -1.17 | -1.25 | 0.36 | thin (62% of -11.6R) |

Weeks total 88 · green 53 (60%) · worst -12.85R · best +26.10R.

Correlation of weekly book R with the weekly SPY intraday sum: **+0.361**; with IWM: **+0.396**; with SPY daily sd: **-0.009** (n=88 weeks).

---

## 4. Sequence within the day

| ordinal | TRAIN n | TRAIN R/tr | VAL n | VAL R/tr | TEST n | TEST R/tr | ALL n | ALL R/tr |
|---|---|---|---|---|---|---|---|---|
| 1 | 247 | +0.164 | 102 | +0.681 | 68 | +0.148 | 417 | +0.288 |
| 2 | 240 | +0.050 | 102 | +0.245 | 68 | +0.045 | 410 | +0.098 |
| 3 | 227 | +0.006 | 100 | +0.079 | 68 | -0.260 | 395 | -0.021 |
| 4+ | 398 | +0.038 | 227 | +0.033 | 162 | -0.067 | 787 | +0.015 |

| entry hour (ET) | TRAIN n | TRAIN R/tr | VAL n | VAL R/tr | TEST n | TEST R/tr | ALL n | ALL R/tr |
|---|---|---|---|---|---|---|---|---|
| 09:xx | 623 | +0.052 | 361 | +0.311 | 265 | -0.032 | 1249 | +0.109 |
| 10:xx | 287 | +0.087 | 98 | -0.013 | 56 | -0.135 | 441 | +0.036 |
| 11:xx | 101 | -0.022 | 33 | +0.098 | 27 | -0.109 | 161 | -0.012 |
| 12:xx | 54 | +0.140 | 24 | -0.098 | 9 | +0.395 | 87 | +0.101 |
| 13:xx | 46 | +0.145 | 15 | -0.140 | 9 | +0.017 | 70 | +0.067 |
| 14:xx | 1 | -0.073 | 0 | - | 0 | - | 1 | -0.073 |

### The two declared rule cells

Both are SUBSET filters on the already-booked trades: dropping a trade does NOT free its slot for a candidate the book passed over, so these are a lower bound on what a re-booked run would show.

| cell | TRAIN n | TRAIN R/tr | TRAIN total | VAL n | VAL R/tr | VAL total | TEST n | TEST R/tr | TEST total |
|---|---|---|---|---|---|---|---|---|---|
| as booked | 1112 | +0.0620 | +69.0 | 531 | +0.2066 | +109.7 | 366 | -0.0420 | -15.4 |
| first two entries only (seq <= 2) | 487 | +0.1076 | +52.4 | 204 | +0.4632 | +94.5 | 136 | +0.0966 | +13.1 |
| no entries after 11:00 (entry_min < 660) | 910 | +0.0627 | +57.1 | 459 | +0.2417 | +110.9 | 321 | -0.0502 | -16.1 |

---

## 5. Avoidance arithmetic

Only two of the eight classes are decidable AT THE FILL: **(e) late/extended** (the entry is >8% above the 09:30 open — known from the fill price and the first bar) and **(d) spread/thin** (the 5-min $ volume before the fill). The others need the future tape. Below: drop each entry-knowable class from the book (again a subset, not a re-book), and both together. $/month at $300 risk = total net R x $300 / months in the split (TRAIN 12, VAL 5, TEST 3.13).

| rule | TRAIN n | TRAIN R/tr | TRAIN $/mo | VAL n | VAL R/tr | VAL $/mo | TEST n | TEST R/tr | TEST $/mo |
|---|---|---|---|---|---|---|---|---|---|
| as booked | 1112 | +0.0620 | $1,725 | 531 | +0.2066 | $6,584 | 366 | -0.0420 | $-1,473 |
| drop (e) late/extended | 1024 | +0.0761 | $1,948 | 495 | +0.2013 | $5,979 | 331 | -0.0530 | $-1,680 |
| drop (d) spread/thin | 624 | +0.0278 | $433 | 333 | +0.1937 | $3,870 | 244 | -0.0581 | $-1,358 |
| drop both (e) and (d) | 570 | +0.0446 | $636 | 314 | +0.1773 | $3,340 | 216 | -0.0709 | $-1,468 |

Share of TOTAL loser R carried by each entry-knowable flag:

- e_late · TRAIN: 55/626 losers, -32.8R of -470.0R (7.0% of the loss); those same trades among winners: 33 for +23.9R
- e_late · VAL: 14/280 losers, -8.2R of -226.0R (3.6% of the loss); those same trades among winners: 22 for +18.3R
- e_late · TEST: 16/221 losers, -10.5R of -172.4R (6.1% of the loss); those same trades among winners: 19 for +12.6R
- d_thin · TRAIN: 267/626 losers, -213.5R of -470.0R (45.4% of the loss); those same trades among winners: 221 for +265.2R
- d_thin · VAL: 101/280 losers, -82.0R of -226.0R (36.3% of the loss); those same trades among winners: 97 for +127.3R
- d_thin · TEST: 74/221 losers, -63.4R of -172.4R (36.8% of the loss); those same trades among winners: 48 for +62.2R
---

## 6. Two follow-ups the tables forced

### 6a. Is "first entry of the day" anything but the 09:3x bar?

The book is FIRST-COME (12/day, 4 concurrent), so `seq` is just entry order in time. Cross-tab of net R per trade, seq ordinal x entry hour, ALL splits pooled and then per split for the 09:xx column only.

| ordinal | 09:xx n | 09:xx R/tr | 10:xx n | 10:xx R/tr | 11:xx+ n | 11:xx+ R/tr |
|---|---|---|---|---|---|---|
| 1 | 386 | +0.284 | 22 | +0.237 | 9 | +0.592 |
| 2 | 329 | +0.093 | 57 | +0.199 | 24 | -0.076 |
| 3 | 261 | -0.019 | 103 | -0.040 | 31 | +0.025 |
| 4+ | 273 | +0.003 | 259 | +0.014 | 255 | +0.028 |

Within the 09:xx hour only, per split:

| ordinal | TRAIN n | TRAIN R/tr | VAL n | VAL R/tr | TEST n | TEST R/tr |
|---|---|---|---|---|---|---|
| 1 | 222 | +0.142 | 96 | +0.706 | 68 | +0.148 |
| 2 | 175 | +0.022 | 90 | +0.278 | 64 | +0.025 |
| 3 | 121 | -0.029 | 81 | +0.164 | 59 | -0.250 |
| 4+ | 105 | +0.002 | 94 | +0.065 | 74 | -0.073 |

### 6b. The worse-than-1R losses

A stop exit books `min(stop, bar open) x 0.999`. When the next bar opens BELOW the stop, the loss is bigger than the 1R the stop nominally risked. The rule floor is `R >= 1% of entry`, so a 1.1%-wide stop on a thin tape turns a 4% gap-down bar into a -3.6R print.

- trades with net R <= -1.5: **16** of 2009 (0.8%), **-34.5 R** — 4.0% of ALL loser R, against a book total of +163.4 R.
- their median stop distance **1.65%** of entry vs **6.13%** for the book; median 5-min $ volume **$34,769** vs **$336,948**.
- 15 of 16 had a stop closer than 3% of the entry price; 15 were thin.

| stop distance band | n | R/trade | total R | mean net R of its losers |
|---|---|---|---|---|
| < 2% | 65 | +0.052 | +3.3 | -1.461 |
| 2-3% | 87 | +0.518 | +45.1 | -1.169 |
| 3-4% | 102 | +0.497 | +50.7 | -0.988 |
| 4-6% | 698 | +0.069 | +48.2 | -0.761 |
| 6-9% | 733 | +0.012 | +9.1 | -0.713 |
| >= 9% | 324 | +0.021 | +6.9 | -0.525 |

**Declared cell: a minimum stop distance** (known at the fill — the stop is the running low, the entry is the fill). Subset filter, not a re-book.

| rule | TRAIN n | TRAIN R/tr | TRAIN $/mo | VAL n | VAL R/tr | VAL $/mo | TEST n | TEST R/tr | TEST $/mo |
|---|---|---|---|---|---|---|---|---|---|
| as booked | 1112 | +0.0620 | $1,725 | 531 | +0.2066 | $6,584 | 366 | -0.0420 | $-1,473 |
| stop distance >= 2% of entry | 1074 | +0.0719 | $1,929 | 518 | +0.1735 | $5,392 | 352 | -0.0200 | $-674 |
| stop distance >= 3% of entry | 1029 | +0.0783 | $2,015 | 493 | +0.1013 | $2,997 | 335 | -0.0467 | $-1,499 |
| stop distance >= 4% of entry | 990 | +0.0676 | $1,673 | 457 | +0.0402 | $1,102 | 308 | -0.0683 | $-2,015 |

**Declared cell: stop floor combined with the sequence rule.**

| rule | TRAIN n | TRAIN R/tr | TRAIN $/mo | VAL n | VAL R/tr | VAL $/mo | TEST n | TEST R/tr | TEST $/mo |
|---|---|---|---|---|---|---|---|---|---|
| seq <= 2 | 487 | +0.1076 | $1,310 | 204 | +0.4632 | $5,669 | 136 | +0.0966 | $1,259 |
| stop >= 3% | 1029 | +0.0783 | $2,015 | 493 | +0.1013 | $2,997 | 335 | -0.0467 | $-1,499 |
| seq <= 2 AND stop >= 3% | 426 | +0.0984 | $1,048 | 179 | +0.2343 | $2,516 | 116 | +0.0792 | $880 |
| seq <= 2 AND stop >= 3% AND 09:xx | 339 | +0.1056 | $895 | 161 | +0.2501 | $2,416 | 112 | +0.0688 | $739 |

Tail check on the surviving rule (`seq <= 2 AND stop >= 3%`): mean net R with the top 1% and top 5% of trades removed, and with winners capped at +3R.

| split | n | R/tr | ex-top-1% | ex-top-5% | winners capped +3R |
|---|---|---|---|---|---|
| TRAIN | 426 | +0.0984 | +0.0154 | -0.1582 | -0.0038 |
| VAL | 179 | +0.2343 | +0.1025 | -0.0570 | +0.0962 |
| TEST | 116 | +0.0792 | +0.0287 | -0.1326 | +0.0293 |

Monthly net R of `seq <= 2 AND stop >= 3%`:

| 2025-01 | 2025-02 | 2025-03 | 2025-04 | 2025-05 | 2025-06 | 2025-07 | 2025-08 | 2025-09 | 2025-10 | 2025-11 | 2025-12 | 2026-01 | 2026-02 | 2026-03 | 2026-04 | 2026-05 | 2026-06 | 2026-07 | 2026-08 | 2026-09 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -7.6 | +9.5 | +17.2 | +5.3 | +6.7 | -5.1 | +9.2 | -10.8 | +15.1 | +3.1 | +5.2 | -5.9 | +19.8 | +10.0 | -0.1 | +16.0 | -3.7 | -2.7 | +8.1 | +1.7 | +2.1 |

Months green **14/21**; worst month **-10.8 R** (= $-3,229 at $300 risk); mean month **+4.43 R** ($1,329).

### 6c. Tail dependence — the book and every surviving rule

| rule | split | n | R/tr | ex-top-1% | ex-top-5% | winners capped +3R |
|---|---|---|---|---|---|---|
| as booked | TRAIN | 1112 | +0.0620 | -0.0195 | -0.1564 | -0.0054 |
| as booked | VAL | 531 | +0.2066 | +0.0793 | -0.1164 | +0.0414 |
| as booked | TEST | 366 | -0.0420 | -0.1093 | -0.2480 | -0.0897 |
| seq <= 2 | TRAIN | 487 | +0.1076 | +0.0027 | -0.1964 | -0.0389 |
| seq <= 2 | VAL | 204 | +0.4632 | +0.3077 | +0.0015 | +0.1370 |
| seq <= 2 | TEST | 136 | +0.0966 | +0.0411 | -0.1742 | -0.0112 |
| stop >= 3% | TRAIN | 1029 | +0.0783 | +0.0115 | -0.1091 | +0.0325 |
| stop >= 3% | VAL | 493 | +0.1013 | +0.0149 | -0.1160 | +0.0329 |
| stop >= 3% | TEST | 335 | -0.0467 | -0.0913 | -0.2193 | -0.0691 |
| seq <= 2 AND stop >= 3% | TRAIN | 426 | +0.0984 | +0.0154 | -0.1582 | -0.0038 |
| seq <= 2 AND stop >= 3% | VAL | 179 | +0.2343 | +0.1025 | -0.0570 | +0.0962 |
| seq <= 2 AND stop >= 3% | TEST | 116 | +0.0792 | +0.0287 | -0.1326 | +0.0293 |

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
