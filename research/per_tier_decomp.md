# Per-tier realized-R decomposition (10%-frame, current shipping config)

Total trades loaded (conv>=1.4): 625
Source caches: /tmp/s1_looser/cache_base_{2025,q1}.csv

Tier classification: A ≥ 20%, E in [10%, 20%), edge < 10%.

## 1. Per-tier summary (all splits combined)

| Tier | n | WR | mean R | median R | Total PnL | pnl@1x |
|---|---:|---:|---:|---:|---:|---:|
| A | 171 | 44.4% | +0.238R | -1.048R | $+37,266 | $+12,209 |
| E | 442 | 40.3% | +0.041R | -1.056R | $+7,024 | $+960 |
| edge | 12 | 66.7% | +0.526R | +0.658R | $+3,964 | $+1,442 |

## 2. Per-tier × split breakdown

| Tier | Split | n | WR | mean R | Total PnL |
|---|---|---:|---:|---:|---:|
| A | TRAIN | 77 | 51.9% | +0.426R | $+32,748 |
| A | VAL | 46 | 41.3% | +0.203R | $+8,593 |
| A | HOQ1 | 48 | 35.4% | -0.031R | $-4,075 |
| E | TRAIN | 170 | 39.4% | +0.027R | $-9,672 |
| E | VAL | 141 | 36.2% | +0.012R | $+3,006 |
| E | HOQ1 | 131 | 45.8% | +0.091R | $+13,691 |
| edge | TRAIN | 5 | 80.0% | +1.240R | $+3,819 |
| edge | VAL | 5 | 60.0% | +0.070R | $+295 |
| edge | HOQ1 | 2 | 50.0% | -0.121R | $-151 |

## 3. Per-tier × MACD zone (most important table)

MACD 1.0 = normal zone. 1.5 = strong (pos or neg). Current shipping multiplies strong bucket 1.5× sizing. This table shows whether the 1.5 bucket actually HAS edge per tier.

| Tier | MACD | n | WR | mean R | Total PnL | pnl@1x |
|---|---|---:|---:|---:|---:|---:|
| A | 1.0 (normal) | 83 | 44.6% | +0.134R | $+6,432 | $+3,397 |
| A | 1.5 (strong) | 88 | 44.3% | +0.336R | $+30,834 | $+8,812 |
| E | 1.0 (normal) | 219 | 37.0% | -0.055R | $-14,734 | $-7,999 |
| E | 1.5 (strong) | 223 | 43.5% | +0.135R | $+21,758 | $+8,960 |
| edge | 1.0 (normal) | 7 | 57.1% | +0.306R | $+865 | $+382 |
| edge | 1.5 (strong) | 5 | 80.0% | +0.833R | $+3,099 | $+1,060 |

## 4. Per-tier rule β (realized-R lift when rule fires vs not)

Positive β = rule has edge IN THAT TIER. Negative β = rule is noise or counter-signal. Compare magnitudes: if β_A >> β_E, the rule is A-tier-specific.

| Rule | A β | A n_fires | E β | E n_fires | edge β | edge n |
|---|---:|---:|---:|---:|---:|---:|
| r1 (pole gain) | -0.134R | 96 | +0.227R | 271 | +0.648R | 7 |
| r2+ (flag tight) | +0.110R | 63 | -0.111R | 154 | +0.628R | 5 |
| r2- (flag loose) | +0.099R | 33 | +0.178R | 98 | -1.032R | 4 |
| r3 (vol ratio) | +0.047R | 115 | -0.226R | 326 | -1.395R | 10 |
| r4+ (SPY good) | -0.150R | 88 | +0.101R | 206 | -0.156R | 6 |
| r4- (SPY bad) | +0.191R | 19 | +0.193R | 60 | -0.549R | 2 |
| r5 (retracement) | -0.203R | 80 | +0.124R | 278 | -0.888R | 4 |
| r7 (vwap dist) | +0.087R | 136 | -0.016R | 403 | +1.748R | 11 |
| r8 (gap fading) | -0.260R | 5 | -1.160R | 1 | n/a | — |
| r9 (V-reversal) | +1.031R | 41 | n/a | — | n/a | — |

## 5. Per-tier conviction-decile breakdown

Shows realized R by conv bucket within each tier. Helps identify whether high-conv trades generalize across tiers.

| Tier | ConvBucket | n | WR | mean R | Total PnL |
|---|---|---:|---:|---:|---:|
| A | <1.5 | 17 | 52.9% | +0.559R | $+3,591 |
| A | 1.5-1.8 | 54 | 35.2% | -0.159R | $-4,682 |
| A | 1.8-2.2 | 61 | 54.1% | +0.520R | $+17,686 |
| A | 2.2-2.6 | 31 | 32.3% | +0.007R | $+4,671 |
| A | ≥2.6 | 8 | 62.5% | +0.978R | $+16,000 |
| E | <1.5 | 34 | 50.0% | +0.206R | $+1,776 |
| E | 1.5-1.8 | 171 | 39.8% | +0.076R | $+8,002 |
| E | 1.8-2.2 | 169 | 37.3% | -0.098R | $-3,334 |
| E | 2.2-2.6 | 57 | 45.6% | +0.285R | $+4,698 |
| E | ≥2.6 | 11 | 36.4% | -0.144R | $-4,117 |
| edge | <1.5 | 1 | 0.0% | -1.034R | $-427 |
| edge | 1.5-1.8 | 5 | 80.0% | +0.402R | $+834 |
| edge | 1.8-2.2 | 5 | 80.0% | +1.282R | $+4,166 |
| edge | 2.2-2.6 | 1 | 0.0% | -1.077R | $-609 |

## 6. Top rule patterns per tier


### Tier A: Top-10 rule-firing patterns

| Pattern (r1,r2,r3,r4,r5,r7,r8,r9) | n | mean R | Total PnL |
|---|---:|---:|---:|
| (+,0,+,+,+,+,0,0) | 8 | -0.560R | $-2,453 |
| (0,0,+,0,0,+,0,0) | 8 | -0.143R | $-816 |
| (0,0,+,+,0,+,0,0) | 7 | -0.125R | $-695 |
| (0,+,+,+,+,+,0,0) | 5 | +1.320R | $+5,590 |
| (0,0,0,+,0,+,0,0) | 5 | +0.564R | $+1,799 |
| (+,0,+,0,0,+,0,0) | 4 | -0.423R | $-841 |
| (+,0,+,0,0,0,0,0) | 4 | -0.560R | $-1,201 |
| (+,+,+,+,+,+,0,0) | 4 | +0.023R | $+2,962 |
| (+,0,0,+,0,+,0,0) | 4 | +0.188R | $+373 |
| (+,-,0,0,+,+,0,0) | 4 | +0.306R | $+252 |

### Tier E: Top-10 rule-firing patterns

| Pattern (r1,r2,r3,r4,r5,r7,r8,r9) | n | mean R | Total PnL |
|---|---:|---:|---:|
| (+,0,+,+,0,+,0,0) | 20 | +0.080R | $+1,571 |
| (+,+,+,0,+,+,0,0) | 19 | +0.159R | $+569 |
| (+,+,+,-,+,+,0,0) | 17 | -0.071R | $+1,022 |
| (+,+,+,-,0,+,0,0) | 15 | +0.482R | $+2,398 |
| (0,0,+,0,0,+,0,0) | 15 | -0.658R | $-4,414 |
| (+,0,+,0,0,+,0,0) | 15 | +0.365R | $+2,792 |
| (+,0,+,-,+,+,0,0) | 15 | -0.011R | $-366 |
| (+,0,+,0,+,+,0,0) | 14 | -0.656R | $-3,473 |
| (0,-,+,0,+,+,0,0) | 13 | +0.110R | $+681 |
| (0,-,+,+,+,+,0,0) | 13 | +0.124R | $+2,346 |

### Tier edge: Top-10 rule-firing patterns

| Pattern (r1,r2,r3,r4,r5,r7,r8,r9) | n | mean R | Total PnL |
|---|---:|---:|---:|
| (+,+,+,0,0,+,0,0) | 2 | +1.566R | $+1,766 |
| (0,-,+,+,0,+,0,0) | 2 | -0.123R | $-94 |
| (0,-,+,0,+,+,0,0) | 1 | -1.034R | $-427 |
| (0,0,+,0,0,+,0,0) | 1 | +1.387R | $+555 |
| (+,+,0,+,0,+,0,0) | 1 | +2.503R | $+1,971 |
| (+,+,+,-,+,+,0,0) | 1 | -0.099R | $-2 |
| (0,-,+,+,+,+,0,0) | 1 | +0.632R | $+263 |
| (+,+,+,+,0,0,0,0) | 1 | -1.077R | $-609 |
| (+,0,+,-,+,+,0,0) | 1 | +0.236R | $+111 |
| (+,0,0,+,0,+,0,0) | 1 | +0.874R | $+430 |
