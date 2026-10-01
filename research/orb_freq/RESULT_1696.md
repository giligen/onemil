# RESULT 1,696 -- rank_not_selected at full coverage (run started 2026-10-01T15:34:02.744017)

PREREG: research/orb_freq/PREREG_1696.md (FROZEN). Written incrementally per step; coverage first.

## Independent check (replay + live PDR/G1/range-size vetoes vs book_csv)

ranking-stage survivors (pre-veto) 2763; after replaying PDR->G1->range-size veto (same functions, same order, no refill): 633 vs book_csv in-range 633; agree 633 (100.00%), only-in-book 0, only-in-replay 0.

## Step 2 -- rank_not_selected (1937) split by real reason

- score_threshold: 1149
- slot_cap: 517
- skip_q1: 259
- dedup: 12

## Step 1 -- coverage after backfill

rank_not_selected + G1_veto candidates: 2585; bar coverage now 98.8% ok (was 16.3% / 12.7% respectively before the backfill).

By old-vs-new reason:

_bucket
G1_veto             99.4% ok (n=648)
dedup               100.0% ok (n=12)
score_threshold    98.0% ok (n=1149)
skip_q1             98.8% ok (n=259)
slot_cap            99.6% ok (n=517)

## Step 3 -- per reason x per half (taken book beside it)

| reason | half | n | n_ok | cov% | mean R | day_t | runner>=3R | ex_top5 | $/yr@375 |
|---|---|---|---|---|---|---|---|---|---|
| score_threshold | pooled | 1149 | 1126 | 98.0 | -0.065 | -3.06 | 0.135 | -0.300 | $-15,944 |
| score_threshold | 2025 | 576 | 560 | 97.2 | -0.128 | -2.92 | 0.136 | -0.382 | $-15,739 |
| score_threshold | 2026 | 573 | 566 | 98.8 | -0.002 | -1.28 | 0.134 | -0.218 | $-205 |
| skip_q1 | pooled | 259 | 256 | 98.8 | +0.241 | 1.02 | 0.133 | -0.137 | $+13,494 |
| skip_q1 | 2025 | 111 | 108 | 97.3 | -0.127 | -0.44 | 0.148 | -0.344 | $-2,999 |
| skip_q1 | 2026 | 148 | 148 | 100.0 | +0.509 | 1.13 | 0.122 | -0.009 | $+16,493 |
| dedup | pooled | 12 | 12 | 100.0 | -0.439 | -1.57 | 0.083 | -0.621 | $-1,153 |
| dedup | 2025 | 8 | 8 | 100.0 | -0.705 | -1.78 | 0.125 | -1.029 | $-1,236 |
| dedup | 2026 | 4 | 4 | 100.0 | +0.094 | -0.06 | 0.000 | -0.033 | $+82 |
| slot_cap | pooled | 517 | 515 | 99.6 | +0.154 | -2.60 | 0.155 | -0.089 | $+17,360 |
| slot_cap | 2025 | 192 | 190 | 99.0 | -0.209 | -1.35 | 0.168 | -0.435 | $-8,700 |
| slot_cap | 2026 | 325 | 325 | 100.0 | +0.366 | -2.34 | 0.148 | +0.118 | $+26,060 |
| G1_veto | pooled | 648 | 644 | 99.4 | -0.092 | -1.44 | 0.107 | -0.306 | $-12,942 |
| G1_veto | 2025 | 372 | 370 | 99.5 | -0.267 | -0.68 | 0.105 | -0.497 | $-21,627 |
| G1_veto | 2026 | 276 | 274 | 99.3 | +0.145 | -1.84 | 0.109 | -0.041 | $+8,685 |
| TAKEN | pooled | 471 | 471 | 100.0 | +0.297 | 2.65 | 0.227 | -0.059 | $+30,609 |
| TAKEN | 2025 | 212 | 212 | 100.0 | +0.270 | 1.59 | 0.288 | -0.034 | $+12,558 |
| TAKEN | 2026 | 259 | 259 | 100.0 | +0.318 | 2.14 | 0.178 | -0.076 | $+18,050 |

## Coverage-bias check (rank_not_selected + G1_veto pooled)

| group | n | n_ok | cov% | mean R | day_t | runner>=3R | $/yr@375 |
|---|---|---|---|---|---|---|---|
| originally_covered_16pct | 398 | 398 | 100.0 | +0.339 | 1.10 | 0.128 | $+29,570 |
| newly_covered_after_backfill | 2187 | 2155 | 98.5 | -0.061 | -1.69 | 0.132 | $-28,756 |

## Step 4 -- promotion read (pre-declared: mean R >= +0.15 BOTH halves, runner share >= 15%, pooled day_t >= 2.0)

- **score_threshold**: cov 98.0%, 2025 mean R -0.128, 2026 mean R -0.002, runner 0.135, pooled day_t -3.06 -> fails the promotion bar
- **skip_q1**: cov 98.8%, 2025 mean R -0.127, 2026 mean R +0.509, runner 0.133, pooled day_t 1.02 -> fails the promotion bar
- **dedup**: cov 100.0%, 2025 mean R -0.705, 2026 mean R +0.094, runner 0.083, pooled day_t -1.57 -> fails the promotion bar
- **slot_cap**: cov 99.6%, 2025 mean R -0.209, 2026 mean R +0.366, runner 0.155, pooled day_t -2.60 -> fails the promotion bar
- **G1_veto**: cov 99.4%, 2025 mean R -0.267, 2026 mean R +0.145, runner 0.107, pooled day_t -1.44 -> fails the promotion bar

Buckets meeting the promotion bar: NONE. Per PREREG: nothing changes in orb.yaml without the promoted cell passing both directions AND a rebuild -- not done in this cell.
