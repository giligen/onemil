# ORB created_at -> submitted_at gap: LIVE endpoint vs PAPER (2026-10-17, read-only)

Method: order ids from `data/trades.db` (`strategy='orb'`; live = 2026-05-19..09-23, account NULL; paper = 10/12..10/16,
account 'paper'), each fetched with `get_order_by_id` (live key on the live endpoint, ORB key on paper). BUY entries only.
gap = submitted_at - created_at. 0 orders 404ed or lacked timestamps on either side.

| Account | n buys | gap > 0.5 s | median | p90 | max |
|---|---|---|---|---|---|
| LIVE | 165 | 0 | 0.01 s | 0.04 s | 0.42 s |
| PAPER (10/12-10/16) | 12 | 12 | 7.35 s | 10.12 s | 10.70 s |

Paper sanity check: gaps 4.3-10.7 s, which reproduces the triage figure.

LIVE by month (n / median / p90 / max, seconds):
May 30 / 0.00 / 0.03 / 0.03; Jun 83 / 0.01 / 0.04 / 0.42; Jul 38 / 0.00 / 0.11 / 0.37; Aug 7 / 0.01 / 0.03 / 0.03;
Sep 7 / 0.00 / 0.01 / 0.01.

LIVE by created_at bucket (ET):
| Bucket | n | median | p90 | max |
|---|---|---|---|---|
| 09:34:50-09:35:10 | 102 | 0.01 | 0.06 | 0.42 |
| 09:35:10-09:36:00 | 44 | 0.00 | 0.03 | 0.09 |
| later | 19 | 0.01 | 0.05 | 0.07 |

Other: submitted_at -> filled_at median is 82.5 s on LIVE (n 123 filled) and 873 s on PAPER (n 5 filled); the paper
fill figure rests on only 5 orders.

Verdict: YES, the multi-second created->submitted gap is a paper-endpoint artefact. Live never exceeded 0.42 s in any
month or time bucket (0 of 165 over 0.5 s), while paper was 4.3-10.7 s on 12 of 12 orders.
