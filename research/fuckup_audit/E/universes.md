# Stage E Part 1 — causal universes (counts only)

Generated 2026-09-16T22:11:10+00:00 by `research/fuckup_audit/E/universes.py`.

Membership uses ONLY data known at 09:30 ET on day t: the day's own open, and prior-day close / high / low / volume. No range gate, no hindsight.

| universe | rule |
|---|---|
| U1 gap | open/prev_close - 1 >= 3%, open >= $5, adv20 >= 100,000 sh |
| U2 prior-day range | (prev_high-prev_low)/prev_low >= 8%, open >= $5, adv20 >= 100,000 sh |
| U3 liquid slice | median_20d(close*volume) >= $5,000,000, open >= $5 (no other gate) |
| U4 premarket | pm_dollar_vol >= $500K — NOT computable from a daily panel; built in Part 3 from the fetched 04:00-09:29 bars |

adv20 = `volume.shift(1).rolling(20, min_periods=10).mean()`, dvol20_med = `(close*volume).shift(1).rolling(20, min_periods=10).median()` — the convention of `research/lit_review_2026/build_daily_panel.py:17`, i.e. the one behind `research/bf_zero/universe.csv`.

## Size, per universe per split

| universe | split | symbol-days | distinct symbols | in bf_zero universe (>=5% range) | already in bars_sip.db |
|---|---|---:|---:|---:|---:|
| U1 | TRAIN | 30,871 | 4,825 | 21,891 (70.9%) | 8,829 (28.6%) |
| U1 | VAL | 17,839 | 3,912 | 12,190 (68.3%) | 4,782 (26.8%) |
| U1 | TEST | 13,328 | 3,007 | 9,227 (69.2%) | 4,070 (30.5%) |
| U1 | ALL | 62,038 | 6,050 | 43,308 (69.8%) | 17,681 (28.5%) |
| U2 | TRAIN | 81,168 | 5,053 | 66,653 (82.1%) | 31,865 (39.3%) |
| U2 | VAL | 45,772 | 3,475 | 38,626 (84.4%) | 17,493 (38.2%) |
| U2 | TEST | 33,679 | 3,080 | 28,770 (85.4%) | 13,381 (39.7%) |
| U2 | ALL | 160,619 | 5,976 | 134,049 (83.5%) | 62,739 (39.1%) |
| U3 | TRAIN | 842,396 | 4,754 | 137,819 (16.4%) | 99,074 (11.8%) |
| U3 | VAL | 396,575 | 4,671 | 84,617 (21.3%) | 62,532 (15.8%) |
| U3 | TEST | 273,060 | 4,583 | 58,392 (21.4%) | 43,795 (16.0%) |
| U3 | ALL | 1,512,031 | 5,571 | 280,828 (18.6%) | 205,401 (13.6%) |

**U1 u U2 = 198,318 symbol-days**, of which 73,732 (37.2%) are already in `bars_sip.db`; **124,586 to fetch**.

`bars_sip.db` holds 305,547 symbol-days on this 420-day calendar.

`data/cache.db::intraday_bars_1min` probe on 4,000 random U1uU2 keys: 292 hits (7.3%). **Information only** — cache.db bars are a different fetch provenance (and RTH-only); this stage fetches SIP so the new store is one tape with `bars_sip.db`.

## U3 — not fetched in this stage

U3 is 1,512,031 symbol-days, 1,306,630 of them not in `bars_sip.db`. At the KB/symbol-day this stage measures (see E/REPORT.md) the disk it would need is reported there; it is a later decision, not this stage's.

## Price-scale check (PLAN.md §1)

198 of 200 random keys had a 09:30 ET minute bar in `bars_sip.db`. `diff = panel_open / minute_open - 1`:

| |within 0.01%|within 0.1%|within 0.5%|within 2%|median |diff||p95 |diff||
|---|---:|---:|---:|---:|---:|---:|
| panel open vs 09:30 minute open | 89.9% | 95.5% | 99.0% | 100.0% | 0.0000% | 0.0865% |

Rows: `E/pricescale.csv`.

## Fetch truncation

U1 u U2 minus bars_sip.db = 124,586 keys > the 120,000 cap; the fetched set is bar_date >= 2025-07-01 -> **93,715 keys**. The other 30,871 keys (2025-01-02..2025-06-30, all TRAIN) are written to `E/fetch_keys_overflow.csv` and are NOT fetched by default; `fetch_causal.py --overflow` fetches them into the same store. Nothing is dropped silently, and TRAIN coverage of the causal universe is therefore HALF a year unless the overflow pass is run.

Built in 5.1 min.

