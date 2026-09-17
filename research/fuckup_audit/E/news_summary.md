# Stage E — causal news presence for the U1 u U2 key set

Produced 2026-09-16 by `E/e_news.py` (a thin re-point of `D/d0_news.py` at the Stage-E universe),
verified by `E/e_news_verify.py`. **No trading claim is made here** — this file records what was
fetched, that it is complete, and that it is causal.

## What was fetched

| | |
|---|---|
| key set | `(E/u1_keys.csv u E/u2_keys.csv)` = **198,318** (day, symbol), 410 trading days 2025-01-17 .. 2026-09-04 |
| already covered by `D/news_presence.csv` | 35,665 (18.0%) — D's own file holds 61,902 keys, the other 26,237 are Stage-D-only keys outside U1uU2 |
| fetched here | **162,653** keys, written to `E/news_presence_e.csv` (4.6 MB) |
| window | prev CALENDAR day 15:00 ET -> trade day 14:01 ET (identical to D; PLAN H4 / `research/scripts/orb_news_backfill.py`) |
| columns | `day, symbol, n_prev15_to_0930, last_prev_ts, n_0930_to_1401, mins_0930_to_1401, fetch_ok` — byte-identical header to D's file, so the two concatenate |
| state | `E/news_state.json` (done-days list; the CSV is appended per day, so the job is resumable) |
| log | `E/e_news.log`, ends `EXIT=0` |

## Completeness

| check | result |
|---|---|
| rows written | 162,653 (162,653 unique keys — no duplicates) |
| `fetch_ok == 0` rows | **0** (E), 0 (D) |
| page-walk truncations (`TRUNC`) | **0** — no chunk ever exhausted the 25-page walk, so no premarket article was dropped. This also clears the same risk in D's file, which was fetched with the same chunk size and window. |
| coverage of the E key set by `news_presence_e.csv u D/news_presence.csv` | **198,318 / 198,318 = 100.000%**, 0 missing |
| columns identical to D | yes |

## Causality

The window ends at 14:01 ET (the population's last entry minute) and the file stores, per key, the
count of articles strictly before 09:30 plus the ET minute-of-day of every article between 09:30
and 14:01. A row with signal minute `sig_m` therefore uses `n_prev15_to_0930` (always causal) plus
`#{mins < sig_m}` — nothing at or after the signal minute is ever readable. No article timestamp
is taken from the trade day after 14:01.

## What the data says (descriptive only — this is not a result)

Share of symbol-days carrying at least one article, over the full E key set (E + D files):

| split | keys | pre-09:30 news | share | 09:30-14:01 news | share |
|---|---:|---:|---:|---:|---:|
| TRAIN (2025) | 99,937 | 16,824 | 16.8% | 12,787 | 12.8% |
| VAL (2026-01..05) | 57,058 | 8,413 | 14.7% | 6,810 | 11.9% |
| TEST (2026-06..09-04) | 41,323 | 5,301 | 12.8% | 4,200 | 10.2% |
| **ALL** | **198,318** | **30,538** | **15.4%** | **23,797** | **12.0%** |

The pre-09:30 share declines TRAIN -> VAL -> TEST (16.8 -> 14.7 -> 12.8%). Two candidate causes,
neither settled here: a genuinely thinner news tape in 2026, or vendor/indexing coverage differing
by era. Any model that uses news presence as a feature must therefore be walk-forward, and a
level-shift in the feature's base rate across splits should not be read as a regime signal.

## Timing / cost

410 days, 3,686 API requests, **43.9 min** wall clock, single process, `nice -n 10`,
`ulimit -v 1000000`, ~0.35 s between calls (well under the 200 req/min data-API limit).
D's comparable run: 420 days / 61,902 keys / 28 min.

## Independent spot check (10 + 10 keys re-called one symbol at a time)

`E/e_news_verify.py` re-called the API per symbol (not per 50-symbol chunk, i.e. a different
request shape) and recomputed the counts:

- 10 uniformly random rows: **10/10 exact** (all happened to be no-news rows — a weak check, so a
  second, news-biased draw was run).
- 10 random rows drawn from rows with at least one article: **10/10 exact** on
  `n_prev15_to_0930` and `n_0930_to_1401`, and exact on the `mins_0930_to_1401` minute list for
  every row that has one (SCPH 2025-08-25 4/1, WMS 2025-02-07 0/1, PGY 2025-07-17 1/2,
  BW 2026-08-11 8/1, ODFL 2025-10-29 2/2). The five rows the first pass flagged "MISMATCH" were an
  artefact of the checker: an empty `mins` field reads back as NaN under this tree's
  `keep_default_na=False, na_values=['']` convention, and every one of those rows has
  `n_0930_to_1401 == 0`. Confirmed globally: exactly 144,056 rows have an empty `mins` field and
  exactly 144,056 rows have `n_0930_to_1401 == 0`.

## One deliberate deviation from `D/d0_news.py`

If a 50-symbol chunk's window were deeper than 25 pages x 50 articles, the `sort=desc` page walk
would silently drop the OLDEST articles — i.e. exactly the premarket ones this feature is for.
`e_news.py` detects that condition, splits the chunk in half and re-fetches. It never fired
(0 `TRUNC` lines in 3,686 requests), so the file is identical to what D's code would have produced.

## How to consume

```python
RD = dict(keep_default_na=False, na_values=[''])          # the ticker NA
news = pd.concat([
    pd.read_csv('research/fuckup_audit/E/news_presence_e.csv', dtype={'day': str, 'symbol': str}, **RD),
    pd.read_csv('research/fuckup_audit/D/news_presence.csv',  dtype={'day': str, 'symbol': str}, **RD),
], ignore_index=True).drop_duplicates(['day', 'symbol'])
```
Join on `(day, symbol)`; `has_news_premarket = n_prev15_to_0930 > 0`. For a row with signal minute
`sig_m`, add `sum(int(m) < sig_m for m in str(mins).split())` for the intraday-so-far count.

## Resume / re-run

`python3 research/fuckup_audit/E/e_news.py` is idempotent at day granularity: it re-reads
`E/news_state.json` and skips finished days. To re-fetch a day, delete it from the state file's
`done` list and drop that day's rows from the CSV.
