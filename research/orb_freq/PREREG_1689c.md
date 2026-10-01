# PREREG 1,689c — quiet-window in-regime read, ORB sub-pools 24/25/26/30 (FROZEN 2026-10-01 18:xx UTC, before any 1689c number is read)

## Mechanism
RESULT_1689b.md VOIDed pools 24/25/26/30's in-regime leg "by design": their candidates' bars sit
in cache.db (production's intraday_bars_1min), and study_orb_pipeline_static_lock.py's full
backtest hung 10+min at 0% CPU on the LIVE trading service's own cache.db lock (write-lock wait,
not a slow read). That file's "What the original cell did NOT license" section prescribes two
fixes: (a) avoid cache.db-default mode entirely, or (b) wait for a quiet window. 1689c takes (b):
the live service is down until 12:30 UTC on 10/2. queue_1689c.sh runs the SAME unchanged
_run_pipeline_once/_pipeline_env_for (1689b_pools.py) against the SAME already-built features
files (1689b's stage_prep succeeded for these pools — only the backtest subprocess hung), with a
watchdog that SIGSTOPs the subprocess at 12:25 UTC 10/2 and SIGCONTs at 20:05 UTC 10/2 so it never
overlaps the live session. Pools 21/27/28 already have a real in-regime read (1689b's same-day
update) — reused unchanged, not recomputed.

## Pool definitions (unchanged from 1689b_pools.py)
21 gap>=4%, price $50-200, prevVol>=1M (fresh-build). 24 gap 3-5%, price $3-30, needs premkt+own-
range (cheap/WIDE-seed). 25 gap 3-5%, $3-30, needs premkt, no range req (cheap). 26 gap 3-5%,
$3-30, needs premkt+range, compression variant (cheap). 27 gap>=5%, price $1-3, prevVol>=2M
(fresh-build). 28 gap>=5%, price $30-100, prevVol>=500K (fresh-build). 30 gap 3-5%, $3-30, needs
range, no premkt req, opening-drive variant (cheap). Pools 22/29 stay VOID (no data source
exists) — not re-attempted.

## Window and metrics
In-regime 2025-01-01..2026-09-26, split into calendar-year halves (2025, 2026); out-regime 2024H2
reused from 1689b unchanged (always bars_sip.db, never cache.db-contended). Per pool, per half: n
fills, fills/week, mean R, iid t, day-clustered t, ex-top-5% mean, weekly P10 per fill, worst week
$ at $375; plus union with the production book (union $, worst week $, P10). Exactly 1689b's own
metric set — no new metric invented for this cell.

## Pass bar (identical to 1689b)
Own meanR>=+0.05 AND dc_t>=2.0 in-regime, AND (0 fills OR meanR>=0.0) out-of-regime, AND
ex_top5>0.

## Paper-candidacy decision rule (NEW, pre-committed, separate from the pass bar above)
A pool is a candidate for paper only if mean R >= +0.10 in BOTH halves (2025 AND 2026) AND
>=2 fills/week AND ex-top-5% mean >= 0. Anything weaker stays research-only regardless of the
pass bar above.

## Multiplicity
7 pools x 3 windows (in-regime 2025, in-regime 2026, out-regime 2024H2) = 21 cells scored this
round, on top of the 16 cells 1689b already spent.

## Scope limits
No new exit, gap/price/volume band, or filter — LIVE exit only, as scoped by 1689b. No new fetch
(24/25/26/30 need none; 21/27/28 already fetched). FROZEN before any 1689c number is read.
