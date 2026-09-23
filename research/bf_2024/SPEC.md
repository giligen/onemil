# SPEC — bull-flag P1 on 2024H2 (PREREG.md). For a Sonnet implementer, ≤ 40 tool calls

Read `research/bf_2024/PREREG.md` (frozen). Grep/offset reads only; never a file > 300 lines in full. Write only under
`research/bf_2024/`. Do not commit; do not touch `trading/`, `config.yaml`, `orb.yaml`, crontab, `data/cache.db`, or
the production bull-flag cache. No Alpaca fetch and no heavy compute between 13:25 and 20:05 UTC (live market).

1. Read how `batch_backtest.py` builds Stage 1: `find_big_movers` (~line 756), `fetch_daily_bars_cached` (~865),
   `get_1min_bars_cached` (~929), the `--build-cache` path, `BT_CACHE_PATH_OVERRIDE`, and how Stage 2 reads the P1
   profile. Write `research/bf_2024/run_bf_2024.py` that monkeypatches those data functions (pattern:
   `research/orb_2024/build_features_2024.py` and `research/orb_seed_wide/build_wide_features.py`): daily bars from
   `data/research/databento/equs_daily_2024H2.parquet` (map '+'→'.WS', drop '-' preferreds; if a lookback needs June
   2024, fetch Alpaca daily bars for those symbols into `research/bf_2024/daily_june.parquet`), minute bars from
   `research/bf_2024/bars.db` (fetch from Alpaca SIP on first need, schema of `research/day_breadth/y2024/bars.db`;
   reuse `research/day_breadth/y2024/bars.db` rows when present), then call the Stage-1 build for 2024-07-02..2024-12-31
   into `BT_CACHE_PATH_OVERRIDE=research/bf_2024/cache_2024.csv`, then Stage 2 with the normal flags.
2. Smoke on 3 days first; confirm movers, bars, cache rows and Stage-2 trades are sane.
3. Launch the full chain detached so it survives you: `nohup setsid nice -n 19 ionice -c3 bash research/bf_2024/chain.sh`
   (fetch → Stage 1 → Stage 2 → score), with a guard that sends SIGSTOP to its own process group at 13:25 UTC and
   SIGCONT at 20:05 UTC if it is still running then (a small `at`-free loop in the script is fine). Log to
   `research/bf_2024/chain.log`; the final step writes `research/bf_2024/REPORT.md` with n trades, mean R, t
   (day-clustered), total $, ex-top-5 %, monthly $, the P1 2025–26 book beside it, and the PREREG verdict.
4. Return as soon as the full chain is running and past its first 10 % (do not wait for it to finish).

Reply ≤ 150 words: the smoke numbers, the movers count and availability share, every seam/parity problem, the
chain's PID and expected finish time.
