# PREREG — Does the LIVE bull-flag P1 book survive a different regime? 2024-07-02 .. 2024-12-31. Cell 1,417

Why: ORB, run unchanged on the untouched survivorship-free 2024H2 half-year, was flat (−0.007 R/fill vs +0.272 in
2025 — `research/orb_2024/REPORT.md`), and the risk-on base entry reversed there. Bull flag P1 is live (stage L0,
effective risk $270–900 per trade through its conviction and MACD multipliers) and ALL its evidence is 2025–26.
Frozen before any 2024 bull-flag number exists.

## Population and data
Movers per `batch_backtest.find_big_movers`'s own definition, with daily bars from the Databento EQUS 2024H2
point-in-time parquet (delisted included; '+' warrants → '.WS'; preferreds excluded) instead of `cache.db`. 1-minute
SIP bars from Alpaca into `research/bf_2024/bars.db` (never `cache.db`). Stage-1 cache into a side file via
`BT_CACHE_PATH_OVERRIDE` (never the production cache). Availability rail ≥ 80 % of movers with bars, else VOID.

## Book
Stage 2 with the LIVE P1 profile exactly as `config.yaml` configures it today (filters, 50 % partial at +2 R, unified
trail, conviction + MACD multipliers as the BT models them), `--capital 50000 --risk 2000 --max-shares 10000`.
Reported in R (R = pnl / (shares × |entry − stop|), the ramp's R basis) and in $.

## Verdict (per the ORB PREREG's logic — survival, not proof)
SURVIVES iff mean R per trade > 0 AND total $ > 0 AND ex-top-5 % mean R > 0. RED FLAG iff mean R ≤ −0.10 → recommend
the owner pause bull flag live pending review. Beside it: the P1 2025–26 book (`research/bf_frequency/runs/P1.csv`).

## Not allowed
Changing any P1 parameter, the multipliers or the verdict after a 2024 number exists; any other cell on this data.
Programme count 1,417.
