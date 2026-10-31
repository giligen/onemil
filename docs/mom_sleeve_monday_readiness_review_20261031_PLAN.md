# MOM sleeve — Monday 2026-11-02 plan rehearsal (Sat 2026-10-31, real API, NO orders)

Commands (spec Part B; never `--submit`):
1. `nice -n 19 python3 scripts/momentum_sleeve.py --force --verbose` — paper plan, fetch included (09:19–09:24 UTC, 0 ERROR lines; the run pruned `daily_20261009.parquet` as expected and wrote `daily_20261030.parquet` + its completeness file).
2. `python3 scripts/momentum_sleeve.py --live --force --skip-fetch --equity-start 10000 --verbose` — live plan, dry (rc 0, 0 ERROR lines, `NOTHING WRITTEN (dry run)`).
State files, ledgers and weekly CSVs keep their 10/19 mtimes (untouched).

## Data and gates (asof Fri 2026-10-30)
- TXSE path: `get_all_assets: SDK rejected the payload (... unknown exchange values {'TXSE'} — returned 14407 assets) — raw REST fallback` (WARNING, works on both accounts).
- COMPLETENESS: requested 13,509, 13,198 with bars, 13,191 with a 2026-10-30 bar (98 %), **LOST 310 (2.3 %, max 5 %)**, **liquid 770/770 (100.0 %, min 98 %)** → no refusal.
- Guard: none in the top 40.
- Gate line: `gate OFF (p83) size 100%  ratio 0.9346 (VIX 19.42 / VIX3M 20.78, 2026-10-30)`.
- Rail line (LIVE): sleeve equity $9,447 (start $10,000, rail at $7,500) → not hit; `paused` absent in both state files; weeks since a strong week 1 (warn at 8). Paper: equity $16,905 vs peak $20,103 (dd −15.9 %, kill rule at −40 % not hit).
- LIVE buying power $243,136 vs buy notional $1,633: OK. 0 non-sleeve positions on the live account at the time of the run.

## The 20 selected names (same for paper and live)
TXG, AXTI, SNDK, TWST, MU, MXL, LITE, RVMD, VLO, UMC, ASX, STX, MRNA, ILMN, MPC, FRO, TRGP, DELL, PSX, HPE

vs the 10/19 holdings: **OUT DINO, IOVA, WDC** · **IN HPE, MRNA, MXL** · 17 kept (re-weighted to 1/20).

## Deltas — PAPER (equity $16,904.82, target $845.24/name), 22 orders
SELL: ASX $11.43 · AXTI $32.55 · **DINO $801.10 (full exit)** · FRO $114.50 · **IOVA $541.77 (full exit)** · LITE $34.86 ·
MU $18.88 · RVMD $106.01 · STX $17.55 · TRGP $109.17 · TWST $156.29 · TXG $53.77 · **WDC $931.88 (full exit)**
BUY: DELL $29.83 · **HPE $845.24** · ILMN $50.10 · MPC $31.91 · **MRNA $845.24** · **MXL $845.24** · SNDK $104.04 · UMC $113.37 · VLO $56.04
(PSX within $5 of target — no order.)

## Deltas — LIVE slice (equity $9,446.63, target $472.33/name), 22 orders
SELL: ASX $6.80 · AXTI $10.50 · **DINO $447.67 (full exit)** · FRO $63.24 · **IOVA $303.53 (full exit)** · LITE $15.71 ·
MU $8.14 · RVMD $63.79 · STX $6.10 · TRGP $62.75 · TWST $94.33 · TXG $30.78 · **WDC $516.54 (full exit)**
BUY: DELL $18.33 · **HPE $472.33** · ILMN $22.50 · MPC $17.33 · **MRNA $472.33** · **MXL $472.33** · SNDK $62.77 · UMC $63.27 · VLO $31.59
`LIVE PLAN: sells $1,630 buys $1,633 | sleeve cash (state) $0.20 | buying power OK`

Monday's numbers will differ by the weekend/Monday-open marks (sizing uses broker marks at submit time).

## Telegram that WOULD go on Monday (no message is sent by a dry run)
- 11:50 UTC prefetch (dry): none (ERRORs only; there were none today).
- 13:45 / 13:46 UTC (08:45 ET, EST): none — new INFO no-op `submit window opens 09:31 ET (now 08:45 ET) — no-op; the later cron line submits` (before the fix: a `momentum_sleeve crashed (RuntimeError)` ERROR on Telegram from each line).
- 14:45 UTC paper, format: `[MOM] 2026-11-02 equity $16,9xx (dd -15.x% from peak) names 20 in 3 out 3 turnover $~5,850 fills 22/22 gate OFF (p83) size 100%`
- 14:46 UTC live, format: `[MOM LIVE] 2026-11-02 equity $9,4xx (dd -5.x% from peak) names 20 in 3 out 3 turnover $~3,260 fills 22/22 week ±x.xR gate OFF (p83) size 100%`
- A refusal would read `[MOM] …` / `[MOM LIVE] 2026-11-02 REFUSED: completeness … -- no orders placed, book unchanged` (the LIVE label is new).
