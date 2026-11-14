# Gap-fade short — PAPER instrument spec (from cells 1,773 → 1,776, rebuilt 2026-11-14)

Status: SPEC. Engine work for a Sonnet agent (TDD, full suite before commit, boot rehearsal before Monday). Paper only (HOD
paper account `ALPACA_HOD_API_KEY/SECRET`); live only after ≥ 2 weeks of paper + locate-ledger parity AND the owner's GO
(real locate dollars). ONE helper module shared by BT and live (`trading/gap_short_rules.py`), parity test against
`research/signal_search/1776R_build.py` on the 2026 book (same picks on (day, symbol), same L1 cutoff).

## The rule (every number from the rebuilt cell, nothing tuned)
1. Universe at 09:25 ET: symbols with a daily bar yesterday (daily_bars), prev close ≥ $1; gap = pre-market last trade / prev close
   − 1 ≥ +10 %; drop gap > 200 %; drop `^Z[A-Z]ZZT$`; drop names absent from daily_bars.
2. Liquidity L1 = pre-market dollar volume 04:00–09:25 ET (sum price × size of pre-market trades/bars) ≥ **$7.3 M** (the TRAIN
   top-tercile cutoff, fixed; refit never without a PREREG).
3. Rank by gap desc; take the top 10 (fewer if fewer qualify). No other filter.
4. Borrow: ETB → trade. HTB → paper: trade and LOG the advisory locate fee (`GET /v1/locates/quotes`, live keys, never POST);
   live: POST locate only after the owner's GO, fee cap 40 bp of notional, else skip the name. Threshold-list → skip.
5. Entry: market-on-open sell short, `time_in_force=opg`, qty = floor($300 / pre-market last). Under $5 the broker's 100 %
   maintenance applies — size is $ not R; one doubler costs $300.
6. Exit: at 10:30:00 ET a marketable limit buy-to-cover at ask + 10 bp (fallback market at 10:31 if unfilled). No stop. Hard
   backstop only: if the position is +100 % against us intraday (halt-up), cover on resume.
7. Portfolio rails: ≤ 10 names/day, ≤ $3,000 gross short, paper day kill at −$300 realised, pause the instrument after 2 consecutive
   red weeks below the weekly P10 of the BT (−0.66 R at $150 unit).
8. Ledger: `logs/gap_short_ledger.csv` one row per name/day: picked, L1, gap, borrow_status, locate fee bp (advisory), MOO fill,
   10:30 ask at decision, cover fill, slip vs ask, net bp, cost lines; Friday 17:05 ET scorer vs the BT line (+145 bp net VAL,
   5 names/day) with the count-matched placebo; `[GAPSHORT]` Telegram prefix; ERROR only on real failures.

## What the BT says (rebuild, prose-only, both placebos cleared at the measured ask)
VAL 2026 n 496, +145 bp net/trade (fees 4 + locate 9 + borrow 100 %/360 + measured ask), day-t 2.7, ex-top-5 % ≈ 0,
worst-5 % −17 %, weekly P10 −0.66 R, 5 names/day; TRAIN 2025H2 +188 (t 3.5), halves +282/+88. Expected at $300/name ≈ $20/day
≈ $400/mo paper-equivalent; the instrument's job is to measure real fees, fills and the tail, not to earn yet.
Known gaps: the squeeze tail (1,775 news veto pending), real gap-morning locate fees (ledger from Monday), MOO fill vs daily open.

## Architecture decision (2026-11-14 evening): standalone cron script, NOT inside the trader service
`scripts/gap_short_paper.py` on the **MOM paper account** (`ALPACA_MOM_API_KEY/SECRET`, weekly long book → least intraday
overlap; HOD/ORB paper accounts trade the same gappers at 09:35–09:40 and would net against us). No systemd change, no boot
rehearsal of the live service. Modes (all idempotent, all with `--dry-run`):
* `--pick` at 09:26 ET: build the list per rules 1–3 (pre-market trades via Alpaca data API snapshots/bars 04:00–09:25; L1 ≥ $7.3 M;
  gap ≥ 10 %, ≤ 200 %; top 10 by gap), skip any symbol with an existing position or open order on the account, skip threshold-list
  names, log borrow_status + advisory locate fee from the LIVE quotes endpoint (GET only), submit `sell` `market` `opg`
  qty = floor(300 / pre-market last), `client_order_id = gapshort-<date>-<sym>`. Paper may reject HTB shorts → log, no retry.
* `--cover` at 10:30 ET: for every open `gapshort-` short, limit buy-to-cover at ask + 10 bp (`day`); poll 60 s; unfilled →
  cancel and market. `--flatten` at 10:40 ET: safety — any remaining `gapshort-` short is market-covered and logged as ERROR.
* `--score` at 16:15 ET: ledger row per name (MOO fill vs daily open, cover fill vs 10:30 ask, slip, net bp, cost lines, borrow
  status, advisory fee bp); `--weekly` Friday 17:10 ET: vs the BT line (+145 bp, 5 names/day), count-matched placebo from the
  day's non-picked gappers, `[GAPSHORT]` Telegram.
* Rails in code: ≤ 10 names, ≤ $3,000 gross, no new picks if the ledger's realised week ≤ −$300 (pause flag file), ERROR to Telegram
  only on order rejections / API failures / flatten events. Shared rule helper `trading/gap_short_rules.py` (pure functions:
  gap, L1, ranking, sizing) with a parity test against `research/signal_search/1776R_events.parquet` picks for 10 sampled 2026 days
  (same (day, symbol) set given the same inputs).
Crons (ET block, standing-authorized): 26 9, 30 10, 40 10 (1-5); 15 16 (1-5); 10 17 (5). Tests: unit (mocked client with spec=),
integration (paper account read-only: assets/borrow_status, quotes GET), and a `--dry-run --pick` system run Monday 09:26 before
the first real paper order (the first live-paper day is Tuesday 11/17 unless the dry run is clean and the owner says Monday).
