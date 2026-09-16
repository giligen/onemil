# RUNBOOK — literature-driven hypothesis tests on the honest tape (for a cheaper model to execute)

Written 2026-09-16 by the session that ran the 9/15–9/16 audit. Read this whole file before running anything.
Working directory: `/home/ec2-user/onemil`. Python: `/usr/bin/python3` (3.11, pandas 2, numpy 2, alpaca-py). Git branch
`fix/spy-regime-shared-helper`; commit and push code and `.md` results only — data files are git-ignored (see `.gitignore`).

## 0. What this is, and the rules you must not break
- Goal (owner 9/16): "go back to square one … literature review … hypothesis tables … find it. 10R was just a target, data will speak."
  Every published rule is a HYPOTHESIS with the paper's number as the prior. It is tested OUT OF SAMPLE on our tape with
  costs, on a fixed split, with a pre-committed falsification rule. Nothing is tuned to the test split. If nothing passes,
  say so with the closest miss. Never manufacture a number.
- Three look-aheads were found in this repo in one week — read them before writing any test (research/bf_zero/REPORT.md §6b,
  research/bf_zero2/REPORT.md, research/bf_zero2/score_tables_VOID_no_floor.md): (1) a study population selected by a
  scanner in hindsight; (2) a one-publisher tape (EQUS.MINI) that could never pass a volume gate; (3) a universe rule
  "day range ≥ 5%" that is only known at the close. Rule: every input of a signal must be known at the decision minute,
  and the UNIVERSE the signal is evaluated on must be defined by information available before the decision.
- Splits are FIXED: TRAIN = 2025-01-02..2025-12-31, VAL = 2026-01-01..2026-05-31, TEST = 2026-06-01..2026-09-11 for anything
  on the 2025–26 stores; for the 2016→ ETF store, IS = 2016–2023 (the papers' samples), OOS = 2024-01→today. Choose on
  TRAIN/IS, confirm on VAL, read TEST/OOS once and report it whatever it says.
- Costs are charged per trade, always: US large caps and index ETFs 1 bp per leg (SPY's real spread is ~0.1 bp; 1 bp is
  conservative and the papers' per-share fees are smaller — report BOTH when the paper states its own cost); $5–20 small
  caps with ≥5% range: 40 bps spread → half in, half out on non-target exits (research/bf_zero/spread_study_clean.csv medians);
  daily-bar books by 20-day dollar volume: ≥$50M 6 bps, ≥$10M 12 bps, ≥$5M 25 bps, else 40 bps round trip.
- Book rule for anything with several names a day: `trading.hod_break.run_book(rows, max_per_day=12, max_concurrent=4)` —
  causal slot freeing, symbol tie-break. Do not write another one.
- Machine rules: 7.8 GB RAM, 2 cores, the live trader runs on this box (never touch `data/cache.db` writes, never restart
  services, never place orders). Run every Python job as `setsid nohup bash -c "cd /home/ec2-user/onemil; ulimit -v 4500000;
  nice -n 10 python3 <script> > <log> 2>&1; echo EXIT=\$? >> <log>" > /dev/null 2>&1 < /dev/null &` and poll the log —
  the tool harness kills foreground/background commands when system memory is low; a detached process survives. One heavy
  process at a time. Never `pkill -f <pattern>` with a pattern that matches your own shell; anchor it (`^python3 …`).
- Provenance gate for any NEW bar store: `research/bf_zero/parity_review/tape_provenance_check.py` (50 random keys vs a
  fresh Alpaca REST call, ≥ 99% bar-exact) before any number is reported from it.

## 1. Data stores (all built; do not rebuild unless a section says so)
| store | path | schema | content | built by |
|---|---|---|---|---|
| Daily PIT panel | `research/lit_review_2026/daily_panel.parquet` | one row per symbol-day: symbol, bar_date, open, high, low, close, volume, prev_close, prev_volume, adv20, dvol20, ret_on (open/prev_close−1), ret_id (close/open−1), ret_cc, ret_*_next (t+1), range_pct, vol_ratio (volume/adv20), ret5, ret20, high52, dow | every US symbol 2025-01→2026-09 incl. delisted (Databento EQUS.SUMMARY); the daily `open` matches the 09:30 1-min open on liquid names (checked, 0 of 32 > 10 bps off) | `build_daily_panel.py` (needs ulimit 5.5 GB) |
| ETF 1-min | `research/lit_review_2026/etf_1min.db` | bars(symbol, t ISO-UTC, o,h,l,c,v,n,vw); done(symbol, month, n) | SPY QQQ TQQQ SQQQ IWM DIA SOXL UVXY, 2016-01→2026-09, extended hours included (filter 09:30–15:59 ET) | `fetch_etf_1min.py` (resumable; add symbols as argv) |
| Stock 1-min, ≥5%-range days | `research/bf_zero/bars_sip.db` | bars(symbol, day, t ISO-UTC, o,h,l,c,v); fetch_log(symbol, day, src, n_bars, fetched_at) | every universe symbol-day 2025–26 with (high−low)/low ≥ 5% and open ≥ $5 that `data/cache.db` did not hold (~305K symbol-days incl. the +5% superset); `data/cache.db intraday_bars_1min` holds the rest (the BF scanner's mover days). Loader: `research/bf_zero/build_candidates.py::load_bars` with env `BFZ_SIP_STORE=/home/ec2-user/onemil/research/bf_zero/bars_sip.db` | `refetch_thin_tape.py --fetch --keys …` |
| Liquid open window | `research/lit_review_2026/liquid_open.db` | window(symbol, day, o5,h5,l5,c5,v5) = the 09:30–09:34 bars for every liquid name (prior close > $5, ADV14 ≥ 1M, ATR14 > $0.50) each day; days(symbol, day, t, o,h,l,c,v) = full-day bars of the daily top-20 by relative volume | 2025-01→2026-09 | `fetch_liquid_open_window.py window` then `days`; `liquid_top20.csv` = the daily picks with rv |
| Candidates (pass 1/2) | `research/bf_zero2/candidates.csv` (4.67M rows, 2 GB), `candidates_full.csv` (3.0M rows at price ≥ 5, entry ≤ 14:01, with rv_profile/rv_clock/cohort) | one row per (day, symbol, family, config): entry_m, entry, stop, r_pct, rr_e1..e4 (exits), rr_e1c/exit_m_e1c (the live exit), features | 26 entry-family configurations on the whole universe | `research/bf_zero2/build_candidates2.py` → `pass2b.py` |
| Spread study | `research/bf_zero/spread_study_clean.csv` | historical NBBO at signal time for 6,847 signals | cost model source | — |

## 2. Test harnesses that exist (read each docstring; they are short)
| script | tests | inputs | output | runtime |
|---|---|---|---|---|
| `research/lit_review_2026/test_daily_hyps.py` | daily-bar hypotheses: D1 volume-shock overnight, D2 large-loser reversal, D12 intraday-component reversal (+ overnight control), overnight-continuation proxy, Pritamani-Singal continuation, 52-week-high breakout — top-4-per-day book, net of costs | daily panel | `daily_hyps.log`, `daily_hyps_results.csv` | 3 min |
| `research/lit_review_2026/test_etf_intraday.py` | A1 market intraday momentum (first-half-hour → last-half-hour, 4 variants), A2 simplified band momentum, A3 5-min ORB on TQQQ/QQQ (0.1 ATR stop, hold to close) | etf_1min.db | `etf_intraday.log`, `etf_intraday_results.csv` | 10 min |
| `research/lit_review_2026/test_zarattini_spy.py` | the FAITHFUL Zarattini-Aziz SPY/QQQ noise-band strategy + ablations | etf_1min.db | `zarattini_spy.md` | 1 min |
| `research/lit_review_2026/test_eod_reversal.py` | H-B5 end-of-day reversal: prior-close→15:00 ≤ −8% → buy 15:30, sell close; winner tail as control | daily panel + both 1-min stores | `eod_reversal.md`, `eod_reversal_trades.csv` | ~60 min |
| `research/lit_review_2026/test_stocks_in_play.py` | H-B1 Zarattini stocks-in-play ORB OOS replication (top-20 RV, 5-min ORB, 0.1 ATR stop, hold to close) by RV bucket, at 1 bp and 20 bps | liquid_open.db + liquid_top20.csv | `stocks_in_play.md`, `_trades.csv` | 5 min |
| `research/lit_review_2026/test_rv_monotone.py` | H-B2 relative-volume monotonicity of ORB/HOD P&L on our population, with and without the causal floor | bf_zero2/candidates_full.csv | stdout | 2 min |
| `research/bf_zero2/score2.py` | every pass-1 family on the executable book, net, causal floor | candidates_full.csv | `score.log`, `score_tables.md`, `results.csv` | 10 min, 4 GB |

## 3. Results so far (2026-09-16 05:00 UTC) — do NOT re-run these unless you change the spec; extend them
| hypothesis | result | verdict |
|---|---|---|
| A1 market intraday momentum (Gao 2018), SPY, all four variants | −2 to −3 bps/day in EVERY year 2016–2026, t ≈ −3 (IS) and −2.4 (OOS) | dead; the 0DTE-era anti-momentum the review predicted |
| A2/H3 Zarattini SPY noise bands, faithful | IS 2016–23: 10.9 bps/day, t 3.2, SR 1.03 unlevered at paper costs (reproduces); OOS 2024→: 0.6 bps/day, t 0.09, 2026 YTD −21% at the 2% vol target; the semi-hourly check cadence is the load-bearing ingredient | dead on SPY out of sample |
| same on QQQ | OOS 10.1 bps/day, t 1.59, 8.4%/yr unlevered, SR 0.99 | marginal; ≈ $5K/yr on a $60K account unlevered ≈ 1R/week at $100 risk |
| A3 5-min ORB TQQQ (0.1 ATR stop, hold to close) | IS +7.4 bps/day t 1.9 SR 0.71; OOS +3.6 bps/day t 0.7 SR 0.44; 80% of days stopped | not significant OOS; ≈ $110/week on $60K notional |
| A3 on QQQ | ≈ 0 both periods | dead |
| D1 volume-shock overnight (top-4 by volume/ADV20, $5+, $2M+/day) | gross +2 / +12 / −15 bps (T/V/T), net negative, hit 32% | dead at a 4-name book in our band |
| D2 large-loser next-day reversal (≤ −8% open→close on 2× volume) | net −105 / −113 / −9 bps | dead (the paper's "no-news" filter is not modeled — see queue) |
| D12 bottom-decile intraday-return reversal | net −174 / −93 / −137 bps | dead |
| overnight-continuation proxy, Pritamani-Singal continuation, 52-week-high breakout (daily) | all net negative on all splits | dead |
| bf_zero2 26 family configs on ≥5%-range days (F1–F10) | best +0.003/+0.007R per trade TRAIN, fail VAL; gross means ≤ +0.18R | dead (research/bf_zero2/REPORT.md) |
| H-B2 RV monotonicity | with the causal floor, no RV pattern (all ≈ 0R); without it the inflated population shows a DECREASING pattern | the paper's monotonicity is not in our population |
| H-B1 / M8 stocks-in-play ORB, OOS 2025–26 on the liquid universe (the strongest paper claim, at its own 1 bp cost) | gross −0.12 / −0.25 / −0.17R per trade, hit 10%, stop-rate 89%, no relative-volume monotonicity, both sides negative | dead out of sample: the paper's 2016–23 in-sample edge did not survive 2025–26 |
| H-B5 / M20 end-of-day loser reversal (≤ −8% by 15:00, buy 15:30, MOC) | gross −0.7 / +40 / −3 bps, net negative in 2 of 3 splits; the winner control moves with it (a period effect, not reversal); deepest tail +20 bps gross < cost | dead at our costs; the untested tail the review hoped for is +20 bps gross |

## 4. The test design — master ids from `HYPOTHESES.md` (54 rows, 5 reviews, ~150 sources)
Read `HYPOTHESES.md` §2 for every rule's exact spec, prior, falsification rule and cost sensitivity; §3 is the priority order;
§4 is the honest dollar translation (index timing tops out at 1–3R/week at 2× on $60K; the only sleeves that could reach
10R/week are M22 at a breadth we cannot hold and the untested M20 tail). The queue below is §3 turned into executable steps.
Status codes: DONE (result in §3 above / the files named), RUNNING, TODO (write the script), SKIP (§3 "not worth testing").

| # | M-id | rule (one line) | store | status | script | falsification (pre-committed) |
|---|---|---|---|---|---|---|
| 1 | M6 | Zarattini SPY noise area, faithful | ETF1m | DONE — SPY dead OOS (0.6 bps/day, t 0.09); QQQ OOS +10.1 bps/day t 1.59 SR 0.99 | `test_zarattini_spy.py` → `zarattini_spy.md` | holdout < 3 bps/day or t < 1.5 → SPY fails; QQQ is marginal (t 1.6 on 2.7 yrs) |
| 2 | M1 M2 M3 M12 | last-30-min sign timing (4 variants) | ETF1m | DONE — negative every year 2016–2026 | `test_etf_intraday.py` A1 | success ≤ 52% or ≤ 1 bp/day → fails |
| 3 | M22 | volume-shock overnight, top-4 book | DAILY | DONE — gross +2/+12/−15 bps, net negative | `test_daily_hyps.py` D1 | top-decile close→open < +20 bps net → fails (it did) |
| 4 | M7 | 5-min ORB QQQ/TQQQ (0.1 ATR stop, hold to close) | ETF1m | DONE — TQQQ OOS +3.6 bps/day t 0.7; QQQ ≈ 0 | `test_etf_intraday.py` A3 | t < 2 OOS → fails |
| 5 | M9 | RV monotonicity of ORB/HOD P&L in our universe | RANGE1m (candidates_full) | DONE — no pattern with the causal floor | `test_rv_monotone.py` | non-monotone → fails |
| 6 | M36 M37 | large-loser / intraday-component reversal | DAILY | DONE — net −100..−170 bps (no VIX split, no news filter yet) | `test_daily_hyps.py` D2/D12 | as run: fails; TODO add the VIX-regime split (use SPY first-30-min realized vol tercile from ETF1m as the regime, VIX itself is not in our stores) and a news filter (Alpaca news API, prev 15:00→open) before calling M36 finally dead |
| 7 | M20 | end-of-day loser reversal, ≤ −8% at 15:00 → buy 15:30, sell MOC | RANGE1m + DAILY | DONE — FAILS: loser tail gross −0.7 / +40.2 / −3.2 bps (T/V/T), net −21 / +20 / −23 after a 20-bps entry cost; the WINNER control is +0.5 / +17.0 / −5.6, i.e. VAL's positive is a period-wide last-30-min drift, not reversal; the deepest bucket (−30..−15%) is +20 bps gross, below cost | `test_eod_reversal.py` → `eod_reversal.md` | < +40 bps net in either year → FAILED |
| 8 | M8 | stocks-in-play ORB OOS 2025–26, liquid universe | TOP20 | DONE — FAILS: 6,205 fills (16/day), gross −0.12/−0.25/−0.17R per trade (T/V/T), hit 10%, 89% stopped, no RV monotonicity (RV>30 bucket: +2.6R on 35 TEST trades, −0.08/−0.74 on TRAIN/VAL), long and short both negative, weekly −14..−23R at 1 bp | `test_stocks_in_play.py` → `stocks_in_play.md` | gross mean R ≤ 0 at RV ≥ 1 → FAILED |
| 9 | M16 | DONE (see RESULTS.md) — the single-stock gap table (measurement, not a strategy): P(close > open), mean open→close, P(same-day fill of the gap), P(half-fill) by gap size (2/5/10/20%+) × dollar-volume band × prior-day range | DAILY (+ RANGE1m for the intraday path) | TODO: `test_gap_table.py` — pure tabulation on `daily_panel.parquet` (gap = ret_on, path = ret_id, fill = low ≤ prev_close), 2025–26, all symbols ≥ $5, splits shown but nothing selected | 30 min to write, 3 min to run | none — it is a measurement; report the table and which cells (if any) have mean open→close ≥ +50 bps with n ≥ 200 in BOTH years |
| 10 | M18 | DONE — CONFIRMED as a veto (−60/−26/−20 bps open→10:30) — no-chase / delayed entry on prior-day attention names: mean open→10:30 and 09:30→09:35 returns of names whose PRIOR day was in the top decile of abs(return)×volume ratio | DAILY + RANGE1m | TODO: `test_open_fade.py` — select on day t−1 (causal), read day t's 1-min path from the two 1-min stores (loader in `test_eod_reversal.py`), report mean/median open→09:35, →10:00, →10:30, →close by dollar-volume band | 1 h | if mean open→10:30 > −20 bps (small caps) the fade is absent; if < −20 bps, the HOD/ORB books should not enter before 10:30 in these names (a veto, not a book) |
| 11 | M11 M12 | DONE — fails t≥2 (OOS +3.6 bps t 1.5) — SPY overnight premium (buy close, sell open) and the conditional reversal after a bottom-quintile intraday day | ETF1m | TODO: extend `test_etf_intraday.py` with a daily loop: overnight = 09:30 open / prior 15:59 close; unconditional and conditional on the prior day's open→close quintile; 1 bp/leg | 20 min | unconditional < 1 bp/night → dead (expected); conditional needs t ≥ 2 on 2016–26 AND positive 2024–26 |
| 12 | M5 M4 | DONE for M5 — dead (−3.1/−1.7 bps on gated days) — volatility/gamma-gated last-30-min timing | ETF1m (+ OI for M4, not available) | TODO for M5 only: re-run item 2 restricted to days whose SPY 09:30–10:00 realized vol (sum of squared 1-min returns) is in the top tercile of the trailing 250 days; M4 SKIP (no options OI) | 20 min | gated mean ≤ 1 bp/day or t < 2 → fails |
| 13 | M10 | DONE — dead (−21/−28 bps/day) — QQQ VWAP flip | ETF1m | TODO only if free: 1-min close vs running VWAP, flip on every cross, 1 bp/leg; count flips/day | 20 min | net < 0 (expected) |
| 14 | M29 | DONE — fails on TEST (+7.4/+6.7/−13.4 bps net) — large-cap overnight continuation: rank on the trailing-20-day mean overnight return, hold the top decile close→open | DAILY (≥ $50M/day as the large-cap proxy; no market cap in the panel) | TODO: `test_overnight_xs.py` — form the signal on days t−20..t−1, hold t's overnight; top-decile and top-4; 6 bps round trip | 30 min | < 5 bps/night or t < 2.5 → fails |
| 15 | M30 M31 M38 M39 M40 M41 | DONE — none passes (M41 flips on TEST) — cheap DAILY add-ons listed in HYPOTHESES.md §3 item 14 (read each row's spec) | DAILY | TODO in one script `test_daily_addons.py`, same book/cost conventions as `test_daily_hyps.py` | 2 h | each row's own criterion in §2 |
| 16 | M50 M44 M47 M48 | engineering on the HOD-break (RV-scaled stops, L1-imbalance filter, exit-timing diagnostics) | RANGE1m, MBP-1 | SKIP for this program: there is no HOD-break book to improve (research/bf_zero2/REPORT.md) | — | — |
| — | M13 M14 M15 M21 M23 M33 M42 M43 M45 M46 M49 M52 M53 M6b M19 | not worth testing (reasons in HYPOTHESES.md §3) | — | SKIP | — | — |

## 5. Decision rules (pre-committed; do not move them after seeing a result)
1. A hypothesis PASSES only if: TRAIN (or IS) mean net > 0 with t ≥ 2; VAL (or the first OOS year) mean net > 0; TEST (or the
   last OOS window) read once, positive, with weeks green ≥ 60% for book-style rules; AND the realistic prior in `HYPOTHESES.md`
   §2 is not contradicted by more than 1 SE. Fewer than 200 events in TRAIN = "underpowered", not a pass.
2. Count every cell you look at (variants, thresholds, splits). Report the count. Raise the VAL bar by one standard error of
   the weekly mean for every 10 cells that passed TRAIN.
3. Costs are never lowered to make something pass. If a result depends on the cost assumption, show both cost rows and say so.
4. Never select the universe with information from after the decision time (the ≥5%-range store is such a universe for
   anything except entries ≥ 5% above the open or prior-day-selected signals; the DAILY panel and the ETF store are clean).
5. A pass becomes a BOOK only after a live dry run reproduces the signals (the engine parity machinery exists:
   `scripts/hod_break_miss_audit.py` is the template) and the owner says go.

## 6. What to deliver
- One file `research/lit_review_2026/RESULTS.md`: the queue table above with every row's result filled in (n, gross, net, t,
  hit, weekly R, weeks green, worst week per split), the cell count, and a one-paragraph verdict per row in plain words.
- `git add` the `.md` and `.py` files only; commit with a message that names the rows tested; push. Data files stay ignored.
- One Telegram via `python3 scripts/send_telegram_alert.py "<text>"` (no bare '(' characters; under 30 lines): what passed,
  what failed, what is underpowered, the honest weekly-R translation from `HYPOTHESES.md` §4 for anything that passed.
- Never tell the owner a number that is not in a file.

## 7. Pitfalls already hit in this program (each cost hours)
- Nasdaq TEST symbols (ZVZZT, ZWZZT, ZXZZT…) are in the daily feed: drop `^Z[VWX]ZZ|^ZZ` and any one-day move beyond ±50%.
- `pd.read_csv` on 2 GB of candidates OOMs at 3 GB; use `usecols`, `chunksize`, float32, categories (see `pass2b.py`).
- A fixed `lookback_minutes=420` REST window loses the morning after 16:30 UTC; compute the window from the clock.
- Alpaca rejects a whole 200-symbol request if ONE symbol is not in its symbology (`^[A-Z]{1,5}(\.[A-Z]{1,2})?$`).
- `limit=10000` in `StockBarsRequest` truncates a month of minute bars; omit it and let the SDK page.
- The harness kills your shell when free memory is low; detach jobs with `setsid nohup … &` and poll the log.
- `pkill -f` with an unanchored pattern kills the shell that issued it.
- Two bash chains reading one log can race on a stale EXIT line; wait on the PROCESS (`pgrep`) or use a fresh log name.
- The universe file `research/bf_zero/universe.csv` is "(high−low)/low ≥ 5% that day" — a hindsight filter. Any signal
  evaluated on it must guarantee membership causally (entry ≥ 5% above the open) or use the DAILY panel instead.
- Never read `research/bf_zero/bars.db` or `pit_bars_1min.db`: deleted (thin one-publisher tape). `topup.db` is mixed: do not use.


## 8. State at 2026-09-16 08:10 UTC
All fifteen queue rows have been run (RESULTS.md). The only rows left in the program are the ones that need data we do not hold:
M25/M26/M27 (own-ticker news at the decision minute — an Alpaca news pull per symbol-day), M24 (halt reopens), M28 (IPO age and
float), M32 (closing-auction deviation, needs quotes), M4 (options open interest for the gamma gate), M44 (MBP-1 windows).
Nothing in the tested set clears the decision rules in section 5. Two usable findings, neither a book: the QQQ noise-band
sleeve (~1R/week, t 1.6, needs a dry run before belief) and the open-fade veto (do not buy prior-day attention names before
10:30). If the program continues, the honest order is: the news rows (M25/M26/M27) because news is the one input the review
found repeatedly decisive and we do not have it, then M24 halts.
