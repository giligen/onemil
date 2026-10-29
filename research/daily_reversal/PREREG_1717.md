# PREREG — cells 1,717–1,719: one-day cross-sectional reversal on liquid US stocks (Nagel 2012 liquidity provision; overnight/intraday decomposition after Lou-Polk-Skouras 2019 / Aboody et al. 2018). FROZEN 2026-10-29 23:05 UTC (Fable), before any number.

Owner's ask 10/29 evening: "mine different strat" / "check the new ideas". Selection rule (pre-committed): untested in the
programme (grep: no daily-reversal or gap-fade cell exists; 1,713 was the WEEKLY reversal, rejected on VAL 2020–22 with
the gross premium gone, +0.13 %/mo), data on disk, a published mechanism, DAILY events (power — the monthly sleeves'
MDE of 1.4 %/mo was the problem), executable long on Alpaca (short leg reported, long leg must stand alone), fills at
auctions (no spread assumption). Programme count 1,716 → 1,719.

## Mechanism (published)
Nagel (2012, RFS) "Evaporating liquidity": the 1-day/1-week reversal in liquid stocks is the return to liquidity
provision; it is largest at the ONE-DAY horizon and in high-VIX regimes. Lou, Polk, Skouras (2019, JFE): overnight and
intraday return components behave differently (overnight persists, intraday reverses); Aboody, Even-Tov, Lehavy,
Trueman (2018, JFQA): a large overnight return is firm-specific sentiment and reverses. Hence the decomposition cells.
Why the weekly rejection does not close this: most of the 5-day reversal premium accrues in days 1–2, and 1,713 charged
20 bps RT per week on a 5-day hold; at one day the turnover is 5× but so is the event count (power), and the cost per
hold is lower at the auctions.

## Data (on disk)
`research/momentum_weekly/panel_2016_2026.parquet` through the SAME loader and screens as `research/sleeves/fiftytwo_wh.py`
(`load_panel`, asset-map + fund-name + name-pattern exclusions, `SEAL_END`): raw Alpaca daily bars (open/high/low/close/
volume). Splits: TRAIN 2016-01 .. 2019-12; VAL 2020-01 .. 2022-12; TEST 2023+ SEALED (the loader refuses it).
Universe each session t: top 1,000 by 20-session median dollar volume, close ≥ $10, ≥ 15 of the last 20 bars present.
Report per split: sessions, mean universe size, share of picked name-days with no exit price (counted 0, as 1,713).
Price scale: the panel is raw — exclude from the SIGNAL any name whose |1-day close move| > 50 % (split/corporate action;
causal: known at the close) and report the count; apply the 1,713 price-scale table (+500 % / −80 %) on picks.

## Cells (signal at the close of session t; ENTRY at the official open of t+1 by MOO — obtainable: the signal is known at
16:00, orders rest before 09:28 — EXIT primary = official open of t+2 (MOO); secondary = official close of t+1 (MOC))
All signals are market-adjusted: x_i − mean over the universe that day.
* **1,717 — close-to-close reversal:** x = close_t / close_{t−1} − 1. Long the lowest decile (100 names), short the
  highest decile, equal weight, 1x long / 1x short. Report long-only and short-only legs separately.
* **1,718a — overnight-component reversal:** x = open_t / close_{t−1} − 1 (long low, short high).
* **1,718b — intraday-component reversal:** x = close_t / open_t − 1 (long low, short high).
* **1,719 — report-only:** decile table x → mean next-day return (monotone = the mechanism) for each signal; per-year
  mean and t; by 20-day realised-vol tercile of the universe EW return (Nagel's VIX state); top-500 universe
  sensitivity; quintile instead of decile; the 1,717 book ex names in the 1,718a top/bottom decile (overlap).
Costs: auctions, no quoted spread; charge 5 bps per side at every entry/exit (the opening auction's price impact for
names ranked 500–1,000 by $vol; placeholder, not a measured NBBO — stated as such), sensitivity 3 and 10 bps/side.
Turnover is ~100 %/day so the hurdle is ~10 bps/day on the gross — report gross beside net always.

## Pass bar (frozen; VAL; each cell; days are the units, one obs per session)
mean net long-short ≥ +5 bps/day AND iid t ≥ 2.0 (also year-block bootstrap 95 % CI, reported) AND ex-top-5 % of days > 0
AND winner-capped (+2 %/day) > 0 AND the LONG leg alone net > 0 (the executable leg) AND TRAIN same sign with t ≥ 1 AND
≥ 3 of 4 TRAIN years positive AND green-week share ≥ 55 % and above a count-matched random-decile null (1,000 draws,
same universe, random 100/100 names each day). Cadence bar (`docs/cadence_bar.md`): R := the VAL sd of daily net LS
return; weekly table (P10, green share, strong weeks ≥ +5R, gap between them). Power: sd(daily decile LS) ≈ 1.2 % →
se ≈ 4.4 bps on ~755 VAL days → MDE ≈ 9 bps/day at t 2 — detectable, and above the cost hurdle, so a null here is a
detected absence at the tradable size.
Undetermined: net > 0 but t < 2 → reported, no ladder. Reject: net ≤ 0 or ex-top-5 % ≤ 0 or long leg ≤ 0.

## Ladder (pre-committed)
Pass → independent rebuild from this prose by an agent that has not seen `cell_1717.py`, compared day by day (date,
mean LS return) and on picks (date, symbol) → TEST read once beside VAL → paper 20 sessions (account chosen then; MOO at
09:28, 100 names long only at $20K, no short leg on paper) → live $10K long-only after 20 clean paper sessions →
short leg only after the borrow probe (ETB share of the top decile on a real day). Kill: 20 sessions realized < 0 or one
session < −3 %. Nothing to the owner before the rebuild.

## Builder contract (Sonnet; one agent; ≤ 40 calls; no network, no Databento, no writes outside `research/daily_reversal/`)
Script `research/daily_reversal/cell_1717.py` (argparse: --split train|val, --cost-bps 5), vectorised on the [D, S] arrays
(no per-name loops over days); writes `RESULT_1717.md` (sections: data check incl. exclusions and universe size, per-cell
per-split table with EVERY pass-bar item, gross/net, legs, decile tables, per-year, vol-tercile, weekly table, sensitivities,
caveats read as an adversary) and `daily_1717_<cell>_<split>.csv` (date, n_long, n_short, long_ret, short_ret, ls_gross,
ls_net) plus `picks_1717_<cell>_<split>.parquet`. TEST must not be run or read. Log to `cache/run_1717.log`.
