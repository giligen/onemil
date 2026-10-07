# PREREG — leveraged-crypto-ETF close rebalance flow in BTC spot (2026-10-07)

**Question.** Daily-reset leveraged BTC ETFs (2× since 2024-06: BITX, BITU; ETH: ETHU; 3× newly approved) must rebalance
into the 16:00 ET close in the direction of the day's move. Is there a tradeable 15:30→16:00 ET drift in BTC/USD spot in
the direction of the 09:30→15:30 ET move, that did NOT exist before the leveraged products traded?

**Data (free, no purchase).** BTC/USD 1-min bars from Alpaca's crypto data API (`alpaca.data.historical.CryptoHistoricalDataClient`,
no key needed) 2022-01-01 → 2026-10-06, cached under `research/crypto_rebalance/data/` (never data/cache.db). ETH/USD the
same. Check `df -h` ≥ 5 GB before fetching. Count LOST days (completeness gate ≥ 95 % of trading days, else VOID).
Sessions = NYSE trading days only (ETFs do not trade weekends); use `pandas_market_calendars` or the exchange calendar in
`trading/` if present.

**Rule (frozen before any number is read).**
- Signal day d: r_day = close(15:30 ET) / close(09:30 ET) − 1 on BTC/USD. Position = sign(r_day) from the 15:30 bar close
  to the 16:00 bar close (30 min). Flat otherwise. Size proxy variant: |r_day| ranked terciles.
- Primary period: 2024-06-04 → 2026-10-06 (2× products live). Placebo period: 2022-01-01 → 2024-06-03 (no leveraged BTC
  ETF): the SAME rule must be ≈ 0 there, else what we measure is plain intraday momentum, not flow.
- Placebo window: same rule for 11:30→12:00 (signal 09:30→11:30) in the primary period.
- Cost: report gross, at 5 bp round trip (futures/maker venue), and at 30 bp (Alpaca spot 15 bp maker per side). The pass
  bar is at **30 bp** (our venue): mean net > 0, t > 2.5 (iid AND day-clustered are the same here, one obs/day), same
  sign in both halves of the primary period, ex-top-5 % of days still > 0, placebo period |t| < 1.5.
- MDE: state the minimum mean per-day return detectable at t = 2.5 with n days; state the per-day gross mean against it.
- Also report: the first-hour variant (16:00→16:30? no — BTC keeps trading; measure the REVERSAL 16:00→16:30 as the
  flow-impact signature: a true flow effect reverts after the close; momentum does not).

**Outputs.** `research/crypto_rebalance/run.py` (one script, flags for the windows; verbose progress), `results.csv`
(per-day rows: date, r_day, ret_1530_1600, ret_1600_1630, period), `REPORT.md` ≤ 60 lines: the table (gross / 5 bp / 30 bp
× primary / placebo-period / placebo-window × all / terciles / halves / ex-top-5 %), the MDE, the completeness count, the
verdict against the bar, and its own caveats (bar-close executability: a 15:30 bar-close entry is the next bar's open).

**Cells.** This is cell 1 of this programme; every variant above is pre-declared, nothing else is read.

**Rules for the agent.** python via `bash scripts/research_run.sh -m 2500M python3 …`; never write outside
`research/crypto_rebalance/`; never touch config/.env/crontab/services/caches; no orders; no git. Budget ≤ 40 calls.
Return ≤ 120 words (verdict, the primary 30 bp line, the placebo lines, completeness). This task IS the owner's request.
