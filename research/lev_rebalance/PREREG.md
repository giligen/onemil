# PREREG — leveraged-ETF close-rebalance drift (lev_rebalance)

- Mechanism: a 2x (or −2x) single-stock ETF rebalances at the close in the direction of the underlying's daily move; the flow ≈ 2 × wrapper assets × |return|. Late-day price pressure in the direction of the day's move on the UNDERLYING.
- Universe: underlyings that have at least one leveraged wrapper in `trading/orb_asset_class.py` (grep for the lev-family sets / `underlying_anchor` / the offline map `data/research/orb_asset_class_map_20260711.csv`); point-in-time (research/scripts/pit_listings.py); price ≥ $5. Exclude `^Z[A-Z]ZZT$`.
- Data: cache.db `daily_bars` (ADV, prior close) and `intraday_bars_1min` (timestamps UTC; 15:00 ET = 19:00 UTC in EDT, 20:00 in EST — handle DST with zoneinfo). Report coverage of underlying-days that have 1-min bars (≥ 80% rail, winner/loser missingness gap ≤ 5 pp).
- Signal at 15:00 ET (known by then): r = close_15:00 ÷ prior close − 1, |r| ≥ 5%; flow proxy F = 2 × |r| × (sum over the name's wrappers of wrapper dollar-ADV20) ÷ underlying dollar-ADV20 (all from daily_bars, prior 20 sessions). Rank by F each day; take F ≥ the TRAIN median of F among |r| ≥ 5% days (state the number); cells: 1,291 LONG on r > 0, 1,292 SHORT on r < 0 (Reg SHO: if r ≤ −10% the short needs an uptick — model fill = the 15:01 bar's open only if that open > that bar's low, else skip; report the share), 1,293 combined.
- Entry: the 15:01 bar's OPEN (obtainable), charged the 15:00 minute-of-day half-spread (grep research/mature_method for the hs table). Exit: MOC at the official close (daily_bars.close; zero spread). Stop: none inside the window (report the P&L distribution instead; also report a variant with a 2% stop as a diagnostic, counted as cell 1,294). R for the cadence scorer = 2% of entry price (a fixed unit, state it).
- Control: same rule, same |r| ≥ 5% and same clock, on movers WITHOUT any wrapper, matched by day and |r| bucket. Claim = signal − control.
- Splits: TRAIN 2025 (both halves), VAL 2026-01-01..05-31, TEST ≥ 2026-06-01 SEALED. Note: wrappers listed after 2026-04-04 were backfilled on 9/5 (`scripts/backfill_wrapper_universe.py`) — say whether the wrapper set is point-in-time (a wrapper counts only from its listing date; check via pit_listings or the wrapper's first daily bar).
- Book: 5 concurrent, 1% of $66K risk per position with R = 2% of price (so ≈ $33K notional per position → cap notional at 1× equity total; report positions/day and the unlimited per-trade stats).

PASS BAR (VAL): signal net ≥ +0.10 R (= +0.20% of price) with day-clustered t ≥ 2; signal − control ≥ +0.10 R with t ≥ 2; TRAIN halves same-signed; ≥ 3 entries/week; cadence bar C1–C5 pass (run `python scripts/cadence_bar.py --trades <csv> --split VAL` and `--split TRAIN`, columns date,pnl_R,symbol; paste both). Diagnostics: gross, cost, WR, ex-top-1%/5%, top-5 share, MDD, F-quintile monotonicity (does higher F mean more late-day drift?), the 15:00→15:30 vs 15:30→close split of the move, MDE beside every null, iid and clustered SE.

The ONE caveat that alone could explain the headline; any mid-run change recorded.

## Pass 2 — data pull, pre-registered before rescoring

Pass 1 was VOID on availability: `cache.db intraday_bars_1min` covered only 12.3% of
candidate underlying-days and only TSLA/MSTR/NVDA were tried (the FAMILIES scope cut in
FREEZE.md #1). This pass fixes availability by pulling the missing bars and running the
SAME scorer (same PREREG above, unchanged) over the FULL wrapper-underlying universe.

- **Pull filter (superset of the signal)**: `|daily high / prior_close - 1| >= 5% OR
  |daily low / prior_close - 1| >= 5%` from `daily_bars` — any day whose price could have
  been >=5% away from prior close at 15:00 gets an intraday pull, whether or not the
  CLOSE ended up >=5% away. The signal itself is still computed at 15:00 from the pulled
  bars exactly as PREREG specifies (`r = close_15:00/prior_close - 1`, `|r| >= 5%`) — the
  pull filter only decides what gets FETCHED, never what qualifies as a trade.
- **Universe**: every underlying resolvable from the full wrapper set in
  `data/research/orb_asset_class_map_20260711.csv` (asset_class='wrapper', 6,136 rows) via
  `trading/orb_asset_class.py::underlying_anchor` (parses each wrapper's fund name,
  validates the anchor token against the class map's STOCK rows) — generalizing pass 1's
  FAMILIES-only 3-name list to every complex the classifier can resolve. Point-in-time
  wrapper listing: a wrapper's ADV contributes to F only from its first `daily_bars` row
  (same proxy as pass 1 FREEZE.md #4 — `pit_listings` still lacks exact IPO dates).
  Price >= $5 (prior close). Test tickers excluded (`is_test_ticker`).
- **Data pull**: 1-min bars 14:55-16:00 ET for every (underlying, candidate day) not
  already in `cache.db` (read-only), fetched from Alpaca SIP via
  `AlpacaClient.get_1min_bars_range_multi`, batched per day, written to a NEW sqlite db
  owned by this study (`research/lev_rebalance/bars_1500_1600.db`) — `cache.db` is never
  written. Control pool sized up from pass 1's 60 symbols to 400 (the pull is cheap;
  more power), same non-wrapper `stock`-class sampling, same seed family.
- Everything else — signal/control definition, cost model, MOC exit, 2% stop diagnostic,
  SHO uptick gate, splits, cadence bar, book — is UNCHANGED from the PREREG above. The
  ONLY change from `run_study.py` to `run_study2.py` is the bars source (merged: this
  study's own pulled db, falling back to `cache.db` where it already had the window) and
  the universe (full wrapper map instead of TSLA/MSTR/NVDA). No analysis choice was
  tuned after any result was read; the F-threshold rule (TRAIN median) and the pass bar
  are exactly as stated above.
- **Process note**: the data-engineering steps (universe build, Alpaca pull) ran before
  this section was committed, but no P&L or signal-vs-control comparison was computed or
  viewed before this text was written and committed — only population sizes (row counts,
  coverage %) were seen, which this section already discloses.
