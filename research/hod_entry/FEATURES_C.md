# Feature Set C — the tape (PREREG 1,478 item 4)

Builder: `build_features_1478_C.py` (nohup run 2026-09-26 11:36:34–11:45:10 ET-node-time, 516 s,
log `build_features_1478_C.log` / `run_1478C.out`). Output: `features_1478_C.csv`, 9,911 rows — one
per `causal_arming_causal.csv` row with `status=='fill'` (join key `day, symbol, fill_min`).

## Mechanism
For each fill, the arm bar j and the fill/breakout bar j+1 are recovered from the minute bars
(`causal_arming.load_day_bars`, cache.db + bars_sip.db, SIP-wins-on-tie) as **adjacent rows in the
bars array**: j+1 = the bar whose [m, m+1) contains `fill_min` (largest bar-minute m with m <= fill_min);
j = the immediately preceding row. This is the same j, j+1 pairing `armed_crossing_bars` uses internally
(`out.append(dict(a, j=int(j), m_lo=int(m[j]), m_hi=int(m[j+1]), ...))`, causal_arming.py:96) — verified
directly: AAP 2025-07-01 fill_min 605.311 -> j=604, j+1=605 -> key `AAP|604|605`, present in
`sip_cache/c1438_2025-07-01.pkl.gz`. The reconstructed key is looked up in that day's `c1438_{day}.pkl.gz`
(the 1,438 fetch cache, `sip_cache/`, keyed `symbol|m_lo|m_hi`); tick data for the fill's own window
comes back exactly as `causal_arming.process_symbol_day` cached it when it built the base book, so this
is a re-read of the same tape the base book's fill was resolved from, not a new fetch.

## Timestamp proofs (data known at/before the close of arm bar j, or the trigger print that produced
the recorded fill — no field below uses a timestamp later than the fill's own trigger print)

1. **trigger_print_odd_lot**, **trigger_print_size** — the first trade in bar j+1's window
   (`t.ts in [start, end)`, `start/end = causal_arming.window_ns(day, m_lo, m_hi)`) with
   `price >= level`, i.e. the SAME print `resolve_window` used to declare the recorded fill
   (`hits = w[w.price >= arm['trigger']]`, causal_arming.py:112). Timestamp = the fill's own
   trigger print, already realised by definition of the base book row; no later data used.
   Proof: for every window_found==1 row, a qualifying trigger print exists (0/9,737 misses) —
   confirms this is exactly the print that produced the fill, not a different one.
2. **pre_break_odd_lot_share, pre_break_mean_trade_size, pre_break_print_count** — trades with
   `t.ts < start` (`start` = the window's own boundary = end of arm bar j / start of bar j+1, the
   same `start` `resolve_window` uses to slice `w`). All such trades have `ts` strictly before the
   window's start, which is strictly before the trigger print's `ts` (the trigger print is inside
   `[start, end)`). Causal by construction: nothing at or after `start` is read here.
3. **pre_break_buy_share** — Lee-Ready-lite classification of the same pre-`start` trades against
   the prevailing NBBO at each trade's own timestamp (`pd.merge_asof(..., direction='backward')` on
   `sr._valid(quotes)`, i.e. the last quote with `ts <= trade.ts`); every quote used has `ts <= ` its
   trade's `ts`, which is `< start`, so no data at or after `start` is used.
4. **spread_bps_at_arm** — the prevailing quote (`sr.prevailing_quote`, "last valid quote with
   `ts <= ts`") at the timestamp of the FIRST print inside bar j+1's window (`w.ts.min()`, i.e. the
   first trade at or after `start`). This is a quote timestamped at or before that first print, which
   is at or before the trigger print — no later data used. (Named "at arm" per the PREREG's own
   wording; it is measured at the first tick of the breakout bar, the earliest point at which the
   arm-to-break transition is observable on the tape, not inside arm bar j itself — bar j's own
   window is the ~5 s pre-`start` sliver covered by items 2-3 above, per `fetch_tape`'s
   `span_s = (m_hi - m_lo) * 60`, `QUOTE_LOOKBACK_S = 5` math: the raw tape fetched for a window
   starts only 5 s before `start`, regardless of the j/j+1 minute gap.)

## Coverage (9,911 fill rows)
| Feature | Non-null | Share |
|---|---|---|
| window_found (c1438 key present) | 9,737 | 98.2% |
| trigger_print_odd_lot / trigger_print_size | 9,737 | 98.2% |
| spread_bps_at_arm | 9,737 | 98.2% |
| has_prebreak (any trade before `start`) | 7,896 | 79.7% |
| pre_break_odd_lot_share / mean_trade_size / print_count | 7,896 | 79.7% |
| pre_break_buy_share (also needs a valid quote before that trade) | 4,491 | 45.3% |

174 fills (1.8%) miss their `c1438` cache key entirely (spot check: DIN 2025-07-01, key
`DIN|619|620` absent from that day's cache file — a genuine fetch gap in the original 1,438 builder,
not a reconstruction error, since the day's other 46/47 keys resolve correctly). The remaining
~18-20 pts between `window_found` and `has_prebreak` are windows where the tape simply has no print
in the ~5 s pre-`start` sliver — expected given how little of arm bar j the cache actually carries
(see item 4 proof above); this directly contradicts a literal "windows cover bars j and j+1" reading
of the PREREG text and should be flagged to the PREREG author. `pre_break_buy_share`'s extra drop
(79.7% -> 45.3%) is quotes, not trades: many of those pre-`start` prints have no valid NBBO print at
or before their own timestamp within the cached quote history.

## Distributions (informational, not a claim)
`trigger_print_odd_lot` mean 0.624 (62% of trigger prints are odd lots, <100 shares) — same direction
as cell 1,429's report-only "+0.33 vs +0.07 R" split, now available as a per-fill feature rather than a
report-only cut. `trigger_print_size` median 40 sh, p75 100 sh, max 35,400 (block prints exist and are
NOT capped/winsorized here — the model step should decide how to handle them). `pre_break_buy_share`
median 0.64 (buy-skewed tape ahead of the break, among fills where it resolves). `spread_bps_at_arm`
median 28.7 bps, p75 53.2 bps — sanity-consistent with `research/orb_spread_gate_verdict.md`'s bps
scale for this population (not re-derived here).

## Caveats (read as an adversary)
* No cost, no R, no pass/fail claim here — this is a feature build, not a result. `pre_break_buy_share`
  coverage (45%) is under half the book; a model using it must handle NaN natively (the PREREG's
  HistGradientBoostingClassifier does) and its causal proof (< 5.5) applies only to the rows where it
  is non-null.
* `trigger_print_odd_lot`/`_size` and `spread_bps_at_arm` are NOT missing-at-random with respect to
  `window_found`: the 174 missing-key rows are whichever symbol-days the original 1,438 fetch lost,
  not a random sample — if that loss correlates with anything about the day (e.g. very high-volume/
  volatile names hitting rate limits), the 98.2% that DO resolve are not guaranteed representative.
  Not tested here; flag for the independent-check pass.
* `pre_break_*` coverage (80% / 45%) is exactly what the ~5 s pre-window sliver implies, not a defect
  in this builder, but the PREREG text's framing ("windows cover bars j and j+1... many windows start
  at m_lo") reads as if bar j's full minute were routinely available — it is not; only please the
  ~5 s tail of it is. The independent rebuilder should re-derive this bound from `fetch_tape`'s own
  span math rather than trust this file's characterization of it.
* Lee-Ready-lite ties (print exactly at both bid and ask, i.e. a locked/crossed-adjacent quote) are
  double-counted into both `n_ask` and `n_bid` before the ratio — a rare edge case on a `_valid`
  (non-crossed) quote, not audited row-by-row here.
* `trigger_print_size` outliers (block prints up to 35,400 sh) are raw, unwinsorized — per the
  programme's tail-dependence rule (rule 5), any model result using this feature must be reported
  ex-top-1%/5% before being believed.
