# LENS B — Is the cell 1,427 fill obtainable? (adversarial review)

**Claim under review:** buy-stop-limit (trigger = level + $0.01, limit = level x 1.0015) filled at the
prevailing SIP NBBO ask at the first consolidated print >= trigger, TEST +0.330R net (n=972 fills,
28-34% fill rate, ~34 fills/wk). Source: `research/hod_entry/sip_rebuild.py::simulate_entry`,
`research/hod_entry/sip_rebuild_test.csv`.

**Method.** Sampled n=230 (seed 42) of the 972 TEST `status=='fill'` rows. For each, re-fetched Alpaca
SIP quotes (`feed='sip'`, window [t_hit-5s, t_hit+1.1s]) **with size** (`as`/`bs`) — a field
`sip_rebuild.py`'s own cache builder (`fetch_tape`, `_fetch_quotes`) never stores; it keeps only
`bp`/`ap`. 216/230 returned usable data (7 network-lost, 7 no quote in the re-fetch window — both
counted as attrition, not as evidence either way). **Sanity check: the re-fetched ask at t_hit matched
the CSV's recorded `ask_at` in 216/216 (100%, <0.5c), so the re-fetch pipeline reproduces exactly the
quote the frozen sim used** — the findings below are about that same quote, not a different one.
Script: not committed (per instructions), ran from `/tmp/.../scratchpad/lensb.py`; raw per-row output
in `/tmp/.../scratchpad/lensb_results.csv` (230 rows) if it needs to be re-run.

## Finding 1 — the model has NO size check at all. This is a spec gap, not just a friction cost.

`simulate_entry` (sip_rebuild.py:340-364) fills the **entire** order at `ask` the instant a qualifying
print arrives, with no reference to displayed size anywhere in the module (grep confirms: no `as`/`bs`
field is even fetched). A real stop-limit becomes a **limit order at the ask**, filled only up to the
displayed size; the rest waits or walks the book. Sized against the position the book would actually
buy (`shares = risk / (fill - stop)`, using the CSV's own `R` = fill-stop column as the risk denominator):

| Risk | median shares needed | median displayed ask size | **% of sampled fills where the ask alone can't cover the size** |
|---|---|---|---|
| $100 | 141 | 250 | **23.1% (50/216)** |
| $375 | 528 | 250 | **58.8% (127/216)** |

At $375 risk, a **majority** of the sampled fills are oversized for the touch. Worst case in the
sample: ZLAB 2026-08-07 10:25 ET, level 23.10, displayed ask 23.09 x 200 shares, but $375/R($0.20) =
1,875 shares needed — the order would need to walk **9x** the displayed size.

## Finding 2 — the quote used is not durable; real order latency alone kills ~1/3 of the "fills."

A real order has ~100-300ms of wire+matching latency between the triggering print and the exchange
book seeing the order. Recomputing the ask **250ms after** the first qualifying print:

- **32.4% (70/216)** of sampled fills already show `ask(250ms) > limit` — i.e. **would not have filled
  at all** at 250ms of latency, vs the model's assumption of zero latency.
- Of the 44.0% (95/216) of fills where the ask crosses above the limit at some point within the 1.1s
  window after the trigger, the median time-to-cross is **3.8ms** and 76/216 (35.2%) cross within 250ms
  — the window in which the limit stays fillable is often single-digit milliseconds, not the ~1s a
  human/API round-trip would need.
- **15.7% (34/216)** of sampled fills show the ask exceeding the limit **under 1ms** after the quote
  the sim used — i.e. the NBBO was already flapping between venues at the exact instant of the fill;
  the sim's "prevailing ask at t_hit" is a coin flip between two SIP-merged quotes a microsecond apart,
  not a stable, standing offer an order could actually reach.

**Revised mean raw R** (same `exit_price`/`stop` as the frozen run — this changes only the entry leg;
target-dependent exits that a worse fill might have missed are NOT re-walked, so this is a
**lower bound on the damage**, flagged explicitly):

| Variant | n | mean raw R | delta vs modeled |
|---|---|---|---|
| Modeled fill (as reported), this subsample | 216 | **+0.505** | — |
| (i) Latency only: fill @ ask +250ms, subset still fillable | 146/216 (67.6%) | +0.539 (orig fill on same subset: +0.555) | **-0.016** on survivors |
| (ii) Size only ($100 risk): walk to limit when size insufficient | 216 | +0.481 | **-0.024** |
| (ii) Size only ($375 risk): walk to limit when size insufficient | 216 | +0.457 | **-0.048** |
| (iii) Combined: fill @ +250ms, then size check at $100 risk | 146/216 | +0.520 (same-subset orig: +0.555) | **-0.035** |

**The per-fill R damage is modest (-0.02 to -0.05R) — the real hit is to volume, not edge sign.** 32.4%
of nominal fills don't survive 250ms of latency; naively scaled to the full TEST population that implies
the honestly obtainable fill rate is closer to **19-23%** of signals than the reported 28-34%, and
~34 fills/week is closer to **~23/week** — this is an extrapolation from a 216-row sample at one
latency assumption (250ms), not a re-run of the full book, and should be treated as an estimate, not
a number to publish as-is. Note also: the excluded (missed) trades average a *lower* modeled R
(+0.401) than the survivors (+0.555) — the trades an execution-latency haircut removes are not
obviously the best ones, so this isn't disguising a worse book, but it is a real cut to trade
frequency, which is a cadence-bar input (`docs/cadence_bar.md`: >=3 fills/week at the LIVE config).

## Finding 3 — the triggering print itself is often not a print a market-maker's depth reflects.

Checked the print that satisfies `price >= trigger` (from the already-cached trades tape, no
re-fetch): size is the *trade* size, not book depth, but its thinness is still informative about how
representative the "signal instant" is.

- Median trigger-print size: **33.5 shares**; only **25.5%** are round lots (size >= 100 and a
  multiple of 100); **63.9%** are under 100 shares.
- **14.4% (31/216)** trigger off a literal **1-share** print (e.g. NNE 2026-06-01, VSH 2026-06-01, LITE
  2026-06-01 at $886.60).
- **6.0% (13/216)** are sub-penny prints (e.g. SATS 2026-06-11 trig 123.9481, FSLR 2026-08-03 trig
  229.1750) — non-standard increments typically carrying odd-lot/average-price/cross conditions that
  are frequently excluded from trade-through/NBBO-quality analysis; this rebuild keeps them as valid
  triggers with no condition-code filter (conditions were never fetched — `fetch_tape` drops `c`).
  Under Reg NMS a stop legitimately triggers off any consolidated print regardless of lot size, so this
  is not a correctness bug in the trigger logic, but it does mean a meaningful minority of "signal
  instants" are single micro-prints rather than a liquid crossing, which is consistent with Finding 2's
  observation that the quote at that instant is often unstable.

## Bottom line

The +0.285/+0.238/+0.330R headline is not fabricated, but it is priced off a **zero-latency,
infinite-depth NBBO fill** with no size check anywhere in `simulate_entry`. Empirically, against
resting-order economics that any live implementation must actually clear:
- **~1/3 of nominal fills don't survive one order round-trip of latency** (250ms) — this is the
  dominant obstacle, not per-fill slippage.
- **23-59% of surviving fills (risk-size dependent) are oversized for the displayed touch** and would
  walk the book or partial-fill.
- Per-fill raw R degrades a modest **-0.02 to -0.05R** under these corrections, so the edge's *sign*
  survives every variant tested here — but the **obtainable fill rate is materially below 28-34%**,
  which is a direct cadence-bar and $/month input, not a cosmetic one.

**This does not "lock" the strategy as specced.** Before this cell is treated as live-ready, the shared
BT/live entry helper needs an explicit latency haircut (>=250ms) and a size-walk rule fed by real
`as`/`bs` quote data (which the SIP-quote fetcher must start storing — it currently discards it), and
the TEST number needs a re-run under those rules, not a discount applied post hoc to the current
headline.

## Provenance
- Population: `research/hod_entry/sip_rebuild_test.csv` (3,522 signals; 972 fill / 2,525 nofill / 24
  no_tape).
- Sample: n=230, `random.seed(42)`, `random.sample(range(len(fills)), 230)` over the 972-row fill
  frame; 216 usable (7 `LOST_REFETCH`, 7 `NO_QUOTE_IN_WINDOW`).
- Re-fetch: `alpaca.data.historical.StockHistoricalDataClient.get_stock_quotes`, `feed='sip'`,
  per-signal window `[t_hit - 5s, t_hit + 1.1s]`, fields `t, bp, ap, bs, as`.
- Trigger-print size/price: read from the already-fetched cache
  `research/hod_entry/sip_cache/{day}.pkl.gz` (trades leg), no re-fetch.
- All aggregate numbers above are computed directly from `/tmp/.../scratchpad/lensb_results.csv`
  (230 rows, not committed — regenerate with the script left in the same scratchpad dir if needed).
- `R_dollar` used for `shares = risk / R_dollar` is the CSV's own `R` column (`sip_rebuild.py`'s
  `trade_result()` defines it as `fill - stop`, confirmed by reading the function).
