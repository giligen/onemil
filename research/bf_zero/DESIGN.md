# Bull-flag from zero — pre-registered design (2026-09-13, written before any pattern was scanned)

Owner (9/13): "BF is profitable only because of the 6312, so this is BS! And so few trades is
also BS! Start from scratch, read the literature, remove the filters, the quartiles, everything,
and re-build this in the BT in a way that generates 5+ trades per week and a 1:2 R:R and profit."
Then: "start from scratch, no gaps, nothing, clean sheet."

## What "clean sheet" means here
- No inherited filters: no gap requirement, no 20%/10% mover threshold, no conviction score, no
  MACD tiers, no two-tier composite, no quintiles, no regime multiplier, no VWAP gate, no price
  cap, no pole minimum carried over. Every one of those, if it reappears, reappears as a
  HYPOTHESIS scored below, never as a precondition.
- The universe is the whole market, point-in-time (Databento EQUS daily, delisted names
  included), not the cache universe and not the old scanner's list.
- The only preconditions are TRADABILITY, stated here and tested for sensitivity:
  price ≥ $1 at the open (sub-dollar names are not tradable at size), prior-20-day average
  volume ≥ 100K shares (causal: prior days only; sensitivity at 50K / 200K reported), and the
  day's range ≥ 5% (an exact mathematical superset — a 5% pole cannot exist on a day whose
  range is under 5% — not a selection; the 3% floor would triple the data and is deferred).
- Bars: 1-min RTH from cache.db + the two point-in-time side DBs + a new Databento fetch for
  the 289,575 symbol-days none of them held (`missing_keys.csv`, `bars.db`). Universe:
  `universe.csv` (647,796 symbol-days, 2025-01-02 → 2026-09-11).

## Literature, read 9/13 (what it says, and what it implies for the design)
- Zarattini, Barbon, Aziz (SSRN 4729284, rev. Apr 2025): 5-min ORB across 7,000 US stocks
  2016–2023 — profitable ONLY on "stocks in play" (abnormal relative volume / news); across all
  stocks the raw breakout is not. → Relative volume at the moment of entry is a first-class
  hypothesis (H-RV), not a filter to remove.
- Gao, Han, Li, Zhou (JFE 2018, "Market intraday momentum"): first-half-hour return predicts
  the last half-hour; small stocks CONTINUE intraday, large stocks half-reverse. → Opening drive
  continuation in small caps has an academic basis (H-DRIVE).
- Bulkowski, flags (updated 1,028-trade study): average rise 39%, failure 15%, far below the
  retail claims of 67–85% win rates; "high-tight" flags do best. → Flag TIGHTNESS (retrace ≤ 50%,
  low-range consolidation) is a hypothesis (H-TIGHT); win-rate claims are not evidence.
- Lottery-stock literature (Bali/Cakici/Whitelaw; Kumar): positively skewed small caps are
  overpriced on average — the average momentum trade in this universe should be ≤ 0. → The
  ex-tail metric is mandatory; a book that is positive only through its skew is the ignition
  book again.
- Cameron (Warrior Trading): pole, 2–3 red-candle pullback retracing ≤ 50%, above VWAP, entry
  on the first candle to make a new high, stop just below the pullback low, target the retest of
  the high and beyond; "micro pullback" = 1–3 bar pause. → These are the F1/F2 definitions.

## Entry families (each yields at most ONE trade per symbol per day per family-config, the
## first qualifying event; all quantities computed on bars ≤ the entry bar)
- F1 Bull flag (Cameron): pole = close-to-high gain ≥ P% within ≤ 8 bars; flag = 2–6 bars,
  each high ≤ prior high or within 0.2%, retrace ≤ 50% of the pole (H-TIGHT variant: ≤ 33%);
  entry = first bar whose high exceeds the flag's high, at that level × (1 + slip); stop = flag
  low. P ∈ {5, 8, 12}.
- F2 Micro pullback: same pole; 1–3 bar pause (no new high); entry = break of the prior bar's
  high; stop = pause low. P ∈ {5, 8, 12}.
- F3 Opening drive first pullback: ≥ P% from the 9:30 open by minute M (M ∈ {5, 15, 30}); first
  pullback of ≥ 2 bars; entry = break of the prior bar high; stop = pullback low.
- F4 VWAP bounce: after being ≥ 5% above VWAP, price touches within 1% of VWAP and the next
  bar's high breaks the touch bar's high; stop = touch bar low.
- F5 High-of-day break after consolidation: ≥ K bars (K ∈ {5, 10, 15}) all within X% of the
  day's high (X ∈ {2, 4}); entry = HOD break; stop = consolidation low.
- F6 Red-to-green: opened below the prior close; entry = first cross above the prior close;
  stop = the day's low so far.
- F7 Premarket-high break (needs PM bars; coverage reported): first break of the PM high after
  9:30; stop = the low since 9:30.
- F8 Opening range break at 15 and 30 min (the 5-min ORB exists as a book; 15/30 are new):
  entry = break of the range high; stop = range low.
- Entry-time windows as a hypothesis: 9:30–10:30 / 10:30–12:00 / 12:00–15:30.

## Exits (applied to every entry; the owner's 1:2 is E1)
- E1 Fixed target +2R, stop −1R. Target fills only if the bar HIGH trades through target × 1.002
  (no touch-fills); stop fills at min(stop, bar open) × 0.999 (gap-through modeled).
- E2 Partial 50% at +2R, stop to entry, remainder trails the higher-low structure (the P1 exit).
- E3 Fixed target +3R (1:3).
- E4 Hold to 15:55 with the −1R stop (control: what the pattern's continuation is worth).
- Entry slip +0.3% (sensitivity +0.6% — live BF entries have run worse than the model).
- Force flat 15:55. No re-entry in a symbol the same day within a family.

## Hypotheses scored on top of every family (as splits, never as preconditions)
H-RV relative volume at entry (cum volume to entry ÷ same-clock 20-day average… when
unavailable: bar volume ÷ prior-bars mean) — the Zarattini "in play" idea.
H-DRIVE distance from the open at entry (0–5 / 5–15 / 15–30 / 30+%).
H-TIGHT flag retrace ≤ 33% vs 33–50%.
H-VWAP entry above vs below VWAP.
H-PRICE $1–2 / 2–5 / 5–10 / 10–20 / 20+.
H-RMIN stop distance as % of price: < 1 / 1–3 / 3–6 / 6+ (spread and noise).
H-TIME the entry-time window.
H-DAY2 prior-day range ≥ 10% vs not (continuation vs fresh).
H-GAP open gap ≤ 0 / 0–5 / 5–15 / 15+ (the old gap rule, now a split).
H-VOL prior-20-day ADV 100–500K / 500K–2M / 2M+.
H-SPY SPY 5-min return at entry sign; SPY 3-day range tercile.
H-ANCHOR wrapper vs common; sibling that already triggered the same family today (causal
cohort only — `coh_by_t`; the day-level cohort is banned, see ignition_zero/REPORT.md §3).

## Splits — fixed, never moved
TRAIN 2025-01-02 → 2025-12-31 · VALIDATE 2026-01-01 → 2026-05-31 · TEST 2026-06-01 → 2026-09-11.

## Metrics (every family-config × exit × split)
trades/week, mean R, mean R ex-tail (R ≥ 3 removed; under E1 every winner is +2R so the tail
term is empty and mean R is the whole story), WR, profit factor, weeks green / weeks, worst
week (R), max drawdown (R), and the TEST quarter week-by-week.

## Selection rule (pre-committed)
1. On TRAIN only: keep family-configs with ≥ 5 trades/week AND mean R > 0 AND worst week ≥
   −10R. Then, on TRAIN only, at most THREE hypothesis splits may be added to a config, each
   required to raise mean R by ≥ 0.05 while keeping ≥ 5 trades/week.
2. VALIDATE: the config must still show mean R > 0 and ≥ half its weeks green. Configs that
   fail are dropped; nothing is re-tuned on VALIDATE.
3. TEST is read once, week by week, tail removed. A book is a candidate only if TEST mean R > 0
   with ≥ half the weeks green and trades/week ≥ 5.
4. Every table is published, failures included. If nothing passes, the answer is "nothing
   passes", with the closest miss shown.
5. Sensitivities reported, never selected on: slip 0.6%, ADV floor 50K/200K.

## Known limits (stated now)
- Quotes/spreads are not available historically; H-RMIN and H-PRICE are the proxies.
- The 3% range floor is deferred (data cost); P = 5 is the smallest pole tested.
- Premarket bars are sparse; F7 reports coverage.
- One-minute bars: intrabar sequencing of stop vs target within the same bar is resolved
  conservatively (stop first).
