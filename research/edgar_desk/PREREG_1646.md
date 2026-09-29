# PREREG — cells 1,646–1,648: PRE-ANNOUNCEMENT RUN-UP (the earnings announcement premium, expected date from the firm's own 8-K history)

FROZEN 2026-09-29 06:15 UTC before any number. Programme count: 1,645 → 1,648. Free queue #4 (`research/ideas_web/
RANKED_20260928.md`; idea 7 of `research/IDEAS_20260928.md`). No paid data.

## Mechanism (documented: Frazzini & Lamont 2007; Barber, De George, Lehavy & Trueman 2013; Savor & Wilson 2016)
Stocks earn an abnormal return in the days around their scheduled earnings announcement — attention-driven buying
ahead of the event and a premium for bearing announcement risk. The date is predictable from the firm's own history
(the same fiscal quarter's release one year earlier, usually within three days), so the window is knowable a month
ahead without any estimate data. The premium is concentrated in names whose announcement-window volume jumps.

## Data (on disk)
`research/edgar_desk/events_raw.csv` (8-K item 2.02 acceptance datetimes, UTC → ET; 116 k events 2019–2026); daily
bars `research/overnight_high/alpaca_daily_2019_2024H1.parquet` + `panel_2024_2026.parquet` (zero-OHLCV rows dropped);
universe price ≥ $3 and 20-day dollar volume ≥ $5M on session E−6. Splits as 1,633: TRAIN 2019–2022, VAL 2023–2024H1,
TEST 2024H2–2026-09 sealed (one read for the single best passing cell). Test tickers `^Z[A-Z]ZZT$` excluded.
MDE printed with the verdict: ≈ 2,000 VAL events at a 5-session SD of ≈ 6 % give SE ≈ 13 bps, so t ≥ 2.5 detects
≈ 33 bps — reachable for the 40-bps bar; recompute on the realised n and SD before reading.

## Signal and trades
Expected date E = the ET date of the 8-K 2.02 for the same fiscal quarter one year earlier (+ the median year-over-
year drift of the firm's last two same-quarter releases when both exist; skip the firm-quarter when the last two
years' dates disagree by more than 7 days). Only filings accepted before session E−6 are used to form E (causality).
* 1,646 PRE-WINDOW: buy at the open of session E−5 (MOO), sell at the close of session E−1 (MOC) — no announcement
  exposure by construction; if the actual 8-K arrives earlier than E−1 the position is closed at that session's close
  (the fill is what the auction gives; the early-arrival share is reported).
* 1,647 THROUGH-EVENT: the same entry, sell at the close of session E+1 (the classic premium including the event).
* 1,648 report-only: 1,647 split by the prior-year announcement-window volume ratio (tercile) — the mechanism.
Costs: 5 bps per auction leg. Report per cell and split: n events, events/week (with the seasonality), mean net bps,
day-clustered t (entry session), ex-top-5 % / ex-top-1 %, winner-capped +30 %, SPY-adjusted beside raw, the
expected-date hit rate (share of actual releases inside [E−1, E+1]), size split (≤ $1B vs larger), per-quarter table,
the MDE line.

## Pass bar (frozen; VAL, per cell)
Mean net ≥ +40 bps per event, day-clustered t ≥ 2.5, ex-top-5 % > 0, ≥ 20 events/week in season, TRAIN same sign
t ≥ 1, SPY-adjusted ≥ +25 bps, the volume-tercile table monotone on both halves, hit rate ≥ 70 %.

## Independent check and consequences
Rebuild from the prose (event set Jaccard ≥ 0.98, bps within 2); refuters: the expected-date estimator's causality
(no actual date used to select or to time), UTC→ET acceptance mapping, survivorship (delisted inside the hold kept at
−100 %), splits inside the hold, duplicate and amended 8-Ks, tails and month concentration, the auction fills on
illiquid names (the $5M floor). PASS → a daily MOO/MOC leg on the paper account for 6 weeks, then live at $3K per
event on the owner's word. FAIL → closed on this population with the tercile table on record.

## Not allowed
Choosing the window, the hold or the volume floor on VAL; more than one TEST read; any use of the actual date.
