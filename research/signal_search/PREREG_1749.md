# PREREG 1,749 — high-volume return premium, volume shock WITHOUT a price move (candidate H1) — Stage 1, rejection-only

FROZEN 2026-11-13 16:05 UTC before any number. Programme cell count: 1,748 → 1,749. Rules as PREREG_1746 (write
only `research/signal_search/1749_*`, `RESULT_1749.md`; no git / service / orders; budget ≤ 50 calls; no network
needed — the bars are on disk). This task IS the owner's request; do not pivot.

## Claim under test (Gervais–Kaniel–Mingelgrin JF 2001 — URL in the H1 block of CANDIDATES_20261113.md)
Stocks whose daily volume is extreme relative to their own recent history earn a positive abnormal return over the
following 1–20 days (up to +50 bps/week, 1962–2005); the premium is larger when the volume shock is NOT accompanied
by an extreme return. Mechanism: visibility raises the investor base. Post-2005 persistence is the open question.
Distinct from the closed ignition/momentum bursts (those required a PRICE burst; this selects volume without one).

## Data (on disk — reuse, no new fetch unless a gap is found)
Daily bars already cached by 1,746 (liquid tier, ≥ $100 M and $10–100 M) and 1,748 (ranks 1,001–3,000); their
universe files and point-in-time dollar-volume ranks; survivorship caveat as before (rejection-only).

## Signal and books (the paper's construction, daily resolution)
Reference window = the 49 trading days before t (the paper uses a 50-day window incl. t). Volume shock on day t =
volume_t in the top decile of the stock's own 50-day distribution (i.e. the highest or second-highest of 50);
"low-volume" = bottom decile. No-price-move filter: |close-to-close return on t| ≤ the stock's own 50-day median
absolute return × 1.5 (secondary: no filter). Entry at the OPEN of t+1 (auction), exits at the close of t+5
(primary) and t+20 (secondary). Equal weight. GROSS. Reads: HIGH leg vs the universe mean on the same days;
HIGH − LOW; by tier (liquid / $10–100 M / 1,001–3,000); the no-price-move subset vs the extreme-return subset
(the paper's prediction: the former larger).

## Splits
TRAIN 2016-01 → 2020-12 (halves 2016–18 / 2019–20); VAL 2021-01 → 2023-12; TEST 2024-01 → 2026-11 sealed.

## Numbers (per split, per tier)
n signals/wk, mean gross excess (HIGH − universe) over 5 d, day-clustered and NW(5) t, HIGH − LOW, ex-top-5 %,
winner-capped +10 %, monthly green share, year-by-year sign, cost from the 1,746/1,748 measured half-spread tables
(+ 2 bp per leg), net, MDE at 80 % power beside the kill threshold.

## Kill (pre-committed — any one kills the primary = liquid-tier HIGH − universe, 5 d, no-price-move)
VAL gross excess ≤ +15 bp or |t| < 2; TRAIN halves opposite sign; ex-top-5 % ≤ 0 on VAL; 2022 negative; net of
cost ≤ 0 on VAL; no-price-move subset not above the extreme-return subset on TRAIN and VAL (mechanism fails);
< 3 signals/week. Survive → TEST once on PIT bars → independent rebuild → paper-sleeve spec.

## Output
`1749_score.py`, `1749_signals.parquet`, `1749_summary.csv`, `RESULT_1749.md` ≤ 350 words. Return ≤ 120 words:
signals/wk by tier, VAL primary gross / t / net, HIGH − LOW, mechanism check, year signs, KILL/SURVIVE.
