# PREREG — cells 1,483–1,485: WHY is it moving (LLM catalyst attribution) and CAN it afford to (XBRL cash runway) — the two structural features the mover books never had

FROZEN 2026-09-26 before any number. Programme count on the HOD line: 1,482 → 1,485. Owner 9/26: "News sentiment sounds
like something everyone is doing… looking for different angles if LLMs / SOTA ML for selection… you are the expert."

## Why these two, and why they are not sentiment
Every price-derived feature at the arm bar failed to separate the big-day cohort (cells 1,445–1,478). Two things a
discretionary trader knows at 10:30 and none of our models had: the CAUSE of the move (an earnings beat continues; a
no-news pump fades; a financing headline is a sell) and the company's ability to survive it without selling shares
(a name with two quarters of cash raises money into strength — the overnight collapse pattern of 9/25). Neither is a
sentiment score; the first is a classification of the day's own-name news, the second is arithmetic on the last 10-Q.

## Data (point-in-time)
* News: Alpaca news API (Benzinga, free, history to 2015) for each fill's symbol, window = 16:00 ET the previous session
  → the arm instant; own-name only (articles listing ≤ 3 symbols). Entities MASKED before classification (the symbol and
  company name replaced by "the company"; other tickers by "another company") — Glasserman & Lin 2023. Classification by
  Haiku in harness batches (no API key on the node) into ONE of: earnings/guidance, FDA/clinical, M&A/strategic,
  contract/product/partnership, financing/dilution (offering, ATM, convertible, S-3), analyst action, legal/regulatory,
  sector/sympathy (own news absent, a peer/sector headline present), no news. Materiality 1–3. The classifier's prompt and
  every raw article are stored; a 200-item human-readable audit sample is written for the refuter.
* Runway: SEC XBRL companyfacts (`data.sec.gov/api/xbrl/companyfacts/CIK…json`, free, 1 req/s, UA set): cash and
  equivalents and quarterly operating cash flow from the latest 10-Q/10-K FILED before the fill date (`filed` field, PIT);
  runway_q = cash / max(quarterly cash burn, 0) (∞ if cash-flow positive); NaN if no XBRL (foreign filers, funds, wrappers)
  — coverage reported; wrappers and ETFs excluded from these cells by construction (no XBRL, no catalyst).
* Base book and standard: the 9,911 fills of 1,438, net R under the 1,478 standard (corrected cost + stop-limit exit).

## Cells (filters on the base fills; hypotheses stated in advance)
| cell | kept | hypothesis |
|---|---|---|
| 1,483 CATALYST | own-name HARD catalyst = earnings/guidance ∨ FDA/clinical ∨ M&A ∨ contract/product, materiality ≥ 2 | continuation; the "no news" and "financing" classes are reported as the expected losers (report-only, with their own means) |
| 1,484 RUNWAY | runway_q ≥ 4 (or cash-flow positive) | no dilution pressure → continuation; runway_q < 2 reported as the expected loser and, report-only, its close→next-open return after the fill day (the offering gap) |
| 1,485 JOINT | 1,483 ∧ 1,484 | the discretionary trader's checklist |

## Pass bar (frozen; per cell on VAL)
Kept mean net R ≥ +0.15, day-clustered t ≥ 2.5, ex-top-5 % > 0, ≥ 3 fills/week at 12/4, dropped mean < kept mean on
both holdouts, TRAIN-H2 same sign t ≥ 1, count-matched null percentile ≥ 99, feature coverage ≥ 70 % of fills (else VOID),
kept set's cache-only share within 5 pp of 19.5 %. Classification robustness: a second, independently prompted Haiku pass
on a 500-item sample must agree with the first on ≥ 85 % of class labels, else the catalyst cells are VOID. TEST once
for the single best passing cell.

## Independent check and consequences
Rebuild: a second agent re-pulls the news windows and XBRL facts from the raw sources and recomputes the flags from this
prose (the classification batches are re-run with its own prompt); kept-set Jaccard ≥ 0.90 (classification noise is
allowed up to that), means within 0.03 R. Refuters: look-ahead (article timestamps vs the arm instant; XBRL `filed` vs
the fill date; the masked-entity check), data (coverage by class; wrappers; symbol concentration), statistics. PASS → the
live engine gets the two columns at arm (news classification via the Alpaca stream + a nightly XBRL runway table) as a
gate: dry 5 sessions with the columns logged, then $50 real orders under the 9/25 fixes. FAIL → both columns still ship
to the dry ledger as instruments (they cost nothing), and the next angle (economic-link spillovers, own PREREG) proceeds.

## Not allowed
Re-labelling classes after a number exists; changing the runway thresholds; sentiment scores of any kind as a feature;
unmasked entities in the classifier prompt; reading TEST for more than one cell.
