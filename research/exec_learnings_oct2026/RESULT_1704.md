# RESULT 1,704 (+ amendment) — HOD forward book vs all-armed population, 2026-09-25..10-22
Data: book = data/trades.db `trades` strategy='hod_break', pnl not null: n 182, 18 days, pnl -$338.7. Population = `dry_trades`
hod_break, source NOT IN {backfill_journal, replay_capped} (the hod_dry_ledger "ALL-ARMED" set = counterfactual_all 9/25-9/28 +
live_dry 9/29+): n 960 closed, 19 days (open rows excluded). Scripts run_1704.py, run_1704b.py; tables hod_1704_book.csv, hod_1704_population.csv.
**R definition (no stop field in pattern_data; used `stop_loss_price`):** book R = pnl / (shares*|fill-stop|); 14 rows (ops_flatten,
over_exit_short, adopted/odd stops, risk<$0.5) fall back to pnl/50. Book risk is NOT $50: median $37, mean $35, 41 fills < $20.
Population R = the ledger's r_multiple (risk_usd 50). Sensitivity R50 = pnl/50 (same denominator as population).

## 1,704 frozen statistics (book)
- Mean R +0.121 (iid SE 0.138, day-clustered SE 0.146, 18 days), CI95cl [-0.165,+0.407]. R50: -0.037 (clSE 0.082) [-0.198,+0.123].
  Dollar P&L is NEGATIVE (-$338.7) while mean R is positive: small-risk fills carry R (risk<$20: +0.82 R n 41; risk>=$20: -0.08 n 141).
- vs prior -0.21: CI [-0.165,+0.407] EXCLUDES -0.21 (z_cl 2.27) on R. On R50 the CI [-0.198,+0.123] also excludes -0.21 only marginally
  (upper side clearly; -0.198 is inside by 0.01). Pre-committed read on the headline R: NOT "consistent with the prior".
- Ex-top-5 % (10 fills): mean R -0.134 (R50 -0.156); ex-top-1 % +0.007; winsor +-3 R: +0.028. Ex-top-5 % pnl -$1,338. max R +10.6, min -5.4.
  Tail-dependent: the positive R lives in the top ~5 % of fills.
- By exit (n, mean R, sum pnl): target 44, +2.58, +$2,628 | stop_loss 77, -1.12, -$2,709 | stop 11, -1.41, -$609 | eod 37, +0.29, +$357 |
  eod_ops_ah 4, +0.47, +$87 | ops_flatten 7, +0.05, +$18 | over_exit_short 1, -1.84, -$92 | stop_loss_unconfirmed 1, -1.03, -$19.
  Stops 49 %, targets 24 %, eod 20 % of fills (population: 60 / 22 / 18 %).
- Weekly (Fri-ending): 9/25 n3 sumR -0.9 -$140 | 10/2 n50 +6.0 -$556 | 10/9 n52 +7.8 -$76 | 10/16 n42 +5.7 +$265 | 10/22 n35 +3.5 +$169.
  Dollar P&L rises monotonically week over week; 4 of 5 weeks positive in R.

## Amendment
(a) Same days: book R +0.121 [-0.165,+0.407] vs population -0.227 (n 903, 17 days, clSE 0.045, CI [-0.314,-0.139]); diff +0.348, day-bootstrap
  CI [+0.096,+0.606]. With the same denominator (R50 vs population): diff +0.190, CI [+0.053,+0.339]. Population overall -0.208 (n 960, CI [-0.301,-0.114]),
  ex-top-5 % -0.341. The book is ~20 % of arms (10.1 fills vs 50.2 arms/day) and is better than the population it samples from.
(b) Within-day arm order in population: slot 1 -0.271 (n19), slot 2 -0.185 (n19), slot 3+ -0.207 (n922). Slot1-2 minus 3+ = -0.021 CI [-0.236,+0.215];
  halves +0.055 / -0.045 (signs differ). NO first-break mechanism (n 38 is tiny; MDE ~0.4 R). The book actually takes ~10/day, so wider buckets:
  slots 1-2 -0.228, 3-10 -0.016, 11-30 -0.198, 31+ -0.282; slot<=10 minus >10 = +0.186 CI [-0.027,+0.400]; halves +0.146 / +0.273 (same sign,
  CIs include 0). Suggestive, not significant (multiplicity: 3 slot cuts + 2 halves post-hoc).
(c) Last 2 weeks (10/12-10/22) n 77, +$433.5 (+$5.63/fill) vs book sd $47.2/fill: z vs zero 1.05; vs whole-window mean (-$1.86) 1.39; vs earlier
  105 fills (-$7.35) 1.88; day-clustered (8 days) t 0.93. Ex-top-3 fills (3 x ~$104) the fortnight is +$122. Not distinguishable from noise.
(d) Ramp rule recorded: $50 -> $100 R only when realized book P&L since 9/25 > 0 AND fills >= 200; never under water. Status: pnl -$338.7, fills 182 -> NOT met on either leg. No knob changes.

## Adversarial caveats
1. Headline R and dollars disagree in sign; R is inflated by small-risk fills (R denominator = shares*|fill-stop|, corr(risk,R) -0.20). Dollars/R50 are the honest read.
2. Same-window population is paper-dry (idealised fills at arm) vs book paper fills (7 s Alpaca paper queue); not the same instrument, so (a) is a selection+execution gap, not an edge proof.
3. Population slot order uses dry-ledger entry_ts; the book's actual selection rule (caps, filters, preplace) is not slot order, so (b) tests the amendment's mechanism, not the book's.
4. Rows: 5 adopted_on_boot, 4 ops rows with no stop (R from pnl/50), 1 row with negative risk_per_share in the DB (SPCF, recomputed by |fill-stop|).
5. 18 days, 5 weeks: day-clustered CIs are wide (MDE ~0.3 R); the improving trend is post-hoc. Cell count this programme not recounted here.
