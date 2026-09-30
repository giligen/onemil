# PREREG — cell 1,681: HOD mid-flight exit decisions — 35 hypotheses, one by one, then the joint rule (FROZEN 2026-09-30 17:50 UTC)

Owner 17:45 UTC: "can't be that HOD cannot use the candle-shape story line or the trained model or a combination to
make better exit decisions mid-flight. Set up 20–50 hypotheses, go over them one by one with cheaper agents, collect
the insights and land the secret sauce."

## Ground rules (every hypothesis)
* Book: the 1,663 join, floored r_pct ≥ 1.5 % (n 5,506); halves `split`; the completed bar store; per-fill × minute
  tables already on disk (`1670_per_fill_k.csv`: open-at-k, mtm, MFE/MAE, exits; `1676_features.csv`: rolling 5/10/15-
  minute candle features per k; the persisted models `models/1676_G7noclose_k*`, `models/1678_*`, `models/ff10_k1_*`;
  the rolling evaluation loop and decomposition in `1677_take_profit.py` / `1678_remaining_r.py`).
* Exits execute at the NEXT bar's open − 6 bps; partials book in original-R units; a rule never uses bars after its
  decision bar; a rule that changes the stop or target changes it from that bar on. Every hypothesis is a fixed rule
  (no fitted threshold); the thresholds below are frozen.
* Reads per hypothesis: paired ΔR vs the base (the live rule) on the whole book, iid t, day-clustered t, ex-top-5 % ΔR,
  MDE, share fired, give-back saved / continuation forgone / cost (the identity must reconcile), AND the consistency
  lens (owner 17:25): green-day share, green-week share, weekly P10, weekly Sharpe, max drawdown — base vs rule. Rules
  that use a model are read on BOTH scorings (TRAIN→VAL and the swap); rules without a model on both halves.
* Insight line per hypothesis (the deliverable of "one by one"): what it did to which trades, and WHY the number came
  out as it did (which side of the decomposition dominated, which trades it touched).

## The 35 hypotheses
F1 stale-trade / time × progress
 H1 exit at 30 min if mtm < +0.25 R      H2 exit at 60 min if mtm < +0.5 R      H3 exit at 90 min if mtm < +0.75 R
 H4 exit when > 20 min since the last new high AND mtm < +0.5 R      H5 after reaching +1 R, exit if no new high in 15 min
F2 shape story line (rolling candles, unaligned, from 1,676 G7)
 H6 exit on a bearish engulfing 5-min candle while ≥ +0.5 R      H7 exit on a close below the trailing 10-min candle's low while ≥ +0.5 R
 H8 exit on a 5-min close below the day's VWAP while ≥ +0.5 R    H9 exit on an upper-wick rejection (wick ≥ 2 × body) while ≥ +1 R
 H10 exit on a climax bar (volume ≥ 3 × mean, CLV ≤ 0.5) at any profit      H11 exit when the trailing 15-min candle turns red after +1 R
 H12 exit on two consecutive lower 5-min highs while ≥ +0.5 R
F3 model-based (persisted models only, never re-fit)
 H13 exit when P(+1 R next 15) < 0.3 while ≥ +0.5 R (rolling, from k = 5)      H14 exit when P(stop after k) > 0.7 while ≥ +0.25 R
 H15 exit when H13's and H14's conditions both hold      H16 the 1,677 reference: k ≥ 60, mtm ≥ +1 R, P < 0.3
 H17 P-weighted partial: at +1 R sell the fraction 1 − P(+1 R next 15) (rounded to quarters)
F4 partials / scale-outs
 H18 50 % out at +1 R, rest as the live rule      H19 33 % at +1 R, 33 % at +2 R, rest trails MFE − 1 R
 H20 50 % out at +1 R only if P(+1 R next 15) < 0.5      H21 50 % out at +1 R and stop to breakeven for the rest
 H22 25 % out at +0.5 R, rest as the live rule
F5 target / stop reshaping mid-flight
 H23 target to +1.5 R when the 30-min trailing candle is red (CLV < 0.5)      H24 target to +3 R when P(+1 R next 15) > 0.7 at the moment +1 R is reached
 H25 stop to breakeven at +1 R (exit-lab X5 replicate)      H26 stop to +0.5 R at +1.5 R
 H27 trail = the trailing 10-min candle's low once ≥ +1 R      H28 trail = the day's VWAP once ≥ +1 R
F6 volume / flow mid-flight
 H29 exit when the last 5 bars' mean volume < 0.3 × the break bar's AND mtm < +0.5 R after 20 min
 H30 exit on a red bar with volume ≥ 3 × mean while ≥ +0.5 R      H31 hold to +1 R only while volume of up-bars > volume of down-bars over the last 10 bars, else exit at ≥ +0.5 R
F7 combinations (pre-declared "secret sauce" candidates)
 H32 = H13 AND H7      H33 = H18 then H24 on the remainder      H34 = H4 AND H14      H35 = H21 AND H8

## Synthesis (the joint rule) — fixed before any number
Score per hypothesis on TRAIN only: S = ΔR (R) + 0.5 × (green-week share gain, in R-equivalents: 1 pp = 0.01) with the
constraints ex-top-5 % ΔR > −0.02 and weekly P10 not worse. The joint rule = the top-scoring hypothesis on TRAIN plus
every other hypothesis whose TRAIN score is positive AND which is compatible (does not act on the same trigger) — at
most three rules combined, precedence by TRAIN score. Read ONCE on VAL (and the swap for model rules). Pass bar for
the joint: paired ΔR ≥ +0.05 R, day-clustered t ≥ 2.5, ex-top-5 % > 0 on the held-out read, green-week share ≥ base
+ 5 pp, weekly P10 not worse. A pass → independent rebuild from prose → HOD PAPER as the one mechanics change of a
session. A null reports the best single hypothesis with its insight line and the MDE.

## Multiplicity
35 hypotheses × 2 reads (halves or scorings) × 2 lenses + 1 joint. Programme count on the HOD line: > 5,300. The
joint is the only "chosen" object and it is chosen on TRAIN alone.

## Not allowed
Changing a threshold after seeing its number; re-fitting any model; using the fill bar's future; VAL numbers in the
selection; reporting the joint's TRAIN read as a result.

## Output
`research/hod_entry/RESULT_1681.md` (≤ 200 lines: the 35-row table with the insight lines, the synthesis, the joint's
held-out read), `1681_reads.csv`, `1681_insights.md`, `1681_per_fill.csv`, `1681_hypotheses.py`, `1681_hypotheses.log`.
