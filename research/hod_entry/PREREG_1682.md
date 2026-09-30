# PREREG — cell 1,682: the literature's and the practitioners' exit rules, on the 1,681 harness (FROZEN 2026-09-30 18:40 UTC)

Source: `research/ideas_web/EXITS_LIT_20260930.md` (32 rules; 20 new vs H1–H35). Same book, tables, harness, reads,
lenses, statistics and pass bar as PREREG_1681 (paired ΔR vs the live rule; both halves / both scorings; decomposition;
consistency lens; insight line per rule). All thresholds fixed here. Runs after the 1,681 harness exists.

## The rules (E-ids from the review)
 H37 (E1 Kaminski–Lo) serial-correlation gate: ρ1 = lag-1 autocorrelation of 1-min returns over bars fill+1..fill+15;
     if ρ1 > 0 apply the breakeven lock at +1 R (H25), else the base — the gate is the hypothesis.
 H38 (E12/E18) change of character, confirmed: exit only when ≥ 2 of {bearish engulfing 5-min candle, close below the
     trailing 10-min candle's low, upper-wick rejection (wick ≥ 2 × body), 5-min close below the day's VWAP} fire within
     5 minutes of each other, while ≥ +0.5 R.
 H39 (E13) doldrums de-risk: at 12:30 ET, if mtm ≥ +0.5 R sell 50 %; if mtm < 0 and the trade is ≥ 60 min old, exit.
 H40 (E23) late trim: at 14:30 ET sell 50 % of any open position with mtm ≥ 0.
 H41 (E10/E11) vol-regime-gated mechanism: day's ATR14 % in the top TRAIN tercile → trail = MFE − 1.0 × ATR14 (in
     price) once ≥ +1 R; otherwise the base.
 H42 (E14/E3) analytic target: target = fill + 2 × σ̂ × √T_rem, σ̂ = the stock's 1-min realized vol over bars fill+1..
     fill+15, T_rem = minutes to 15:55 ET, capped to [+1 R, +4 R]; stop unchanged.
 H43 (E7) Bayesian de-risk: Beta posterior of the book's win rate from the last 20 fills before this one (prior
     Beta(1, 2)); if the posterior mean < 0.28, exit at +0.5 R instead of the base target.
 H44 (E4) chandelier: once ≥ +1 R, stop = highest high since the fill − 1.5 × ATR14 (price), never below breakeven.
 H45 (E8 Sweeney) MAE ceiling: exit when running MAE exceeds the 80th percentile of winners' MAE on the TRAINING half
     (computed once there, applied to the other half; the swap likewise).
 H46 (E15) large first partial: 75 % out at +1 R, 25 % runs with the base exits.
 H47 (E16 Warrior) half at +1 R, then exit the rest on a 1-min close below the 9-EMA of closes.
 H48 (E17 SMB) session give-back: when the day's HOD book (realized + open, in R) has been ≥ +3 R and falls back to
     +1.5 R, flatten all open HOD positions for the day (book-level; per-day aggregation in fill order).
 H49 (E19) half at +3 R (no exit at 2 R), the rest trails MFE − 1 R.
 H50 (E20) swing-low trail: after +1 R, stop = the lowest low of the last three completed 5-min candles, updated each
     5-min close, never below breakeven.
 H51 (E22) measured-move target: target = fill + (level − day's open), capped to [+1 R, +4 R].
 H52 (E25) new-high volume fade: when a bar makes a new high since the fill on volume < 0.5 × the previous new-high
     bar's volume, sell 50 %.
 H53 (E5) parabolic SAR trail (step 0.02, max 0.2 on 1-min bars) once ≥ +1 R, only while ADX(14) on 1-min > 20.
 H54 (E13, mirror) power-hour hold: from 15:00 ET, replace the 2 R target by the 15:55 close for trades with mtm ≥ +1 R.
 H55 (E15 + E4 + E18 combination) 75 % at +1 R, then the confirmed change-of-character exit (H38) or the chandelier
     (H44), whichever first, on the remainder.
 H56 (E9/E2 book-level) portfolio hard stop: flatten all open HOD positions and take no new fills for the day once the
     day's book is ≤ −3 R (book-level; reads as ΔR per fill with the day's later fills counted as 0).

## Synthesis
As PREREG_1681: score on TRAIN, joint of ≤ 3 compatible positive rules, read once on VAL (+ swap where a model or a
TRAIN-derived parameter is used: H41, H43, H45). If the 1,681 joint passed, the 1,682 joint is also read stacked on it.
Pass bar unchanged. A pass → independent rebuild from prose → HOD PAPER as one mechanics change.

## Multiplicity
20 rules × 2 reads × 2 lenses + 1 joint (+ 1 stacked). Programme count on the HOD line: > 5,400.

## Not allowed
Tuning any threshold (12:30, 14:30, 1.5 × ATR, 80th percentile, 75 %, 0.28, 3 R); bars after the decision bar;
using VAL in the selection.

## Output
`research/hod_entry/RESULT_1682.md`, `1682_reads.csv`, `1682_insights.md`, rules added to `1681_hypotheses.py` as
H37–H56 (harness reuse), `1682_hypotheses.log`.
