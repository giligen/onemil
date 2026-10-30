# PREREG — cell 1,726: (A) passive squeeze, (B) uncensored afternoon path of the runners (owner 2026-10-30)

Owner's correction to 1,725: (A) a position with zero expected drift should be sold with a RESTING offer (earn the
spread) not a marketable order (pay it); (B) "the hold side does not keep rising" was measured to the LIVE exit, which
the 2 R target censors — a runner at +1.5 R at 14:00 can show at most +0.5 R. Frozen before any number.

## Data
`variants_1724.csv`, `runners_1725.csv` (67 positions open at 14:00, R1/R2 flags, features at the 13:59 close) and the
cached 1-min bars `cache/bars_1724/`. Tick = $0.01. Exit cost for marketable exits 12 bps (measured). A passive fill is
credited only when a LATER bar's high ≥ limit + 1 tick (the touch alone does not fill); no quotes exist in the bars, so
the offer is placed at the 13:59 close + 1 tick, which is a conservative proxy for the ask on these names.

## (A) Passive squeeze — three frozen variants, paired vs the live replay on the same 231 fills
A1: squeeze side of R1 at 14:00 (46 positions): rest a sell at 13:59 close + 1 tick until 15:54; unfilled → live rule
(stop or 15:55 flat). Hold side unchanged.
A2: as A1 but the offer is 13:59 close + 0.5 × (13:59 bar high − low), floor 1 tick (a wider "let it come to me").
A3: passive EOD for EVERY position still open at 15:30 (the live flat sells marketable at 15:55): rest at the 15:29
close + 1 tick until 15:54, unfilled → live 15:55 flat. Independent of (A1/A2).
Report per variant: fills via the offer (count, share), Δ R/trade pooled over all 231 fills, t, ex-top-5 %, A/B halves,
and the Δ restricted to the touched positions. Pass: Δ ≥ +0.02 R/trade pooled (this is a cost lever; the bar is the
measured 12 bps × touched share), same sign in both halves, ex-top-5 % > 0. A1/A2: the better one ships if both pass.

## (B) Uncensored afternoon path — descriptive, then one add-on variant
For the R1 hold side (21) and the squeeze side (46), from the 14:00 open to the 15:55 close: return in bps and in R
(original R per share), MFE and MAE after 14:00 in R, share closing above the 14:00 price; A/B halves; same for R2.
B1 **double down**: at the 14:00 open, the R1 hold side adds the ORIGINAL share count (2× position) at the 14:00 open
+ 12 bps; the add's stop is the 13:59 close − 1.0 R (original R per share), the add exits with the position (live
target/flat — if the live target triggers, the add is sold at the same bar; the ORIGINAL leg keeps its 2 R target);
report the add leg's own R (its risk = 1 R of the original R per share), t, ex-top-5 %, A/B, and the combined position
Δ vs live. Pass: add-leg mean ≥ +0.10 R, same sign in both halves, ex-top-5 % > 0.

## (C) Pre-committed decision on a dedicated 14:00 runner strategy (owner's item 3)
If (B) shows the R1 hold side's 14:00 → 15:55 return ≥ +30 bps net of 24 bps round trip in BOTH halves, PREREG cell
1,727: a universe-level 14:00 scan (all morning HOD breakers near their highs at 14:00, not only our fills) on free
Alpaca bars, TRAIN/VAL by month. Otherwise item 3 is answered by (B) and no universe pull is made.
