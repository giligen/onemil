# Insights — cell 1,682: the literature's and the practitioners' exit rules (H37-H56)

Per-rule notes on the 1,681 harness (same book n=5,506, same cost convention, same TRAIN-H2/VAL halves). "Touched" =
fired_share x n on the TRAIN-side read (TRAIN-H2 directly, or VAL->TRAIN-H2 for model rules H41/H43/H45). saved/
forgone are the decompose() conditional means (R, on fired fills only); ex5 = ex-top-5% dR (both reads); "cons." =
green-week pp gain / weekly P10 delta, TRAIN then VAL.

**H37** Kaminski-Lo serial-correlation gate. Touched 282/2,349 (12.0%). saved +0.93R, forgone -0.23R, net dR +0.0031
(TRAIN) / -0.0041 (VAL) -- sign flips. ex5 -0.023/-0.032 (tail-carried). cons. 0/0pp, P10 +0.49/+1.27. Why: rho1>0
over bars 1-15 selects a small, noisy subset; the breakeven lock helps occasionally on a big winner and costs more
often on trades that would have continued. No real signal.

**H38** Confirmed change of character (>=2/4 signals in 5min). Touched 716/2,349 (30.5%). saved +1.20R, forgone
-0.98R, net dR -0.0226/-0.0143 (both reads negative). ex5 -0.119/-0.112. cons. -7.4/-22.7pp (VAL green weeks hit
hard). Why: matches the exit lab's standing finding almost exactly -- the confirmation gate fires on normal
continuation pauses; forgone trades outnumber saved ones even though saved's mean is larger.

**H39** Doldrums de-risk at 12:30 ET. Touched 456/2,349 (19.4%). saved +0.56R, forgone -0.46R, net dR +0.0021 (iid)
but day_t -0.54 (day-clustering flips the sign), VAL -0.0053. ex5 -0.041/-0.046. cons. 0/0pp, P10 +2.03/-3.31. Why:
the 50%-out branch gives away upside on trades that keep running; the full-exit-if-red branch locks in losses that
often would have partially recovered by EOD. A wash, day-clustered-negative.

**H40** Late trim at 14:30 ET. Touched 346/2,349 (14.7%). saved +0.33R, forgone -0.28R, net dR +0.0046 (day_t 0.44)
/ +0.0054 (day_t 1.37) -- **both reads positive**, the only rule of the 20 that is. ex5 -0.019/-0.019 (passes the
-0.02 TRAIN gate by a hair). cons. +3.7/+4.5pp (both positive), P10 +0.74/-2.51 (TRAIN better, VAL worse). Why:
banking half of a winner ahead of the final-hour chop is a cheap, small, consistency-positive trim. **Top TRAIN
scorer (S=+0.0231), joint member.**

**H41** Vol-regime trail (TRAIN-top-tercile ATR14% -> MFE-1.0xATR14 after +1R). Touched only 351/2,349 (14.9%, gated
on both the ATR tercile AND +1R). dR ~0 both reads (-0.0002/+0.0026), ex5 ~0. Why: top-tercile-ATR names in this
population already have wide bases -- a 1xATR trail rarely differs from the existing stop/target. Matches the
project's standing "breakout-thermometer sizing FAILED" finding (cells 1,420-1,421): volatility-regime gating does
not move this book's per-fill edge.

**H42** Analytic target (2*sigma*sqrt(Trem), capped [1R,4R]) -- REPLACES target0 (fixed after discovering
run_reshape's min-only accumulation silently no-ops a WIDER proposed target; see RESULT_1682 methodology note).
Touched 1,729/2,349 (73.6%). dR +0.0094 (t 0.17) / +0.0146 (t 0.02) -- positive but ~zero t. ex5 -0.095/-0.089
(strongly tail-carried). cons. 0/+4.5pp. Why: the uncapped formula often hits the +4R ceiling early in the day
(large T_rem), letting a few trades run far past the base's ~2R target; a handful of monster winners carry the
whole positive point estimate -- textbook lottery shape (CLAUDE.md gate #5).

**H43** Bayesian de-risk (posterior win rate <0.28 from the last <=20 fills, same-half rolling -> target capped to
+0.5R). Touched ~700/2,349 (29.5%). saved +0.60R (n-weighted ~72% of fired), forgone -1.31R (~28% of fired). dR
+0.0189 (t 1.08) / +0.0133 (t 0.94) -- modestly positive both reads, but t<2. ex5 -0.061/-0.066 (tail-carried: a few
of the "saved" cases are large). cons. 0/+9.1pp, P10 +5.34/+4.44 (both better). Why: capping the target after a run
of losses mostly banks a modest win (~72% of the time it fires) at the cost of occasionally giving up a much bigger
continuation -- directionally interesting (best P10/green-week combo among the model rules) but not TRAIN-eligible
on ex5.

**H44** Chandelier (HH-1.5xATR14 after +1R, floor breakeven). Touched 1,109/2,349 (47.2%). dR -0.0223/-0.0126. ex5
-0.083/-0.070. cons. -3.7/-13.6pp (worse both reads). Why: 1.5xATR trails tighter than this book's own ~2R target
dynamic once +1R is reached -- cuts winners before target more than it saves on losers.

**H45** MAE ceiling (exit past TRAIN winners' P80 running MAE). Touched the majority of fills: 1,908/2,349 (61%)
VAL->TRAIN-H2, 1,292/3,157 (41%... wait 54.9% VAL). dR -0.0381 (t -1.51) / -0.0051 (t 0.23). ex5 -0.080/-0.043.
cons. **-18.2/-3.7pp -- the single worst consistency reading of the 20 rules.** Why: the P80(winners' own MAE)
cutoff is loose enough, or the running-MAE metric normal enough, that it fires on the majority of trades including
completely ordinary pullbacks that go on to work -- the Sweeney MAE-ceiling idea does not transfer to this book.

**H46** 75% out at +1R. Touched 1,068/2,349 (45.5%). dR -0.0028/-0.0076, ex5 -0.094/-0.096. cons. +3.7/-9.1pp
(inconsistent sign). Why: same shape as H18 (50% at +1R, already in the 1,681 set) but deeper -- gives up more of
the common continuation to the ~2R target for a larger variance cut, echoing cell 1,680's finding that scaling out
harder disproportionately cuts the runners that build strong weeks.

**H47** Warrior half+9EMA -- remainder REPLACES target0 with the EMA trigger (fixed for the same run_reshape-cannot-
widen reason as H42/H49/H51; target0 is kept only before the +1R leg). Touched 1,068/2,349 (45.5%). dR
-0.0113/-0.0155, ex5 -0.120/-0.124. cons. -3.7/-13.6pp. Why: letting the back half run against a 9-EMA stop instead
of the ~2R target exposes more giveback than the fixed target on this book's typical path shape -- same family as
H44/H50, all three trailing-stop cousins read negative here.

**H48** SMB session give-back (book: +3R->+1.5R flattens open HOD positions; book-level, aggregated per day in fill
order). Touched 916/2,349 (39.0%) of FILLS on days that cross the give-back trigger. dR -0.0426 (TRAIN, the worst
raw point estimate of the 20) / -0.0114 (VAL). ex5 -0.117/-0.086. cons. -14.8/-4.5pp. Why: flattening every open
position once the day's book gives back from +3R to +1.5R cuts still-developing winners (fills entered later that
day are often mid-flight, not yet resolved) and the give-back itself is frequently just normal volatility around a
good session, not a reversal -- a blunt, costly circuit breaker on this book.

**H49** Half at +3R, no 2R exit, rest trails MFE-1R -- REMOVES target0 until the +3R leg fires (fixed after a first
run_partial-based build read 0% fired, because the unmodified target0 was still closing every trade at ~2R before
the 3R condition could ever be reached; see RESULT_1682 methodology note). Touched only 350/2,349 (14.9%). saved
+1.14R, **forgone exactly 0.0000R on TRAIN** (structural: a fired fill by definition already beat the ~2R base, so
every fire is a guaranteed improvement pre-trail). dR +0.0250 (day_t 1.11) / +0.0132 (day_t -0.68, day-clustering
flips sign on VAL despite a positive mean). ex5 -0.068/-0.076 (tail-carried by construction -- the whole mechanism
only ever pays off on the rare trades that run past 3R). Why: real R on the trades it touches, but touches too few
(15%) and the entire effect lives in the top slice.

**H50** Swing-low trail (last 3 rolling-5-bar-window lows after +1R, floor breakeven). Touched 956/2,349 (40.7%). dR
-0.0117/-0.0117 (remarkably consistent negative). ex5 -0.102/-0.098. cons. -3.7/-4.5pp. Why: a 3-window swing low is
typically tighter than this book's ~2R target/stop dynamic -- same trailing-stop-too-tight pattern as H44/H47.

**H51** Measured-move target (fill+(level-day_open), capped [1R,4R], level proxied by entry -- no separate level
field exists in this population) -- REPLACES target0 (same run_reshape fix as H42/H49). Touched 2,345/2,349 (99.8%,
virtually every fill gets a materially different target). dR **+0.0376, day_t 2.52 (TRAIN, the strongest day-t of
the batch)** / -0.0080, day_t -0.72 (VAL, sign flips). ex5 -0.069/-0.114 (both strongly tail-carried). cons.
+3.7/+4.5pp, P10 -0.08/**-11.66** (VAL tail much worse). Why: HOD breaks often print well above the day's open, so
the capped target frequently sits at +4R -- occasionally a monster trade (driving the TRAIN t=2.52) but the sign
reverses out-of-sample and VAL's worst weeks get much worse. A TRAIN-only mirage; the entry-for-level proxy adds
further noise CLAUDE.md would flag on its own.

**H52** New-high volume fade (<0.5x prior new-high volume -> sell 50%). Touched 1,391/2,349 (59.2%, a very common
trigger). dR -0.0024/+0.0053 (sign flips, both tiny). ex5 -0.057/-0.045. cons. 0/0pp. Why: saved (+0.59R) and
forgone (-0.59R) are near-mirror images -- the volume-based new-high signal carries ~zero information on this
population's continuation, consistent with the standing "HOD attention-rank refuted" finding (cells 1,351-1,354).

**H53** Parabolic SAR trail while ADX(14)>20, after +1R. Touched 718/2,349 (30.6%). dR -0.0041/-0.0046 (small,
day_t -1.08/+0.09). ex5 -0.098/-0.099. cons. +3.7/-13.6pp (inconsistent). Why: a reasonably selective gate (ADX>20
sustained is uncommon) but where active, SAR trails tighter than the base's own management -- a small, flat drag,
no edge either direction.

**H54** Power-hour hold (drop the ~2R target for the 15:55 close if mtm>=+1R at 15:00 ET). Touched only 162/2,349
(6.9%, rare: still open AND >=+1R AND that late). dR +0.0036 (day_t **1.98**, 2nd-highest of the batch) / -0.0016
(day_t -1.70, sign flips). **ex5 -0.003/-0.004 -- the smallest tail-dependence of any of the 20 rules, genuinely not
a lottery shape.** cons. 0/0pp, P10 +0.05/-0.61 (both flat). Why: clears the TRAIN ex5/P10 gates cleanly (with H40,
the only one of 20 to do so) and isn't tail-carried, but 6.9% firing is too rare to resolve a sign that flips on
VAL -- frequency, not confidence, is the open question. **Joint member.**

**H55** 75% at +1R, then H38 (confirmed CoC) or H44 (chandelier) whichever first on the remainder. Touched
1,068/2,349 (45.5%). dR -0.0119/-0.0118 (very consistent negative). ex5 -0.122/-0.121. cons. -3.7/-9.1pp. Why:
inherits the same negative shape as both components (H38 and H44 individually also read negative) compounded onto
a 75%-at-1R base leg (H46's own shape) -- stacking two already-negative mechanisms stacks their flaws, not their
strengths.

**H56** Portfolio hard stop (book<=-3R flattens + blocks new fills that day; book-level, per day in fill order).
Touched 646/2,349 (27.5%) of fills on stopped-out days. dR -0.0190/-0.0124. ex5 -0.086/-0.079. cons. -3.7/-9.1pp.
Why: blocking new fills and flattening once the book hits -3R removes some of the worst day's further losses by
design, but also blocks the V-shaped recoveries common after a rough morning on this book -- same blunt-circuit-
breaker pattern as H48, net costly here.

## Methodology note: three rules needed a bespoke walk, not run_reshape

H42, H49 and H51 each REPLACE the base target (H49 explicitly: PREREG's own "(no exit at 2R)"; H42/H51 compute a
genuinely different target formula). `run_reshape`'s `cur_target = min(cur_target, new_target)` accumulation can
only ever TIGHTEN a target -- a first build of all three under run_reshape silently no-opped whenever the proposed
target was above target0 (H49 read exactly 0% fired, the tell). Fixed by writing these three as bespoke walks (same
stop/target/EOD precedence, target replaced outright) mirroring H47/H54's existing pattern in the 1,681 set for the
same reason (H54 also needs to WIDEN, not tighten). Verified by re-running before/after: H49 0%->14.9% fired, H42
and H51's dR changed materially. H37/H38/H39/H40/H41/H43/H44/H45/H46/H48/H50/H52/H53/H55/H56 needed no such fix
(stop-only trails tighten correctly under run_reshape; H39/H40/H46/H52 are partials that intentionally preserve the
base target per their own PREREG text, e.g. H46 "25% runs with the base exits").
