# frames17 — PREREG

**Cell F52 — the discovered-mirror SHORT on a RESTING limit entry.** frames16 arm3 (F49) found the
mirror short (volume elevation *with* a price move, shorted) beats both placebos on gross, on both
splits (TRAIN +0.238 R t +6.19, VAL +0.201 R t +3.53 on S3/UP2; TRAIN +0.153/VAL +0.129 on S1/MIR2),
and dies on the MEASURED entry leg: every fill in 1,252 prior cells is a *reacting* order at the next
bar's open that crosses the spread by construction (entry leg 0.379 % of price at the median vs an
exit leg of 0.161 %). frames16 §3.4 already found that the signals the no-chase cap SKIPS returned
**+0.241 R** against **+0.151 R** filled — the passive side of this book selects the BETTER half, the
opposite of the halt-resume dip-buy signature that killed a passive entry there. F52 asks whether a
genuinely PASSIVE entry (a resting limit that is paid the spread, not one that crosses it) keeps the
gross edge net of costs, or whether it selects adversely once actually walked and priced.

Programme cell count: **1,252 → 1,254** (2 cells, both scored; this is the only frame run this pass).

## 1. Population (unchanged object, new entry mechanic)

Exactly frames16 arm3's **MIR2** signal object (`hrv_h >= 3` and `|hour_ret| > 2 %` at a session-hour
close, on a name whose session high already reached open x 1.05 by that hour's close — `gate5`,
causal membership), re-used verbatim from `frames16/sw_*.csv`: `f_mir2 & gate5 & price >= $5 &
ex-wrapper & day < 2026-06-01` (TEST stays sealed exactly as frames16 froze it). MIR2, not UP2/S3, is
used because it is the cell frames16 actually measured NBBO for and the one this frame's caveat #1
was written against; UP2 is not re-run here (a second signal family is a different cell count, not
this frame).

Short-feasibility rail: Alpaca ETB (`shortable AND easy_to_borrow`, `borrow_flags.csv`, today's
snapshot, absent = not shortable) — same rail frames16 applied. SSR's fill-blocking mechanic
(frames16 §5.5.4: a reacting marketable sell needs an uptick to fill under SSR) is **NOT applied to
this frame**, declared here before scoring: that mechanic exists because a REACTING order must
already be priced at/above the current bid at the instant it is forced to fill, and under SSR that is
only true on an uptick. A RESTING limit priced `k > 0` above `ref` is, by construction, already
displayed above the last trade — it is not the object Reg SHO 201 restricts, and no version of SSR
prevents a short from being *offered* above the market, only from being executed *at* the bid. This
is a declared assumption, not a measurement; it is listed as a caveat in the report regardless of the
verdict.

## 2. The entry — the actual test

A resting **SELL limit** at `signal_price x (1 + k)`, `signal_price = ref` (the last 1-minute close
before the hour cut — the exact same reference frames16 used for its own cap), `k in {0.3 %, 0.6 %}`
— **two cells only**, no diagnostic sweep beyond these two. The order is placed at the close of the
signal-hour cut bar (the same bar frames16's reacting order filled at, one bar earlier than its own
open-of-next-bar fill) and is **live for 5 minutes** (the cut bar and the 4 bars after it, 5 bars
total). It fills **AT THE LIMIT** the first bar in that window whose **HIGH >= limit** — an obtainable
fill, a touch of a resting order, never a reacting cross. A window with no bar reaching the limit is
**UNFILLED, 0 P&L, never a loss**.

**Unfilled counterfactual, pre-committed:** frames16's own `rr_a_bare` for the identical
(day, symbol, hour) row — the P&L the SAME signal already earned under the reacting fill — is carried
through unchanged as "what the unfilled signals would have returned." This is not a new number; it is
the frames16 arm3 object re-used as the counterfactual by construction, so it cannot be gamed by a
different walk on the unfilled side.

## 3. The stop, R, and exit — unchanged from arm3 spec A

`stop = entry x 1.02` (entry = the limit fill price, not `ref`), `R = stop - entry`. **bare** exit
only (EOD at `m >= 955` -> stop -> nothing else — S1's exit, the one this frame's caveat was written
against; S2's +2R bracket is out of scope here, a second exit family is a different cell). Walked from
the fill bar + 1 with `frames16/short_walk.py::swalk` **imported, not re-implemented** — same EOD
clock, same stop slip (`x 1.001`, one slip against us), same priority (EOD -> stop). Parity with
frames16's exit code is by construction (one shared function), not by a second implementation.

## 4. Cost — the actual claim under test

**Entry: charged ZERO.** A resting limit that is touched and filled earns the spread; it is not
charged half of anything. This is the entire point of the frame — F52 exists because 3.4/caveat #1
in frames16 found the passive side selects the better outcomes, and this pass converts that into a
priced book rather than a diagnostic.

**Exit: the measured NBBO half-spread, unchanged in KIND from arm3** — `0.5 x measured_spread(exit
day, symbol, exit_m) / rpct`, in R. Source: `frames16/nbbo16.csv` where the (day, symbol, exit_m) leg
was already measured there (it frequently is: an EOD exit at `m=955` lands on the same clock-time bar
regardless of which bar filled the entry, and EOD is arm3's dominant exit reason); for any
(day, symbol, exit_m) leg absent from `nbbo16.csv`, this frame measures it fresh with the same Alpaca
SIP-quotes method `frames16/nbbo.py` used (mean/median ask-bid over the exit minute), written to
`frames17/nbbo17.csv`. No leg is priced from the F45 imputed table — RUNBOOK step 3 ("measured cost,
never the band") applies here exactly as it did in frames16 arm3 §3.3.

## 5. THE DECIDING TABLE (printed before any pass/fail language)

**Filled vs unfilled**, both splits, both k: mean `rr_reacting` (frames16's own number) on the
UNFILLED rows vs mean **net** `rr` (gross minus the measured exit half-spread, entry charged zero) on
the FILLED rows. **If filled-net < unfilled-net on either split, adverse selection is confirmed and
the form is dead** — a passive fill that only happens on the days the short performs worse is not an
edge, it is the halt-resume signature restated on the short side, and F52 is refuted regardless of
what the filled-only book shows in isolation.

## 6. Pre-committed pass bar (BOTH k, BOTH splits, unless stated)

1. Net R (gross − measured exit half-spread, entry = 0) **>= +0.10 R** on TRAIN AND on VAL.
2. Same sign in both TRAIN halves (`day < 2025-07-01` / `>= 2025-07-01`) — frames16's own halves
   split, unchanged.
3. Ex-top-5 % (drop the best 5 % of `rr` by value) still **positive** — frames16's kill #3, applied
   here exactly as there.
4. Green-week share (count-matched permutation null, `common15.null_green`, unchanged) **above** the
   null p50 band on at least one split, reported on both.
5. The deciding table (§5) does **not** show adverse selection on either split — this is checked
   FIRST, before 1-4 are read as a pass/fail, per §5's own rule.

A cell that fails §5 is reported as refuted regardless of 1-4. A cell that passes §5 but fails any of
1-4 is reported as a lead, not a pass — the same standard frames16 held S3/S4 to at the measured cost.

## 7. What this pass will not do

Exactly frames16 §6: no TEST, no touch to `config.yaml` / `orb.yaml` / any engine / systemd / cron /
order. A SHIP-TO-DRY verdict (unlikely at this size; +0.10 R x ~15/wk x $100 risk is single-digit
weekly dollars) is a recommendation, never a change made in this pass.

## 8. Reported for every cell

n (signal x k rows) · fill rate · n filled · gross R (filled) · exit-leg measured-coverage % · net R
(filled) · net R day-clustered t · TRAIN halves · VAL · ex-top-5 % · green % vs null · weekly $ at
$100 risk · the deciding table (§5) · the unfilled counterfactual n and mean.
