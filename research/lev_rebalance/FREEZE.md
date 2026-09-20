# FREEZE — lev_rebalance (post-PREREG operationalization, before reading VAL results)

Frozen 2026-09-20, before any pass/fail number was read.

## Deviations from PREREG (forced by the 45-tool-call / single-agent budget)

1. **Signal universe narrowed to the three FAMILIES-verified single-stock leveraged
   complexes** in `trading/orb_correlation.py::FAMILIES` — `tsla_leveraged` (8 wrappers),
   `mstr_leveraged` (8 wrappers), `nvda_leveraged` (7 wrappers). PREREG also named the
   offline-class-map / `underlying_anchor` route (6,136 wrapper candidates); extracting
   and querying that full set was infeasible in the budget (would need tens of thousands
   of per-symbol-day intraday queries). This is a scope cut, not a methodology change —
   the FAMILIES sets are the code-verified, unambiguous single-stock complexes.
2. **Daily prefilter before intraday fetch**: |daily close/prior_close − 1| ≥ 3% gates
   which underlying-days get an intraday query at all (bounds query volume). A 15:00
   move ≥ 5% that reverses to < 3% by the close would be missed. Applied identically to
   signal and control.
3. **Control pool**: 60 symbols sampled (seed 1291) from the offline class map's `stock`
   rows, not a scan of the whole market — budget-bounded, and intraday fetches for the
   control pool were hard-capped at 900 (reached: pool likely truncated for some
   later-appearing symbols).
4. **Point-in-time wrapper listing**: proxied by each wrapper's first `daily_bars` row
   (not `pit_listings`, which only carries monthly listing snapshots, not exact IPO
   dates) — a wrapper's ADV20 contribution to F is zeroed before its first cached daily
   bar.
5. **15:00→15:30 vs 15:30→close split**: NOT computed (would need one more intraday
   fetch per trade) — dropped under budget, flagged here rather than silently omitted.
6. Half-spread source: `research/mature_method/frames14/f45_minute_table.csv`, row
   `clock_m=900` (15:00 ET), `med=0.2331%` **full** spread (confirmed via `f45.py:73`,
   `sp_pct = sp_mean/price*100`) → half-spread charged = 0.11656% of price, one-sided,
   at entry only. Exit (MOC) is zero-spread per PREREG.

None of these were changed after VAL was read (see coverage collapse below, discovered
during the SAME run that produced the trades — there was no re-run with a different
prefilter or universe after seeing results).
