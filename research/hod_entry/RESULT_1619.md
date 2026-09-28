# RESULT -- cells 1,619 / 1,620 (Frame B: burst fade)

PREREG: `research/hod_entry/PREREG_1617.md` (FROZEN 2026-09-28 17:00 UTC). Builder run (FULL, n=9,911 base fills), elapsed 28s.

## Population funnel (both cells share entry/exclusion; cover target differs)
* base fills (status==fill, causal_arming_causal.csv): 9911
* excluded, not shortable or missing from borrow_flags.csv: 3680
* eligible (shortable) population: 6231
* entry search outcome on the eligible population: {'fill': 3750, 'no_fill': 12, 'no_fill_gap': 1840, 'no_tape': 629}
  (`no_fill` = confirmed absent burst print, a real result; `no_fill_gap` = some minute in the 3-minute entry window was not cached, absence unconfirmed; `no_tape` = the fill minute itself has no cached tape, excluded from the population)

## Cell 1619

### TRAIN (holdout)
* population (eligible, shortable): 2655; short entries filled: 1604 (fill share 0.604)
* cover share within 15 min: 0.749 (1201/1604)
* stop share: 0.251 (403/1604)
* runner cohort (no cover within 15 min, n=403): mean net % of price = -0.8131
* mean net R_f = -0.1241; mean net % of price = -0.0743; day-clustered t (R_f) = -5.93
* ex-top-5% mean net R_f = -0.1480
* median R_f as % of price = 0.5991 (rail: >= 0.5 or NOT SHIPPABLE)
* fills/week (raw, unslotted -- see caveats) = 59.41 over 27 weeks
* ambiguous bars (both stop and cover touched in one un-taped bar, stop-first tie-break applied): 26; halt candidates (>=3 consecutive missing RTH bar-minutes): 8; EOD-fallback exits (no bar at/after 15:55): 0

### VAL (holdout)
* population (eligible, shortable): 3576; short entries filled: 2146 (fill share 0.600)
* cover share within 15 min: 0.776 (1666/2146)
* stop share: 0.222 (476/2146)
* runner cohort (no cover within 15 min, n=480): mean net % of price = -0.7946
* mean net R_f = -0.0750; mean net % of price = -0.0450; day-clustered t (R_f) = -4.24
* ex-top-5% mean net R_f = -0.0963
* median R_f as % of price = 0.5991 (rail: >= 0.5 or NOT SHIPPABLE)
* fills/week (raw, unslotted -- see caveats) = 97.55 over 22 weeks
* ambiguous bars (both stop and cover touched in one un-taped bar, stop-first tie-break applied): 35; halt candidates (>=3 consecutive missing RTH bar-minutes): 12; EOD-fallback exits (no bar at/after 15:55): 0
* **PASS BAR (frozen, VAL only binds): FAIL**
* TRAIN-H2 same-sign check: TRAIN-H2 mean net R_f = -0.1241, VAL mean net R_f = -0.0750 -> SAME sign (both negative -- the losing direction)

## Cell 1620 (report-only)

### TRAIN (holdout)
* population (eligible, shortable): 2655; short entries filled: 1604 (fill share 0.604)
* cover share within 15 min: 0.656 (1053/1604)
* stop share: 0.341 (547/1604)
* runner cohort (no cover within 15 min, n=551): mean net % of price = -0.7877
* mean net R_f = -0.0687; mean net % of price = -0.0412; day-clustered t (R_f) = -2.82
* ex-top-5% mean net R_f = -0.1029
* median R_f as % of price = 0.5991 (rail: >= 0.5 or NOT SHIPPABLE)
* fills/week (raw, unslotted -- see caveats) = 59.41 over 27 weeks
* ambiguous bars (both stop and cover touched in one un-taped bar, stop-first tie-break applied): 17; halt candidates (>=3 consecutive missing RTH bar-minutes): 13; EOD-fallback exits (no bar at/after 15:55): 0

### VAL (holdout)
* population (eligible, shortable): 3576; short entries filled: 2146 (fill share 0.600)
* cover share within 15 min: 0.682 (1464/2146)
* stop share: 0.307 (659/2146)
* runner cohort (no cover within 15 min, n=682): mean net % of price = -0.7429
* mean net R_f = 0.0039; mean net % of price = 0.0023; day-clustered t (R_f) = 0.17
* ex-top-5% mean net R_f = -0.0265
* median R_f as % of price = 0.5991 (rail: >= 0.5 or NOT SHIPPABLE)
* fills/week (raw, unslotted -- see caveats) = 97.55 over 22 weeks
* ambiguous bars (both stop and cover touched in one un-taped bar, stop-first tie-break applied): 21; halt candidates (>=3 consecutive missing RTH bar-minutes): 19; EOD-fallback exits (no bar at/after 15:55): 0

## Rails
R-vs-spread rail (corr(net_pct, half_entry/level) = -0.113):
  * Q1 (tight): n=938, mean net % = 0.0009
  * Q2: n=937, mean net % = -0.0340
  * Q3: n=937, mean net % = -0.0336
  * Q4 (wide): n=938, mean net % = -0.1633

Mirror check (n=3750): short mean net % = -0.0575, base long mean net % (outcome_R * R_pct) = -0.0467, corr(short, base long) = -0.103 (expect negative if the fade is a real mirror)

## Caveats (read as an adversary before relaying)
* **SSR not applied**: borrow_flags.csv (research/fuckup_audit/O_halt/PASSIVE/) is a static one-row-per-symbol snapshot (symbol, tradable, shortable, easy_to_borrow, exchange) with no `day` column and no SSR field. Only the static `shortable` flag is excluded here; a per-day SSR trigger (10% intraday decline) is NOT modeled. If the true HOD-break population has meaningful SSR incidence this book is optimistic by that share.
* **fills/week is unslotted**: cell_1445.fills_per_week applies research/hod_consol/run_consol.simulate_slots (first-12/day, 4-concurrent). That module was not imported here (to avoid its wider dependency graph inside the 40-call builder budget); the fills/week reported above is the raw count / distinct ISO weeks, an UPPER BOUND on the slotted number.
* **R_pct inferred**: the mirror check assumes features_1478_A.csv's `R_pct` column is the base long's R expressed as a percent of price (base_net_pct = outcome_R * R_pct) by name and units alone -- not independently confirmed against its builder script.
* **Halt handling is approximate**: a halt is inferred as >=3 consecutive missing RTH minutes in bars_fills_1478.db; the walk resumes at the next available print/bar (the reopen), which is what "the position is marked at the reopen print" was read to mean, but no explicit halt flag was cross-checked against an independent halt calendar.
* **Ambiguous bars use stop-first**, mirroring sip_rebuild.walk_path's documented tie-break for the long, sign-flipped; this is conservative (never overstates the fade's edge) but is a modeled assumption on bars where tape was unavailable, not an observed fact.
* **No independent reimplementation yet**: this is the BUILDER only, per CLAUDE.md's independent-check protocol a rebuild from this prose by an agent that has not read cell_1619.py is required before any number here is relayed to the owner.
* Price-scale (split/adjustment) check was not run: this is a same-day, intraday-only book (entry and exit both inside one session), so a raw-vs-adjusted daily-bar mismatch cannot fabricate the R -- the usual multi-day risk in CLAUDE.md item 3 does not apply the same way here.