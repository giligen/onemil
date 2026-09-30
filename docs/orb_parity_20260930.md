# ORB paper-vs-backtest parity — 2026-09-30 (+ 2026-09-29)

Read-only diagnostic for the ORB live-GO decision. Sources: live `journalctl` (9/30) /
`logs/session_archive/2026-09-29.log` (9/29), `data/trades.db` (ro), `analysis_results/orb_bplus_book.csv`
+ `orb_features_20260930_2052.csv` / `orb_features_20260929_2053.csv`, `study_orb_broad.load_broad_universe`,
`data/cache.db` (ro).

## 2026-09-30

| Live (preplace 09:34:57-09:35:07 ET) | BT book / features (22-name universe that day) |
|---|---|
| **ASTX** rh=$10.32 Q4 → filled 322sh@10.34/10.35, stop 10.01, exit `stop_loss`/`market_fallback`, **-$106.26** | **NBIL** Q4, gap 5.61%, no_fill |
| **AEHG** rh=$9.60 Q4 → preplaced, order `time_stop_canceled` (never filled) | **CRWG** Q5, gap 5.47%, no_fill |
| (only 2 slots preplaced; no PDR/G1 veto logged against NBIL/CRWG/NEBX) | **NEBX** Q3, gap 5.13%, no_fill (+19 more no_fill/eod/target names) |

ASTX and AEHG are **absent from every 9/30 BT row** — not even a no_fill row exists for either.

**Root cause (confirmed in `data/cache.db`, read-only):** `study_orb_broad.load_broad_universe` admits a
(symbol, date) pair only if `(open-prev_close)/prev_close*100 >= MIN_GAP_PCT` with `MIN_GAP_PCT=5.0`,
`MIN_PREV_DAY_VOL=500_000`, price $3-$30 — the *same numbers* live logs (`ORB: snapshot universe —
.../2796 symbols pass (gap>=5.0%, vol>=500,000, $3.0-$30.0)`). But the two systems price "today's open"
differently:
- ASTX: daily_bars 9/30 open=9.95, 9/29 close=9.76 → gap **+1.95%** (fails BT's 5% floor).
- AEHG: 9/30 open=9.27, 9/29 close=9.10 → gap **+1.87%** (same failure).

Both names had large moves the *prior* day (ASTX opened $11.65 on 9/29 off a $10.29 close, settled back at
$9.76) and still cleared live's gate at 09:30 ET — live scores gap off a real-time snapshot; BT recomputes
gap off the settled `daily_bars.open` written after the fact. Identical threshold, different input price ⇒
two different candidate universes by construction. NEBX is separately excluded from live by price alone
(BT entry $30.44 > live's $30.00 ceiling). NBIL ($29.69) and CRWG ($19.28) pass live's basic gate but lost
the two preplaced slots to ASTX/AEHG's ranking — no veto line names them.

**Fix:** not a threshold bug — both already agree on 5.0%/500K/$3-$30. Parity needs live to persist the
exact snapshot price/timestamp it used to pass the gap gate, so BT can be rebuilt against live's *observed*
open instead of `cache.db`'s settled open. Absent that, "BT misses live's post-spike-reversion gappers" is a
structural, bounded gap — a 9/30-style BT "nothing filled" does not contradict a live fill on ASTX/AEHG.

## 2026-09-29

| | AXTL |
|---|---|
| BT (`orb_features_20260929_2053.csv:13342`) | entered=1, entry 3.4579, exit `eod`, pnl **+$99.61 (+0.20%)** |
| Live (`trades.db` id 382) | filled 963sh@$3.46, stop 3.21→floored 3.3372, exit `force_close`@$3.36, exited **2026-09-29T19:45:07Z (15:45 ET)** — 15 min before the 16:00 ET close BT's `eod` assumes, pnl **-$96.30** |

Selection agrees on 9/29 — both land on AXTL (`[ORB] SUBMIT LATENCY sym=AXTL ... preplaced=1`; BT row
entered=1). ASTX also appears in BT's 9/29 file (no_fill; that day's gap is 13.2%, easily clearing
MIN_GAP_PCT — consistent with the 9/30 mechanism above). No `[ORB PREPLACE]` line survives for 9/29 to show
the live rank order: the daily archive cron's grep (`\[ORB\]|ORB SCORED|IGNITION|VETO|Q1 filter|WOULD
BUY|ENTRY SUBMITTED|FILLED|LOCK|kill|\[HOD`) does not match `[ORB PREPLACE]` (no bracket immediately after
"ORB"), so that tag is silently dropped from `logs/session_archive/*.log` every day.

Root cause for 9/29 is **not** universe/ranking — it is the exit. The ORB process restarted at least **7
times** while AXTL was open (PIDs 197526→237564→251615→269877→327519→332119→360967, 13:35-18:09 ET),
re-arming the same ATR-floor stop / scale-out order at each restart; `filled_at` in `trades.db` reads
18:09:21 (the last restart's re-registration), not the true 13:35:07 fill — the same fill-persistence
defect class fixed for HOD in `1b3584f` (`_on_live_fill` not persisting fill fields), evidently not carried
to the ORB path. Grepping the whole day for "force" (case-insensitive) finds exactly one hit, an unrelated
HOD/CONI line — **the ORB force-close trigger for AXTL is not logged under any greppable tag**, so which
restart (or a separate kill-switch) caused it cannot be confirmed from logs; only the DB's `exit_reason` and
15:45 ET timestamp are authoritative.

BEZ (flagged as a divergence "yesterday") has no row in `orb_bplus_book.csv` for 9/29 — its last book row is
9/28 (entered, eod, +5.6%); it does not bear on 9/29 and was not re-investigated here.

**Fix:** (1) give the ORB force-close path its own greppable tag (`[ORB] FORCE CLOSE <sym>`) and add it to
the archive cron's pattern; (2) apply the same persist-fill-fields-on-registration fix as `1b3584f` to the
ORB entry path so `filled_at`/`fill_price` survive restarts; (3) separately root-cause the 7 restarts on
9/29 (out of scope here) before trusting any BT `eod` exit as live-achievable on an unstable day.

## Bottom line

Parity does **not** hold on either day, for two unrelated reasons: 9/30 is a universe-input-timing gap
(live's real-time gap snapshot vs BT's settled daily-bar gap) that is structural, not a bug; 9/29 is an
operational-stability defect (restarts + an unlogged force-close + a fill-persistence bug) that IS a bug and
is fixable. Neither day's BT `no_fill` or `eod` result is the live-achievable outcome until both are fixed.
