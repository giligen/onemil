# Cell 1,733 — ORB selection parity (live paper vs nightly BT) — RESULT 2026-11-11

Read-only. Scripts `sel_1733.py` (BT re-rank + journal parse) and `cls_1733.py` (classes + re-sim). Rows:
`selection_1733.csv` (278 symbol-days: BT top-8 under today's params AND point-in-time params, BT book, live
provisional / final / submitted / fills, live vs BT composite and features). Window 9/29–11/9, 30 trading days.
Journal holds 11/3+ only; earlier days come from `logs/session_archive` (9/30 missing there).

## 1. Main finding: the nightly BT ranks the past with today's parameters
`scripts/orb_weekly_refit.py` rewrites the z-params and `quintile_cutoffs` in orb.yaml every Sunday
(`logs/orb_refit_history.jsonl`). Live scores each week with that week's fit. The pipeline
(`study_orb_pipeline_static_lock.py:955-966`) reads the CURRENT orb.yaml once and re-scores all of history with it.
- With today's params, live composite − BT composite = +0.013 mean on 164 shared names, a per-week step (0.024 in the
  week of 10/12 → 0.000 on 11/9). Quintiles agree on 91.6 %.
- With the point-in-time (PIT) refit in force that week: |diff| ≤ 5e-5 and quintiles agree on **100 %**.
- Because the rank order is Q4 > Q5 > Q3 (`:142`), a 0.013 shift across the 0.29 cutoff moves a name from rank 25
  to rank 4 (SNXX 11/3).
- Live's final top-8 equals the PIT BT top-8 on **11 of 17** days with a reconcile line, and the published-param
  top-8 on 7 of 17. The other 6 days are all universe differences (below) or a truncated live field (10/30 had 7 ranked
  and 11/10 had 4, the sweep failures fixed in bd282be/265603a).

So for class R, **the live engine was right and the BT book is wrong**.

## 2. Divergences (fills; K = 8 on both sides, no K class)
| side | class | n | $ | stops | names |
|---|---|---|---|---|---|
| live-only | R, BT stale params (live correct) | 3 | −327 | 2 | CORD 10/16, SNXX, AAOX 11/3 |
| live-only | U, universe | 4 | −322 | 3 | ASTX 9/30, ASTY 10/12, SK 10/23, SKHX 11/2 (displaced by the BT-only AEYE) |
| live-only | U?+T | 1 | −96 | 0 | AXTL 9/29: BT rank 11, before SCORED logging (unresolved). It **filled 14:09 ET, 3.5 h after its 60-min time stop** |
| live-only | V, veto input | 1 | −124 | 1 | SKUU 11/2: BT range 2.209 % ≤ 2.221 → BT range-size veto; live range above the cut |
| live-only | P, add-on pool | 2 | +179 | 1 | APLX 10/6, SMCZ 10/8 (no BT book) |
| BT-only | R, BT stale params | 2 | −215 | 2 | RAM, SKHL 11/3 (not in the PIT top-8) |
| BT-only | U | 4 | −406 | 2 | DFDV 10/5, AAOZ 10/13, SAIQ 10/23: their slots went to live-only names that were then vetoed, with no refill (BABX/GDS, INFY, SK/LNOK). AEYE 11/2: BT feature gap 3.07 % |
| BT-only | T, no live fill | 2 | +748 | 0 | BEG, CRDU 11/9: submitted (reconcile at 09:35:34, CRDU replaced) and never filled. This is entry obtainability, not selection |

Universe detail. Daily bars exist for every U name.
- **Live admits names whose official daily gap is below 5 %:** ASTX 9/30 1.9 %, INFY 3.2 %, BABX 3.3 %, GDS 3.8 %. Live gates on snapshot.open, then re-checks bar1.open (`orb_engine.py:3890-3906`).
- **The BT admits feature-gap < 5 % names** (AEYE 3.07 %, BNC 9/29 4.6 %) because it has no phantom-gap re-check.
- **The builder never emits some names that pass every universe gate:** ASTY, SK 10/23, LNOK, SSG, BNC 10/20, RAMZ, OIS, SOPH, PMN. Gap ≥ 5 %, prev volume ≥ 500K, and the drop reason is not traced.

**Preplace.** About 19 provisional orders were placed at ~09:35:05 on a partial field (11/3: 106 candidates had no range yet). All were cancelled at reconcile, and `n_filled_before_reconcile` = 0 on every logged day. This is legitimate by design, but only while the final reconcile converges to the BT set.

## 3. Re-sim, exact rows, day-clustered over 30 days
| book | sum | $/day | SE | t | stops |
|---|---|---|---|---|---|
| live actual | −88 | −2.9 | 70.0 | −0.04 | 16/32 |
| live on the published-BT names (matched → live $, BT-only → BT row) | +729 | +24.3 | 73.3 | 0.33 | 13/29 |
| live on the PIT-BT names | +618 | +20.6 | 73.5 | 0.28 | 13/30 |
| BT published realized | +1,007 | +33.6 | 60.2 | 0.56 | 12/29 |

Selection parity is worth about +$700 over 6 weeks in this window, but **none of these is distinguishable from zero**. Parity is a correctness fix, not an edge. RESULT_1732's +$603 counted the matched names only; adding the BT-only rows gives the +$729 above.

## 4. Fixes, in priority order
1. **PIT params in the BT** (R, BT defect), `study_orb_pipeline_static_lock.py:955-1066`:
   ```
   hist = load_jsonl('logs/orb_refit_history.jsonl')          # ts, params, cutoffs
   for day, g in df.groupby('date'):
       h = last(x for x in hist if x.ts[:8] < day)            # FATAL if none and day >= first refit
       df.loc[g.index,'_composite'] = composite_score(g, h.params)
       df.loc[g.index,'_quintile']  = assign_quintile(df.loc[g.index,'_composite'], h.cutoffs)
   ```
   The same applies to `orb_weekly_refit.py`'s own fit, which reads the BT book.
2. **One universe gate** (U): a shared `trading/orb_universe.admit(prev_close, open_px, prev_vol, price, min_gap)` for the builder and live. The BT re-checks `gap_pct >= universe.min_gap_pct` after `:1066`, the twin of `orb_engine.py:3900`. The builder writes a per-symbol drop reason so ASTY/SK/SSG-type omissions get named.
3. **Range parity** (V): log the live final H/L per candidate and compare with the BT range. Any veto decided by less than 1 tick of difference is flagged `range_source`.
4. **Time-stop leak** (AXTL): a resting order filled 3.5 h after `entry.time_stop_minutes`. Audit the cancel path.
5. **Add-on pool**: build the nightly `ORB_BT_POOL=addon_gap35_range5` book (`:659-690`) or run the pool dry.
6. **T (BEG/CRDU)**: the BT entry model should be the live stop-limit rule (cell 1,426). This is not selection.

## 5. Parity test (extends `scripts/orb_selection_observer.py`, nightly after the BT)
For each day:
- **BT side:** PIT top-8 through the shared helpers, with the refit `ts` it used.
- **Live side:** reconcile final top-8, submissions and fills from the journal plus the archive.
- **Rule:** assert `set(live_final) == set(bt_pit_top8)` and `live_comp == bt_comp ± 1e-4` per shared name.
- **Named reasons:** every difference must get one of `universe_live_only | universe_bt_only | veto_input(range/pdr) | param_ts_mismatch | addon_pool | provisional_fill | no_fill | range_missing`.
- **Alarms:** Telegram ERROR on any unexplained difference, or if a live fill is not in the final top-8 (a provisional or post-time-stop leak).

Legitimate by design: provisional preplace (it must converge), add-on pools (with their own book), T no-fills (logged). Defects: stale BT params, the two gap definitions, the builder dropping names, the time-stop leak.
