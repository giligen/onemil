# Lens E — Live vs backtest parity and frequency

**Verdict: parity is weak over the only window available.** Live has run the current config
(catalyst_veto OFF, 8 slots, $375 nominal risk/trade) for 4 trading days (2026-09-21 to 09-24).
In the 3 of those days the BT book (`research/thermo/book_2025_26.csv`) also covers, BT claims 5
fills; live took 3 of them and missed the other 2 — including the day's best trade. Live also ran
below BT's claimed frequency, and the position sizing live actually uses does not match the
sizing arithmetic behind the claim's R units. n is far too small (3 days) to be dispositive, but
every discrepancy found points the same direction: away from the claim, not toward it.

## 1. Live fills, `data/trades.db` WHERE strategy='orb' AND trade_date >= '2026-09-21'

| date | symbol | entry | exit | exit_reason | pnl | shares | total_risk |
|---|---|---|---|---|---|---|---|
| 2026-09-21 | — | — | — | — | **0 trades** | — | — |
| 2026-09-22 | CRCA | 24.72 | 23.62 | stop_loss | -147.40 | 134 | 168.46 |
| 2026-09-22 | VNCE | 9.91 | 9.78 | stop_loss | -43.68 | 336 | 134.28 |
| 2026-09-23 | GDXD | 18.43 | 18.80 | force_close | +72.00 | 180 | 114.32 |
| 2026-09-24 | — | — | — | — | **0 trades** | — | — |

Confirmed via `logs/session_archive/2026-09-2{1,2,3,4}.log` (`ATR FLOOR <SYM>` is the live entry
marker): 0 entries 9/21, 2 entries 9/22, 1 entry 9/23, 0 entries 9/24. `onemil-trader` is
`inactive` right now (11:50 UTC 9/25, pre-open) and no 9/25 session log exists yet, so today
contributes nothing.

## 2. BT picks for the same calendar dates, `research/thermo/book_2025_26.csv`

Read with `trading.orb_csv.read_orb_csv` (no NA coercion). The file's own max date is
**2026-09-23** — it does not yet cover 9/24, so 9/24 can't be cross-checked either direction.

| date | symbol | entered | entry_price | exit_reason | _sized_pnl (R×375) |
|---|---|---|---|---|---|
| 2026-09-21 | ETRA | 1 | 14.293 | tag_bb | -40.08 |
| 2026-09-22 | CRCA | 1 | 24.724 | stop | -126.01 |
| 2026-09-22 | BIAF | 1 | 8.509 | lock | **+38.38** |
| 2026-09-22 | VNCE | 1 | 9.910 | tag_bb | -3.21 |
| 2026-09-23 | GDXD | 1 | 18.425 | eod | +68.02 |

## 3. Symbol-by-symbol match

- **ETRA, 2026-09-21 — MISSED, and not by any logged veto.** Live scored ETRA at 13:35:25/13:35:29
  (`ORB SCORED: ETRA comp=0.4225 Q4`) alongside ~35 other candidates that morning. Grepping the
  full day's log for `PDR VETO`, `G1 VETO`, and `Q1 filter dropped` turns up 7 PDR vetoes (WBD,
  BITU, BITX, AIP, ALM, IMSR, PRTH), 1 G1 veto (NNE), and a 5-name Q1 drop (RGTX, ETOR, KORU,
  XRPZ, MEDS) — **ETRA is in none of them.** Live entered **zero** ORB trades all day. There is no
  log line explaining why a Q4-composite, un-vetoed candidate was not submitted; the log simply
  goes silent on `[ORB]` after the last PDR veto at 13:36:00. I could not root-cause this in
  budget (would need `trading/orb_engine.py` source, not just its logs) — flagging as an open
  parity gap, not a confirmed mechanism.
- **CRCA, 2026-09-22 — MATCHED on entry, diverged on exit magnitude.** Entry price live 24.72 vs
  BT 24.724 (agree). Both exit via a stop (`stop_loss` / `stop`). Live pnl -$147.40 vs BT's sized
  -$126.01 — same sign, same ballpark, plausibly slippage/spread (BT's R-unit and live's actual
  risk aren't the same dollar base either, see §5).
- **BIAF, 2026-09-22 — MISSED, and it was the day's best trade.** Live scored BIAF at 13:35:25
  (`comp=0.2652 Q3`), same as ETRA it appears in no PDR/G1/Q1 veto line, and live never entered
  it. In BT, BIAF is the only winner that day: +$575.76 raw / +$38.38 sized — the single biggest
  swing in the 3-day comparison window. A book built from `entered==1` rows is, by construction,
  crediting live's book with a trade live's own logs show it never took.
- **VNCE, 2026-09-22 — MATCHED on entry, diverged on exit *mechanism*.** Entry live 9.91 vs BT
  9.910 (agree). But BT exits VNCE via `tag_bb` for a near-scratch -$3.21 sized (-0.009R on the
  $375 unit); live exits via `stop_loss` for -$43.68 (-0.325R on live's own $134.28 total_risk) —
  a ~14x larger loss and a different exit trigger entirely. This is the one clean same-symbol,
  same-day, same-entry pair in the sample, and its exit doesn't match. Worth its own lens; out of
  scope to fully diagnose here (would need the minute bars around the VNCE exit).
- **GDXD, 2026-09-23 — MATCHED, no real discrepancy.** Entry 18.43 vs 18.425 (agree). Live shows a
  `SCALE OUT ARMED GDXD: 72/180sh at $20.14 (+3.0R)` order, but the trades row's
  `partial_exit_*`/`scale_*` columns are all NULL and the full 180 shares closed via
  `force_close` at $18.80 (never reached $20.14) — consistent with a scale order that armed but
  never filled, then EOD close. BT's single-exit `eod` model (+$68.02 sized) vs live's
  `force_close` (+$72.00) agree in sign and magnitude. No parity issue here.

**Net: live captured 3 of BT's 5 claimed fills in the overlapping window (60%), zero fills live
took that BT didn't, one exit-mechanism mismatch (VNCE) on a matched entry, and the one miss with
a real dollar impact (BIAF) removed the week's only winner from live's realized book.**

## 4. Frequency

BT, `entered==1` rows in `book_2025_26.csv`, 2025-07-01 through 2026-09-23 (388 days, the claim's
own OOS-2025H2-2026 slice): **389 fills / 63.9 weeks = 6.09 fills/week**, close to the 6.3/week
figure this lens was asked to check — order of magnitude and direction confirmed, exact figure not
independently reproduced (I get 6.09, not 6.3; whole-2025-2026 n=473/89.9wk = 5.26/week).

Live, same-config days only (2026-09-21 → 09-24, 4 trading days): **3 fills / 4 days ≈ 3.75/week**
at a 5-day week — about 60% of BT's claimed rate. Restricted to the exact 3 days BT can be checked
against (9/21-9/23): BT claims 5 fills in 3 days (8.3/week pace), live realized 3 (5.0/week pace,
and only because 2 of BT's 5 landed on the same names).

**This is a 4-day, 3-day-comparable sample — not remotely enough to reject the frequency claim
statistically. But the sign is consistent across every check in this lens: live is running colder
than BT, not matching it, in its first live days under the current (9/21) config.** This is also
the very first live window after the 9/21 corpse-gate fix (stale-snapshot gate that had been
silently killing ~2,950 symbols/day) — if the zero-entry 9/21 and the ETRA/BIAF misses trace to a
residual variant of that defect or a related universe-completeness gap (the class of bug flagged
in `feedback_fetch_completeness_gate.md`: bulk fetches silently dropping symbols), that would be a
live infrastructure defect, not evidence about the strategy's edge — but I did not have budget to
trace `trading/orb_engine.py`'s universe-build path to confirm or rule this out. **Next step, not
done here: re-run this same comparison after 2-3 more live weeks, and separately audit why ETRA/
BIAF were scored but never submitted.**

## 5. Sizing: live's real dollar risk does not match the claim's $375 R-unit

None of the 3 live fills risked anything close to $375:

| symbol | shares | risk_per_share×shares | total_risk (logged) | nominal risk_per_trade_usd |
|---|---|---|---|---|
| CRCA | 134 | $122.07 | $168.46 | $375 |
| VNCE | 336 | $86.99 | $134.28 | $375 |
| GDXD | 180 | $89.93 | $114.32 | $375 |

`account_budget_usd: 26666.67` ÷ 8 slots = **$3,333.33 per-position cap**. CRCA: 3333.33/24.72 =
134.8 shares → matches live's 134 exactly. This is the $3,333/slot position-*value* cap binding,
not the $375 *risk* cap — for all 3 live fills, the position-value cap was reached before 375/risk-
per-share would have been, so live took on 30-45% of the nominal risk unit.

The claim's R is `_sized_pnl / 375` (`research/thermo/thermo.py:228`), and `_sized_pnl` itself is
built as `_rp_pnl * mults[quintile]` — a quintile-adaptive multiplier on raw per-position P&L
(confirmed by grep across `study_orb_*.py`; the exact multiplier table wasn't traced further here
for budget). **Nothing in that formula reproduces live's per-slot dollar-value cap.** So the R the
claim reports is not computed the same way live's realized R would be: live's true risk-per-trade
varies $87-$168 in this sample, not a flat $375, which means live's dollar P&L per unit of *actual
risk taken* will differ from what `_sized_pnl/375` implies even when the raw trade outcome
matches. This is a structural sizing-parity gap, independent of the entry/exit-matching issues in
§3, and it sits on top of Lens B's finding that the OOS window isn't clean and Lens A's finding
that the edge is tail-concentrated.

## 6. What's clean

- orb.yaml's live literals match the claim's stated config: `risk_per_trade_usd: 375`,
  `catalyst_veto.enabled: false`, `account_budget_usd: 26666.67` (8 × $3,333.33).
- Every matched entry price (CRCA, VNCE, GDXD) agrees with BT to the cent — the core breakout-scan
  and composite-scoring logic that decides *what a candidate looks like* is consistent between BT
  and live when live does act.
- GDXD's scale-out-armed-but-unfilled-then-EOD-close sequence reconciles cleanly with BT's simpler
  `eod` exit model; no discrepancy there.

## Provenance

- Live fills: `sqlite3 data/trades.db` — `SELECT ... FROM trades WHERE strategy='orb' AND
  trade_date >= '2026-09-21'` (3 rows) and `SELECT MIN/MAX(trade_date) ... WHERE strategy='orb'`
  (2026-05-18 → 2026-09-23, confirming no live ORB trades exist for 9/24 or 9/25 beyond the
  0-count already shown).
- Live picks/vetoes: `logs/session_archive/2026-09-{21,22,23,24}.log`, grepped for `[ORB]`,
  `ORB SCORED`, `PDR VETO`, `G1 VETO`, `Q1 filter dropped`, `ATR FLOOR`, `SCALE OUT ARMED`, `error`,
  `exception`, `WARNING` (none found beyond the one latency-tripwire WARNING per day, unrelated).
  Used session_archive instead of a live `journalctl --since 2026-09-21` scan (which took >120s
  and was killed) — session_archive covers the identical days and is the documented substitute
  per `reference_dry_ledger_archive_sep2026`.
- BT picks: `research/thermo/book_2025_26.csv` via `trading.orb_csv.read_orb_csv`, filtered
  `entered==1` and `date` between 2026-09-21 and 2026-09-24; file's own max date is 2026-09-23.
- Frequency: same file, `date >= 2025-07-01`, n=389, span 63.9 weeks → 6.09/week; full 2025-2026
  span n=473, 89.9 weeks → 5.26/week.
- Sizing constants: `orb.yaml` lines ~148-160 (catalyst_veto), ~306-308 (account_budget_usd,
  risk_per_trade_usd); `_sized_pnl` assignment pattern confirmed via
  `grep -rn "_sized_pnl'\] *=" *.py` (10+ hits, all `_rp_pnl * mults[_quintile]` form);
  `research/thermo/thermo.py:228` for `R = _sized_pnl / 375`.
- Checked `date -u` at task start: 2026-09-25 11:50 UTC, before the 13:25 cutoff — no bars DB or
  cache.db was read at any point in this lens; only `data/trades.db` (live fills table, not a bars
  cache) and CSV/log files were used.
