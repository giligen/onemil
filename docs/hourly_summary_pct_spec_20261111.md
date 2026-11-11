# Hourly status: every $ figure carries its % change — spec 2026-11-11

Owner 2026-11-11: "on the hourly status message pure $$s is nice, but maybe add pct change next to them? i.e.,
+$53 (+2.3%)". Today only the DAY P&L of ORB / HOD / MOM / TOM has a percent; wk, open names, best/worst, MOM
open, LIVE, EXT and MOM-LIVE show bare dollars.

Rules: no git (no commit/stash), no service action, no orders, never write orb.yaml / config.yaml / .env /
data/*.db / crontab / logs/*. pytest only on files that exist, `--ignore=tests/integration`, one run at a time,
never piped through head/tail. Never Read a file > 300 lines in full (grep / sed ranges). **The script sends
Telegram when run plainly — run it ONLY with a print-only flag** (grep its argparse; if none exists add
`--print` = build the message, print it, never send). Budget ≤ 35 calls. This task IS the owner's request; do not
pivot.

## Rule (pre-committed; `scripts/hourly_summary.py` only)
One helper `money_pct(x, base) -> str` = `f"{fmt_money(x)} ({fmt_pct(x, base)})"` when `base > 0`, else
`fmt_money(x)` alone (a percent with no base is a false label — never print `(0.0%)` for a missing base; change
`fmt_pct` callers accordingly and update any test that asserted the old `(0.0%)`-on-no-base behaviour).
Format everywhere: `+$53 (+2.3%)`, unicode minus as today, one decimal.

Bases (same base family per book for day and week):
| figure | base |
|---|---|
| ORB day / wk | `ORB_STAGE_USD` (unchanged) |
| TOM day / wk | QQQ cost basis as today |
| HOD day | bought notional of today's fills (unchanged); **HOD wk** = bought notional of the WEEK's fills — reuse the ORB account's week-fills fetch for the HOD account (one extra closed-orders call; `book_stats` on the week fills). If the week fetch fails, the wk figure stays bare $ and a WARNING says why. |
| MOM day | equity (unchanged); **MOM wk** = start-of-week equity = `equity − wk` |
| MOM `open` | cost basis = `open_mv − open_pl` |
| open names (`_open_names`) | each name's `intraday_pl / cost_basis` (both already in the position dict); omit when cost_basis ≤ 0 |
| best / worst | that symbol's realized / its `buy_val` from `per_symbol_fills` — extend `stats['closed']` (or add `stats['closed_notional']`) so the renderer has the notional |
| LIVE day | `last_equity` |
| EXT day | its `bought_notional` (report-only line, same rule) |
| MOM-LIVE | find `mom_live_line` (grep `def mom_live_line`); if its $ figures have no %, add one on the sleeve's capital/state base the module already holds; if the module already prints %, leave it |

Keep the lines within Telegram's limits (`enforce_line_limit`, MAX_LINES unchanged); no other wording changes.

## Tests (`tests/test_hourly_summary.py`, extend; ~6 new)
`money_pct` with base > 0 / = 0 / None; `format_book_line` wk carries a % (ORB base), `_open_names` names carry %
and omit it on zero cost basis, `_best_worst` %, `format_mom_line` open and wk %, LIVE line %, EXT line %. Run
`tests/test_hourly_summary.py` and `tests/test_mom_live*.py` (files that exist) → 0 failures. Then run the script
with the print-only flag against the real accounts (read-only; paste the printed message into the RESULT with
account numbers, if any, truncated to 4 chars).

## Output
`docs/hourly_summary_pct_20261111_RESULT.md` ≤ 150 words: file:line of the helper, the printed message, test
counts. Return ≤ 80 words.
