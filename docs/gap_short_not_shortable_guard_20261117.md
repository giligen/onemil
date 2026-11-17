# GAPSHORT: pre-submit `shortable` guard (2026-11-17)

## The error
`scripts/gap_short_paper.py --pick` 2026-11-17 14:20 UTC (logs/gap_short_paper.log):
```
INFO  pick: LITX entry tif=day (borrow_status=hard_to_borrow)
WARNING pick: order rejected for LITX: {"code":42210000,"message":"asset \"LITX\" cannot be sold short"}
WARNING pick: MVLL skipped (threshold-list)
ERROR pick: ALL 1 submitted picks were rejected by the broker
```
Verified facts (GET /v2/assets/LITX, 14:25 UTC): paper-MOM AND live both return
`tradable=true, shortable=false, easy_to_borrow=false, borrow_status=hard_to_borrow`. The name cannot be shorted on Alpaca
at all; the rejection is deterministic and knowable BEFORE the submit. The picker reads only `borrow_status`
(`locate_ledger.fetch_borrow_status`) and never `shortable`, so it submits, gets rejected, and the "ALL rejected" ERROR
reaches Telegram for a condition that is a skip reason, like `threshold-list`.

## The fix (guard + classifier; no rule change, NO refill of the freed slot — post-ranking veto never refills)
1. `scripts/locate_ledger.py`: add `fetch_asset_flags(session, symbol) -> dict` doing the SAME single GET
   `/v2/assets/{symbol}` and returning `{"borrow_status": str, "shortable": Optional[bool], "easy_to_borrow": Optional[bool]}`
   (`borrow_status` falls back to `"unknown"` as today; on any failure log ERROR as today and return
   `{"borrow_status": "error", "shortable": None, "easy_to_borrow": None}`). Re-implement `fetch_borrow_status` as a thin
   wrapper `return fetch_asset_flags(session, symbol)["borrow_status"]` so every other caller is untouched.
2. `scripts/gap_short_paper.py` pick loop (around line 309): call `ll.fetch_asset_flags` ONCE per pick instead of
   `fetch_borrow_status`; `bs = flags["borrow_status"]`. In the skip chain, after the threshold check and before the
   existing-position check: `elif flags["shortable"] is False: skip = "not shortable (asset.shortable=false)"`.
   `None` (unknown/error) must NOT skip — the broker decides, as today. The row keeps `borrow_status`; the skip logs
   WARNING as the other skips do; skipped rows are not counted in `n_ord`, so the "ALL rejected" ERROR no longer fires
   for this case (a rejection with an unknown message still does).
3. `scripts/gap_short_parity.py`: add `"not_shortable"` to `REASONS`; in `classify_bt_only`, before the `"threshold"`
   check: `if "not shortable" in note: return "not_shortable"`. Confirm `not_shortable` is treated as explained
   (whatever path `rejected_htb` takes for the summary/ERROR decision — mirror it).
4. Tests (TDD): `tests/test_gap_short_paper.py` — (a) pick with flags `shortable=False` → `short_at_open` NOT called,
   row `entry_status == "skipped"`, note contains "not shortable", caplog has NO ERROR; (b) flags `shortable=None` →
   submit proceeds as before; (c) `fetch_asset_flags` parses a 200 body and returns the error dict on HTTP 500 (mock
   the session). `tests/test_gap_short_parity.py` — a ledger row `entry_status=skipped, note="not shortable (...)"`
   → `not_shortable`, not `unexplained`. Existing tests that monkeypatch `fetch_borrow_status` on the pick path must be
   moved to `fetch_asset_flags` (keep the wrapper test if one exists). Any `tests/test_locate_ledger*.py` stays green.
5. `docs/gap_short_paper_instrument_20261114.md`: one paragraph under the "Entry TIF" paragraph: "Shortable guard"
   (what, why, the LITX 11/17 case, parity reason `not_shortable`).
6. Run `python3 -m pytest tests/test_gap_short_paper.py tests/test_gap_short_parity.py tests/test_gap_short_rules.py
   $(ls tests/test_locate_ledger*.py 2>/dev/null) -q --ignore=tests/integration` — all green. Then run
   `python3 scripts/gap_short_paper.py --help` to confirm the module imports.

## Rules
Sonnet, ≤ 40 calls. No git commit/stash, no service restart, no orders, no live-API calls (mock everything; the real
GET facts are above), no writes outside the three scripts, the two test files and the doc. Docstrings on every new
function. Return ≤ 120 words: diff summary per file with line numbers, test counts, and anything you were unsure of,
marked separately.
