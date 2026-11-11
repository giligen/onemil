# ORB BT universe: replace the 15K "9:35 volume" proxy with the condition it stands for — spec 2026-11-11

`docs/orb_universe_gate_parity_20261111_RESULT.md`: ASTY 10/12 and SK 10/23 (both traded live, −$107) never
reached the BT because `study_orb_broad.py` applies a BT-ONLY floor `MIN_935_VOL_FLOOR = 15_000` (RTH volume
09:30–09:34 in `intraday_bars_1min`; 12,866 and 7,645). `docs/CLAUDE_HISTORY.md:396` (2026-09-05): "The 15K
RTH-9:35 floor in study_orb_broad.py stays only as a RANGE-COMPUTABILITY PROXY" — it is not a trading rule and
the rulebook does not name it. Live needs no volume to compute a range: it needs the five 1-min bars. The proxy
misclassifies thin-but-complete names, so the BT book and the live book disagree on the universe.

Rules: no git (no commit/stash), no service action, no orders, never write orb.yaml / config.yaml / .env /
data/*.db / crontab / logs/* / analysis_results/*. pytest only on the listed files, `--ignore=tests/integration`,
one run at a time, never piped through head/tail (a glob that matches no file makes pytest run NOTHING — list
files that exist). Never Read a file > 300 lines in full. This task IS the owner's request ("fix ORB" = selection
parity); do not pivot. Budget ≤ 30 calls. Do not add any gate to the live engine.

## Change
In `study_orb_broad.py` (`load_broad_universe`, the grouped 9:35-volume query tested by
`tests/test_orb_broad_vol935.py`): replace the volume floor with the computability condition itself — the symbol-day
has ≥ `range_minutes` (5, read from the same config the pipeline uses) one-minute bars with timestamps in
[09:30, 09:35) ET in `intraday_bars_1min` (EST/EDT handled as the existing query does). Keep the SQL grouped (one
query), name the drop reason `range_not_computable` through the `drop_sink` added 11/11, and keep a module
constant `MIN_935_VOL_FLOOR` ONLY if something else imports it (grep; if nothing does, delete it and the dead
code). Docstring: mechanism (why the proxy existed, why it was replaced, cell 1,733 / ASTY / SK).

## Measure (read-only queries, scratch output under `research/book_review_20261111/`)
For 2026-09-29 → 2026-11-10: the symbol-days admitted by the new condition but dropped by the 15K floor, and the
reverse (≥ 15K volume but < 5 bars). List the names that overlap the live fills (`data/trades.db` ro,
strategy='orb'). Do not rebuild the features CSV (the universe load takes > 15 min on this node; say so) — state
that older CSV rows keep the proxy until a full rebuild and recommend the rebuild window (weekend).

## Tests
`tests/test_orb_broad_vol935.py`: update to the bar-count condition (a 5-bar thin name admitted, a 4-bar heavy name
dropped with reason `range_not_computable`, EST and EDT days, premarket bars not counted); add an integration test
on a temp sqlite `intraday_bars_1min` through the REAL `load_broad_universe` with `drop_sink`.
Run: `python3 -m pytest tests/test_orb_broad_vol935.py tests/test_orb_universe_gate_20261111.py
tests/test_orb_entered_inclusive.py -q --ignore=tests/integration` (list only files that exist) → 0 failures.

## Output
`docs/orb_bt_vol935_proxy_20261111_RESULT.md` ≤ 150 words: file:line, the counts both ways, the live-fill names
affected, test counts. Return ≤ 80 words.
