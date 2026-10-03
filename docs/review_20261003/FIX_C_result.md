# FIX C result — momentum sleeve (scripts/momentum_sleeve.py, tests/test_momentum_sleeve.py)

* C1: `COMPLETENESS_MIN = 0.90`; `completeness_info/path/write/read`, `completeness_refusal`. Fresh fetch writes
  `daily_<asof>.parquet.completeness.json` {asof, requested, with_asof_bar, ratio, lost}; skip-fetch reads it.
  Refuse (exit 2, ERROR `MOM REFUSED: completeness …`, `[MOM] … REFUSED` Telegram on --submit, no sells/buys) when
  file missing / other asof / ratio < 0.90 / any held or top-20 name in `lost`. --force does not override.
  Gate is evaluated after the plan, before any order (also in dry-run, which prints the line and exits 2).
  `prune_caches` removes the matching JSON.
* C2: `last_rebalance` + `last_rebalance_run` saved right after `execute()`, before sync/ledger/marking.
* C3: `apply_fractionable`: non-fractionable buy -> whole qty floor(notional/price), 0 -> skipped with WARNING,
  residual shown in the plan line; lookup failure = WARNING + notional order. Selection unchanged.
* C4: `gate_label(info, asof)`: `gate n/a (full size)` / `gate STALE ON|OFF (pNN)`. n/a emits WARNING from
  `shadow_gate_info` AND from `ms.gate_scale` (2 lines, both pre-existing, trading/ file out of scope).
* Tests: 55 -> 68 (13 new). Dry-run (Saturday, so `--force --asof 2026-10-02 --skip-fetch`; no --submit):
  `COMPLETENESS: requested 13510 symbols, 13202 with bars, 12122 with an 2026-10-02 bar (90%), LOST 0`
  `MOM REFUSED: completeness completeness file missing for 2026-10-02` (Friday's cache predates the JSON).
* AMENDED dry-run (Friday cache): `COMPLETENESS: ... LOST 0 (0.0%, max 5%), liquid 788/788 with an asof bar (100.0%, min 98%)`
  then `MOM REFUSED: completeness completeness file missing for 2026-10-02` (only because the JSON does not exist yet;
  Monday's 11:50 prefetch writes it). The 0.897 raw ratio is normal (dormant listings) and no longer gates.
