# frames18 FREEZE

PREREG frozen before any scoring of the F55 grid.

* git HEAD at freeze: `432979e098fdf7c5b8f8ca365217d8d5d8e53022`
* branch: `fix/spy-regime-shared-helper`
* frozen at (UTC): 2026-09-20T12:11:27Z
* PREREG sha256: 600fcd1d7583fb847310f1469665fb70a10a1dda6c716f8a43a2b7ea62c4d9ea
* Cells declared: 12 (k in {0.4,0.6,0.8,1.0}% x w in {3,5,10} min). Programme count 1,254 -> 1,266.
* TEST window `day >= 2026-06-01` sealed; scorer asserts max scored day < 2026-06-01.
* Inputs re-used verbatim, not re-derived:
  - `research/mature_method/frames16/sw_*.csv` (signal population, reacting counterfactual)
  - `research/mature_method/frames16/short_walk.py::swalk` (exit walk, imported)
  - `research/mature_method/frames16/nbbo16.csv`, `research/mature_method/frames17/nbbo17.csv`
  - `research/fuckup_audit/O_halt/PASSIVE/borrow_flags.csv` (ETB rail)
  - `research/mature_method/frames15/common15.py` (clustered t, null_green, week_shape)
  - `data/cache.db::daily_bars` (prev close for the Reg SHO 201 rail)
