# frames20 FREEZE

PREREG.md frozen at git HEAD `4ff9e674092092e2b134e067bf719bb77a3271c5` (branch `fix/spy-regime-shared-helper`) on 2026-09-20T12:33:26Z.

sha256 PREREG.md: 625fc416b26d6c70bd289612a164eb8adbac6d08d118a5eae3f06615d20f192b

Frozen inputs (not rebuilt by this frame):
* `research/mature_method/frames18/grid18.csv` sha256 42c4d0a9745a445d1e2924f2d7d06c55011f6484efe0104d5509e26434ce7300
* `research/mature_method/frames16/sw_*.csv` (17 files, unchanged)
* quote tables `frames16/nbbo16.csv`, `frames17/nbbo17.csv`, `frames18/nbbo18.csv`
* `research/mature_method/frames16/short_walk.py::swalk` (imported verbatim)
* `research/bf_zero/bars_sip.db` (read-only), `data/cache.db` (read-only)

Only NEW artifact: `p120.csv` (the P1x5 control walk, seed 57).

TEST (`day >= 2026-06-01`) is sealed; the walker and the scorer both assert it.
Nothing under trading/, config.yaml, orb.yaml, systemd or cron is touched by this frame.
