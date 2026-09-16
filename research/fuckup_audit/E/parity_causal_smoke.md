# Stage E — parity vs `B/candidates4.csv`

`/home/ec2-user/onemil/research/fuckup_audit/E/candidates_causal_smoke.csv` vs `/home/ec2-user/onemil/research/fuckup_audit/B/candidates4.csv`, days 2025-07-02..2025-07-15 (3), families F13, F6, F8.

- rows: mine **3,326**, Stage B (same days+families) **10,332**
- symbol-days: 3,493; served by `data/cache.db` in Stage B 1,240; present in `bars_sip.db` 1,966; **tape-identical 1,966**
- signal rows on tape-identical keys present in BOTH files: **1,231**
- signal rows only in mine: 1,083; only in Stage B: 8,089 (the two universes are different by design — this is not an error)

| column | n compared | max abs diff | n diff > 1e-6 |
|---|---:|---:|---:|
| `sig_m` | 1,231 | 0 | 0 |
| `level` | 1,231 | 0 | 0 |
| `stop` | 1,231 | 0 | 0 |
| `range_so_far_pct` | 1,231 | 0 | 0 |
| `next_entry` | 992 | 0 | 0 |
| `next_entry_m` | 992 | 0 | 0 |
| `next_rr_2r` | 992 | 0 | 0 |
| `rest_entry` | 840 | 0 | 0 |
| `rest_entry_m` | 840 | 0 | 0 |
| `rest_rr_2r` | 840 | 0 | 0 |

**Task bar (entry_next / rr_2r to 1e-6): max abs diff 0, 0 row(s) over tolerance.**

