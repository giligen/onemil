# Stage E — parity vs `B/candidates4.csv`

`/home/ec2-user/onemil/research/fuckup_audit/E/candidates_causal.csv` vs `/home/ec2-user/onemil/research/fuckup_audit/B/candidates4.csv`, days 2025-01-17..2026-09-04 (410), families F13, F6, F8.

- rows: mine **583,769**, Stage B (same days+families) **1,636,862**
- symbol-days: 542,876; served by `data/cache.db` in Stage B 214,511; present in `bars_sip.db` 289,225; **tape-identical 289,225**
- signal rows on tape-identical keys present in BOTH files: **217,576**
- signal rows only in mine: 132,846; only in Stage B: 1,185,939 (the two universes are different by design — this is not an error)

| column | n compared | max abs diff | n diff > 1e-6 |
|---|---:|---:|---:|
| `sig_m` | 217,576 | 0 | 0 |
| `level` | 217,576 | 0.0005 | 13 |
| `stop` | 217,576 | 0 | 0 |
| `range_so_far_pct` | 217,576 | 0 | 0 |
| `next_entry` | 177,747 | 0 | 0 |
| `next_entry_m` | 177,747 | 0 | 0 |
| `next_rr_2r` | 177,747 | 0 | 0 |
| `rest_entry` | 142,257 | 0.0003 | 3 |
| `rest_entry_m` | 142,257 | 0 | 0 |
| `rest_rr_2r` | 142,257 | 3.1e-05 | 3 |

**Task bar (entry_next / rr_2r to 1e-6): max abs diff 0, 0 row(s) over tolerance.**

