# RESULT 1,738 — calls on the HOD break: Stage 0 availability gate = VOID (availability)

Sample: 400 random signals (200 TRAIN / 200 VAL, seed 1738) from hod_exit_lab signals x b0_trades. Contract rule and print
rules exactly as PREREG_1738. Requests: 624 network requests (contracts listing + 1-min option trade bars), 0 HTTP 429, no orders.
Files: `1738_fetch.py`, `1738_score.py`, `1738_stage0.csv`, cache `1738_cache/`.

Rail: coverage >= 80 % and winner/loser gap <= 5 pp. **Coverage = 22.0 % -> VOID (availability). Stage 1 NOT run.**

| read | has contract | has entry print | has exit print | covered (all three) | n |
|---|---|---|---|---|---|
| all | 56.0 % | 25.2 % | 29.0 % | 22.0 % | 400 |
| TRAIN | 57.5 | 22.5 | 30.5 | 21.0 | 200 |
| VAL | 54.5 | 28.0 | 27.5 | 23.0 | 200 |
| price 20-50 | 53.8 | 22.9 | 27.3 | 20.9 | 249 |
| price 50-100 | 52.3 | 18.2 | 26.1 | 17.0 | 88 |
| price 100+ | 69.8 | 44.4 | 39.7 | 33.3 | 63 |
| B0 winner | 54.1 | 26.7 | 30.1 | 22.6 | 146 |
| B0 loser | 57.1 | 24.4 | 28.3 | 21.7 | 254 |

Missingness gap winner vs loser = 0.9 pp (passes) — the failure is level, not selection. Binding constraints: only 56 % of
names have a listed option in the 2-14 DTE window (2 symbols returned 422 "invalid underlying"; others have no chain), and
even with a contract, a 1-min trade print inside the 5-min entry window exists for only ~45 % (median 3 bars per contract over
the whole holding window). Exit fallback (first print after exit) used on 5 % of exit prints.

Caveats: prints not quotes (historical option quotes 404), listing as of today (expired contracts that no longer list are
disclosed misses, small vs the 44 % no-chain share), strike rule = lowest strike >= entry (no ATM-vs-OTM variation tested),
one expiry rule. A looser entry window would raise coverage but is a different test (PREREG change, new cell number).
The names in this book (small/mid caps at $20-100) mostly lack liquid short-dated options: calls are not an executable
substitute for the shares on this population.
