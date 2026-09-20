# FREEZE — ORB short mirror

PREREG.md and `build.py` were committed **before the first scoring run**:

| file | git hash at freeze |
|---|---|
| `research/orb_short/PREREG.md` | `9ff94465d67918a4fc21c343c4790576876ccd57` |
| `research/orb_short/build.py`  | `9ff94465d67918a4fc21c343c4790576876ccd57` |
| `research/orb_short/nbbo.py`   | `9ff94465d67918a4fc21c343c4790576876ccd57` |
| `research/orb_short/score.py`  | committed in the report commit; written **before** its first run, after
  `build.py` produced `sig.csv` / `ctl.csv` and before any net-of-cost number existed |

Artifacts produced under this freeze:

- `sig.csv` — 2,615 triggered short signals (TRAIN 1,751 · VAL 864), entered-inclusive (no-fill rows kept)
- `ctl.csv` — 1,407 matched controls (gap-down, no break in the 60-min window, shorted at the 09:36 open)
- `legs.csv` / `nbbo_short.csv` — 1,806 distinct (day, symbol, minute) legs, Alpaca SIP consolidated quotes
- `book_stage_a.csv`, `book_stage_b.csv`

TEST (>= 2026-06-01) was **never queried** — `build.py` hard-stops the universe SQL at `2026-05-31`.

Pre-registered deviation, declared in PREREG §2 before scoring: the trigger test is `bar.low <= L`
(the resting stop-limit's own level), not `bar.low < range_low`.
