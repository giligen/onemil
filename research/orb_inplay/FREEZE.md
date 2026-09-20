# orb_inplay — FREEZE

PREREG committed at: **40b4e6eebd514ac1c4874b8f3cefdf98bd624160**
Tree at freeze: **40b4e6eebd514ac1c4874b8f3cefdf98bd624160**  branch `fix/spy-regime-shared-helper`
Frozen: 2026-09-20T14:29:36Z

## Frozen artefacts (built BEFORE any P&L was looked at)
| file | what |
|---|---|
| `fetch_open.py` | 09:30-09:34 ET tape for the whole ADV>=1M universe, Alpaca SIP, RAW adjustment |
| `universe.parquet` | per-day universe with causal ADV20 / ATR14 / prev_close from `daily_bars` |
| `open5.parquet` | the open tape |
| `select_picks.py` | RVOL -> top-20 stocks in play, direction from the 5-min candle |
| `fetch_bars.py` | 09:35-15:55 ET bars for the selected picks |
| `nbbo_sample.py` | stratified measured-NBBO sample -> `hs_table.json` |
| `score.py` | simulator + the PREREG statistics |

## Data-source decision recorded at freeze (mid-run change #1)
`cache.db::intraday_bars_1min` covers only **~10 %** of the ADV>=1M universe
(173/1,724 names on 2025-03-05; 233/2,237 on 2026-03-04) — it is the gap-up/mover seed.
Ranking the paper's universe off that table would have reproduced exactly the look-ahead
population that killed `research/bf_zero`. The open tape is therefore fetched from
**Alpaca SIP, adjustment=RAW**, for every universe name on every day. `daily_bars`
(ADV20, ATR14, prev close) is still the cache's. This deviation is declared here, before scoring.

## Seal
TEST = **>= 2026-06-01**. `fetch_open.py` and `fetch_bars.py` hard-stop at 2026-05-31;
no TEST day has been fetched, ranked or scored in this run.
