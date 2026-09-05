# Ignition — point-in-time universe / survivorship check (2026-09-05)

**Verdict: survivorship is NOT load-bearing for the ignition book.** The
candidate symbol-days the cache-based universe never saw would have added
104 river trades worth **−$11.2K** (on a −$567K ungated river) and **9
complex-confirmed trades worth +$3.2K** (on the +$254K CC proxy book,
+1.2%). The delisted-name cohort proper is 40 trades / −$2.6K / 0 in CC.
No headline in `RESTING_MODEL.md` moves by more than ~1%.

## Method

- Point-in-time daily universe: Databento `EQUS.SUMMARY` ohlcv-1d, every
  US equity that existed on each date (delisted included), Jan-25 → Sep-26
  (`data/research/databento/equs_daily_2025_2026.parquet`; note the
  vendor's daily `ts_event` is 00:00 UTC of the trade date — a New York
  conversion shifts every bar one day back; fixed 9/5).
- Same capcheck prefilter applied to both sources
  (`build_universe_pit.py`): open ≥ 1.95, gap < 5.5%, high ≥ open×1.09,
  volume×high ≥ $2M. PIT 45,589 symbol-days vs cache-based 44,478;
  **2,776 (6.1%) invisible to the cache** (Jan-25 13% → Mar-26 0.5% →
  Aug-26 11.5%).
- Every invisible symbol classified against Alpaca's full asset list
  (all statuses, `data/research/databento/alpaca_assets_all_20260905.csv`;
  `pit_missing_classified.csv`):

| class | symbol-days | meaning |
|---|---|---|
| survivorship (unknown to Alpaca / inactive) | 1,751 | delisted or renamed before the cache was seeded — the true bias |
| wrapper_policy (active, dropped by `_is_common_stock`) | 1,080 | new 2× single-stock ETFs (AXTX, LITZ, SNDQ, CBRG…) — deliberately excluded by the universe builder's filter; live never sees them either |
| ticker_reuse / vendor disagreement (active tradable common, absent) | 989 | e.g. LAZR = Tema ETF since 2026-06-25 (old Luminar rows are survivorship); a few vendor OHLC disagreements at the threshold |
| correct exclusion (not tradable at Alpaca, test ticker) | 292 | ZVZZT + names Alpaca will not trade |

- 1-min bars for the 2,575 invisible symbol-days pulled from Databento
  `EQUS.MINI` ohlcv-1m into `topup.db` ($0.45; `fetch_missing_databento.py`)
  and simulated with the unchanged `capsim.py` (`CAPSIM_UNIVERSE`
  override → `trades_PIT.csv`); rejects: 1,723 `u_dollar_2M`, 279
  level-not-crossed, 228 no bars, 156 gap-ORB-territory.
- Anchor cohorts recomputed on the union (`pit_analyze.py`,
  `trades_union_pit_annotated.csv`): 7 baseline trades flip CC status.

## Ledger

| book | baseline | PIT additions | union |
|---|---|---|---|
| river (ALL) | 9,515 tr / −$567,395 / WR 42.8% / 278 monsters | 104 / −$11,191 / 46.2% / 6 | 9,619 / −$578,586 |
| CC proxy | 1,077 / +$254,323 / 51.3% / 37 | 9 / +$3,175 / 66.7% / 2 | 1,086 / +$257,498 |

By bucket (river): survivorship 40 tr −$2.6K (WR 45%); ticker-reuse/vendor
20 tr −$12.3K (WR 35% — six full stops, three of them on 2025-01-02);
wrapper_policy 43 tr +$2.4K (WR 51%, all 9 CC trades live here: SPCU
+$3.9K, SPCH −$2.6K). Per era: 25H1 −$3.5K, 25H2 −$7.5K, 2026 −$0.2K.

## Caveats

- Same shared biases as the cap study: no news-leg catalyst gate, trade-
  level (no slots), Databento consolidated minute bars vs Alpaca SIP.
- Only 50/104 PIT symbols resolve an anchor (delisted names are absent from
  the asset-class map) — the CC delta is a floor, but the whole cohort is
  small enough that no plausible resolution changes the verdict.
- The wrapper_policy bucket is a UNIVERSE-RULE question, not survivorship:
  `AlpacaClient._is_common_stock` drops new leveraged wrappers while older
  ones (NVDL-era) sit in daily_bars from earlier cache builds — an
  accidental inconsistency to resolve deliberately (owner call), shared by
  live and BT for all three books.
