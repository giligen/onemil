# ORB BT: 15K 9:35-volume proxy replaced by the bar-count condition (2026-11-11)

Change: `study_orb_broad.py` `load_broad_universe` last gate is now >= `range_minutes` (ORB_RANGE_MINUTES, 5) one-minute
bars in [09:30, 09:35) ET (`opening_range_bar_counts_for_pairs`, one grouped query); drop reason `range_not_computable`
via `drop_sink`; `MIN_935_VOL_FLOOR` deleted (nothing imported it).

Counts 2026-09-29..11-10 (1,217 SQL candidates; `research/book_review_20261111/vol935_proxy_measure.out`):
old 15K floor 1,163 kept, new 1,107 kept. Admitted only by new: 11 (live fills ASTY 10/12, SK 10/23). Dropped
only by new (>= 15K volume, < 5 bars): 67, no live fills; most are 11/02-11/10 cache gaps (bars missing in the
cache, live could range them) - BT cannot simulate those either way.

Tests: 52 passed (vol935 7, gate 37, entered_inclusive 8).

Features CSV not rebuilt (universe load > 15 min): older rows keep the proxy until a full rebuild. Recommend a
weekend rebuild of 2025-01-01..present, after refilling the 11/02-11/10 bar gaps.
