"""Read-only: symbol-days admitted by the bar-count condition vs the old 15K 9:35-volume floor, 2026-09-29..11-10."""
import sqlite3, re
import study_orb_broad as sb

CACHE = 'file:data/cache.db?mode=ro'
conn = sqlite3.connect(CACHE, uri=True)
src = open('study_orb_broad.py').read()
q = re.search(r'query = """(\n    WITH daily_ranked.*?)"""', src, re.S).group(1)
cur = conn.execute(q, ('2026-09-29', '2026-11-10', sb.MIN_GAP_PCT, 1e18, sb.MIN_PREV_DAY_VOL,
                       sb.MIN_OPEN_PRICE, sb.MAX_OPEN_PRICE))
cands = [(s, str(d)) for s, d in cur.fetchall()]
vol = sb.rth_volume_by_935_for_pairs(conn, cands)
bars = sb.opening_range_bar_counts_for_pairs(conn, cands, 5)
old = {p for p in cands if vol[p] >= 15_000}
new = {p for p in cands if bars[p] >= 5}
add, lost = sorted(new - old), sorted(old - new)
tc = sqlite3.connect('file:data/trades.db?mode=ro', uri=True)
live = set()
for r in tc.execute("SELECT symbol, trade_date FROM trades WHERE strategy='orb' AND fill_price IS NOT NULL"):
    live.add((r[0], r[1]))
print(f"candidates {len(cands)} | old(15K) {len(old)} | new(5 bars) {len(new)}")
print(f"ADMITTED by new, dropped by 15K: {len(add)}; live-fill overlap: {[p for p in add if p in live]}")
print(f"DROPPED by new, passed 15K (>=15K vol, <5 bars): {len(lost)}; live-fill overlap: {[p for p in lost if p in live]}")
print("added:", [(s, d, vol[(s, d)], bars[(s, d)]) for s, d in add])
print("lost:", [(s, d, vol[(s, d)], bars[(s, d)]) for s, d in lost])
