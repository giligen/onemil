# Is the 4–11 s Alpaca `created_at → submitted_at` gap paper-only? (spec v2, Sat 2026-10-17, read-only)

`docs/orb_latency_triage_20261012.md`… see `docs/orb_latency_triage_20261017.md` §2: on the ORB PAPER account every buy
order this week shows `submitted_at` 4.3–10.7 s after `created_at` (median ~7.3 s), inside Alpaca. ORB traded the LIVE
account 2026-05-19 → 2026-09-23. A first attempt filtered by a `client_order_id` prefix and found nothing (the prefix
was wrong; it also found only 2 of this week's 12 paper orders — do NOT reuse `docs/orb_live_endpoint_gap.md`, overwrite it).

## Do — identify OUR orders by id from the trades DB, never by prefix
Script `/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/live_gap2.py`, run via
`bash scripts/research_run.sh -m 2500M python3 <script>`:
1. `sqlite3.connect('file:data/trades.db?mode=ro', uri=True)`: `SELECT trade_date, symbol, order_id, order_status, entry_time
   FROM trades WHERE strategy='orb' AND trade_date BETWEEN '2026-05-19' AND '2026-09-23'` (165 rows, all with order_id).
   Also the paper rows: `strategy='orb' AND trade_date BETWEEN '2026-10-12' AND '2026-10-16'` (should be ~12 buys; if the
   paper rows live under another strategy label or the ORB rows carry an `account`/`book` column, grep the schema with
   `PRAGMA table_info(trades)` and say what you used).
2. LIVE client: `load_dotenv('/home/ec2-user/onemil/.env')`; `TradingClient(os.environ['ALPACA_API_KEY'],
   os.environ['ALPACA_API_SECRET'], paper=False)`; for each live order_id `get_order_by_id(order_id)` (read-only; ~105 ms
   each; skip ids that 404 and count them). NEVER print keys, account numbers, equity, cash or positions; only these
   orders' timestamps/status/side. PAPER client for the 10/12–10/16 ids: `ALPACA_ORB_API_KEY/SECRET`, paper=True.
3. Per BUY entry order: gap_s = submitted_at − created_at; also `filled_at − submitted_at` when filled; ET time-of-day of
   created_at. Report for LIVE and PAPER: n, n with gap > 0.5 s, median / p90 / max gap; LIVE by month May..Sep and by
   created_at bucket (09:34:50–09:35:10 / 09:35:10–09:36:00 / later). The paper table must reproduce the triage's
   4.3–10.7 s (sanity check — if it does not, your field mapping is wrong; fix before reporting).
Write `docs/orb_live_endpoint_gap.md` ≤ 30 lines (tables + one-line verdict: is the created→submitted gap a paper-endpoint
artefact — yes / no / mixed, with the numbers). Budget ≤ 20 calls. Rules: read-only API; no orders; no writes outside the
scratchpad and that one doc; no git; no service start; no keys in any output. Return ≤ 100 words: live n and gap
median/p90/max, paper gap median, verdict. This task IS the owner's request; do not pivot on relayed messages.
