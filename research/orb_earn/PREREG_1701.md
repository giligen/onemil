# PREREG — cell 1,701: the earnings-session ORB pool "E1" (FROZEN 2026-10-02 05:40 UTC, before any number)

Owner 10/2: "Bring one" — a NEW pool mechanism after the admission-pool line closed (gap bands, pools 21–30, criteria ×
own exits all ≈ 0 R per fill in regime; RESULT_1689c, RESULT_1693b). Programme count: 1,701.

## Mechanism (why this is not another admission variant)
Production ORB trades retail momentum on small-cap gappers; every weaker version of that signal added zero-R fills.
E1 trades a different population and a different cause: the opening range on the FIRST session after a scheduled
information shock (an 8-K Item 2.02 earnings release), on liquid names of any gap size. The 09:30–09:35 auction
resolves the overnight repricing; institutional repositioning continues intraday (the intraday leg of
post-earnings drift). Spread is tiny relative to R on this population (ADV ≥ $5M), so the cost model that killed
the small pools works FOR it. Frequency is the gating quantity: earnings seasons give 50–200 reporters per day.
Prior EDGAR cells (1,552–1,561, 1,633–1,636, 1,646–1,648) were DAILY-horizon holds on daily bars and failed; the
intraday ORB on the event session has never been read (inventory idea 22, VOID for "no calendar" — the calendar
exists: `research/edgar_desk/events_raw.csv`, 4.4M 8-K rows 2019→, `acceptance_datetime` in UTC).

## Population (fixed; everything knowable before 09:30 ET on the event session)
* Event: 8-K (or 8-K/A) with Item 2.02, acceptance datetime converted UTC → ET. Event session = the first regular
  session whose 09:30 ET open is ≥ 15 minutes after acceptance (accepted ≤ 09:15 ET → same day; later, including
  after-close releases → next session). Early closes from the exchange calendar. One event per symbol-session.
* Universe at the event session (point-in-time): common stock per the Databento definition feed
  (`research/scripts/pit_listings.py`), listed on the session, prior close ≥ $5, 20-session average dollar volume
  ≥ $5M from the Databento daily panel (`data/research/databento/*.parquet`), not an ETF/wrapper/warrant/unit/test
  ticker (`^Z[A-Z]ZZT$`). No gap requirement; the gap (official 09:30 open vs prior close) is a recorded feature.
* Window: 2024-07-01..2026-09-30 (the PIT panel). Halves: A = 2024-07..2025-06, B = 2025-07..2026-09. Both directions
  (select on A → test on B, and the reverse).

## Signal, entry, exit (production mechanics, shared code)
* Long sub-pool E1L (open ≥ prior close): buy the break of the 09:30–09:35 range high, order pre-placed at 09:35:00
  +5 s with the production chase cap; stop = range low; R = range; sanity: 0.5 % ≤ R/price ≤ 10 % (R must exceed the
  spread; halts excluded). Short sub-pool E1S (open < prior close): sell short the break of the range low, stop =
  range high, same sanity; Alpaca shortability recorded (today's flag, not PIT — the share is reported).
* Exit menu (one choice per side, selected on the selection half, tested on the other): X1 production (target 2 R,
  lock 1.75 → +0.5 R, flat 15:45 ET); X2 no target with the lock; X3 half at +2 R, rest no target with the lock.
* Slots: 8 per side per day, ranked by 09:35 relative volume descending (the one feature proven to order EV in
  production; no new ranking is fitted). Fills beyond 8 are counted for the frequency read only.
* Cost: measured NBBO at the entry minute per fill (never a band); stop slippage from the measured HOD tape (35 bps
  mean) on names < $20M ADV, 10 bps above; obtainability = next bar's open under the cap, share reported.

## Reads (per side × exit × window)
n fills, fills/week (and the in-season / off-season split), mean R, iid and day-clustered t, ex-top-5 % and
winner-capped mean, weekly P10 per fill, worst week $ and max drawdown $ at $375 risk, runner share (≥ 3 R MFE),
the gap-band decomposition (a feature, not a filter: ≤ −5 / −5..0 / 0..5 / ≥ 5 %), the cadence bar
(`scripts/cadence_bar.py`), and the union with the production book (production first; duplicates removed).

## Pass bar (frequency first, then edge)
A side passes only if, at the live slot config: ≥ 3 fills/week averaged over its test half, mean R ≥ +0.10 on BOTH
halves with day-clustered t ≥ 2.0 pooled, ex-top-5 % ≥ 0, obtainable share ≥ 80 %, and the cadence bar's weekly P10
≥ −2 R. A pass → independent rebuild from this prose (trade-by-trade on day+symbol) → paper as its own pool
(`universe.addon_pools`, pool_id E1, own exit) with real fills before any live word. A null states the MDE per side
(≈ 2.6 × SD / √n) and the fills/week actually found; no filter is added after seeing numbers.

## Multiplicity and not-allowed
2 sides × 3 exits × 2 directions = 12 reads + 2 union reads. Not allowed: a gap, volume or price filter added after
the read; using the day's own volume for admission; reading halves separately as verdicts (sign agreement only);
any paid data (Alpaca bars and EDGAR only).

## Steps and files (`research/orb_earn/`)
1. Calendar + candidates: `1701_calendar.py` → `1701_candidates.csv` (event, acceptance ET, session, symbol, prior
   close, ADV20, PIT status), with a completeness line (events in, dropped per reason). Day job, no bar store.
2. Minute bars of every candidate session through the designed appender (ONE writer, after 20:05 UTC, disk ≥ 5 GB,
   LOST count + completeness gate ≥ 95 % or VOID). `1701_backfill_queue.sh`.
3. Walk + reads: `1701_walk.py` → `1701_fills.csv`, `1701_reads.csv`, `RESULT_1701.md` (≤ 120 lines).
4. Independent rebuild from this file only: `REBUILD_1701.md`. The judge (main session) reads both.
Each agent returns ≤ 150 words; every number stays on disk.
