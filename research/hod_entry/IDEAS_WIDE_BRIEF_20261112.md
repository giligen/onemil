# Wide idea search for the HOD-break book — brief 2026-11-12 (owner: "go wide, search for ideas")

Task: a WIDE external search (web: academic papers, SSRN/arXiv, Quantpedia, practitioner blogs and books, prop-desk
playbooks, market-microstructure literature, options-overlay and liquidity-provision literature, broker/exchange
mechanics) for ways to make money from, around, or against an intraday high-of-day breakout signal in US small/mid
caps. Output ≥ 50 ideas that are NOT in `research/hod_entry/IDEAS_HOD_20261112.md` (read it first; those 60 are
mine — do not repeat them, extend them). This task IS the owner's request; do not pivot. Budget ≤ 60 calls. No git.
Write only `research/hod_entry/IDEAS_WIDE_20261112.md`. Never Read a file > 300 lines in full.

## What the machine is (facts, so every idea maps onto something real)
* Signal: a stock in the gapper/momentum universe makes a new high of day after a ≥ 5-bar consolidation within 4 %
  of the HOD, relative volume 1–5× its normal pace, stop distance ≥ 1 % of price; a resting buy stop-limit at the
  level (+15 bps cap), stop under the consolidation (1.5 % floor), target 2 R, stop-limit exits 20 bps under the stop,
  flat at 15:55 ET. ~50 fills/week on paper at $2K notional caps; the engine is live-capable (Alpaca, options level 3
  approved on both accounts, 5,209 names shortable).
* Measured: +0.075 R GROSS per fill at its own levels, ≈ 0.15 R cost (entry chase 8 bps, stop slip 13–35 bps on a
  41 % stop-out rate, spread). Backtest book 2025-01..2026-09: −0.22 R/fill net TRAIN, −0.30 VAL.
* Data on disk: 1-min SIP bars for every fill day (`bars_sip.db`), per-minute paths of 15,656 signals, NBBO
  half-spread per signal, L2 depth snapshots at entry on live rows, a retest dataset (8,973 rows), an arm/replay
  table (8,248 arms incl. unfilled), Databento PIT listings and daily bars, EDGAR tooling. Free: Alpaca SIP, Alpaca
  option 1-min trade bars (not quotes), FINRA/OCC/CBOE public files. Paid (needs owner GO): Databento MBO/quotes.
* CLOSED (do not propose again; cite only to extend): per-fill filters on this population (2,400+ cells: attention
  rank, OFI 5-min, candle shapes, vol-regime, SPY-regime, time-of-day, catalyst LLM), exit variants with a hard stop
  (35 cells ±0.04 R), no-stop/soft-stop/time-stop (1,735: +0.2 R paired, book still red), short the failed retest
  (1,736: −0.75 R), post-entry failure model short (1,675), options on this universe (1,738: 22 % have prints),
  overnight hold of closers at the high (tail-carried), PEAD/EDGAR event desk, index/ETF ORB and intraday
  momentum, daily reversal, 52-week-high sleeve, pre-earnings run-up.

## What each idea must carry (one block of ≤ 6 lines per idea)
Name · Source (URL / paper / book / practitioner, with the year) · Mechanism (WHY money changes hands) · Evidence
the source gives (numbers, sample, period; "anecdote" if none) · Mapping (which of our assets/engines it uses, what
data or permission is missing) · Cost sanity (does the edge survive ~15–40 bps round trip on a 1–2 % stop?).
Group by family (e.g. liquidity provision, microstructure, event-driven intraday, cross-sectional/relative, options
and volatility, horizon changes, portfolio construction, counterparty/behavioural, data edges, execution venues/
order types, risk/sizing, human-in-the-loop, markets other than US small caps). Mark each idea with a prior you
assign: STRONG (peer-reviewed or replicated), MEDIUM (practitioner with numbers), WEAK (anecdote). End with your
top 10 ranked by (mechanism strength × fit to our data × cost survival), one line each with the reason.

Search broadly and specifically: "opening range breakout" literature (Crabel, Zarattini-Aziz 2023-24 ORB papers,
their critiques), "intraday momentum" (Gao-Han-Li-Zhou 2018, Baltussen-Da-Lammers-Martens 2021 "hedging demand and
market intraday momentum"), "breakout failure / false breakout" studies, "stop-loss hunting" and "liquidity
sweeps" evidence, "market making small caps", "retail order flow / PFOF / odd-lot" studies, "short-sale
constraints and overpricing intraday", "SSR uptick rule effects", "LULD halt resume returns", "closing auction
imbalance", "options gamma and intraday price paths / pinning", "day-trader herding Reddit / social sentiment
intraday", "high-frequency VWAP execution for retail", "IEX D-Limit / midpoint orders for retail brokers",
"Ross Cameron / Warrior Trading statistics", "SMB Capital playbooks", "Quantpedia intraday strategies", "Kris
Abdelmessih options for directional ideas", "pairs / sector-neutral intraday momentum", "earnings-day intraday
drift", "pre-market high break", "crypto 24h high breakout" (different microstructure, same engine).
