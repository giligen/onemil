# PREREG — cell 1,700z: extend the momentum-sleeve backtest through October 2026 and reconcile it with the paper/live sleeve (Fable, 2026-10-31; owner: "Extend the bt")

## Question
The live rule (guarded sleeve, `REBUILD_1700tu.py`, panel through 2026-09-30) is not scored for October. The paper
sleeve ($20K, since 10/2) made −15.5 % in October (SPY −1.8 %); the live slice ($10K, since 10/19) −5.5 %. Is that
what the RULE did (parity), or did selection, the MISSED 10/26 rotation (TXSE enum bug, held the 10/19 names two
weeks) and execution add to it? No fitting, no new rule, no parameter is touched — this is a measurement.

## Data (free; no Databento pull — owner's standing rule)
* Existing panel `research/momentum_weekly/panel_2016_2026.parquet` (READ ONLY — never rewrite it) — first learn its
  conventions: columns, adjusted or raw closes, the universe/common-stock filter source (`PREREG_1700b.md`, PIT
  definitions `data/research/databento/pit_definition/def_YYYYMM.parquet`, `security_type == 'C'`), volume/ADV fields.
* Extension: Alpaca daily bars 2026-09-01 → 2026-10-30 for every symbol present in the panel in 2026-09 (overlap
  September to CHECK the price-scale convention bar-by-bar: report the share of (symbol, day) closes that differ by
  > 0.1 % between the panel and Alpaca in September; if > 2 %, stop and report — do not guess an adjustment).
  Keys: `ALPACA_MOM_API_KEY/SECRET` via `load_dotenv('/home/ec2-user/onemil/.env')` (never print keys), data API
  `https://data.alpaca.markets/v2/stocks/bars`, `timeframe=1Day`, `adjustment` set to match the panel's convention,
  batch ≤ 200 symbols per request, `feed=sip` if the key allows else `iex` (say which). Completeness gate: count LOST
  symbols; refuse to score if > 5 % of the September universe is missing (CLAUDE.md fetch rule). October listings:
  the PIT definitions end 2026-09 — use the September definitions for October membership (knowable at the signal
  date; the only approximation, state it). `df -h /` ≥ 5 GB before fetching; write
  `research/momentum_weekly/panel_ext_202610.parquet` (new file, atomic write), never `--build-cache`, never data/*.db.

* REUSE, don't rewrite: `scripts/momentum_sleeve.py` already fetches `adjustment=ALL`, `feed=SIP` daily bars with
  the LOST/completeness machinery (`fetch_batch_resilient` l.142, `fetch_panel` l.176, `COMPLETENESS_MIN`/`LOST_MAX`
  l.77–78) into a parquet cache (`KEEP_CACHES = 2`; find the cache path/glob in the script — the two most recent
  Monday caches, 10/19 and 10/26, already hold adjusted bars through those dates). Import those functions. The
  research panel may be RAW (Databento) while the sleeve is adjusted=ALL — that is exactly what the September
  overlap check decides; if they disagree only on split/dividend names, say so and use the sleeve's convention
  for the extension with the affected names listed.

## Cells (frozen)
Z1 **Rule replay.** Run the guarded sleeve exactly as `REBUILD_1700tu.py` does (import its functions; no copy-edit of
   the rule) on panel + extension for the signal Fridays 10/2, 10/9, 10/16, 10/23 (trades Mondays 10/5, 10/12,
   10/19, 10/26) with the cost model of the rebuild. Output weekly and October returns at $20K; SPY beside it. Also the
   half-gate book for reference (the live gate is shadow-only, scale 1.0 — `logs/momentum_sleeve_shadow_gate.csv`).
Z2 **Selection parity.** The 20 BT names per Monday vs the names the paper sleeve actually held (`logs/momentum_sleeve_weekly.csv`
   `names_in/names_out` from 10/2; `logs/momentum_sleeve_ledger.csv`): overlap count per Monday, and the names that
   differ with their ranks/scores in the BT. ≥ 18/20 every Monday = parity; otherwise list the cause per name (data,
   universe filter, tie convention).
Z3 **Missed-rotation cost.** BT variant that SKIPS the 10/26 rotation (holds the 10/19 names through 10/30) vs Z1:
   the $ difference at $20K = what the TXSE bug cost (or saved).
Z4 **Execution.** BT fills at Monday's official open with 15–20 bp vs the paper ledger's actual fills
   (`slip_bps_vs_open`: paper buys −10 bps mean n 90, sells −28 bps n 63; live buys +55 bps mean n 20 on 10/19, p90
   309). Translate the ledger's measured slip into October $ at $20K; state the live 09:45-vs-open gap separately.
Z5 **Reconciliation.** paper October −15.5 % = Z1 rule + Z2 selection + Z3 missed rotation + Z4 execution + residual.
   Residual > 2 pp → say "unexplained" and list the candidates; never force the sum.

## Pass bar / reading
Nothing ships from this cell. Parity = Z2 ≥ 18/20 each Monday AND |Z5 residual| ≤ 2 pp. Report Z1's October number
with SPY and with the BT's own monthly distribution 2017–2026 (what percentile is a −15 % month for this rule;
`REBUILD_1700tu_weekly.csv` has the series). Independent check: this cell IS the check of the live book against the
rule; a second rebuild is not required because no claim about edge is made.

## Agent (Sonnet; ≤ 40 calls; one python process at a time; no commit/stash, no restart, no orders, no writes to
## data/*.db, config.yaml, orb.yaml, .env, crontab, logs/*; never Read a file > 300 lines in full; this task IS the
## owner's request, do not pivot on relayed messages)
Script `research/momentum_weekly/cell_1700z.py` (fetch step resumable, verbose progress), outputs
`1700z_weekly.csv`, `1700z_names.csv`, `RESULT_1700z.md` (every cell, the convention check, LOST count, deviations).
Return ≤ 150 words.
