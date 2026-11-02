# `/books` HTML report — design + build spec (Fable, 2026-11-02; owner: "do also an html version! proper design,
tl;dr at the top, drilldown per strategy, collapsible")

Two scripts, one data contract. `scripts/book_status.py --json` is the ONLY data source (extend it, section 1);
`scripts/book_status_html.py` renders that JSON to ONE self-contained HTML file (section 2). Fable publishes the file as
a claude.ai artifact from the `/books` skill; the HTML must therefore follow the artifact page contract (section 3).

## 1. Data contract — extend `book_status.py --json` (text output unchanged)
Verify every column with `pragma table_info(trades)` before use; never guess names. Add:
- `generated_at_et` (ISO, ET) and `session_date`.
- `hod.fills[]` and `orb.fills[]`: `symbol, entry_ts_et (HH:MM:SS or null), fill_price, stop, target, shares,
  risk_usd, pnl (null while open), exit_reason (null while open), exit_ts_et, entry_quote_spread_bps (null if absent),
  pricing_method`. Source: `trades` rows with `trade_date = session_date`, `strategy` in (`hod_break`, `orb`).
- `hod.daily[]` / `orb.daily[]`: last 12 sessions `{date, pnl, fills}` from `trades` (closed rows), plus
  `hod.cum_since_start` (list of `{date, cum_pnl}` since 2026-09-29) — the sparkline.
- `hod.gate`: `{arm_skip_r_pct, arm_skip_min_price, arm_skip_symbols, cap_hits, squeeze_lines, squeeze_enabled
  (from the gates line if present), concurrency_cap, per_day_cap, risk_usd, dry_run}`.
- `orb.vetoes[]`: `{symbol, reason}` parsed from `PDR VETO` / `Q1 filter` lines; `orb.ramp` as today.
- `mom.paper` / `mom.live`: `{equity, start_equity, pct_vs_start, pause_rail, halve_rail, last_rebalance, names_in[],
  names_out[], holdings_n, weekly[] (last 8 rows of the weekly CSV: date, equity, week_pct if derivable from
  consecutive equities, spy_pct if the CSV has it), parity[] (last 4 rows of logs/mom_bt_parity.csv: date, names_n,
  resid_pp)}`.
- `ops`: `{service_active, boot_utc, errors_today, warnings_today, journal_lines_today (the count from one
  journalctl pass — reuse the pass already made), crons[] ({name, schedule, log, last_write, errors_today})}`.
- `flags[]`: pre-committed attention rules, each `{severity: 'crit'|'warn'|'info', book, text, why}`:
  crit — `errors_today > 0`; any MOM equity below its pause rail; HOD/ORB closed P&L today ≤ day-kill.
  warn — MOM equity within 10 % of the pause rail; `squeeze_enabled and squeeze_lines == 0 and any HOD position
  open after 14:00 ET`; a cron whose log's last write is older than its most recent scheduled run; a WARNING text
  repeated ≥ 3 times today (give the text, trimmed to 80 chars); service inactive during 09:30–16:00 ET.
  info — a book with 0 fills on a session day; ARM SKIP symbols > 100.

## 2. The page — `scripts/book_status_html.py [--json path | runs book_status --json] --out reports/book_status_YYYYMMDD.html`
Add `reports/` to .gitignore. Pure stdlib (json, html.escape, datetime); no templating library; inline SVG charts; one
Google Fonts link; no other external resource. ≤ 400 KB.

### Visual identity (write these as the `:root` tokens, dark-first: this is a trading desk's blotter read in the
evening; light theme designed with equal care, not inverted)
- Color tokens: `--bg #0e1218`, `--panel #151b24`, `--panel-2 #1b2330`, `--line #263041`, `--fg #e6e9ee`,
  `--muted #8a94a6`, `--accent #e3a33a` (desk-lamp amber: headings' eyebrows, the "now" marker, focus rings),
  semantic `--gain #3fb67a`, `--loss #e5604f`, `--warn #e3a33a`, `--crit #ff6b5b`, `--info #5b9dd9`.
  Light: `--bg #f6f4ef`, `--panel #ffffff`, `--panel-2 #f0ede6`, `--line #ddd8cc`, `--fg #1a1f27`, `--muted #5f6775`,
  same semantic hues darkened 10 %. Three-state theme CSS exactly as the artifact contract requires (dark on bare
  `:root` + `color-scheme: dark`; light under `(prefers-color-scheme: light)` guarded `:root:not([data-theme="dark"])`
  and under `:root[data-theme="light"]`), `body { background: var(--bg) }`.
- Type: Google Fonts `IBM Plex Sans Condensed` (display: masthead, tile numbers' labels), `IBM Plex Sans` (body),
  `IBM Plex Mono` (every number, table cell, timestamp; `font-variant-numeric: tabular-nums`). Fallback stacks.
  Scale: 12 / 13 / 15 / 18 / 28 / 40 px. Uppercase eyebrows 11 px, letter-spacing 0.08em.
- Layout: single column, max-width 1100 px, 16 px side gutter at every width, `padding-block` for vertical space.

### Structure, top to bottom
1. **Masthead** — "OneMil books" (display), right: `Mon 2 Nov 2026 · 15:20 ET` and a service pill
   (`● active · boot 12:30 UTC · 0 errors`; the dot is gain/crit). One line under it: the session state
   ("session open, 40 min to the flat" / "after the close" / "no session (weekend)") from the clock.
2. **TL;DR tiles** — CSS grid `repeat(auto-fit, minmax(230px, 1fr))`, one tile per book: HOD paper, ORB paper, MOM
   paper, MOM live; a fifth compact tile "Paused" listing BF and R2G. Each tile: eyebrow `HOD · PAPER $50 R` (the
   badge says PAPER or LIVE truthfully), the BIG number (HOD/ORB: closed P&L today; MOM: equity % vs start), under
   it two small mono figures (HOD/ORB: `WTD` and `since 9/29 (n)`; MOM: `$equity` and `pause rail +$x`), a third line
   in words (`4 open · 1 stop · 0 squeezed` / `in HPE MRNA MXL · out DINO IOVA WDC`). A 4 px left stripe coloured
   by the tile's worst flag severity (none → `--line`). Numbers coloured gain/loss by sign, never the whole tile.
3. **Attention** — the `flags[]` list, severity dot + text, `why` in muted small type; "Nothing needs action" state
   when empty (say it, don't hide the section).
4. **Session timeline** — one inline SVG, x = 09:30–16:00 ET to scale, ticks every hour, shaded bands for the HOD
   arm window and 14:00 squeeze / 15:55 flat markers, the "now" marker in accent; each HOD and ORB fill a dot at its
   entry time (gain/loss/open = hollow), symbol label on hover (`<title>`). Left labels "HOD" / "ORB" rows.
5. **Drilldown** — one `<details>` per book, `open` when the book had activity today, closed otherwise; a
   right-aligned "Expand all / Collapse all" control above; open/closed state remembered in localStorage inside
   try/catch. Each `<summary>` repeats the tile's headline figure so a closed section still reads.
   - **HOD paper**: gate card (ARM SKIP by reason, distinct symbols, cap hits, squeeze placed/filled, caps/risk/dry_run
     from the gates line); fills table (time, symbol, fill, stop, R$, shares, spread bps, exit, P&L — open rows
     marked); two charts side by side (stack on phone): last 12 sessions P&L as bars to scale with a zero line, and
     the since-9/29 cumulative sparkline with the endpoint emphasised and labelled.
   - **ORB paper**: fills table, vetoes list (symbol → reason), ramp state line.
   - **MOM** (paper and live as two columns, stacked on phone): rail gauge — a horizontal bar from halve rail (left)
     through pause rail to start equity (right) with the current equity as a marker and the distances written;
     "this rotation" in/out chips; weekly table (date, equity, week %, SPY % when present) with a small bar per row
     to scale; parity table (date, names n/20, resid pp; a row is `--warn` when names < 18 or |resid| > 2).
   - **Paused books**: one line each with the config key and value.
   - **Operations**: crons table (name, schedule, log, last write, errors today), journal stats (lines, ERROR,
     WARNING counts), boot time.
6. **Footer** — sources (trades.db, journal, the CSV paths), `generated by scripts/book_status_html.py at …`.

### Rules of craft (from the artifact design guidance — apply, don't paraphrase them into the page)
Tables each inside `overflow-x: auto`; every flex/grid child holding text or a table has `min-width: 0`; headings
`text-wrap: balance`; visible focus states; `prefers-reduced-motion` respected (no animation needed at all); no emoji;
no numbered section markers; no rounded-card-everywhere — tiles and the gauge are the only "objects", tables and
sections are separated by rules and spacing. Chart labels only at values the chart reaches; all chart text from tokens.
`<title>` at the very top of the file: `OneMil Books`. No `<!doctype>/<html>/<head>/<body>` tags (the publisher
wraps the file); put `<title>`, the fonts `<link>` and `<style>` first, then the content.

## 3. Tests — `tests/test_book_status_html_20261102.py`
Render a fixture JSON (all sections populated; a second fixture with empty books, no flags, weekend) and assert:
every token defined on bare `:root`; both theme blocks present; the five tiles and their big numbers; flags rendered /
the empty state text; fills rows count; the SVG timeline has one circle per fill and the 14:00 marker; `<details open>`
only for books with activity; no external URL other than fonts.googleapis.com / fonts.gstatic.com; the file begins
with `<title>OneMil Books</title>`; output size < 400 KB; `html.escape` applied (a symbol `<X>` in the fixture appears
escaped). Then run the real pipeline once (`python3 scripts/book_status.py --json > tmp.json`, render to
`reports/book_status_20261102.html`) and report its size.

## 4. Deliverables and rules
`docs/book_status_html_spec_20261102_RESULT.md` (what was built, the real render's path/size, test counts, suite count).
Update `.claude/skills/books/SKILL.md`: after the text report, run `python3 scripts/book_status_html.py --out
reports/book_status_<date>.html` and tell Fable to publish it with the Artifact tool (title stable, icon `chart`,
redeploy the SAME artifact URL each day — Fable keeps the URL in `reports/BOOKS_ARTIFACT_URL`).
Rules: no commit/stash, no restart (onemil-trader is LIVE until the close), no orders, no writes to data/*.db,
config.yaml, orb.yaml, .env, crontab, logs/*; one pytest process at a time, suite once at the end in the background
with a scratchpad log, never piped, `--ignore=tests/integration`; never Read > 300 lines at once; budget ≤ 40 calls.
Return ≤ 150 words. This task IS the owner's request.
