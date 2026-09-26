# AI-native trading for a $65K account — from scratch (judge's plan, 2026-09-26)

Owner: "We have AI that was never available before for small traders. How can this be leveraged for a trading bot that
gets us a nice passive income on average?" This is the first-principles answer. The literature sweep
(`research/lit_review_2026/SOTA_AI_TRADING_2023_2026.md`, running) will confirm or amend the citations marked [lit].

## 0. What AI changes, and what it does not
| does NOT change | DOES change |
|---|---|
| the size of market inefficiencies (a 0.3 %/month effect is still 0.3 %) | the cost of READING: every filing, press release, transcript, docket, registry — for cents, in minutes |
| cost per trade (25–30 bps on movers, 2 bps on SPY, 1–2 bps at the auctions) | JUDGMENT at 5–30 s latency: "is this press release material, and which way?" — keyword bots cannot |
| capacity: a fund's 0.3 %/month on $100M is $300K; on $65K it is $200 | research throughput: 1,480 cells in two weeks; nulls found before money is lost |
| the MDE wall: tiny per-trade edges accumulate across capital and slots, not across a small account's trades | plumbing: data pipelines that used to take a quant team a quarter |

Consequence: the only edges worth a small account's time are **public information that is unread or slowly read**, where
the underreaction lasts hours to days (not milliseconds) and where the size that moves the price is small enough that
institutions are absent. That niche exists precisely because it is too small for them. AI is the reader.

## 1. The trap that is new: LLM look-ahead
A model scoring historical news has read the future (Glasserman & Lin 2023; Sarkar & Vafa 2024 [lit]). Lopez-Lira &
Tang's result held only on post-cutoff data. Rule for every LLM-scored cell here: **score only news dated AFTER the scoring
model's training cutoff**, or mask entities and dates; VAL and TEST must be post-cutoff. With Haiku 4.5 that means roughly
2025-07 → 2026-09 (≈ 300 sessions) — enough for a daily signal, not for a monthly one. `ANTHROPIC_API_KEY` is empty in
`.env`: scripted scoring needs the owner to add a key (API spend ≈ $20–60 per 100K headlines), else the scoring runs as
harness batches of Haiku agents.

## 2. Candidate legs, ranked by (documented mechanism × retail feasibility × testability now)
| leg | what the AI does | published evidence [lit] | retail feasibility | when |
|---|---|---|---|---|
| **A. Overnight news sentiment → open** | LLM scores each firm-specific headline released 16:00–09:25 ET; long the clearly positive, short the clearly negative (small/mid caps), MOO entry, MOC (or next-open) exit | Lopez-Lira & Tang 2023: strong gross long-short 2021–22, mostly the short side of small caps; decaying; Chen-Kelly-Xiu 2023 LLM news factor | auction fills (≈ 2 bps), free Benzinga history, borrow flags known; latency irrelevant (decision before the open) | PREREG this weekend; lexicon baseline (free) vs Haiku on post-cutoff months |
| **B. LLM event desk on EDGAR + PR wire** | classify every 8-K / 424B / S-3 / press release into a small catalogue: dilution after a run-up, reverse split, going-concern, auditor change, guidance change, contract/FDA/M&A; trade only classes with a MEASURED multi-day drift | "Lazy Prices" (Cohen-Malloy-Nguyen 2020) for text change; reverse-split and SEO drifts (older literature); the dilution-after-pump pattern is folk-documented, unmeasured here | EDGAR is free and 1-min lagged; holds days–weeks; short side needs borrow | medium build (2–3 days); after A |
| **C. News-spike fast leg** | LLM materiality verdict within ~10 s of a low-float PR; enter the first minutes of the reaction | none peer-reviewed; the retail "news catalyst" trade — competitors are keyword bots | possible only on the tape's terms: halts, 50–200 bps spreads, huge variance; must be measured with `fetch_window` at a 10–15 s latency before belief | PREREG after A; small size |
| **D. Income base: variance risk premium** | none of the AI's reading; AI runs execution, rolling and risk | the most robust documented premium (CBOE PUT / BXM; ≈ 6 %/yr excess over T-bills, −20 % months in crashes — `E1_vrp_options.md`) | needs options enabled on the account; SPY/QQQ only; $300–450/month expected at $65K with a fat left tail | owner decision |
| **E. Intraday ML on our own tape** | the running big-day predictor (cells 1,478–1,480) | Avramov-Cheng-Metzker 2023: ML return gains sit in illiquid names and vanish after costs | measured tonight | verdict pending |
| not legs | LLM "trading agents" (TradingAgents, FinMem, FinRobot): no honest out-of-sample evidence; time-series foundation models on returns: negative; chart CNNs: small and decaying | | | |

## 3. The bot, AI-native (what gets built if A or B pass)
Ingest (Alpaca news websocket + EDGAR RSS) → LLM classifier (Haiku, structured output: entity, class, direction,
materiality 1–5, dilution flag) → **measured drift table per class** (from the research loop; nothing trades without its
own VAL/TEST) → sizing by expected move ÷ cost → execution (MOO/MOC for the daily legs; the StopMonitor stop-limit exit
for anything intraday) → the live guardrail ledger → forward ledger → the same PREREG / rebuild / refute loop run by
agents on the forward data. The AI sits in three places: reading, judging materiality, and running the falsification.
It does NOT predict prices from prices — that is where two weeks of evidence say the edge is smallest.

## 4. Honest economics at $65K (average month / bad month)
D ≈ +$300–450 / −$10K. A, if a third of the published effect survives costs and look-ahead: ≈ +$500–1,500 / −$2K (ten
names a day, $3K each, 30 bps net). B ≈ +$200–800 / −$1.5K (few events, larger moves). C: unknown until measured. Two
surviving legs ≈ **$1–2K/month on average with red months** — the owner's bar; 10 %/month is not on this map, and no
honest map at this size has it.

## 5. Next seven days
1. Leg A: PREREG (post-cutoff windows, lexicon baseline vs Haiku, auction costs, borrow flags), news history pull, run,
   rebuild, refute. Needs the API key or the harness-batch route (owner: the key is a one-line `.env` edit).
2. Leg D: owner decision (options on the account); if yes, the passive-income plan's E1 sleeve gets its engine.
3. Leg C: tape-measurement PREREG once A's classifier exists.
4. Leg E: the predictor verdict tonight, with its measured AUC.

## 6. Literature update (from `research/lit_review_2026/SOTA_AI_TRADING_2023_2026.md`, 49 items, 2023–2026)
* **Leg A survives with two design changes**: strip entity names before scoring (Glasserman & Lin 2023: the named
  version is contaminated by what the model knows about the company, not only by look-ahead) and score only post-cutoff
  news (Sarkar & Vafa 2024, ICML 2025). Lopez-Lira & Tang's own updates show the drift decaying as adoption spread and
  concentrated in small-cap negative news — the short side, where borrow and cost bite. Expect a fraction of the published
  effect; the VAL window decides.
* **The most-cited "GPT beats analysts" paper (Kim, Muhn & Nikolaev 2024) was withdrawn by its authors in Feb 2025**
  after a co-author could not replicate it. Nothing in this plan rests on it; it is the reason every number here needs
  an independent rebuild before it reaches the owner.
* **The only AI mechanism with real live evidence is augmentation, not selection**: AI reads faster and wider (Cao et al.
  JFE 2024; Sheng et al. RFS). That is legs A and B. Leg E (a model picking the big day from price features) gets no
  support anywhere: Avramov, Cheng & Metzker 2023 — ML return edges live in illiquid names and vanish net of costs;
  Nagel 2025 — the "virtue of complexity" gain is volatility-timed momentum in disguise.
* **Never deploy an LLM agent framework or pick a model by leaderboard**: independent re-tests fail 80 % of published
  multi-agent schemes; live arenas show GPT-5-class agents losing to buy-and-hold. Real AI money (Eurekahedge AI index
  2009–24: 9.8 %/yr vs S&P 13.7 %; AIEQ) underperforms passive equity — the base rate for "AI trading bot" is below the
  index. Leg D (the variance risk premium) remains the only leg whose premium does not depend on beating anyone.
