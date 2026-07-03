# ORB Live Roll-Out Plan — Cushion-Gated Capital Ramp

**Last updated:** 2026-04-24  
**Status:** Active playbook. Follow mechanically. Advance on operational gates + loss floor (revised 2026-07-06), not vibes.

---

## Philosophy

Scale ORB's live capital up gradually, with each stage advance gated by **accumulated realized P&L ("cushion")**. Bad stages auto-demote. No emotional overrides.

The 16-month BT baseline shows **$342K P&L / $18K max DD on a $100K account budget** (Calmar 18.90x). Live will haircut some amount — usually 10-30% for paper→live on thin-gap ORB strategies. The ramp protects the account while live-parity is demonstrated.

---

## Starting position (edit these inputs if your cash changes)

| | Value | Notes |
|---|---:|---|
| Cash | $80K | Real account, not paper |
| Margin-enabled | Yes | Required for 4× intraday buying power |
| Intraday BP (~4× cash) | ~$320K | ORB closes 15:45 ET so overnight limits don't apply |
| Overnight BP (2× Reg T) | ~$160K | N/A for ORB |
| **Max intraday deployment goal** | **$320K** | i.e., willing to use full intraday BP at final stage |

> **Framework note (2026-06-05)**: FINRA retired the Pattern Day Trader rule
> and Alpaca replaced it with a real-time intraday margin framework. Prior
> versions of this doc referenced PDT classification and `daytrading_buying_power`;
> those concepts are deprecated. Under the new framework the same ~4× intraday
> BP is available on margin-enabled accounts (minimum equity now $2K, not $25K),
> margin enforcement happens at order submission (server-side pre-trade
> rejection) and via real-time margin calls. Sustained unmet margin calls
> within 5 business days can lead to 90-day account restriction — see the
> `trading/account_state_monitor.py` halt mechanism for our automated
> protection.

If your cash level changes materially (> ±20%), recompute the stages table below and update this doc.

---

## The ramp — Pre-Stage-0 + 5 stages

Each stage = one row. Switch by editing `orb.yaml` (see "How to apply" below) and restarting `onemil-trader`.

| Stage | `account_budget_usd` | `risk_per_trade_usd` | `daily_loss_limit_usd` | Max daily $ deployed worst-case 4×Q4 | % of $80K cash |
|:-:|---:|---:|---:|---:|---:|
| **Pre-0** — Live data collection | **15,000** | **500** | **−750** | $28K | 35% (no margin) |
| **0** — Launch | **30,000** | **1,000** | **−1,500** | $55K | 69% (no margin) |
| **1** | **50,000** | **1,500** | **−2,500** | $92K | 115% (light margin) |
| **2** | **80,000** | **2,400** | **−4,000** | $147K | 184% (moderate margin) |
| **3** | **120,000** | **3,600** | **−6,000** | $221K | 276% (heavy margin) |
| **4** — Full | **174,000** | **5,200** | **−8,800** | $320K | 400% (full intraday BP) |

**Never change** (across all stages):
- `max_concurrent: 4`
- `old_position_reference_usd: 50000` — hard-coded BT fit constant.
- `min_stop_pct: 1.0`
- Anything in `adaptive_mults`, `quintile_cutoffs`, `filter.features`, `ranking.order`, `dedup`, `exit`, `conflict` (notifications.telegram.prefix is the only exception — see Pre-Stage-0 below).

---

## Pre-Stage-0 LIVE — Data Collection Phase

**Purpose:** collect real-money execution data (fill quality, spread experience,
operational stability, BT-vs-live parity) before committing to formal Stage 0.
Paper data has structural limits — synthetic Alpaca paper fills don't capture
real venue queue position, market impact, or rejection-rate distribution.
Half-size live caps the cost of learning at ~$3K worst-case while exposing
the strategy to real execution conditions.

### Config (apply to `orb.yaml`)

```yaml
sizing:
  account_budget_usd: 15000      # half of Stage 0
  risk_per_trade_usd:   500      # half of Stage 0
notifications:
  telegram:
    prefix: "[ORB-LIVE-PRE0]"   # visually distinct from paper [ORB] alerts
risk:
  daily_loss_limit_usd: -750     # half of Stage 0
```

Everything else stays at production defaults.

### Entry criteria (ALL must be true to launch)

- [ ] Q1 filter validated **≥ 1 paper session** (post-shipment 2026-04-25)
- [ ] **No P0/P1 incidents** in preceding 3 trading days
- [ ] Composite drift investigated — root cause identified (does NOT have to be
      fully fixed; we just need to know what's causing the ~0.09 BT-vs-live gap)
- [ ] Paper cumulative P&L **> −$5,000** (we're at −$2,320 currently — passes)
- [ ] One ORB strategy change at a time — no other simultaneous strategy edits

### Duration

**10–15 trading days** at half-size. After 15 days mandatory review:
either advance to full Stage 0 (criteria below) OR demote back to paper.

### Promotion criteria (Pre-Stage-0 → Stage 0)

ALL must be true:
- [ ] **≥ 10 trading days** completed at half-size
- [ ] **Cumulative cushion ≥ +$1,000** realized
- [ ] **Live round-trip slippage ≤ 60 bps** measured (entry ≤ 45 bps mean,
      exit ≤ 25 bps mean — 1.5× BT assumption max)
- [ ] **Zero P0 incidents** during the window
- [ ] **Composite drift held < 0.05** sustained (live `ORB SCORED` vs BT)

### Demotion to paper (ANY triggers immediate revert)

- Cumulative P&L drops to ≤ **−$3,000** (hard stop, override-proof)
- 2 P0/P1 incidents within 5 days
- Single day worse than **−$1,000** (24h pause + trade-by-trade review before resume)
- Composite drift exceeds **0.20** sustained over 3+ days

### Reset procedure (after demotion, before re-launching Pre-0)

The launch-date discovery in `scripts/orb_pre0_daily.py` reads from git
log for `ORB ramp: Pre-Stage-0 LIVE` commits. After a demotion + relaunch,
two attempts share the same git history, which means days-in-stage and
cumulative cushion mechanics get confused.

To re-launch cleanly:

```bash
# 1. Document the demotion + reason in docs/ramp_log.md (one line)
echo "$(date -I)  Pre-0 demoted: <trigger>"  >> docs/ramp_log.md

# 2. Take 2 full trading days off (emotional decompression)

# 3. When re-launching, use a NEW commit message tag so the script picks
#    up the latest launch:
git commit -m "ORB ramp: Pre-Stage-0 LIVE (relaunch attempt 2 - <reason>)"

# 4. The daily monitor will use the most-recent matching commit. If you
#    want to be explicit, pass --launch-date YYYY-MM-DD.
```

### Telegram prefix change

`orb.yaml::notifications.telegram.prefix: "[ORB-LIVE-PRE0]"`

This is the ONLY notification config that should change between paper and
half-size live. Distinct prefix prevents misreading paper vs live alerts.
After Pre-Stage-0 → Stage 0 promotion, change to `"[ORB-LIVE]"`.

### Daily monitoring

Run `python3 scripts/orb_pre0_daily.py` after market close each day. Reports:
cushion, days-in-stage, today's slippage vs BT, eligibility status, demotion
triggers fired (if any).

### What this phase WILL teach you

- True live fill prices (vs paper synthetic)
- Real spread experience at exit, especially for cheap stocks
- Order rejection rates from real venues
- Operational bug surfacing (paper hides some bugs)
- Q1 filter behavior under real market conditions

### What this phase WILL NOT teach you

- Stage 4 capacity behavior (sizes too small to test market impact)
- Margin / intraday-BP utilization stress (using ~5% of available BP)
- Most operational risk (most bugs are size-independent)

Capacity questions wait for Stage 3-4 actual scaling.

---

## Advancement gates — REVISED 2026-07-06 (owner-approved)

> Policy change record: the original cushion (profit-target) gates below
> were REPLACED by operational gates + a loss floor. Rationale in
> `docs/ramp_policy_proposal_jul2026.md`: cushion punished BT-consistent
> variance (two June-2026 demotion flags, both overridden as false
> alarms), and the PDR veto (2026-07-04) halves trade count while
> doubling per-trade quality — cushion accrual slows exactly as the
> strategy improves. We scale unless performing WORSE than the validated
> loss distribution, not once profitability is proven at the size where
> the edge is smallest.

**ALL** of these must be true to advance to the next stage:

1. **Operational green — 10 consecutive sessions** with:
   - 0 unexplained BT↔live selection diffs (observer + `logs/orb_selection_audit.jsonl`)
   - 0 touchgo REKEY / negative-age tripwires
   - 0 order-fill mismatches (DB vs broker; reconciler clean)
   - all exits attributed (no `unknown_exit` / stale `exit_pending_verification` rows)
2. **Loss floor** (replaces cushion): cumulative realized stage P&L ≥
   **−1 × (stage daily loss limit × 5)** — i.e. hold only if doing worse
   than a full losing week at stage size (Stage 0: −$7.5K floor).
3. **Slippage parity**: median entry slippage ≤ BT model + 10 bps over
   the stage (`analyze_orb_slippage.py`).
4. **Min trading days in stage**: 10 (Stages Pre-0→0→1→2), 15 (2→3),
   20 (3→4) — unchanged from original plan.

Budget/risk ladder per stage and daily loss limits: **unchanged**.

**Operational incidents** include: stranded limit orders, SL/TP desyncs, missed fills > 5 bps slippage from expected, data-cache pollution requiring manual cleanup, restart loops, Telegram alerts with ERROR level. Any ONE blocks advancement for 5 days after resolution.

<details>
<summary>Superseded cushion gates (pre-2026-07-06, for reference)</summary>

| Advance to → | Cushion needed | Min days | Health check |
|:-:|---:|---:|---|
| Pre-0 → 0 | ≥ +$1,000 realized at half-size | 10 | Slippage ≤ 1.5× BT, no P0 incidents |
| 0 → 1 | ≥ +$5,000 realized | 10 | No operational incidents in last 5 days |
| 1 → 2 | ≥ +$10,000 cumulative | 10 | DD < 8% of peak |
| 2 → 3 | ≥ +$18,000 cumulative | 15 | Same |
| 3 → 4 | ≥ +$30,000 cumulative | 20 | Same |

</details>

---

## Demotion triggers — REVISED 2026-07-06

**ANY** of these → step DOWN one stage immediately:

- **Operational failure**: unexplained selection diff, fill mismatch,
  unattributed exit, stranded position, data desync, critical bug
- Cumulative stage P&L < **−2 × (stage daily loss limit × 5)**
  (Stage 0: −$15K)

**Explicitly NOT a demotion trigger** (codifies the two June-2026
overrides): P&L drawdown consistent with the BT's own percentile bands
at stage scale. The strategy is lottery-with-edge — 60% of trades lose
by design; drawdowns that match the validated distribution are the cost
of the tickets, not a malfunction.

**TWO simultaneous triggers** → drop 2 stages AND halt 24h to investigate before resuming.

---

## Hard stop (do not override)

If cumulative realized P&L **≤ −15% of starting cash** (on $80K: ≤ −$12,000), **halt trading entirely** and rebuild confidence from scratch:
1. Stop the service: `sudo systemctl stop onemil-trader`
2. Run post-close BT on the live trade dates; compare BT projected P&L to live realized per-trade
3. Identify the divergence (data? slippage? filter drift?)
4. Only resume after either fixing the divergence OR explicitly deciding "this is variance, I accept it" in writing

The hard stop is **non-negotiable**. If it triggers and you feel like overriding, stop trading for a week, not an hour.

---

## Expected P&L and DD at each stage

Derived from BT's $342K/16mo baseline on $100K budget, with a **25% live haircut** applied (conservative).

| Stage | Annual P&L expected | Max DD expected | DD as % of $80K cash |
|:-:|---:|---:|---:|
| 0 | $58K | $5.4K | 7% |
| 1 | $97K | $9.1K | 11% |
| 2 | $155K | $14.5K | 18% |
| 3 | $232K | $21.7K | 27% |
| 4 | $336K | $31.5K | **39%** |

Stage 4's 39% cash DD is the price of maximum leverage. If you're uncomfortable with that, stop advancing at Stage 3 — $232K/yr on $80K cash is still 290% annual return with 27% max DD.

---

## Expected timeline

| Transition | Expected days (median) | Expected cumulative realized at advance |
|---|:-:|---:|
| 0 → 1 | ~17 days (3½ weeks) | $5K |
| 1 → 2 | ~10 days (2 weeks) | $10K |
| 2 → 3 | ~10 days (2 weeks) | $18K |
| 3 → 4 | ~12 days | $30K |
| **Total 0 → 4** | **~50 trading days ≈ 2.5 months** | $30K+ |

If live outperforms BT, compress by ~30%. If it underperforms, stages STALL — don't advance on schedule, advance on gates.

---

## How to apply a stage change

Do this **after market close** (after 16:00 ET) or **pre-market** (before 09:00 ET). Never mid-session.

```bash
# 1. Verify eligibility
python3 scripts/orb_ramp_check.py
# Expect: "CURRENT STAGE X, CUSHION $Y, ELIGIBLE FOR STAGE X+1" (or not)

# 2. Edit orb.yaml — update the 3 numbers for the new stage
#    - sizing.account_budget_usd
#    - sizing.risk_per_trade_usd
#    - risk.daily_loss_limit_usd
vim orb.yaml

# 3. Verify config loads cleanly
python3 -c "
import yaml
with open('orb.yaml') as f: c = yaml.safe_load(f)
print('account_budget_usd:', c['sizing']['account_budget_usd'])
print('risk_per_trade_usd:', c['sizing']['risk_per_trade_usd'])
print('daily_loss_limit_usd:', c['risk']['daily_loss_limit_usd'])
"

# 4. Commit the change with an audit-friendly message — the ramp-check
#    script uses this to compute days-in-stage from git log
git add orb.yaml
git commit -m "ORB ramp: Stage N → Stage N+1 (\$X cushion built, Y days in prior stage)"
git push origin master

# 5. Restart live service
sudo systemctl restart onemil-trader

# 6. Confirm the new config is live
journalctl -u onemil-trader --since "1 minute ago" | grep "ORBEngine init"
```

Use the same commit-message prefix `ORB ramp:` for both advancements and demotions — the check script scans for this tag to determine when the current stage started.

---

## Demotion procedure

Same as advancement, but the commit message reads:  
`git commit -m "ORB ramp: DEMOTE Stage N → Stage N-1 (<trigger reason>)"`

On demotion, also:
- Document the trigger reason in `docs/ramp_log.md` (one line)
- Consider taking **2 full trading days off** before any new stage change — emotional decompression matters

---

## FAQ

**Q: Why no Stage 0.5 or smaller increments?**  
A: 5 stages keeps the decision tree light. If you want finer-grain, halve each jump — e.g., insert Stage 1.5 at `account_budget: 65K`. But each decision point is cognitive load; fewer is better.

**Q: What if I have a big green day and my cushion ALREADY covers Stage 2 on day 3?**  
A: Still wait for the minimum-days-in-stage gate (10 days in Stage 0). Big single-day wins are noise; the minimum-days guard is there to rule out a lucky streak.

**Q: What if the strategy outperforms BT significantly?**  
A: Great. Advance as fast as the gates allow (cushion + min days). Don't skip stages. The gates ARE the performance test.

**Q: Can I run the full ramp on paper first?**  
A: Yes — and you should. On a paper account the demotion triggers still fire on their own because the DB records the same P&L. Do a 2-week paper run of Stage 0 → confirm the check-script math → then transition to live.

**Q: Who decides to demote?**  
A: The ramp-check script (`scripts/orb_ramp_check.py`) flags demotion triggers. You apply them. The process is mechanical.

**Q: What if I change my mind about risk tolerance?**  
A: Edit this doc + the stages table. Commit the change. That IS the decision.
