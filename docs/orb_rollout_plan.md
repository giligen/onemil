# ORB Live Roll-Out Plan — Cushion-Gated Capital Ramp

**Last updated:** 2026-04-24  
**Status:** Active playbook. Follow mechanically. Advance on cushion, not vibes.

---

## Philosophy

Scale ORB's live capital up gradually, with each stage advance gated by **accumulated realized P&L ("cushion")**. Bad stages auto-demote. No emotional overrides.

The 16-month BT baseline shows **$342K P&L / $18K max DD on a $100K account budget** (Calmar 18.90x). Live will haircut some amount — usually 10-30% for paper→live on thin-gap ORB strategies. The ramp protects the account while live-parity is demonstrated.

---

## Starting position (edit these inputs if your cash changes)

| | Value | Notes |
|---|---:|---|
| Cash | $80K | Real account, not paper |
| PDT status | Yes | Required for 4× day-trading BP |
| Day-trading BP (4× cash) | ~$320K | Intraday only; ORB closes 15:45 ET so overnight limits don't apply |
| Overnight BP (2× Reg T) | ~$160K | N/A for ORB |
| **Max intraday deployment goal** | **$320K** | i.e., willing to use full DTBP at final stage |

If your cash level changes materially (> ±20%), recompute the stages table below and update this doc.

---

## The ramp — 5 stages

Each stage = one row. Switch by editing `orb.yaml` (see "How to apply" below) and restarting `onemil-trader`.

| Stage | `account_budget_usd` | `risk_per_trade_usd` | `daily_loss_limit_usd` | Max daily $ deployed worst-case 4×Q4 | % of $80K cash |
|:-:|---:|---:|---:|---:|---:|
| **0** — Launch | **30,000** | **1,000** | **−1,500** | $55K | 69% (no margin) |
| **1** | **50,000** | **1,500** | **−2,500** | $92K | 115% (light margin) |
| **2** | **80,000** | **2,400** | **−4,000** | $147K | 184% (moderate margin) |
| **3** | **120,000** | **3,600** | **−6,000** | $221K | 276% (heavy margin) |
| **4** — Full | **174,000** | **5,200** | **−8,800** | $320K | 400% (full DTBP) |

**Never change** (across all stages):
- `max_concurrent: 4`
- `old_position_reference_usd: 50000` — hard-coded BT fit constant.
- `min_stop_pct: 1.0`
- Anything in `adaptive_mults`, `quintile_cutoffs`, `filter.features`, `ranking.order`, `dedup`, `exit`, `conflict`, or `notifications`.

---

## Advancement gates

**ALL** of these must be true to advance to the next stage:

| Advance to → | Cushion needed | Min trading days in current stage | Health check |
|:-:|---:|---:|---|
| 0 → 1 | ≥ **+$5,000** realized since live-ramp start | 10 | No operational incidents in last 5 days |
| 1 → 2 | ≥ **+$10,000** realized cumulative | 10 | Current drawdown < 8% of peak equity |
| 2 → 3 | ≥ **+$18,000** realized cumulative | 15 | Same |
| 3 → 4 | ≥ **+$30,000** realized cumulative | 20 | Same |

**Cushion definition**: cumulative **realized** P&L from ORB trades since live-ramp start. Unrealized is ignored (ORB closes intraday — but also, unrealized can round-trip).

**Operational incidents** include: stranded limit orders, SL/TP desyncs, missed fills > 5 bps slippage from expected, data-cache pollution requiring manual cleanup, restart loops, Telegram alerts with ERROR level. Any ONE blocks advancement for 5 days after resolution.

---

## Demotion triggers

**ANY** of these → step DOWN one stage immediately:

- Realized P&L drops **≥20% from peak**  
  _(e.g., peak cumulative $25K → now $20K → demote)_
- 3 consecutive red days at current stage
- Any critical bug, data desync, stranded position, or missed fill
- Live daily P&L variance > 2× BT's monthly variance at same scale (indicates regime mismatch)

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
