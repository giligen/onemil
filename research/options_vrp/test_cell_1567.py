"""Unit tests for cell_1567.py (PREREG_1567 SCORE stage). No parquet cache required: every test
either exercises a pure function or a minimal FakeCache stand-in."""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cell_1567 as c1567


# --------------------------------------------------------------------------- Black-Scholes

def test_bs_put_price_known_atm():
    # Known reference: S=K=100, T=1, r=0.05, q=0, sigma=0.20 -> put ~ 5.5735 (textbook value)
    price = c1567.bs_put_price(100, 100, 1.0, 0.05, 0.0, 0.20)
    assert abs(price - 5.5735) < 0.01


def test_bs_put_delta_range():
    # Deep OTM put (way below spot) -> delta close to 0; deep ITM -> close to -1
    otm = c1567.bs_put_delta(500, 400, 45/365, 0.045, 0.013, 0.13)
    itm = c1567.bs_put_delta(400, 500, 45/365, 0.045, 0.013, 0.13)
    assert -0.05 < otm < 0
    assert -1.0 < itm < -0.90


def test_implied_vol_roundtrip():
    """IV solved from a BS price must recover the sigma that generated it."""
    S, K, T, r, q, sigma_true = 495.0, 470.0, 45/365, 0.045, 0.013, 0.14
    price = c1567.bs_put_price(S, K, T, r, q, sigma_true)
    iv = c1567.implied_vol_put(price, S, K, T, r, q)
    assert iv is not None
    assert abs(iv - sigma_true) < 1e-3


def test_implied_vol_rejects_arbitrage_violation():
    """A price above the K*exp(-rT) upper bound is a bad print -- must return None, never a
    fabricated IV."""
    S, K, T, r, q = 495.0, 470.0, 45/365, 0.045, 0.013
    bad_price = K * math.exp(-r * T) + 5.0  # above the upper no-arbitrage bound
    assert c1567.implied_vol_put(bad_price, S, K, T, r, q) is None


def test_delta_nearest_selection_direction():
    """Among strikes at fixed S/T, the one with delta closest to a target must be picked -- a
    higher strike (closer to ATM) always has a bigger |delta| for OTM puts."""
    S, T, r, q, sigma = 500.0, 45/365, 0.045, 0.013, 0.13
    strikes = [450, 460, 470, 480, 490]
    deltas = {k: abs(c1567.bs_put_delta(S, k, T, r, q, sigma)) for k in strikes}
    # deltas must increase monotonically with strike for OTM puts below spot
    ordered = sorted(strikes)
    vals = [deltas[k] for k in ordered]
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    target = 0.20
    best = min(strikes, key=lambda k: abs(deltas[k] - target))
    # sanity: the chosen strike's delta really is the closest of all candidates
    assert abs(deltas[best] - target) == min(abs(d - target) for d in deltas.values())


# --------------------------------------------------------------------------- sizing / budget

def test_size_position_floors_and_zero_case():
    contracts, worst = c1567.size_position(alloc_dollars=1083.33, width=5.0, net_credit=1.20)
    # worst_per_contract = (5-1.20)*100 = 380 -> floor(1083.33/380) = 2
    assert worst == 380.0
    assert contracts == 2
    # tiny allocation -> zero contracts, never negative or fractional
    contracts0, _ = c1567.size_position(alloc_dollars=50.0, width=5.0, net_credit=1.20)
    assert contracts0 == 0


def test_size_position_nonpositive_worst_case_returns_zero():
    # net_credit >= width => worst_per_contract <= 0 -> must return 0 contracts, never divide by <=0
    contracts, worst = c1567.size_position(alloc_dollars=1000.0, width=5.0, net_credit=6.0)
    assert contracts == 0
    assert worst <= 0


def test_budget_never_exceeds_B_across_a_run():
    """run_cell's internal assertion must hold: at every open, reserved + new worst_case <= B."""
    cache = FakeCache()
    mondays_ctx = []
    entry_cache = {}
    mondays = [f'2024-0{m}-0{d if d>0 else 5}' for m in (2,) for d in (5,)]  # placeholder, replaced below
    # Build 8 sequential weekly entries, all with generous width so B/6 always fits >=1 contract,
    # to force > 6 concurrently-open positions and exercise the running-budget cap.
    mondays = [f'2024-02-{5+7*i:02d}' if 5+7*i <= 28 else f'2024-03-{5+7*i-28:02d}' for i in range(8)]
    for m in mondays:
        mkt = {'expiry': '2050-01-01', 'dte': 9999, 'iv_atm': 0.20, 'spot': 500.0,
               'strikes': None}
        mondays_ctx.append((m, mkt))
        entry_cache[(m, 0.20, 5.0)] = {
            'expiry': '2099-12-31', 'dte': 300, 'short_strike': 480.0, 'short_symbol': f'S{m}',
            'short_mid': 1.5, 'short_delta': -0.2, 'long_strike': 475.0, 'long_symbol': f'L{m}',
            'long_mid': 0.3, 'net_credit': 1.5 - 0.03 - (0.3 + 0.03),
        }
    cell_def = {'cell': 9999, 'delta': 0.20, 'width': 5.0, 'mgmt': 'B', 'gate': 0}
    warn = {}
    cycles, counts = c1567.run_cell(cache, cell_def, mondays_ctx, entry_cache, warn)
    # every recorded cycle's worst_case, summed over any snapshot in time, must never exceed B
    # (the assertion inside run_cell would have raised already if violated); also check directly
    # that no single position's worst case alone exceeds B.
    for cy in cycles:
        assert cy['worst_case_usd'] <= c1567.B + 1e-6


# --------------------------------------------------------------------------- FakeCache for run_cycle tests

class FakeCache:
    """Minimal stand-in for Cache exposing only what run_cycle / naked_leg_pnl need."""

    def __init__(self, daily=None, next_open=None, spy_expiry=None):
        self.daily = daily or {}          # {(symbol, day): close}
        self.next_open_map = next_open or {}  # {(symbol, after_day): open}
        self.spy_expiry = spy_expiry or {}  # {day: S_T}

    def daily_asof(self, symbol, as_of_day):
        candidates = {d: v for (s, d), v in self.daily.items() if s == symbol and d <= as_of_day}
        if not candidates:
            return None
        return candidates[max(candidates)]

    def daily_next_open(self, symbol, after_day):
        candidates = {d: v for (s, d), v in self.next_open_map.items() if s == symbol and d > after_day}
        if not candidates:
            return None
        return candidates[min(candidates)]

    def spy_16_close(self, day):
        return self.spy_expiry.get(day)


def _entry(net_credit=1.44, short_strike=480.0, long_strike=475.0):
    return {'expiry': '2024-03-22', 'short_symbol': 'SHORT', 'long_symbol': 'LONG',
            'net_credit': net_credit, 'short_strike': short_strike, 'long_strike': long_strike}


def test_precedence_profit_target_before_21dte():
    """Both a 50%-credit close and a 21-DTE close are reachable on the same day: profit target
    must win (it is checked before the 21-DTE check in run_cycle)."""
    entry = _entry(net_credit=1.44)
    monday = '2024-02-05'
    expiry = '2024-03-22'  # 46 calendar days out -> day at expiry-20 is inside 21 DTE window
    day = '2024-03-01'     # 21 days before expiry: also triggers 21-DTE AND mark <= 50% credit
    daily = {}
    d = monday
    import pandas as pd
    for bd in pd.bdate_range(monday, expiry)[1:-1]:
        day_s = bd.strftime('%Y-%m-%d')
        # mark = short - long; make it drop to 50% of credit exactly at day `day` and stay low after
        mark = 0.60 if day_s < day else 0.30  # 0.30 <= 0.5*1.44=0.72 -> profit target fires at `day`
        daily[('SHORT', day_s)] = mark + 0.10
        daily[('LONG', day_s)] = 0.10
    cache = FakeCache(daily=daily, spy_expiry={expiry: 460.0})
    result = c1567.run_cycle(cache, entry, monday, 5.0, 'A', 1, {})
    assert result is not None
    exit_date, exit_reason, pnl_per_share, closed_actively = result
    assert exit_reason == 'profit_50'
    assert closed_actively is True


def test_stop_precedence_over_profit_and_21dte():
    """A day where the mark is >= 2x credit must trigger the STOP path (deferred to next
    session's open) even though the same day would also have satisfied 21-DTE."""
    entry = _entry(net_credit=1.0)
    monday = '2024-02-05'
    expiry = '2024-03-22'
    import pandas as pd
    daily, next_open = {}, {}
    bdays = [b.strftime('%Y-%m-%d') for b in pd.bdate_range(monday, expiry)]
    stop_day = bdays[5]
    after_stop_day = bdays[6]
    for day_s in bdays[1:-1]:
        mark = 2.5 if day_s == stop_day else 0.9  # 2.5 >= 2*1.0 -> stop; otherwise not profit/21dte
        daily[('SHORT', day_s)] = mark + 0.05
        daily[('LONG', day_s)] = 0.05
    next_open[('SHORT', after_stop_day)] = 2.4 + 0.05
    next_open[('LONG', after_stop_day)] = 0.05
    cache = FakeCache(daily=daily, next_open=next_open, spy_expiry={expiry: 460.0})
    result = c1567.run_cycle(cache, entry, monday, 5.0, 'A', 1, {})
    assert result is not None
    exit_date, exit_reason, pnl_per_share, closed_actively = result
    assert exit_reason == 'stop'
    assert exit_date == after_stop_day
    assert closed_actively is True


def test_intrinsic_settlement_management_B():
    """Management B never checks daily marks; it settles at expiry from intrinsic value."""
    entry = _entry(net_credit=1.20, short_strike=480.0, long_strike=475.0)
    monday = '2024-02-05'
    expiry = '2024-03-22'
    # SPY well below the long strike at expiry -> both legs fully ITM, settlement = width (5.0)
    cache = FakeCache(daily={}, spy_expiry={expiry: 400.0})
    result = c1567.run_cycle(cache, entry, monday, 5.0, 'B', 1, {})
    assert result is not None
    exit_date, exit_reason, pnl_per_share, closed_actively = result
    assert exit_reason == 'expiry'
    assert closed_actively is False
    assert exit_date == expiry
    # pnl_per_share = net_credit - (intrinsic_short - intrinsic_long) = 1.20 - (80-0... ) etc
    assert abs(pnl_per_share - (1.20 - 5.0)) < 1e-9  # worst case: max loss = width - credit


def test_intrinsic_settlement_otm_max_profit():
    """SPY finishes above both strikes -> both legs worthless -> full credit kept."""
    entry = _entry(net_credit=1.20, short_strike=480.0, long_strike=475.0)
    cache = FakeCache(daily={}, spy_expiry={'2024-03-22': 500.0})
    result = c1567.run_cycle(cache, entry, '2024-02-05', 5.0, 'B', 1, {})
    exit_date, exit_reason, pnl_per_share, closed_actively = result
    assert exit_reason == 'expiry'
    assert abs(pnl_per_share - 1.20) < 1e-9


def test_void_when_daily_mark_missing():
    """A leg with no daily bar on/before a required day must VOID the cycle, never impute."""
    entry = _entry()
    cache = FakeCache(daily={}, spy_expiry={'2024-03-22': 460.0})
    result = c1567.run_cycle(cache, entry, '2024-02-05', 5.0, 'A', 1, {})
    assert result is None  # management A needs a daily mark every day; none supplied here -> VOID


# --------------------------------------------------------------------------- IV gate

def test_iv_gate_skips_below_threshold():
    assert c1567.IV_GATE_MIN == 0.15
    # gate logic itself lives inline in run_cell; check the threshold constant and comparison sense
    assert (0.12 < c1567.IV_GATE_MIN) and not (0.16 < c1567.IV_GATE_MIN)


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-v']))
