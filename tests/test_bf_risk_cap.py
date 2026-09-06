"""Per-trade risk cap: shared clamp, Stage-2 application, live twin, ordering."""
import pytest

from trading.bf_risk_cap import (
    DISABLED, RiskCapConfig, cap_usd, capped_shares, load_risk_cap_config,
)


class TestShared:
    def test_config_load_and_validation(self):
        assert load_risk_cap_config({}) == DISABLED
        assert load_risk_cap_config(None) == DISABLED
        assert load_risk_cap_config({'risk_cap': {'enabled': True, 'max_risk_mult': 1.5}}) \
            == RiskCapConfig(True, 1.5)
        with pytest.raises(ValueError):
            load_risk_cap_config({'risk_cap': {'enabled': True, 'max_risk_mult': 0}})

    def test_cap_usd(self):
        assert cap_usd(DISABLED, 2000) is None
        assert cap_usd(RiskCapConfig(True, 2.0), 2000) == 4000.0
        assert cap_usd(RiskCapConfig(True, 2.0), 0) is None

    def test_capped_shares(self):
        assert capped_shares(1000, 0.5, None) == (1000, 1.0)        # off
        assert capped_shares(1000, 0.5, 4000) == (1000, 1.0)        # $500 risk, under cap
        assert capped_shares(20000, 0.5, 4000) == (8000, 0.4)       # $10K → $4K
        assert capped_shares(3, 5000.0, 4000) == (1, 1 / 3)         # floor of 1 share
        assert capped_shares(1000, 0.0, 4000) == (1000, 1.0)        # degenerate R untouched
        assert capped_shares(0, 0.5, 4000) == (0, 1.0)


class TestStage2:
    @staticmethod
    def _trade(symbol, shares, price, stop, pnl):
        return {'symbol': symbol, 'date': '2026-01-15', 'entry_time_et': '09:35:00',
                'exit_time_et': '09:40:00', 'entry_price': price, 'stop_loss': stop,
                'shares': shares, 'pnl': pnl, 'pnl_pct': 1.0, 'conviction_mult': 2.0,
                'avg_volume_20d': 1_000_000, 'daily_range_pct': 25.0}

    def _run(self, monkeypatch, enabled, regime_lookup=None):
        from batch_backtest import filter_bull_flag_trades
        from config import Config
        monkeypatch.setattr(Config, '_load_yaml_only', lambda: {
            'scanner': {'min_daily_volume': 0},
            'trading': {'risk_per_trade': 2000, 'risk_tiers': {'enabled': False},
                        'conviction_scoring': {'enabled': False},
                        'risk_cap': {'enabled': enabled, 'max_risk_mult': 2.0}},
        })
        trades = [self._trade('ZZBIG', 20000, 5.0, 4.5, 30000.0),   # risk $10K
                  self._trade('ZZOK', 4000, 5.0, 4.5, 2000.0)]      # risk $2K
        return {t['symbol']: t for t in filter_bull_flag_trades(
            trades, regime_lookup=regime_lookup,
            regime_multipliers={'A': 1.25, 'B': 1.0, 'C1': 1.5, 'C2': 0.0})}

    def test_off_is_identity(self, monkeypatch):
        r = self._run(monkeypatch, False)
        assert r['ZZBIG']['pnl'] == 30000.0 and r['ZZBIG']['shares'] == 20000

    def test_clamps_pnl_and_shares_linearly(self, monkeypatch):
        r = self._run(monkeypatch, True)
        assert r['ZZBIG']['shares'] == 8000
        assert r['ZZBIG']['pnl'] == pytest.approx(30000.0 * 0.4)
        assert r['ZZOK']['pnl'] == 2000.0 and r['ZZOK']['shares'] == 4000

    def test_cap_applies_after_regime_multiplier(self, monkeypatch):
        """C1 day (1.5x): ZZOK risk $2K→$3K stays under the $4K cap; ZZBIG's
        post-regime risk $15K is clamped to $4K — pnl = 30000×1.5×(4000/15000)."""
        r = self._run(monkeypatch, True, regime_lookup={'2026-01-15': 'C1'})
        assert r['ZZOK']['pnl'] == pytest.approx(3000.0)
        assert r['ZZBIG']['pnl'] == pytest.approx(30000.0 * 1.5 * (8000 / 30000))


class TestLiveTwin:
    def _engine(self, cap):
        from trading.trading_engine import TradingEngine
        eng = TradingEngine.__new__(TradingEngine)
        eng.risk_cap_usd = cap
        return eng

    def _plan(self, shares, rps=0.5):
        from trading.trade_planner import TradePlan
        return TradePlan(symbol='ZZ', entry_price=5.0, stop_loss_price=5.0 - rps,
                         take_profit_price=6.5, risk_per_share=rps,
                         reward_per_share=1.5, risk_reward_ratio=3.0,
                         shares=shares, total_risk=shares * rps, pattern=None)

    def test_live_clamp_matches_shared(self):
        eng = self._engine(4000.0)
        p = eng._apply_risk_cap(self._plan(20000), 'ZZ')
        assert p.shares == 8000 and p.total_risk == pytest.approx(4000.0)
        assert p.entry_price == 5.0 and p.stop_loss_price == 4.5
        same = eng._apply_risk_cap(self._plan(4000), 'ZZ')
        assert same.shares == 4000

    def test_live_off(self):
        eng = self._engine(None)
        assert eng._apply_risk_cap(self._plan(20000), 'ZZ').shares == 20000

    def test_ordering_before_bp_ceiling(self):
        import inspect
        from trading.trading_engine import TradingEngine
        src = inspect.getsource(TradingEngine)
        i_cap = src.index('plan = self._apply_risk_cap(plan, symbol)')
        i_bp = src.index('plan = self._apply_bp_ceiling(plan, symbol)')
        i_ud = src.index('# UD risk scaling')
        assert i_ud < i_cap < i_bp
