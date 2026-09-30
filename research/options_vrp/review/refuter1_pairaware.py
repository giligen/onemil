"""Refuter-1 sensitivity: identical pipeline to cell_1567.py except strike selection is PAIR-AWARE
(nearest-delta short among strikes whose long leg short-W ALSO printed at 10:00). Every input is
still the 10:00-10:10 entry-minute window, so the rule stays causal. Output to review/pairaware/."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cell_1567 as C

_orig = C.select_strikes

def select_pair_aware(cache, monday, mkt, target_delta, width, warn_counter):
    """Nearest-delta short strike whose long leg (short-W) also has a 10:00 print."""
    spot, dte, expiry = mkt['spot'], mkt['dte'], mkt['expiry']
    T = dte / 365.0
    strikes = mkt['strikes'].sort_values('strike')
    by_strike = {float(r['strike']): r['symbol'] for _, r in strikes.iterrows()}
    cands = []
    for k, sym in by_strike.items():
        if k > spot:
            continue
        mid = cache.entry_mid(sym, monday, warn_counter)
        if mid is None or mid <= 0:
            continue
        iv = C.implied_vol_put(mid, spot, k, T, C.R_RATE, C.Q_RATE)
        if iv is None:
            continue
        dl = C.bs_put_delta(spot, k, T, C.R_RATE, C.Q_RATE, iv)
        cands.append((abs(abs(dl) - target_delta), k, sym, mid, dl))
    for gap, k, sym, mid, dl in sorted(cands):
        if gap > 0.05:
            break
        lsym = by_strike.get(k - width)
        if lsym is None:
            continue
        lmid = cache.entry_mid(lsym, monday, warn_counter)
        if lmid is None or lmid <= 0:
            continue
        warn_counter['pair_aware_used'] = warn_counter.get('pair_aware_used', 0) + 1
        return {'expiry': expiry, 'dte': dte, 'short_strike': k, 'short_symbol': sym, 'short_mid': mid,
                'short_delta': dl, 'long_strike': k - width, 'long_symbol': lsym, 'long_mid': lmid,
                'net_credit': (mid - C.LEG_SLIPPAGE) - (lmid + C.LEG_SLIPPAGE)}
    return None

C.select_strikes = select_pair_aware
out = os.path.join(HERE, 'pairaware')
os.makedirs(out, exist_ok=True)
sys.argv = ['x', '--out-dir', out]
C.main()
