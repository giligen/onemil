import os, sys, sqlite3, numpy as np, pandas as pd
from datetime import timedelta
sys.path.insert(0, '/home/ec2-user/onemil')
os.chdir('/home/ec2-user/onemil')
import study_orb_pipeline_static_lock as P
from trading.orb_touchgo_filter import find_breakout_bar_ts

cfg = P.load_bt_config('orb.yaml')
WS = cfg.get('winner_stack_enabled', cfg.get('scale_enabled') or cfg.get('atr_floor_enabled'))
print('cfg', {k: cfg[k] for k in cfg if 'scale' in k or 'atr' in k or 'winner' in k}, 'OLD_POS', P.OLD_POS)

book = P.__dict__.get('read_orb_csv', None)
from trading.orb_csv import read_orb_csv
bk = read_orb_csv('analysis_results/orb_bplus_book.csv')
bk['d'] = pd.to_datetime(bk['date'])

con = sqlite3.connect('file:data/cache.db?mode=ro', uri=True)
def bars_for(sym, day):
    df = pd.read_sql_query(
        "SELECT timestamp,open,high,low,close,volume FROM intraday_bars_1min "
        "WHERE symbol=? AND bar_date=? ORDER BY timestamp", con, params=(sym, day))
    if df.empty: return df
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df

atr_lookup = P.build_atr14_lookup(list(zip(bk['symbol'], bk['date']))) if cfg['atr_floor_enabled'] else {}

def sim(bars, entry_p, rh, rl, ets, key):
    shares = max(1, int(P.OLD_POS / entry_p))
    if WS:
        return P.simulate_winner_stack(bars, entry_p, rh, rl, ets, shares,
            atr14=atr_lookup.get(key) if cfg['atr_floor_enabled'] else None,
            atr_floor_enabled=cfg['atr_floor_enabled'], atr_floor_k=cfg['atr_floor_k'],
            scale_enabled=cfg['scale_enabled'], scale_frac=cfg['scale_frac'],
            scale_level_r=cfg['scale_level_r'])
    return P.simulate_static_lock(bars, entry_p, rh, rl, ets)

rows = []
for r in [type('R',(),d) for d in bk.to_dict('records')]:
    key = (r.symbol, r.date)
    b = bars_for(r.symbol, r.date)
    rec = dict(symbol=r.symbol, date=r.date, book_entered=int(r.entered),
               book_pnl_pct=float(r.pnl_pct), pos=float(getattr(r, '_rp_position')),
               book_reason=r.exit_reason, cls='nobars')
    if b.empty: rows.append(rec); continue
    ots = P._session_open_timestamp(b)
    if ots is None: rows.append(rec); continue
    rend = ots + timedelta(minutes=5)
    rb = b[(b['timestamp'] >= ots) & (b['timestamp'] < rend)]
    if len(rb) < 5: rows.append(rec); continue
    rh = float(rb['high'].max()); rl = float(rb['low'].min())
    search = b[(b['timestamp'] >= rend) & (b['timestamp'] < rend + timedelta(minutes=60))]
    ets = find_breakout_bar_ts(search, rh)
    rec.update(rh=rh, rl=rl)
    if ets is None:
        rec['cls'] = 'i_never_broken'; rows.append(rec); continue
    bb = search[search['timestamp'] == ets].iloc[0]
    cap30 = rh * 1.003; cap60 = rh * 1.006
    rec.update(bb_open=float(bb['open']), bb_low=float(bb['low']), cap30=cap30, cap60=cap60,
               ets=str(ets), gap_bps=(float(bb['open'])/rh - 1) * 1e4)
    rec['cls'] = 'ii_gap_through' if float(bb['open']) > cap30 else 'iii_fillable'
    # --- baseline book reproduction (book entry price) ---
    if int(r.entered) == 1:
        ep = float(r.entry_price)
        xp, rsn = sim(b, ep, rh, rl, ets, key)
        rec['repro_pnl_pct'] = (xp - ep) / ep * 100; rec['repro_reason'] = rsn
    # --- B30 ---
    if rec['cls'] == 'iii_fillable':
        b30 = cap30; b30_ts = ets
    elif rec['cls'] == 'ii_gap_through':
        b30 = cap30 if float(bb['low']) <= cap30 else None; b30_ts = ets
    else:
        b30 = None; b30_ts = None
    if b30 is not None:
        xp, rsn = sim(b, b30, rh, rl, b30_ts, key)
        rec.update(b30_entry=b30, b30_pnl_pct=(xp - b30)/b30*100, b30_reason=rsn)
    # --- F1a: cap60 on (ii) ---
    if rec['cls'] == 'ii_gap_through':
        o = float(bb['open'])
        if o <= cap60:
            xp, rsn = sim(b, o, rh, rl, ets, key)
            rec.update(f1a_entry=o, f1a_pnl_pct=(xp - o)/o*100, f1a_reason=rsn)
        # --- F1b: passive re-arm at rh on LATER bars until 10:35 ---
        later = search[search['timestamp'] > ets]
        hit = later[later['low'] <= rh]
        if len(hit):
            hb = hit.iloc[0]; fp = min(rh, float(hb['open']))
            xp, rsn = sim(b, fp, rh, rl, hb['timestamp'], key)
            rec.update(f1b_entry=fp, f1b_pnl_pct=(xp - fp)/fp*100, f1b_reason=rsn,
                       f1b_ts=str(hb['timestamp']))
    rows.append(rec)

out = pd.DataFrame(rows)
out.to_csv('research/orb_frequency/f1_rows.csv', index=False)
print(out['cls'].value_counts().to_dict())
f = out[out.book_entered == 1]
ok = (f['repro_pnl_pct'] - f['book_pnl_pct']).abs()
print('repro filled n=%d maxabs=%.6f  mismatches>1e-6: %d' % (len(f), ok.max(), (ok > 1e-6).sum()))
print(out[out.book_entered == 0]['cls'].value_counts().to_dict())
