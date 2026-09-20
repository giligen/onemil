"""Fit a 3-state Gaussian HMM on SPY daily features (TRAIN 2025 only) and label days CAUSALLY.

Research (2026-09-20). Features: log return, 20-day realised vol (annualised). Fit on 2025 only.
Labels for every day from 2025-01-01 to 2026-05-31 come from the FORWARD filter (posterior of the
state given data up to and including that day — no smoothing, no look-ahead). The rule-based live
regime (trading/regime_helpers) is written beside it for comparison. TEST (>= 2026-06-01) not labeled.
"""
import sqlite3
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

conn = sqlite3.connect('file:data/cache.db?mode=ro', uri=True, timeout=30)
spy = pd.read_sql_query("SELECT bar_date, close FROM daily_bars WHERE symbol='SPY' ORDER BY bar_date", conn)
spy['bar_date'] = pd.to_datetime(spy['bar_date'])
spy['ret'] = np.log(spy['close']).diff()
spy['vol20'] = spy['ret'].rolling(20).std() * np.sqrt(252)
spy = spy.dropna().reset_index(drop=True)
X = spy[['ret', 'vol20']].to_numpy()

train = (spy['bar_date'] >= '2025-01-01') & (spy['bar_date'] <= '2025-12-31')
model = GaussianHMM(n_components=3, covariance_type='full', n_iter=500, random_state=7)
model.fit(X[train.to_numpy()])
order = np.argsort(model.means_[:, 1])          # 0 = calmest vol, 2 = most volatile
rank = {s: i for i, s in enumerate(order)}

# Causal labels: forward filter over the sequence up to each day (the posterior at the last step).
lab = []
upto = (spy['bar_date'] <= '2026-05-31').to_numpy()
idx = np.where(upto)[0]
for i in idx:
    lo = max(0, i - 120)                          # a 120-day filter window is enough to converge
    post = model.predict_proba(X[lo:i + 1])[-1]
    lab.append(rank[int(np.argmax(post))])
out = spy.loc[idx, ['bar_date', 'close', 'ret', 'vol20']].copy()
out['hmm_state'] = lab
try:
    from trading.regime_helpers import classify_regime_from_daily  # optional comparison
except Exception:
    classify_regime_from_daily = None
out.to_csv('research/regime/hmm_labels.csv', index=False)
print("means (ret, vol20) by state:", [tuple(np.round(model.means_[s], 4)) for s in order])
print("transition matrix:\n", np.round(model.transmat_[np.ix_(order, order)], 3))
print(out.groupby(out['bar_date'].dt.year)['hmm_state'].value_counts().unstack(fill_value=0))
print("wrote research/regime/hmm_labels.csv", len(out), "days (TEST not labeled)")
