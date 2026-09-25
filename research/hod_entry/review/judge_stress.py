"""Judge stress test for cell 1,427: does TEST (and each split) keep mean >= +0.10 R and day-clustered t >= 2
under the live slot config and realistic fill/cost haircuts measured by lenses B and C?

Haircuts (per fill, R units), all taken from the lens files, none fitted here:
  touch-stop slip  : a stop exit filled AT the stop (not a gap-through, which B0 already fills at the open) pays extra
                     slippage: plausible = 0.488 x 47 bps (live BF/ORB stop exits, lens C), severe = 79 bps (p10).
  size ($100 risk) : -0.024 R per fill (lens B size-walk, $100 risk);  severe uses the $375 figure -0.048 R.
  latency          : 32.4 % of fills do not survive 250 ms (lens B) -> removed at random (lens B found the removed
                     fills were WORSE than the survivors, so random removal is conservative for the mean), then the
                     12/day 4-concurrent slots are re-run on the survivors.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ec2-user/onemil/research/hod_consol')
sys.path.insert(0, '/home/ec2-user/onemil/research/hod_exit_lab')
from run_consol import simulate_slots          # noqa: E402  the report's own slot rule
from score_cells import day_clustered_t        # noqa: E402  the report's own t

BASE = '/home/ec2-user/onemil/research/hod_entry/'
LAT_DROP = 0.324
N_DRAWS = 100


def load():
    """All fills from the VAL-file (TRAIN-H2 + VAL, quote-history run) and the TEST file."""
    v = pd.read_csv(BASE + 'sip_rebuild_val.csv')
    t = pd.read_csv(BASE + 'sip_rebuild_test.csv')
    df = pd.concat([v, t], ignore_index=True)
    f = df[df.status == 'fill'].copy()
    f['stop'] = f.fill - f.R
    f['touch_stop'] = (f.why == 'stop') & (f.exit_price >= f.stop - 0.0051)
    f['bps_to_R'] = f.exit_price / f.R / 1e4          # 1 bp of exit price in R
    print('splits:', f.split.value_counts().to_dict(), '| stop exits', int((f.why == 'stop').sum()),
          'touch-stops', int(f.touch_stop.sum()), flush=True)
    return f.reset_index(drop=True)


def haircut(f, slip_bps, size_R):
    """Net R after the extra touch-stop slippage and the size haircut."""
    return f.net_R - np.where(f.touch_stop, slip_bps * f.bps_to_R, 0.0) - size_R


def slotted(f):
    """The live config: first 12 per day, 4 concurrent, ordered by fill minute (the report's own rule)."""
    s = f.assign(entry_m=f.fill_min).reset_index(drop=True)
    return s[simulate_slots(s)]


def stats(x, day):
    t, nd = day_clustered_t(x, day)
    return float(x.mean()), t, nd, len(x)


def scenario(f, slip_bps, size_R, latency, slots, rng):
    """One draw: optional latency removal, optional slots, then haircuts; returns (mean, t, n)."""
    g = f
    if latency:
        g = g[rng.random(len(g)) >= LAT_DROP]
    if slots:
        g = slotted(g)
    r = haircut(g, slip_bps, size_R)
    m, t, nd, n = stats(r, g.day)
    return m, t, n


def run(f, label):
    """Every scenario for one cohort; latency scenarios are the median over N_DRAWS random removals."""
    rows = []
    specs = [('base (as reported)', 0, 0, False), ('plausible: slip 22.9bps + size -0.024', 0.488 * 47, 0.024, False),
             ('severe: slip 79bps + size -0.048', 79, 0.048, False),
             ('plausible + latency 32%', 0.488 * 47, 0.024, True), ('severe + latency 32%', 79, 0.048, True)]
    for slots in (False, True):
        for name, slip, size, lat in specs:
            if lat:
                rng = np.random.default_rng(42)
                d = np.array([scenario(f, slip, size, True, slots, rng) for _ in range(N_DRAWS)])
                m, t, n = np.median(d[:, 0]), np.median(d[:, 1]), np.median(d[:, 2])
                p_t2 = float((d[:, 1] >= 2).mean())
            else:
                m, t, n = scenario(f, slip, size, False, slots, None)
                p_t2 = float(t >= 2)
            rows.append(dict(cohort=label, slots='12/4 slots' if slots else 'unslotted', scenario=name,
                             n=int(n), mean_R=round(m, 3), day_t=round(t, 2), share_draws_t_ge_2=round(p_t2, 2)))
        print(f'  done {label} slots={slots}', flush=True)
    return rows


def main():
    """Score TEST, TEST ex its best week, each earlier split, and the pooled book."""
    f = load()
    out = []
    test = f[f.split == 'TEST']
    wk_sum = test.groupby('wk').net_R.sum().sort_values(ascending=False)
    best = wk_sum.index[0]
    print('TEST best week', best, round(wk_sum.iloc[0], 1), 'of', round(test.net_R.sum(), 1), flush=True)
    cohorts = [('TEST', test), (f'TEST ex best week {best}', test[test.wk != best])]
    for sp in [s for s in f.split.unique() if s != 'TEST']:
        cohorts.append((sp, f[f.split == sp]))
    cohorts.append(('POOLED all splits', f))
    for label, g in cohorts:
        out += run(g, label)
    res = pd.DataFrame(out)
    res.to_csv('/tmp/claude-1000/-home-ec2-user-onemil/257c3e2d-cf38-45d5-94e7-4877f8170f44/scratchpad/judge_stress.csv',
               index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(res.to_string(index=False))


if __name__ == '__main__':
    main()
