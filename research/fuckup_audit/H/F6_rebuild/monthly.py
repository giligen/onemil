"""Monthly net-R table per exit rule for the booked trades."""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main(sig='a', stp='i'):
    out = {}
    for key in ['hold', 'r2', 'partial']:
        df = pd.read_csv(os.path.join(HERE, 'trades_%s_%s%s.csv' % (key, sig, stp)),
                         keep_default_na=False, na_values=[''])
        m = df.day.str.slice(0, 7)
        g = df.groupby(m).net_R.agg(['count', 'sum'])
        out[(key, 'n')] = g['count']
        out[(key, 'netR')] = g['sum']
    t = pd.DataFrame(out).fillna(0)
    p = os.path.join(HERE, 'monthly_%s%s.csv' % (sig, stp))
    t.to_csv(p)
    print(t.round(2).to_string())
    print('wrote', p)


if __name__ == '__main__':
    main(*(sys.argv[1:3] or ['a', 'i']))
