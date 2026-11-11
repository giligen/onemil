"""Supplement to analyze.py: ORB accidents priced from broker fills, HOD robust R. Appends to NUMBERS.csv / weekly.csv.

CRDU 10/27: broker bought 861 @5.51 (provisional 9debf1bf, cancel-queue fill) + 599 @5.56 (ledger order 4918c7bf),
sold 1,460 @6.21 at the 15:45 force close. Ledger row 621 books 949.00 = (6.21-5.56) x 1,460: the naked 861 sh are
INSIDE the ledger P&L (+559.65 at the ledger entry price; +602.70 at the broker's 5.51).
BEX 11/9: base 87 @14.22 sold 14:16 @15.16 (stop leg, +81.78); add-on 87 @15.32 at 14:00 unprotected, sold @15.36
(+3.48). Ledger row 737 books 47.85 (neither).
"""
import csv
import statistics as st
import sys
from pathlib import Path

HERE = Path('/home/ec2-user/onemil/research/book_review_20261111')
sys.path.insert(0, str(HERE))
import analyze as an  # noqa: E402

CRDU_LEDGER_ACCIDENT = (6.21 - 5.56) * 861
CRDU_BROKER_ACCIDENT = (6.21 - 5.51) * 861
BEX_BASE = (15.16 - 14.22) * 87
BEX_ADDON = (15.36 - 15.32) * 87


def main():
    """Accident-OUT ORB stats, gate ex-accidents, HOD robust-R stats and weekly R."""
    orb, q, _ = an.ledger_rows('orb')
    out = []
    for r in orb:
        r = dict(r)
        if r['day'] == '2026-10-27' and r['sym'] == 'CRDU':
            r['pnl'] -= CRDU_LEDGER_ACCIDENT
        if r['day'] == '2026-11-09' and r['sym'] == 'BEX':
            r['pnl'] = BEX_BASE
        r['R'] = r['pnl'] / r['total_risk']
        out.append(r)
    qa = q + ' ; CRDU minus 861-sh accident (559.65), BEX = base leg from broker fills (81.78)'
    an.put('accident_CRDU_broker_usd', 'ORB', CRDU_BROKER_ACCIDENT, source='activities FILL', query='861 x (6.21-5.51)')
    an.put('accident_CRDU_inside_ledger_usd', 'ORB', CRDU_LEDGER_ACCIDENT, source='trades id 621 vs fills', query='861 x (6.21-5.56)')
    an.put('accident_BEX_addon_usd', 'ORB', BEX_ADDON, source='activities FILL', query='87 x (15.36-15.32)')
    an.put('ledger_misbook_BEX_usd', 'ORB', 47.85 - BEX_BASE, source='trades id 737 vs fills', query='ledger 47.85 vs base 81.78')
    an.book_stats('ORB', out, '_exAccidents', qa)
    an.weekly_rows_ex = an.weekly('ORB-exAcc', out)
    since = [r for r in out]  # window rows; add AXTL 9/29 (pre-window, in the Amendment-1 count)
    axtl = -96.3
    real = sum(r['pnl'] for r in since) + axtl
    best = max([r['pnl'] for r in since] + [axtl])
    an.put('gate_realized_since_0925_exAccidents', 'ORB', real, window='2026-09-25..2026-11-10', query=qa + ' + AXTL 9/29 -96.30')
    an.put('gate_ex_monster_since_0925_exAccidents', 'ORB', real - best, window='2026-09-25..2026-11-10', query='realized - best fill (LABX 10/27 496.69)')
    an.put('gate_needed_ex_monster_per_fill_next8', 'ORB', -(real - best) / 8, window='to 40 symbol-days', query='-(ex-monster)/8 remaining symbol-days')

    hod, qh, _ = an.ledger_rows('hod_break')
    for r in hod:
        den = abs(r['fill_price'] - r['stop_loss_price']) * (r['filled_qty'] or r['shares'])
        r['R'] = r['pnl'] / den if den else 0.0
        r['risk'] = den
    qr = qh + ' ; R = pnl / (|fill_price - stop_loss_price| x filled_qty) (total_risk is wrong on partial-fill rows)'
    an.put('mean_R_robust', 'HOD', st.mean(r['R'] for r in hod), query=qr)
    ti, tc = an.tstats(hod, 'R')
    an.put('t_iid_R_robust', 'HOD', ti, query=qr); an.put('t_dayclustered_R_robust', 'HOD', tc, query=qr)
    k = max(1, round(0.05 * len(hod)))
    rs = sorted(r['R'] for r in hod)
    an.put('mean_R_robust_ex_top5pct', 'HOD', st.mean(rs[:-k]), query=qr + f' drop top {k}')
    wk = an.weekly('HOD-robustR', hod)
    with open(HERE / 'weekly.csv', 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(wk[0]))
        w.writerows(an.weekly_rows_ex + wk)
    for r in an.weekly_rows_ex + wk:
        print(r, flush=True)
    with open(HERE / 'NUMBERS.csv', 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'book', 'value', 'window', 'source', 'query'])
        w.writerows(an.NUM)


if __name__ == '__main__':
    main()
