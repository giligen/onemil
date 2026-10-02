"""Build recon copies of both builds (logic untouched; dump + env toggles only)."""
import re
R='/home/ec2-user/onemil/research/momentum_weekly/'
a=open(R+'1700g_vol.py').read().split('\n')
a='\n'.join(a[:446])
a=a.replace("OUT / '1700g.log'","OUT / 'recon' / 'A_run.log'")
a+='''
real_B, pools_B, ports_B = run_cell(weekly_cal, ADV_CUTOFF_U2, 'sig_V2', [20])
rows=[]
for e,syms in ports_B[20].items():
    nxt=weekly_cal.loc[weekly_cal.entry_date==e,'next_entry_date'].iloc[0]
    fw=fwd_ret_row(e,nxt)
    sd=weekly_cal.loc[weekly_cal.entry_date==e,'prior_signal_date'].iloc[0]
    if sd not in sig_by_date: continue
    day=sig_by_date[sd].set_index('symbol')
    for s in syms:
        rows.append(dict(rebalance_date=e,symbol=s,weight=1/len(syms),signal=day.loc[s,'sig_V2'],entry_open=open_piv.loc[e,s],next_open=open_piv.loc[nxt,s],wk_ret=fw.get(s)))
pd.DataFrame(rows).to_csv(str(OUT/'recon'/'A_holdings.csv'),index=False)
real_B[20].to_csv(str(OUT/'recon'/'A_weekly.csv'))
print('DONE',flush=True)
'''
open('A_dump.py','w').write(a)

b=open(R+'REBUILD_1700_sleeve.py').read()
b=b.replace('research/momentum_weekly/','research/momentum_weekly/recon/B_').replace('recon/B_panel_2016','panel_2016').replace('recon/B_1700c_assets','1700c_assets')
b=b.replace('import sys\n','import sys, os\nTAG=os.environ.get("B_TAG","base")\nF=set(os.environ.get("B_FLAGS","").split(","))\n',1)
b=b.replace('OUT_WEEKLY = "research/momentum_weekly/recon/B_1700_rebuild_weekly.csv"','OUT_WEEKLY = f"research/momentum_weekly/recon/B_{TAG}_weekly.csv"')
b=b.replace('OUT_BY_YEAR = "research/momentum_weekly/recon/B_1700_rebuild_by_year.csv"','OUT_BY_YEAR = f"research/momentum_weekly/recon/B_{TAG}_by_year.csv"')
b=b.replace('OUT_MD = "research/momentum_weekly/recon/B_REBUILD_1700_sleeve.md"','OUT_MD = f"research/momentum_weekly/recon/B_{TAG}.md"')
# A's exclusion regex
b=b.replace('def log(msg)','''A_RE = re.compile(r'\\bETFs?\\b|\\bETNs?\\b|\\bFUNDs?\\b|\\bTRUSTs?\\b|\\bWARRANTS?\\b|\\bUNITS?\\b|\\bPREFERRED\\b|\\bRIGHTS?\\b', re.IGNORECASE)
if "AEXCL" in F: NAME_EXCL_RE = A_RE
if "ACOST" in F: COST_CAP = 0.0015  # == A: spread proxy clipped at 20bps => cost <= 15bps
def log(msg)''',1)
b=b.replace('''    df["bar_date"] = pd.to_datetime(df["bar_date"])
    for c in''','''    df["bar_date"] = pd.to_datetime(df["bar_date"])
    if "CLEAN" in F:
        n0=len(df); df=df.drop_duplicates(subset=["symbol","bar_date"],keep="last")
        df=df[(df.open>0)&(df.high>0)&(df.low>0)&(df.close>0)].reset_index(drop=True)
        log(f"CLEAN dropped {n0-len(df)} rows")
    for c in''',1)
b=b.replace('bad_sym_pattern = syms.str.match(SYM_ZZZT_RE) | syms.str.contains(r"[./]", regex=True)','bad_sym_pattern = syms.str.match(SYM_ZZZT_RE) | (syms.str.contains(r"[./]", regex=True) if "NODOTS" not in F else False)')
b=b.replace('''        cash_balance = cash  # leftover''','''        if "EQW" in F:  # A's convention: free weekly rebalance back to 1/20
            tot = cash + sum(holdings.values())
            for s_ in holdings: holdings[s_] = tot / len(holdings)
            cash = 0.0
        cash_balance = cash  # leftover''',1)
b=b.replace('''        if nxt is not None:
            pv_end_week = 0.0''','''        sigmap = dict(zip(snap["symbol"], snap["signal"]))
        if nxt is not None:
            pv_end_week = 0.0''',1)
b=b.replace('''                new_val = val * (px_next / px_now)''','''                new_val = val * (px_next / px_now)
                DUMP.append((reb.date().isoformat(), sym, val / max(pv_start_week,1e-9), sigmap.get(sym), px_now, px_next, px_next / px_now - 1))''',1)
b=b.replace('def main():','DUMP=[]\ndef main():',1)
b=b.replace('    log("DONE")','''    pd.DataFrame(DUMP, columns=["rebalance_date","symbol","weight","signal","entry_open","next_open","wk_ret"]).to_csv(f"research/momentum_weekly/recon/B_{TAG}_holdings.csv", index=False)
    log("DONE")''')
open('B_dump.py','w').write(b)
