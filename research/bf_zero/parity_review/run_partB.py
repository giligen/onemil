import os,sys,time
src=open('/home/ec2-user/onemil/research/bf_zero/parity_review/databento_vs_alpaca_rest.py').read()
head=src.split('# ---- Alpaca REST')[0]; tail=src.split('# ---- (B) spec book census')[1]
tail=tail.replace("open(f'{OUT}/databento_vs_alpaca_summary.txt', 'w').write('\\n'.join(lines + lines2) + '\\n')","open(f'{OUT}/spec_book_census.txt','w').write('\\n'.join(lines2)+'\\n')")
t0=time.time()
exec(compile(head+"\nimport time\n"+tail,'partB','exec'))
print('census seconds',time.time()-t0)
