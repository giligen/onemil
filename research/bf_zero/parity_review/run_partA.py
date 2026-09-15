import os,sys,runpy
os.environ['PART']='A'
src=open('/home/ec2-user/onemil/research/bf_zero/parity_review/databento_vs_alpaca_rest.py').read()
src=src.split('# ---- (B) spec book census')[0]
exec(compile(src,'partA','exec'))
