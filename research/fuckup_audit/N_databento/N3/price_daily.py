import os, sys
from dotenv import load_dotenv
sys.path.insert(0,'/home/ec2-user/onemil'); os.chdir('/home/ec2-user/onemil')
load_dotenv('/home/ec2-user/onemil/.env')
import databento as db
c = db.Historical(os.environ['DATABENTO_API_KEY'])
for ds,sch,s,e in [('XNAS.ITCH','ohlcv-1d','2018-05-01','2024-01-01')]:
    cost = float(c.metadata.get_cost(dataset=ds,schema=sch,symbols='ALL_SYMBOLS',stype_in='raw_symbol',start=s,end=e))
    sz = c.metadata.get_billable_size(dataset=ds,schema=sch,symbols='ALL_SYMBOLS',stype_in='raw_symbol',start=s,end=e)
    print(f'{ds} {sch} {s}..{e}: ${cost:.4f}  billable={sz/1e6:.1f} MB', flush=True)
