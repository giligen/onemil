#!/usr/bin/env python3
"""Fetch SEC Form 3/4/5 structured insider-transaction quarterly datasets (2016Q1..2026Q2)
and build research/insider_events/data/purchases.parquet.

Purchase definition (PREREG.md): NONDERIV_TRANS row with
  TRANS_CODE == 'P', TRANS_ACQUIRED_DISP_CD == 'A', TRANS_PRICEPERSHARE > 0
joined to SUBMISSION (ACCESSION_NUMBER) with DOCUMENT_TYPE == '4' (amendments '4/A' excluded),
AFF10B5ONE != '1' (planned 10b5-1 trades excluded), joined to REPORTINGOWNER (ACCESSION_NUMBER)
for the owner's relationship flags / title.

Resumable: skips any quarter zip / extracted-tsv already present. Rate limit <= 1 req/s on SEC.
"""
from __future__ import annotations

import os
import sys
import time
import zipfile
import urllib.request
import urllib.error

import pandas as pd

BASE = 'https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets'
D = os.path.dirname(os.path.abspath(__file__)) + '/data'
ZIPD = f'{D}/zips'
EXTD = f'{D}/extracted'
UA = 'onemil research giligen@gmail.com'
NEEDED = ('NONDERIV_TRANS.tsv', 'SUBMISSION.tsv', 'REPORTINGOWNER.tsv')

TRANS_COLS = ['ACCESSION_NUMBER', 'TRANS_DATE', 'TRANS_CODE', 'TRANS_SHARES',
              'TRANS_PRICEPERSHARE', 'TRANS_ACQUIRED_DISP_CD']
SUB_COLS = ['ACCESSION_NUMBER', 'FILING_DATE', 'DOCUMENT_TYPE', 'ISSUERCIK',
            'ISSUERTRADINGSYMBOL', 'AFF10B5ONE']
OWNER_COLS = ['ACCESSION_NUMBER', 'RPTOWNERCIK', 'RPTOWNER_RELATIONSHIP', 'RPTOWNER_TITLE']


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def quarters(start=(2016, 1), end=(2026, 2)):
    y, q = start
    out = []
    while (y, q) <= end:
        out.append(f'{y}q{q}')
        q += 1
        if q > 4:
            q = 1
            y += 1
    return out


def download(qtr, missing):
    """Download one quarter's zip if not already present. Returns True on success."""
    dst = f'{ZIPD}/{qtr}_form345.zip'
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    url = f'{BASE}/{qtr}_form345.zip'
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    try:
        time.sleep(1.0)  # <= 1 req/s
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        tmp = dst + '.part'
        with open(tmp, 'wb') as f:
            f.write(data)
        os.rename(tmp, dst)
        log(f'{qtr}: downloaded {len(data):,} bytes')
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            log(f'{qtr}: 404 NOT FOUND (dataset does not cover this quarter)')
            missing.append(qtr)
        else:
            log(f'{qtr}: HTTP ERROR {e.code} {e.reason}')
            missing.append(qtr)
        return False
    except Exception as e:
        log(f'{qtr}: ERROR {e}')
        missing.append(qtr)
        return False


def extract(qtr):
    """Extract only the 3 needed tsv files from the quarter zip. Skips if already extracted."""
    outdir = f'{EXTD}/{qtr}'
    if all(os.path.exists(f'{outdir}/{f}') for f in NEEDED):
        return True
    src = f'{ZIPD}/{qtr}_form345.zip'
    if not os.path.exists(src):
        return False
    os.makedirs(outdir, exist_ok=True)
    try:
        with zipfile.ZipFile(src) as z:
            names = set(z.namelist())
            for f in NEEDED:
                if f not in names:
                    log(f'{qtr}: WARNING missing member {f} in zip')
                    continue
                z.extract(f, outdir)
        return True
    except Exception as e:
        log(f'{qtr}: ERROR extracting {e}')
        return False


def load_quarter(qtr):
    """Load+join+filter one quarter's purchase rows. Returns a DataFrame (possibly empty)."""
    outdir = f'{EXTD}/{qtr}'
    p_trans = f'{outdir}/NONDERIV_TRANS.tsv'
    p_sub = f'{outdir}/SUBMISSION.tsv'
    p_own = f'{outdir}/REPORTINGOWNER.tsv'
    if not (os.path.exists(p_trans) and os.path.exists(p_sub) and os.path.exists(p_own)):
        return None

    def _read(path, wanted):
        """Read only the wanted columns that actually exist in this quarter's file (the SEC
        schema changed over time -- e.g. AFF10B5ONE was added starting 2023q1, the Rule
        10b5-1 checkbox). Missing wanted columns are filled with '' and logged."""
        header = pd.read_csv(path, sep='\t', nrows=0).columns.tolist()
        present = [c for c in wanted if c in header]
        missing = [c for c in wanted if c not in header]
        df = pd.read_csv(path, sep='\t', usecols=present, dtype=str, low_memory=False)
        for c in missing:
            df[c] = ''
        if missing:
            log(f'{qtr}: WARNING {os.path.basename(path)} missing column(s) {missing} '
                f'(pre-dates this field in the SEC schema) -- filled blank')
        return df

    trans = _read(p_trans, TRANS_COLS)
    n_trans_raw = len(trans)
    sub = _read(p_sub, SUB_COLS)
    n_sub_raw = len(sub)
    own = _read(p_own, OWNER_COLS)
    n_own_raw = len(own)

    trans['TRANS_PRICEPERSHARE'] = pd.to_numeric(trans['TRANS_PRICEPERSHARE'], errors='coerce')
    trans['TRANS_SHARES'] = pd.to_numeric(trans['TRANS_SHARES'], errors='coerce')

    # --- purchase filter on NONDERIV_TRANS ---
    m = ((trans['TRANS_CODE'] == 'P')
         & (trans['TRANS_ACQUIRED_DISP_CD'] == 'A')
         & trans['TRANS_PRICEPERSHARE'].notna() & (trans['TRANS_PRICEPERSHARE'] > 0))
    trans_p = trans[m].copy()
    n_trans_purchase = len(trans_p)

    # --- SUBMISSION filter: Form 4 only (amendments '4/A' excluded), no planned 10b5-1 ---
    sub_f = sub[sub['DOCUMENT_TYPE'] == '4'].copy()
    n_sub_form4 = len(sub_f)
    aff = sub_f['AFF10B5ONE'].fillna('0').str.strip()
    n_10b5 = int((aff == '1').sum())
    sub_f = sub_f[aff != '1']

    merged = trans_p.merge(sub_f, on='ACCESSION_NUMBER', how='inner')
    n_after_sub_join = len(merged)

    merged = merged.merge(own, on='ACCESSION_NUMBER', how='inner')
    n_after_owner_join = len(merged)

    merged['symbol'] = merged['ISSUERTRADINGSYMBOL'].astype(str).str.strip().str.upper()
    n_no_symbol = int((merged['symbol'].isin(['', 'NAN', 'NONE'])).sum())
    merged = merged[~merged['symbol'].isin(['', 'NAN', 'NONE'])].copy()

    merged['filing_date'] = pd.to_datetime(merged['FILING_DATE'], format='%d-%b-%Y', errors='coerce')
    merged['trans_date'] = pd.to_datetime(merged['TRANS_DATE'], format='%d-%b-%Y', errors='coerce')
    n_bad_date = int(merged['filing_date'].isna().sum())
    merged = merged[merged['filing_date'].notna()]

    rel = merged['RPTOWNER_RELATIONSHIP'].fillna('')
    merged['is_officer'] = rel.str.contains('Officer', case=False, na=False)
    merged['is_director'] = rel.str.contains('Director', case=False, na=False)
    merged['is_ten_pct_owner'] = rel.str.contains('TenPercentOwner', case=False, na=False)
    merged['is_other'] = rel.str.contains('Other', case=False, na=False)

    out = pd.DataFrame({
        'accession': merged['ACCESSION_NUMBER'],
        'filing_date': merged['filing_date'],
        'trans_date': merged['trans_date'],
        'issuer_cik': merged['ISSUERCIK'],
        'symbol': merged['symbol'],
        'owner_cik': merged['RPTOWNERCIK'],
        'owner_relationship': merged['RPTOWNER_RELATIONSHIP'].fillna(''),
        'is_officer': merged['is_officer'],
        'is_director': merged['is_director'],
        'is_ten_pct_owner': merged['is_ten_pct_owner'],
        'is_other': merged['is_other'],
        'owner_title': merged['RPTOWNER_TITLE'].fillna(''),
        'shares': merged['TRANS_SHARES'],
        'price': merged['TRANS_PRICEPERSHARE'],
    })
    out['value'] = out['shares'] * out['price']
    out['quarter'] = qtr

    log(f'{qtr}: NONDERIV_TRANS raw {n_trans_raw:,} -> purchase-coded {n_trans_purchase:,} | '
        f'SUBMISSION raw {n_sub_raw:,} -> Form4 {n_sub_form4:,} (10b5-1 excluded {n_10b5:,}) | '
        f'OWNER raw {n_own_raw:,} | after sub-join {n_after_sub_join:,} -> after owner-join '
        f'{n_after_owner_join:,} | no/invalid symbol {n_no_symbol:,} ({n_no_symbol / max(n_after_owner_join, 1):.2%}) | '
        f'bad filing_date {n_bad_date:,} | final rows {len(out):,}')
    return out


def main():
    os.makedirs(ZIPD, exist_ok=True)
    os.makedirs(EXTD, exist_ok=True)
    qs = quarters()
    missing = []
    log(f'{len(qs)} quarters to fetch: {qs[0]}..{qs[-1]}')
    for qtr in qs:
        ok = download(qtr, missing)
        if ok:
            extract(qtr)

    frames = []
    for qtr in qs:
        if qtr in missing:
            continue
        df = load_quarter(qtr)
        if df is not None and len(df):
            frames.append(df)

    if not frames:
        log('ERROR: no quarters produced any rows -- aborting parquet build')
        sys.exit(1)

    full = pd.concat(frames, ignore_index=True)
    n_before_dedup = len(full)
    full = full.drop_duplicates(subset=['accession', 'owner_cik', 'trans_date', 'shares', 'price'])
    n_after_dedup = len(full)
    log(f'ALL QUARTERS: concatenated {n_before_dedup:,} rows -> deduped {n_after_dedup:,} '
        f'(dropped {n_before_dedup - n_after_dedup:,} cross-quarter duplicates)')

    out_path = f'{D}/purchases.parquet'
    full.to_parquet(out_path, index=False)
    log(f'wrote {out_path} ({len(full):,} rows, {full["symbol"].nunique():,} distinct symbols)')
    log(f'quarters with no data (404 or fetch error): {missing if missing else "none"}')


if __name__ == '__main__':
    main()
