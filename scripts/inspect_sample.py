#!/usr/bin/env python3
"""Print columns and a small sample of the anonymized data."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main(path='data/sample.csv'):
    p = ROOT / path
    df = pd.read_csv(p)
    print('COLUMNS:\n' + '\n'.join(df.columns.tolist()))
    print('\nHEAD:\n', df.head(3).to_string(index=False))

if __name__ == '__main__':
    main()
