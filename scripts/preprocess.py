import argparse
import pandas as pd
import re
from pathlib import Path


def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^\w\s\-\./:@]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_date(s):
    try:
        return pd.to_datetime(s).strftime('%Y-%m-%d')
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw_partner_headlines.csv')
    parser.add_argument('--output', default='data/news_clean.csv')
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    df = pd.read_csv(inp, low_memory=False)

    # try to find headline, ticker, date columns
    cols = {c.lower(): c for c in df.columns}
    headline_col = cols.get('headline') or cols.get('title') or list(df.columns)[0]
    ticker_col = cols.get('ticker') or cols.get('symbol') or None
    date_col = cols.get('date') or cols.get('datetime') or None

    df['headline_clean'] = df[headline_col].apply(clean_text)

    if ticker_col:
        df['ticker'] = df[ticker_col].astype(str).str.upper().str.strip()
    else:
        df['ticker'] = None

    if date_col:
        df['date'] = df[date_col].apply(normalize_date)
    else:
        df['date'] = None

    df[['headline_clean', 'ticker', 'date']].to_csv(out, index=False)
    print('Wrote', out)


if __name__ == '__main__':
    main()
