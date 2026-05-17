import argparse
import pandas as pd
from pathlib import Path


def map_sentiment_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ('positive', 'pos', '1'):
        return 1
    if s in ('negative', 'neg', '-1'):
        return -1
    if s in ('neutral', 'neu', '0'):
        return 0
    # try numeric
    try:
        v = float(s)
        return v
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentiment', default='data/stock_data.csv', help='input sentiment CSV')
    parser.add_argument('--news', default='data/news_clean.csv', help='cleaned news CSV')
    parser.add_argument('--output', default='data/features_aggregated.csv')
    args = parser.parse_args()

    sfn = Path(args.sentiment)
    nfn = Path(args.news)
    out = Path(args.output)

    df_s = pd.read_csv(sfn, low_memory=False)
    df_n = pd.read_csv(nfn, low_memory=False)

    # detect columns
    scols = {c.lower(): c for c in df_s.columns}
    if 'sentiment' in scols:
        label_col = scols['sentiment']
    elif 'label' in scols:
        label_col = scols['label']
    else:
        # try common names
        label_col = list(df_s.columns)[-1]

    # find ticker and date
    ticker_col = scols.get('ticker') or scols.get('symbol') or None
    date_col = scols.get('date') or scols.get('datetime') or None

    if ticker_col is None or date_col is None:
        # try to infer
        ticker_col = ticker_col or list(df_s.columns)[0]
        date_col = date_col or list(df_s.columns)[1]

    df_s['sentiment_num'] = df_s[label_col].apply(map_sentiment_label)
    df_s['ticker'] = df_s[ticker_col].astype(str).str.upper().str.strip()
    df_s['date'] = pd.to_datetime(df_s[date_col], errors='coerce').dt.strftime('%Y-%m-%d')

    # aggregated sentiment per ticker/date
    agg = df_s.groupby(['ticker', 'date'])['sentiment_num'].agg(['mean', 'count']).reset_index()
    agg = agg.rename(columns={'mean': 'sentiment_mean', 'count': 'sentiment_count'})

    # news density from news_clean
    df_n['date'] = pd.to_datetime(df_n['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    news_agg = df_n.groupby(['ticker', 'date']).size().reset_index(name='news_density')

    merged = pd.merge(agg, news_agg, on=['ticker', 'date'], how='left')
    merged['news_density'] = merged['news_density'].fillna(0).astype(int)

    merged.to_csv(out, index=False)
    print('Wrote', out)


if __name__ == '__main__':
    main()
