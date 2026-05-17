"""Advanced text EDA for Project 2
Generates CSVs and plots under report/ for TF-IDF, n-grams, topic modeling (LDA), NER (if spaCy), and temporal token trends.
Usage: python scripts/text_eda.py --data-dir data --out report
"""
import os
import re
import argparse
import logging
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

# optional packages
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

try:
    import spacy
except Exception:
    spacy = None

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def detect_text_col(df):
    candidates = ['headline', 'headline_clean', 'title', 'text', 'content', 'body']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback to first string column
    for c in df.columns:
        try:
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                return c
        except Exception:
            continue
    return None


def clean_text_simple(s):
    s = str(s)
    s = s.lower()
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"\S+@\S+", "", s)
    s = re.sub(r"[^\w\s'-]", ' ', s)
    s = re.sub(r"\s+", ' ', s).strip()
    return s


def top_terms_from_countvec(cv, X, n=50):
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(cv.get_feature_names_out())
    idx = np.argsort(sums)[::-1][:n]
    return list(zip(terms[idx], sums[idx]))


def top_terms_from_tfidf(tfv, X, n=50):
    mean_tfidf = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(tfv.get_feature_names_out())
    idx = np.argsort(mean_tfidf)[::-1][:n]
    return list(zip(terms[idx], mean_tfidf[idx]))


def lda_topics(count_vec, X, n_topics=10, n_top=15, random_state=42):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
    lda.fit(X)
    terms = count_vec.get_feature_names_out()
    topics = []
    for i, comp in enumerate(lda.components_):
        idx = comp.argsort()[::-1][:n_top]
        topics.append((i, [terms[j] for j in idx]))
    return topics


def ner_counts(spacy_nlp, texts, sample=1000):
    counts = Counter()
    sample_texts = texts if len(texts) <= sample else texts.sample(sample, random_state=1)
    for doc in spacy_nlp.pipe(sample_texts.astype(str).tolist(), batch_size=50):
        for ent in doc.ents:
            counts[(ent.label_, ent.text.lower())] += 1
    # aggregate by label
    label_counts = defaultdict(Counter)
    for (label, text), c in counts.items():
        label_counts[label][text] += c
    return label_counts


def token_trends(df, text_col, top_tokens, date_col='date', out_dir='report'):
    if date_col not in df.columns:
        logging.info('No date column for trends')
        return None
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['date_only'] = df[date_col].dt.date
    # build counts per day for selected tokens
    rows = []
    for t in top_tokens:
        pattern = re.compile(r"\b" + re.escape(t) + r"\b", flags=re.I)
        def has_token(s):
            return int(bool(pattern.search(str(s))))
        grouped = df.groupby('date_only')[text_col].apply(lambda s: s.map(has_token).sum())
        tmp = pd.DataFrame({'token': t, 'date': grouped.index, 'count': grouped.values})
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(out_dir, 'token_trends.csv'), index=False)
    # plot few tokens
    if plt is not None:
        try:
            plt.figure(figsize=(10,6))
            sample_tokens = top_tokens[:10]
            for t in sample_tokens:
                sub = out[out['token']==t]
                plt.plot(pd.to_datetime(sub['date']), sub['count'], label=t)
            plt.legend()
            plt.title('Top token trends')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'token_trends.png'))
            plt.close()
        except Exception as e:
            logging.warning('Could not plot token trends: %s', e)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--out', default='report')
    parser.add_argument('--text-col', default=None)
    parser.add_argument('--sample-rows', type=int, default=5000)
    parser.add_argument('--top-n', type=int, default=100)
    parser.add_argument('--ngram-range', default='1,1')
    parser.add_argument('--num-topics', type=int, default=10)
    args = parser.parse_args(argv)

    DATA_DIR = args.data_dir
    OUT = args.out
    ensure_dir(OUT)

    news_path = os.path.join(DATA_DIR, 'news_clean.csv')
    if not os.path.exists(news_path):
        logging.error('news file not found: %s', news_path)
        return

    df = pd.read_csv(news_path, nrows=args.sample_rows)
    text_col = args.text_col or detect_text_col(df)
    if text_col is None:
        logging.error('No text column detected')
        return
    logging.info('Using text column: %s', text_col)

    # basic clean
    df['text_clean'] = df[text_col].astype(str).map(clean_text_simple)

    # CountVectorizer unigram
    ngram_min, ngram_max = map(int, args.ngram_range.split(','))
    cv_uni = CountVectorizer(ngram_range=(1,1), stop_words='english', max_features=50000)
    X_uni = cv_uni.fit_transform(df['text_clean'])
    top_unigrams = top_terms_from_countvec(cv_uni, X_uni, n=args.top_n)
    pd.DataFrame(top_unigrams, columns=['term','count']).to_csv(os.path.join(OUT, 'top_unigrams.csv'), index=False)
    logging.info('Wrote top_unigrams.csv')

    # n-grams
    cv_ng = CountVectorizer(ngram_range=(max(1,ngram_min), max(1,ngram_max)), stop_words='english', max_features=50000)
    X_ng = cv_ng.fit_transform(df['text_clean'])
    top_ngrams = top_terms_from_countvec(cv_ng, X_ng, n=args.top_n)
    pd.DataFrame(top_ngrams, columns=['ngram','count']).to_csv(os.path.join(OUT, 'top_ngrams.csv'), index=False)
    logging.info('Wrote top_ngrams.csv')

    # TF-IDF
    tfv = TfidfVectorizer(ngram_range=(1,1), stop_words='english', max_features=50000)
    X_tfidf = tfv.fit_transform(df['text_clean'])
    top_tfidf = top_terms_from_tfidf(tfv, X_tfidf, n=args.top_n)
    pd.DataFrame(top_tfidf, columns=['term','tfidf']).to_csv(os.path.join(OUT, 'top_tfidf.csv'), index=False)
    logging.info('Wrote top_tfidf.csv')

    # LDA topic modeling
    try:
        cv_small = CountVectorizer(ngram_range=(1,1), stop_words='english', max_features=5000)
        X_topics = cv_small.fit_transform(df['text_clean'])
        topics = lda_topics(cv_small, X_topics, n_topics=args.num_topics, n_top=15)
        with open(os.path.join(OUT, 'lda_topics.txt'), 'w', encoding='utf8') as f:
            for tid, terms in topics:
                f.write(f'Topic {tid}: ' + ', '.join(terms) + '\n')
        logging.info('Wrote lda_topics.txt')
    except Exception as e:
        logging.warning('LDA failed: %s', e)

    # NER via spaCy (optional)
    if spacy is not None:
        try:
            # try load en_core_web_sm
            try:
                nlp = spacy.load('en_core_web_sm')
            except Exception:
                nlp = spacy.load('en_core_web_trf') if 'en_core_web_trf' in spacy.util.get_installed_models() else None
            if nlp is None:
                logging.warning('spaCy model not available, skipping NER')
            else:
                ner_res = ner_counts(nlp, df['text_clean'], sample=2000)
                # save top entities per label
                rows = []
                for label, ctr in ner_res.items():
                    for text, c in ctr.most_common(200):
                        rows.append((label, text, c))
                pd.DataFrame(rows, columns=['label','entity','count']).to_csv(os.path.join(OUT, 'ner_entities.csv'), index=False)
                logging.info('Wrote ner_entities.csv')
        except Exception as e:
            logging.warning('NER processing failed: %s', e)
    else:
        logging.info('spaCy not installed; skip NER')

    # WordCloud for top tokens
    if WordCloud is not None and plt is not None:
        try:
            freqs = dict(top_unigrams[:200])
            wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freqs)
            plt.figure(figsize=(12,6))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(os.path.join(OUT, 'wordcloud.png'))
            plt.close()
            logging.info('Wrote wordcloud.png')
        except Exception as e:
            logging.warning('WordCloud failed: %s', e)
    else:
        logging.info('wordcloud not available; skip')

    # token trends
    top_terms = [t for t,_ in top_unigrams[:20]]
    token_trends(df, 'text_clean', top_terms, date_col='date', out_dir=OUT)

    # plots: top unigrams bar
    if plt is not None:
        try:
            top = pd.DataFrame(top_unigrams[:30], columns=['term','count'])
            plt.figure(figsize=(10,6))
            if sns is not None:
                sns.barplot(x='count', y='term', data=top)
            else:
                plt.barh(top['term'], top['count'])
            plt.title('Top unigrams')
            plt.tight_layout()
            plt.savefig(os.path.join(OUT, 'top_unigrams.png'))
            plt.close()
            logging.info('Wrote top_unigrams.png')
        except Exception as e:
            logging.warning('Top unigrams plot failed: %s', e)

    logging.info('Text EDA complete. Outputs in %s', OUT)


if __name__ == '__main__':
    main()
