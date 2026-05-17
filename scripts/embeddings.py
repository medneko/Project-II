import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--news', default='data/news_clean.csv')
    parser.add_argument('--out_emb', default='data/embeddings.npy')
    parser.add_argument('--out_meta', default='data/embeddings_meta.csv')
    parser.add_argument('--model', default='ProsusAI/finbert')
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    news = pd.read_csv(args.news)
    texts = news['headline_clean'].fillna('').astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    embeddings = []
    for i in range(0, len(texts), args.batch):
        batch_texts = texts[i:i+args.batch]
        enc = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = out.last_hidden_state
            vecs = mean_pooling(token_embeddings, attention_mask)
            vecs = vecs.cpu().numpy()
            embeddings.append(vecs)

    embeddings = np.vstack(embeddings)
    np.save(args.out_emb, embeddings)
    news.reset_index()[['index']].to_csv(args.out_meta, index=False)
    print('Saved embeddings:', args.out_emb)


if __name__ == '__main__':
    main()
