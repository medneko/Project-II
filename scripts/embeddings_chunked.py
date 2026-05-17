import argparse
import os
import json
import time
from pathlib import Path
import numpy as np
import numpy.lib.format as nbf
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def count_lines(path):
    # subtract header
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, _ in enumerate(f):
            pass
    return max(0, i)  # i is zero-based index of last line


def safe_write_meta(meta_path, df_chunk, mode='a'):
    header = not Path(meta_path).exists() or mode == 'w'
    df_chunk.to_csv(meta_path, mode='a', header=header, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--news', default='data/news_clean.csv')
    parser.add_argument('--out_emb', default='data/embeddings.npy')
    parser.add_argument('--out_meta', default='data/embeddings_meta.csv')
    parser.add_argument('--model', default='ProsusAI/finbert')
    parser.add_argument('--chunksize', type=int, default=10000, help='rows per pandas chunk')
    parser.add_argument('--batch', type=int, default=64, help='tokenizer batch size')
    parser.add_argument('--progress', default='data/emb_progress.json')
    args = parser.parse_args()

    news_path = Path(args.news)
    out_emb = Path(args.out_emb)
    out_meta = Path(args.out_meta)
    progress_path = Path(args.progress)

    total = None
    # try to infer total rows from file
    print('Counting total rows in', news_path)
    total = count_lines(news_path)
    print('Total rows (including header):', total+1)
    total_rows = max(0, total)  # count_lines returned last index

    dim = 768
    dtype = np.float32

    # prepare memmap file
    print('Creating memmap at', out_emb)
    # ensure output directory exists
    out_emb.parent.mkdir(parents=True, exist_ok=True)

    # expected raw size (for raw memmap files without .npy header)
    expected_size = total_rows * dim * np.dtype(dtype).itemsize

    # Try to open as a proper .npy memmap first; if that fails, handle raw memmap or backup
    if out_emb.exists():
        try:
            print('Found existing embeddings file; attempting to open with open_memmap')
            mm = nbf.open_memmap(str(out_emb), mode='r+')
            if mm.shape != (total_rows, dim):
                print(f'Warning: existing .npy shape {mm.shape} != expected {(total_rows, dim)}')
        except ValueError:
            # maybe a raw memmap without .npy header (created by np.memmap earlier)
            size = out_emb.stat().st_size
            if size == expected_size:
                print('Existing file appears to be a raw memmap (no .npy header). Opening with np.memmap to resume.')
                mm = np.memmap(str(out_emb), dtype=dtype, mode='r+', shape=(total_rows, dim))
            else:
                # not compatible, back it up and create a fresh .npy memmap
                bak = out_emb.with_name(out_emb.name + f'.broken.{int(time.time())}')
                print(f'Existing file not compatible; renaming to {bak} and creating new .npy memmap')
                out_emb.rename(bak)
                mm = nbf.open_memmap(str(out_emb), mode='w+', dtype=dtype, shape=(total_rows, dim))
    else:
        mm = nbf.open_memmap(str(out_emb), mode='w+', dtype=dtype, shape=(total_rows, dim))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    processed = 0
    chunk_index = 0

    # remove existing meta if any (we append)
    if out_meta.exists():
        print('Removing existing meta file', out_meta)
        out_meta.unlink()

    for df_chunk in pd.read_csv(news_path, chunksize=args.chunksize):
        n_chunk = len(df_chunk)
        print(f'Processing chunk {chunk_index}: rows={n_chunk} (processed so far {processed})')

        texts = df_chunk['headline_clean'].fillna('').astype(str).tolist()

        # batch inside chunk
        all_vecs = []
        for i in range(0, n_chunk, args.batch):
            batch_texts = texts[i:i+args.batch]
            enc = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = out.last_hidden_state
                vecs = mean_pooling(token_embeddings, attention_mask)
                vecs = vecs.cpu().numpy().astype(dtype)
                all_vecs.append(vecs)

        if all_vecs:
            all_vecs = np.vstack(all_vecs)
            start = processed
            end = processed + n_chunk
            mm[start:end, :] = all_vecs
            mm.flush()

        # write meta: preserve index, ticker, date if present
        meta_cols = []
        for c in ('ticker', 'date', 'headline_clean'):
            if c in df_chunk.columns:
                meta_cols.append(c)
        if not meta_cols:
            meta_df = pd.DataFrame({'index': list(range(processed, processed + n_chunk))})
        else:
            meta_df = df_chunk[meta_cols].reset_index(drop=True)
            meta_df.insert(0, 'index', range(processed, processed + n_chunk))

        safe_write_meta(out_meta, meta_df, mode='a')

        processed += n_chunk
        chunk_index += 1

        # write progress
        prog = {'processed': processed, 'total': total_rows, 'chunk_index': chunk_index}
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump(prog, f)

        print('Chunk done. processed=', processed, 'of', total_rows)

    print('All chunks processed. flushing and closing.')
    del mm


if __name__ == '__main__':
    main()
