#!/usr/bin/env python3
"""Convert raw float32 embeddings binary to .npy with shape (n_vectors, dim).
The script will try to detect `dim` from `data/embeddings_sample_with_meta.csv` header.
Use --dim to force a dimension.
"""
import os
import sys
import argparse
import struct

try:
    import numpy as np
except Exception as e:
    print("numpy is required: pip install numpy")
    raise


def detect_dim_from_sample_meta(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        header = f.readline().strip().split(',')
    # header like 'index,0,1,2,...'
    numeric_cols = [c for c in header if c.isdigit()]
    if len(numeric_cols) > 0:
        return len(numeric_cols)
    # fallback: if header includes 'index' then dim = len(header)-1
    if len(header) > 1 and header[0].lower().startswith('index'):
        return len(header)-1
    return None


def is_raw_float32_file(path):
    try:
        with open(path, 'rb') as f:
            b = f.read(16)
            if len(b) < 4:
                return False
            vals = []
            for i in range(0, len(b)//4):
                vals.append(struct.unpack('<f', b[i*4:(i+1)*4])[0])
            import math
            return all(math.isfinite(x) and abs(x) < 1e6 for x in vals)
    except Exception:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', default='data/embeddings.npy')
    parser.add_argument('--outfile', default='data/embeddings_clean.npy')
    parser.add_argument('--dim', type=int, default=None)
    parser.add_argument('--sample-meta', default='data/embeddings_sample_with_meta.csv')
    parser.add_argument('--chunk-rows', type=int, default=50000)
    args = parser.parse_args(argv)

    infile = args.infile
    outfile = args.outfile
    dim = args.dim

    if not os.path.exists(infile):
        print('Input file not found:', infile)
        return 2

    if dim is None:
        dim = detect_dim_from_sample_meta(args.sample_meta)
        if dim is not None:
            print('Auto-detected dim from', args.sample_meta, '->', dim)
        else:
            print('Could not detect dim automatically. Use --dim.')
            return 3

    filesize = os.path.getsize(infile)
    float_bytes = 4
    if filesize % float_bytes != 0:
        print('Warning: file size not divisible by 4 bytes. filesize=', filesize)
    n_floats = filesize // float_bytes
    if n_floats % dim != 0:
        print(f'Warning: total floats {n_floats} not divisible by dim {dim}.')
    n_vectors = n_floats // dim
    print('File size', filesize, 'n_floats', n_floats, 'dim', dim, 'n_vectors', n_vectors)

    if not is_raw_float32_file(infile):
        print('File does not look like raw float32 data. Aborting to avoid corrupt output.')
        return 4

    # Create memmap output
    print('Creating memmap:', outfile)
    out = np.lib.format.open_memmap(outfile, mode='w+', dtype=np.float32, shape=(n_vectors, dim))

    row_bytes = dim * float_bytes
    chunk_rows = args.chunk_rows
    written = 0
    import time
    t0 = time.time()
    with open(infile, 'rb') as f:
        for start in range(0, n_vectors, chunk_rows):
            count = min(chunk_rows, n_vectors - start)
            data = f.read(count * row_bytes)
            arr = np.frombuffer(data, dtype=np.float32)
            try:
                arr = arr.reshape(count, dim)
            except Exception as e:
                print('Reshape failed at start', start, 'count', count, '->', e)
                return 5
            out[start:start+count] = arr
            written += count
            if time.time() - t0 > 1:
                print(f'Progress: {written}/{n_vectors} rows ({written*100/n_vectors:.2f}%)')
                t0 = time.time()

    print('Done. Wrote', written, 'rows to', outfile)
    return 0

if __name__ == '__main__':
    sys.exit(main())
