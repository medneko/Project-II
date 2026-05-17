"""Inspect embeddings.npy and attempt to stack a sample to numeric array.
Saves a sample output to data/embeddings_clean_test.npy if successful.
"""
import os
import sys
import numpy as np

p = 'data/embeddings.npy'
print('path exists:', os.path.exists(p))
if not os.path.exists(p):
    sys.exit(1)
print('size bytes:', os.path.getsize(p))

try:
    a = np.load(p, allow_pickle=True)
    print('loaded type:', type(a), 'shape:', getattr(a, 'shape', None), 'dtype:', getattr(a, 'dtype', None))
    try:
        first = a[0]
        print('first elem type:', type(first))
    except Exception as e:
        print('could not index first element:', e)

    # attempt stacking a small sample
    n = min(100, len(a))
    print('sample n =', n)
    sample = a[:n]
    arrs = []
    for i, x in enumerate(sample):
        try:
            arrs.append(np.asarray(x))
        except Exception as ex:
            print('convert failed at', i, ex)
            raise
    arr = np.vstack(arrs)
    print('stacked sample shape', arr.shape, 'dtype', arr.dtype)
    out = 'data/embeddings_clean_test.npy'
    np.save(out, arr)
    print('saved sample to', out)
except Exception as e:
    print('LOAD FAILED:', repr(e))
    sys.exit(2)

print('done')
