import os
p='data/embeddings.npy'
if not os.path.exists(p):
    print('NOT FOUND')
    raise SystemExit(1)
with open(p,'rb') as f:
    b = f.read(16)
print('first 16 bytes:', b)
# try to see if file is a zip (npz)
if b.startswith(b'PK'):
    print('Looks like a zip/npz file')
else:
    # try to read header using numpy.lib.format
    try:
        import numpy as np
        from numpy.lib import format
        with open(p,'rb') as f:
            version = format.read_magic(f)
            print('npy version:', version)
            header = format.read_array_header_1_0(f)
            print('header:', header)
    except Exception as e:
        print('format header read failed:', e)
print('done')
