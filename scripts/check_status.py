import os, json
import numpy as np
import pandas as pd
lines=[]
emb_path='data/embeddings.npy'
if os.path.exists(emb_path):
    try:
        mm = np.load(emb_path, mmap_mode='r')
        lines.append(f"emb shape: {mm.shape}, dtype: {mm.dtype}, bytes: {os.path.getsize(emb_path)}")
        try:
            nan_rows = int(np.isnan(mm).any(axis=1).sum())
            zero_rows = int((mm==0).all(axis=1).sum())
            lines.append(f"nan rows: {nan_rows}, all-zero rows: {zero_rows}")
        except Exception as e:
            lines.append('Could not compute nan/all-zero rows: '+repr(e))
    except Exception as e:
        lines.append('Error loading embeddings: '+repr(e))
else:
    lines.append('Missing: data/embeddings.npy')
meta_path='data/embeddings_meta.csv'
if os.path.exists(meta_path):
    try:
        m=pd.read_csv(meta_path)
        lines.append(f"meta rows: {len(m)}, columns: {list(m.columns)}")
        lines.append('meta head:\n'+m.head(5).to_string(index=False))
    except Exception as e:
        lines.append('Error reading meta: '+repr(e))
else:
    lines.append('Missing: data/embeddings_meta.csv')
prog_path='data/emb_progress.json'
if os.path.exists(prog_path):
    try:
        prog=json.load(open(prog_path,'r',encoding='utf-8'))
        lines.append('progress file: '+str(prog))
    except Exception as e:
        lines.append('Could not read progress file: '+repr(e))
else:
    lines.append('No progress file found')
fc_path='data/fused_clusters.csv'
if os.path.exists(fc_path):
    try:
        fc=pd.read_csv(fc_path)
        if 'cluster' in fc.columns:
            vc=fc['cluster'].value_counts().to_dict()
            lines.append(f"fused_clusters rows: {len(fc)}, cluster counts: {vc}")
        else:
            lines.append(f"fused_clusters present ({len(fc)} rows) but no 'cluster' column")
    except Exception as e:
        lines.append('Error reading fused_clusters: '+repr(e))
else:
    lines.append('No fused_clusters.csv')
print('\n\n'.join(lines))
