Data pipeline for news + sentiment -> multimodal vectors

Quickstart

1. (Optional) create and activate the virtualenv used by the project.
2. Install dependencies:

```powershell
& "c:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/python.exe" -m pip install -r requirements.txt
```

3. Preprocess news:

```powershell
& "c:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/python.exe" scripts/preprocess.py --input data/raw_partner_headlines.csv --output data/news_clean.csv
```

4. Extract features (aggregated sentiment + news density):

```powershell
& "c:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/python.exe" scripts/feature_engineering.py --sentiment data/stock_data.csv --news data/news_clean.csv --output data/features_aggregated.csv
```

5. Compute embeddings (FinBERT) — may be slow on CPU:

```powershell
& "c:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/python.exe" scripts/embeddings.py --news data/news_clean.csv --out_emb data/embeddings.npy --out_meta data/embeddings_meta.csv
```

6. Fuse & cluster:

```powershell
& "c:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/python.exe" scripts/fuse_and_cluster.py --emb data/embeddings.npy --meta data/embeddings_meta.csv --features data/features_aggregated.csv --out data/fused_clusters.csv --k 10
```

Notes
- The scripts try to auto-detect common column names but may need small edits depending on your CSV schemas.
- Embedding step uses `ProsusAI/finbert` and will download the model from Hugging Face on first run.

## Multimodal clustering pipeline / Pipeline phân cụm đa thể thức

### Tổng quan
Pipeline tích hợp nhiều modal (text embeddings từ news/headlines, numeric features từ stock/sentiment, và metadata) để sinh biểu diễn hợp nhất và thực hiện phân cụm nhằm nhóm các bản tin / sự kiện theo chủ đề hoặc ảnh hưởng thị trường.

### Các bước chính
- Inventory & QC: kiểm tra modal có sẵn, tỉ lệ missing, đồng bộ thời gian. Tham khảo [data/news_clean.csv](data/news_clean.csv) và [data/features_aggregated.csv](data/features_aggregated.csv).
- Tiền xử lý: clean text, dedup, impute và scale numeric features.
- Trích xuất embedding: text → sentence-transformers (vd. `all-MiniLM-L6-v2` hoặc FinBERT cho domain), numeric → scaled vectors hoặc autoencoder.
- Fusion: early fusion (concat → projection MLP) hoặc late fusion (ensemble). Xử lý missing modal bằng masking/imputation.
- Giảm chiều & index: dùng `PCA`/`UMAP` cho trực quan, lưu index ANN bằng `faiss` cho truy vấn nhanh.
- Phân cụm: baseline `KMeans`; khuyến nghị thử `HDBSCAN` cho mật độ biến thiên; so sánh `GMM`/Agglomerative.
- Đánh giá & giải thích: Silhouette, Davies–Bouldin, Coverage, Stability (ARI giữa subsamples), và kiểm tra qualitative (top keywords per cluster).
- Triển khai reproducible: track experiments (MLflow/W&B), version data (DVC), containerize (Docker).

### Mục tiêu phân cụm (cụ thể, đo lường được)
- Silhouette score: tối thiểu ≥ 0.15 (baseline); mục tiêu tốt ≥ 0.25.
- Coverage (tỉ lệ items được gán cụm): ≥ 90%.
- Stability: ARI giữa các subsamples ≥ 0.7.
- Nếu có ground-truth labels: ARI/NMI ≥ 0.35 (conservative), hướng tới ≥ 0.5.
- Business impact: giảm thời gian gán nhãn thủ công ít nhất 30% khi dùng cụm để sample.

### Quick prototype
Bạn có thể bắt đầu bằng pipeline đơn giản: trích text-embedding từ [scripts/embeddings.py](scripts/embeddings.py), nối với [data/features_aggregated.csv](data/features_aggregated.csv), rồi chạy [scripts/fuse_and_cluster.py](scripts/fuse_and_cluster.py) (tham số `--k` hoặc cấu hình HDBSCAN).

### Next steps
- Chạy thử một experiment prototype (notebook/script) với sample nhỏ để kiểm tra metrics trên thực tế.
- Nếu đồng ý, tôi có thể: (A) viết notebook prototype, (B) cài đặt script pipeline modular, hoặc (C) chạy thử experiment mẫu và báo cáo kết quả.

