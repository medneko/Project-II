Luồng xử lý dữ liệu cho tin tức và sentiment → biểu diễn đa thể thức

Bắt đầu nhanh

1. (Tùy chọn) tạo và kích hoạt virtualenv (môi trường ảo) cho dự án.
2. Cài đặt phụ thuộc:

```powershell
& "c:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/python.exe" -m pip install -r requirements.txt
```

3. Tiền xử lý dữ liệu tin tức:

```powershell
& "c:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/python.exe" scripts/preprocess.py --input data/raw_partner_headlines.csv --output data/news_clean.csv
```

4. Trích xuất đặc trưng (sentiment tổng hợp + mật độ tin tức):

```powershell
& "c:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/python.exe" scripts/feature_engineering.py --sentiment data/stock_data.csv --news data/news_clean.csv --output data/features_aggregated.csv
```

5. Tính embeddings (FinBERT) — có thể chậm khi chạy trên CPU:

```powershell
& "c:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/python.exe" scripts/embeddings.py --news data/news_clean.csv --out_emb data/embeddings.npy --out_meta data/embeddings_meta.csv
```

6. Hợp nhất & phân cụm:

```powershell
& "c:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/python.exe" scripts/fuse_and_cluster.py --emb data/embeddings.npy --meta data/embeddings_meta.csv --features data/features_aggregated.csv --out data/fused_clusters.csv --k 10
```

Ghi chú
- Các script cố gắng tự động nhận diện tên cột phổ biến nhưng có thể cần chỉnh sửa nhỏ tùy theo schema CSV của bạn.
- Bước tính embedding sử dụng `ProsusAI/finbert` và sẽ tải mô hình từ Hugging Face khi chạy lần đầu.

## Quy trình phân cụm đa thể thức

### Tổng quan
Quy trình tích hợp nhiều modal (embedding văn bản từ tin tức/headlines, đặc trưng số từ stock/sentiment, và metadata) để sinh biểu diễn hợp nhất và thực hiện phân cụm, nhằm nhóm các bản tin/sự kiện theo chủ đề hoặc theo ảnh hưởng đến thị trường.

### Các bước chính
- Khảo sát & QC: kiểm tra modal có sẵn, tỉ lệ missing, đồng bộ thời gian. Tham khảo [data/news_clean.csv](data/news_clean.csv) và [data/features_aggregated.csv](data/features_aggregated.csv).
- Tiền xử lý: làm sạch văn bản, loại trùng (dedup), impute và chuẩn hóa (scale) các đặc trưng số.
- Trích xuất embedding: văn bản → sentence-transformers (vd. `all-MiniLM-L6-v2` hoặc FinBERT cho domain), numeric → vector đã chuẩn hóa hoặc autoencoder.
- Kết hợp đặc trưng (fusion): early fusion (concat → projection MLP) hoặc late fusion (ensemble). Xử lý modal thiếu bằng masking hoặc imputation.
- Giảm chiều & index: dùng `PCA`/`UMAP` cho trực quan, lưu index ANN bằng `faiss` cho truy vấn hiệu quả.
- Phân cụm: baseline `KMeans`; khuyến nghị thử `HDBSCAN` cho trường hợp mật độ thay đổi; so sánh với `GMM`/Agglomerative.
- Đánh giá & giải thích: Silhouette, Davies–Bouldin, Coverage, Stability (ARI giữa các mẫu con), và kiểm tra định tính (top keywords cho mỗi cụm).
- Triển khai có thể tái tạo: theo dõi thử nghiệm (MLflow/W&B), quản lý phiên bản dữ liệu (DVC), đóng gói bằng container (Docker).

### Thuật toán & Độ đo đề xuất

**Thuật toán phân cụm đề xuất**
- Nhóm phân cấp (Hierarchical): Single Linkage, Ward’s Linkage
- Nhóm dựa trên tâm cụm: K-means (với nhiều giá trị k), K-Medoids (PAM)
- Nhóm dựa trên mật độ: DBSCAN, HDBSCAN
- Nhóm xác suất: Gaussian Mixture Models (GMM)

**Độ đo đánh giá đề xuất**
- Độ đo nội tại:
	- Silhouette Coefficient (ưu tiên tính với khoảng cách cosine cho embedding)
	- Davies–Bouldin Index (DBI)
- Độ đo đa thể thức:
	- Cross-modal Consensus — so sánh mức độ đồng thuận giữa clustering chỉ trên văn bản và clustering trên dữ liệu tích hợp (ví dụ dùng Adjusted Rand Index hoặc NMI)
- Độ đo định tính:
	- Topic Modeling (LDA) để kiểm tra tính rõ rệt và diễn giải các chủ đề chính trong từng cụm

### Pipeline thực thi (tóm tắt)
- Bước 1 — Chuẩn bị dữ liệu: load embeddings từ `data/embeddings_clean.npy` (memmap khi cần), load metadata/đặc trưng từ `data/features_aggregated.csv` và thực hiện aggregation theo `ticker`/`date` nếu cần.
- Bước 2 — Chạy nhiều thuật toán phân cụm theo cấu hình (k, các siêu tham số) và lưu nhãn cụm (CSV) cùng biểu đồ trực quan hóa (PCA/UMAP).
- Bước 3 — Tính độ đo nội tại (Silhouette, DBI) trên mẫu (sampling nếu dataset lớn) trong không gian gốc (cosine cho embeddings).
- Bước 4 — Nếu có modal numeric, chạy clustering trên modal đó và tính Cross-modal Consensus (ARI/NMI) giữa hai phân cụm.
- Bước 5 — Định tính: chạy LDA trên văn bản của từng cụm, lưu top keywords để đánh giá interpretability.
- Bước 6 — Lưu bảng kết quả `report/clustering_results.csv` và các file nhãn `report/cluster_labels_<algo>_<param>.csv`.
- Gợi ý: với dữ liệu lớn, dùng `MiniBatchKMeans`, indexing ANN (`faiss`) cho truy vấn, và sample ngẫu nhiên (ví dụ 10k) để tính các metric tốn kém.

### Mục đích phân cụm

- Mục tiêu chính: gom nhóm các headline/tin tức tương đồng về nội dung hoặc về tác động tới thị trường để hỗ trợ khám phá chủ đề, lấy mẫu gán nhãn hiệu quả, phát hiện sự kiện/ngoại lệ và hỗ trợ truy xuất thông tin (semantic retrieval).

- Căn cứ từ EDA hiện tại: embeddings văn bản (file `data/embeddings_clean.npy`) là nguồn thông tin chính; các đặc trưng số hiện tại ít hoặc không có biến thiên hữu ích (ví dụ `news_density` = 0 cho hầu hết bản ghi; `sentiment_count` hầu hết = 1). Vì vậy, ưu tiên tiến hành phân cụm trên không gian embedding văn bản và chỉ fuse các đặc trưng số sau khi đã aggregate/chuẩn hóa.

- Chiến lược kỹ thuật ngắn:
	- Dùng text embeddings làm input chính; aggregate `sentiment_mean`/`sentiment_count` theo `ticker`/`date` hoặc theo cửa sổ thời gian trước khi fuse, nếu cần.
	- Dùng `HDBSCAN` làm baseline cho trường hợp mật độ biến thiên; so sánh với `KMeans`/`GMM` theo metric đã chọn.
	- Lưu index ANN (ví dụ `faiss`) để scale cho truy vấn gần đúng.
	- Đánh giá metric trên không gian gốc (cosine) hoặc metric tương ứng, không chỉ dựa trên visual 2D (UMAP/PCA).

- KPI tóm tắt:
	- Coverage (tỉ lệ items được gán cụm) ≥ 90%.
	- Silhouette baseline ≥ 0.15 (tốt ≥ 0.25).
	- Stability (ARI giữa các subsamples) ≥ 0.7.
	- Business: giảm thời gian gán nhãn thủ công ≥ 30% khi dùng cụm để lấy mẫu.

### Mục tiêu phân cụm (cụ thể, đo lường được)
- Silhouette score: tối thiểu ≥ 0.15 (baseline); mục tiêu tốt ≥ 0.25.
- Coverage (tỉ lệ items được gán cụm): ≥ 90%.
- Stability: ARI giữa các subsamples ≥ 0.7.
- Nếu có nhãn chuẩn: ARI/NMI ≥ 0.35 (ngưỡng bảo thủ), hướng tới ≥ 0.5.
- Tác động nghiệp vụ: giảm thời gian gán nhãn thủ công ít nhất 30% khi dùng cụm để lấy mẫu (sampling).

### Mẫu thử nghiệm nhanh
Bạn có thể bắt đầu bằng pipeline đơn giản: trích text-embedding từ [scripts/embeddings.py](scripts/embeddings.py), nối với [data/features_aggregated.csv](data/features_aggregated.csv), rồi chạy [scripts/fuse_and_cluster.py](scripts/fuse_and_cluster.py) (tham số `--k` hoặc cấu hình HDBSCAN).

## EDA — Kết quả nhanh & tệp sinh ra

Tôi đã chạy EDA ở chế độ không tương tác và sinh các báo cáo mẫu trong thư mục `report/`. Tóm tắt chính:

- **Embeddings**: chuyển file raw `data/embeddings.npy` → numeric NumPy `data/embeddings_clean.npy` (shape ≈ 1,845,559 × 768) để xử lý hiệu quả và tránh chạy hết RAM.

- **Báo cáo EDA**: [report/eda_summary.txt](report/eda_summary.txt)
- **Các tệp sinh ra**:
	- [report/eda_summary.txt](report/eda_summary.txt) — tổng quan (text)
	- [report/features_describe.csv](report/features_describe.csv) — thống kê đặc trưng số
	- [report/features_corr.png](report/features_corr.png) — ma trận tương quan
	- [report/top_tokens.csv](report/top_tokens.csv) — token phổ biến (mẫu văn bản)
	- [report/text_length.png](report/text_length.png) — phân phối độ dài headline
	- [report/emb_pca.png](report/emb_pca.png) — PCA(2) trên mẫu embeddings
	- [report/cluster_labels_sample.csv](report/cluster_labels_sample.csv) — nhãn cụm trên mẫu
	- `data/embeddings_clean.npy` — embeddings đã chuyển sang float32 ndarray

- **Kết quả clustering mẫu**: chạy baseline (UMAP/PCA → HDBSCAN) trên mẫu cho silhouette ≈ 0.58 và coverage ≈ 97% (tham khảo `report/cluster_labels_sample.csv`). Lưu ý: silhouette được tính trên không gian 2D giảm chiều, chỉ mang tính tham khảo.

## Cách chạy lại EDA & chuyển embeddings (PowerShell)
1) Kích hoạt virtualenv (nếu cần):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& "C:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/Activate.ps1"
```

2) Cài phụ thuộc (matplotlib, seaborn, umap-learn, hdbscan):
```powershell
& "C:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/python.exe" -m pip install --upgrade pip
& "C:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/python.exe" -m pip install matplotlib seaborn umap-learn hdbscan
```
Nếu `hdbscan` khó cài trên Windows, dùng `conda install -c conda-forge hdbscan umap-learn`.

3) (Nếu cần) chuyển embeddings raw sang `.npy` có header:
```powershell
& "C:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/python.exe" scripts/convert_embeddings.py --infile data/embeddings.npy --outfile data/embeddings_clean.npy
```

4) Chạy EDA ở chế độ không tương tác (sinh `report/`):
```powershell
& "C:/Users/ADMIN/Desktop/Project 2/.venv/Scripts/python.exe" scripts/eda.py
```

## Ghi chú & bước tiếp theo
- Silhouette trên báo cáo là chỉ số tham khảo vì được tính trên không gian giảm chiều; để đánh giá chính xác, chạy metrics trong không gian gốc hoặc dùng nhiều metric (Coverage, Stability/ARI, Davies–Bouldin).
- Đề xuất tiếp theo: (A) grid experiments cho UMAP/HDBSCAN và lưu metrics, (B) text EDA nâng cao (TF‑IDF, n‑grams, NER), (C) merge `news` ↔ `features` theo `date`/`ticker` và xử lý duplicates, (D) theo dõi thử nghiệm (MLflow/DVC) và xây pipeline có thể tái tạo.

