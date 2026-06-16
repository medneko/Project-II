### Pipeline thực thi
- Bước 1 — Chuẩn bị dữ liệu: load embeddings từ `data/embeddings_clean.npy` (memmap khi cần), load metadata/đặc trưng từ `data/features_aggregated.csv` và thực hiện aggregation theo `ticker`/`date` nếu cần.
- Lưu ý alignment: chỉ fusion numeric features khi có row/key alignment rõ ràng giữa embedding và feature table. Không dùng modulo repeat, `np.tile`, hoặc repeat tuần hoàn để ép shape; nếu không xác thực được alignment thì chạy text-only.
- Bước 2 — Chạy nhiều thuật toán phân cụm theo cấu hình (k, các siêu tham số) và lưu nhãn cụm (CSV) cùng biểu đồ trực quan hóa (PCA/UMAP).
- Bước 3 — Tính độ đo nội tại (Silhouette, DBI) trên mẫu (sampling nếu dataset lớn) trong không gian gốc (cosine cho embeddings).
- Bước 4 — Nếu có modal numeric, chạy clustering trên modal đó và tính Cross-modal Consensus (ARI/NMI) giữa hai phân cụm.
- Bước 5 — Định tính: chạy LDA trên văn bản của từng cụm, lưu top keywords để đánh giá interpretability.
- Bước 6 — Lưu bảng kết quả `report/clustering_results.csv` và các file nhãn `report/cluster_labels_<algo>_<param>.csv`.

### Tóm tắt kết quả EDA

Dữ liệu là đa thể thức, gồm 2 phần:
- Daily financial news: Gồm 1.84M dòng. Qua mô hình FinBERT, văn bản được chuyển thành vector (1.84M, 768). Nó đại diện cho nội dung, chủ đề, sự kiện xảy ra (ví dụ: tin về báo cáo tài chính, tin CEO từ chức, tin sáp nhập).

- Stock-Market Sentiment: Gồm 5787 dòng, có các chỉ số liên tục như sentiment_mean (tâm lý thị trường tích cực hay tiêu cực) và các chỉ số tần suất như news_density, sentiment_count (sự chú ý của truyền thông lớn hay nhỏ).

Mối quan hệ: Tin tức sinh ra cảm xúc, và tần suất tin tức thể hiện mức độ biến động của cổ phiếu. 

Ví dụ: Một chuỗi tin tức dồn dập có embedding thuộc chủ đề "khủng hoảng nợ" đi kèm với sentiment_mean = -1.0 và news_density cao sẽ định hình rõ ràng một cụm gọi là "Cụm cổ phiếu đang khủng hoảng truyền thông tiêu cực".

Vấn đề: Liệt hoàn toàn biến thiên ở dữ liệu Sentiment.
