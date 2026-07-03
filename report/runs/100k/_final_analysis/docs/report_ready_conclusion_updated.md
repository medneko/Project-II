# Kết luận cập nhật sẵn cho báo cáo

Sau các bước representation tuning, k-sweep, kiểm chứng CLARA true và validation riêng cho HDBSCAN, kết quả cho thấy embedding văn bản gốc vẫn là baseline ổn định nhất cho vai trò conservative full-coverage. Mô hình `text_768_original + minibatch_k16` được giữ làm mốc so sánh chính vì bao phủ toàn bộ 100k quan sát và không phụ thuộc vào đặc trưng phụ.

Với nhóm mô hình full-coverage thực nghiệm, đặc trưng lexical multi-feature giúp cải thiện MiniBatch khi chọn số cụm hợp lý. Mô hình `text_pca64_lexical + minibatch_k40` được chọn làm experimental full-coverage model vì cân bằng tốt giữa silhouette, DBI, phân bố kích thước cụm, rủi ro fragmentation và khả năng diễn giải. Ngược lại, `text_pca64_only + minibatch_k96` không được chọn dù có một số chỉ số tách cụm tốt, vì k quá lớn dễ tạo cụm nhỏ, làm tăng fragmentation và rủi ro cụm bị chi phối bởi metadata/publisher.

Representation tuning cho thấy L2-normalization có thể cải thiện một phần stability và DBI, đặc biệt với `text_768_l2 + minibatch_k40`, nhưng mức silhouette thấp hơn và tỷ lệ negative silhouette cao khiến hướng này chưa đủ để thay thế mô hình experimental đã chọn.

Đối với dense/event detection, mô hình được cập nhật chính thức sang `text_pca64_only + HDBSCAN(min_cluster_size=30, min_samples=20, leaf)`. Mô hình này chạy được trên full 100k, đạt coverage 20.119%, 219 cụm, silhouette cosine 0.608580, negative silhouette 0.69%, không có collapse warning và không quá sparse. Đây không phải mô hình full-coverage, mà là bộ phát hiện các cụm sự kiện dày và rõ hơn.

CLARA true đã được kiểm chứng bằng `sklearn_extra.cluster.CLARA` và không dùng fallback. Kết quả tốt nhất là `text_pca64_lexical_calendar + clara_k16`, nhưng silhouette khoảng 0.033399 và DBI khoảng 4.301401 cho thấy CLARA chỉ nên giữ vai trò diagnostic baseline, yếu hơn MiniBatch cho kết luận chính.

Các hạn chế cần nêu rõ gồm: stability chưa quá cao, một số cụm có publisher dominance, lưới HDBSCAN 768D bị timeout nên kết luận event model chỉ áp dụng cho validated PCA64 path, và thí nghiệm chưa có nhãn market-response hoặc multimodal thực sự để đánh giá tác động thị trường.
