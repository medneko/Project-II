# Kết luận sẵn sàng đưa vào báo cáo

Kết quả thực nghiệm cho thấy biểu diễn text-only embedding là lựa chọn ổn định và thận trọng nhất để làm baseline chính. Mô hình `text-only minibatch_k16` giữ coverage 100%, metric ổn định và dễ so sánh với các thuật toán còn lại.

Nhánh multi-feature không nên bị diễn giải là luôn kém hơn. Ablation cho thấy phần mở rộng nhẹ bằng đặc trưng lexical có thể cải thiện MiniBatch ở cấu hình k=32. Cụ thể, `text_pca64_lexical + minibatch_k32` là mô hình thực nghiệm full-coverage tốt nhất trong nhánh bounded ablation.

Ngược lại, fusion metadata nặng với publisher/stock hoặc all-aux weight lớn có thể làm giảm chất lượng GMM/HDBSCAN hoặc làm cấu trúc cụm bị collapse. Vì vậy, metadata nên được dùng cẩn trọng như tín hiệu bổ trợ và công cụ giải thích hậu nghiệm.

CLARA true đã được kiểm thử bằng `sklearn_extra.cluster.CLARA`, không còn là fallback. Tuy nhiên, CLARA true vẫn yếu hơn MiniBatch và chỉ nên xem là diagnostic baseline.

HDBSCAN phù hợp hơn cho phát hiện cụm dày đặc hoặc cụm kiểu sự kiện, không phải mô hình full-coverage chính.

Khuyến nghị cuối cùng:

- Dùng `text-only minibatch_k16` làm conservative baseline.
- Dùng `text_pca64_lexical + minibatch_k32` làm best experimental full-coverage model nếu phần kiểm tra diễn giải cụm đạt yêu cầu.
- Dùng `text-only hdbscan_minsize50` cho dense/event detection.
