# Algorithm Family Comments

## MiniBatchKMeans

MiniBatchKMeans là nhóm ổn định nhất cho bài toán full-coverage: mọi điểm dữ liệu đều được gán cụm, thời gian chạy phù hợp với quy mô 100k, và các chỉ số cân bằng cụm có thể kiểm soát bằng tham số `k`. Baseline `text_768_original + minibatch_k16` thận trọng vì số cụm vừa phải và ít rủi ro phân mảnh, trong khi `text_pca64_lexical + minibatch_k40` là lựa chọn thực nghiệm tốt hơn khi cần cụm chi tiết hơn nhưng vẫn còn diễn giải được.

Điểm yếu chính của MiniBatchKMeans là phải chọn trước số cụm. Các cấu hình high-k như `text_pca64_only + minibatch_k96` có silhouette/DBI hấp dẫn hơn nhưng tạo quá nhiều cụm nhỏ, nhiều cảnh báo metadata hơn, và dễ làm kết quả khó trình bày trong báo cáo.

## GMM

GMM hữu ích như một baseline xác suất vì cung cấp cách nhìn khác với centroid cứng của KMeans. Trên dữ liệu này, GMM giữ coverage 100% và có DBI cạnh tranh ở một số cấu hình, nhưng silhouette thấp hơn hoặc không vượt rõ MiniBatchKMeans.

Vì vậy GMM nên được giữ làm mô hình tham chiếu, đặc biệt để chứng minh rằng lựa chọn final không chỉ dựa trên một họ thuật toán. Nó không được chọn làm mô hình chính vì không cải thiện đủ rõ chất lượng cụm và tính diễn giải so với MiniBatch.

## HDBSCAN

HDBSCAN phục vụ vai trò khác: phát hiện các cụm dày đặc hoặc nhóm sự kiện, không nhằm coverage 100%. Do đó coverage thấp không phải lỗi nếu các điểm được giữ lại tạo cụm có silhouette cao, negative silhouette thấp và không bị collapse.

Model `text_pca64_only + HDBSCAN(min_cluster_size=30, min_samples=20, leaf)` được chọn cho event detection vì giữ được cụm dày đặc, số cụm phong phú hơn baseline cũ, silhouette cao, và negative silhouette thấp trên kiểm định full 100k. Caveat là một số nhánh 768D HDBSCAN bị timeout, nên kết luận final dựa trên đường PCA64 đã validated.

## CLARA

CLARA được dùng như diagnostic baseline cho phương pháp medoid. Việc chạy CLARA true giúp loại bỏ nghi ngờ rằng kết quả trước đó chỉ là fallback, nhưng các metric vẫn yếu hơn MiniBatchKMeans và GMM trong các cấu hình chính.

Vì chạy trên sample và silhouette/DBI không nổi bật, CLARA không phù hợp làm mô hình chính trong project này. Giá trị của nó nằm ở vai trò đối chứng để cho thấy lựa chọn MiniBatch không phụ thuộc vào một baseline quá yếu.

## Notable Models

| Model | Vai trò | Ưu điểm | Hạn chế | Kết luận |
| --- | --- | --- | --- | --- |
| `text_768_original + minibatch_k16` | Conservative baseline | Coverage 100%, số cụm vừa phải, ổn định | Ít chi tiết hơn k cao | Baseline chính để so sánh thận trọng |
| `text_pca64_lexical + minibatch_k40` | Selected experimental full-coverage | Cân bằng silhouette, DBI, kích thước cụm và diễn giải | Publisher warning cao hơn baseline | Model full-coverage thực nghiệm được chọn |
| `text_pca64_only + minibatch_k96` | Risky reference | Silhouette/DBI mạnh | Phân mảnh, nhiều cụm nhỏ và nhiều warning | Chỉ dùng tham chiếu high-k |
| `text_768_original + gmm_k8` | Probabilistic baseline | Gọn, coverage 100%, hữu ích đối chứng | Không vượt rõ MiniBatch | Giữ làm baseline xác suất |
| `text_pca64_only + HDBSCAN(mcs=30, ms=20, leaf)` | Selected event model | Silhouette cao, negative silhouette thấp, cụm dày đặc | Không full-coverage | Model event/dense detection chính |
| Best CLARA true | Diagnostic baseline | Chạy CLARA thật, có giá trị đối chứng | Sample-level, metric yếu hơn | Không chọn làm final |
