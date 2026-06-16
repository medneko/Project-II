**Các độ đo được sử dụng trong Project**

---

**Ghi chú an toàn khi tính metric**

- Nhãn noise `-1` không được tính là một cụm. Coverage được tính bằng `count(label != -1) / tổng số nhãn`.
- Silhouette và DBI chỉ được tính trên các điểm đã được gán cụm (`label != -1`). Nếu còn dưới 2 cụm hợp lệ, metric trả `NaN` và cần đọc cột `notes`.
- Với HDBSCAN, silhouette cao nhưng coverage thấp không đủ để kết luận thuật toán tốt; luôn đọc kèm coverage/noise ratio.
- Label CSV mới nên dùng schema `row_index,label`; sample labels bắt buộc phải có `row_index` thật trong embedding gốc để tránh bị tính như full labels.

**1. Silhouette**

- Ký hiệu: với một điểm mẫu $i$ thuộc cụm $A$.
- $a(i)$: khoảng cách trung bình giữa $i$ và tất cả điểm khác trong cùng cụm $A$ (intra-cluster distance).
- $b(i)$: giá trị nhỏ nhất của khoảng cách trung bình giữa $i$ và các điểm trong cụm $C \neq A$ (tức là cụm gần nhất kế tiếp).

Độ đo silhouette cho điểm $i$ được định nghĩa bằng:
$$
s(i)=\frac{b(i)-a(i)}{\max\{a(i),\,b(i)\}}.
$$

- Phạm vi: $s(i)\in[-1,1]$. Giá trị gần $1$ cho thấy điểm được phân vào cụm phù hợp; gần $-1$ nghĩa là điểm có thể bị phân nhầm cụm; gần $0$ nghĩa là điểm nằm giữa hai cụm.
- Silhouette score của toàn bộ phân cụm là trung bình của $s(i)$ trên mọi điểm:
$$
S=\frac{1}{n}\sum_{i=1}^n s(i).
$$

**2. Davies–Bouldin Index (DBI)**

- Ký hiệu: có $k$ cụm. Với cụm $i$:
  - $S_i$: một thước đo độ phân tán của cụm $i$ (thường là khoảng cách trung bình giữa các điểm trong cụm và centroid của cụm):
  $$S_i=\frac{1}{|C_i|}\sum_{x\in C_i} d(x,\mu_i),$$
  trong đó $\mu_i$ là centroid của $C_i$ và $d(\cdot,\cdot)$ là hàm khoảng cách (ví dụ Euclid).
  - $M_{ij}$: khoảng cách giữa centroid của cụm $i$ và $j$: $M_{ij}=d(\mu_i,\mu_j)$.

Định nghĩa tỉ lệ tương tự giữa hai cụm:
$$
R_{ij}=\frac{S_i+S_j}{M_{ij}}.
$$

Với mỗi cụm $i$ ta lấy $D_i=\max_{j\neq i} R_{ij}$. Sau đó DBI được tính là trung bình của các $D_i$:
$$
\text{DBI}=\frac{1}{k}\sum_{i=1}^k D_i.
$$

- Diễn giải: DBI nhỏ hơn là tốt hơn (các cụm tách biệt rõ và có phân tán nhỏ). DBI phụ thuộc vào định nghĩa $S_i$ và hàm khoảng cách.

**3. Các chỉ số đối chiếu (Consensus / comparing clusterings)**

Ta xét hai phân cụm (hai gán nhãn) $U=\{U_1,\dots,U_{k_u}\}$ và $V=\{V_1,\dots,V_{k_v}\}$ trên cùng tập $n$ điểm.

- Notation: xây dựng bảng tương quan (contingency table) với $n_{ij}=|U_i\cap V_j|$,
  $a_i=\sum_j n_{ij}=|U_i|$, $b_j=\sum_i n_{ij}=|V_j|$ và tổng $n=\sum_{i,j} n_{ij}$.

3.1 Adjusted Rand Index (ARI)

- Rand Index (RI) đo mức độ đồng thuận dựa trên số cặp điểm được gán giống/khác nhau trong hai phân cụm. ARI điều chỉnh RI theo kỳ vọng ngẫu nhiên.

Sử dụng ký hiệu tổ hợp $\binom{t}{2}=t(t-1)/2$, ARI được tính bằng:
$$
\text{ARI}=\frac{\sum_{ij}\binom{n_{ij}}{2}-\left[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\right]/\binom{n}{2}}{\tfrac{1}{2}\left[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}\right]-\left[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\right]/\binom{n}{2}}.
$$

- Phạm vi: ARI thường nằm trong khoảng $[-1,1]$, với $0$ tương ứng giá trị trùng khớp như mong đợi ngẫu nhiên (tùy theo implementation), và $1$ là trùng khớp hoàn toàn.

3.2 Normalized Mutual Information (NMI)

- Xác suất rời rạc: $P(i)=a_i/n$, $Q(j)=b_j/n$, $P(i,j)=n_{ij}/n$.
- Entropy của phân cụm $U$:
$$
H(U)=-\sum_i P(i)\log P(i).
$$
- Nhận thức thông tin chung (mutual information) giữa $U$ và $V$:
$$
I(U;V)=\sum_{i,j}P(i,j)\log\frac{P(i,j)}{P(i)Q(j)}.
$$
- Một dạng chuẩn hoá phổ biến là:
$$
\text{NMI}=\frac{I(U;V)}{\sqrt{H(U)H(V)}}.
$$

- Phạm vi: NMI nằm trong $[0,1]$ (giá trị cao = hai phân cụm chứa nhiều thông tin chung). Lưu ý có nhiều phép chuẩn hoá khác (ví dụ trung bình điều hoà của entropies), nên khi so sánh kết quả cần biết biến thể NMI được dùng.

3.3 Overlap (giao/overlap giữa các cụm)

- "Overlap" không có một định nghĩa duy nhất trong ngữ cảnh so sánh phân cụm; dưới đây là hai cách thường dùng:

  a) Overlap theo Jaccard giữa hai cụm cụ thể $U_i$ và $V_j$:
  $$
  J(U_i,V_j)=\frac{|U_i\cap V_j|}{|U_i\cup V_j|}.
  $$

  b) Overlap trung bình theo phép ghép (matching): với mỗi cụm $U_i$ tìm $\max_j J(U_i,V_j)$ rồi lấy trung bình:
  $$
  \text{Overlap}(U,V)=\frac{1}{k_u}\sum_{i=1}^{k_u}\max_j \frac{|U_i\cap V_j|}{|U_i\cup V_j|}.
  $$

- Một biến thể khác là tỷ lệ phần trăm điểm được gán cùng nhau theo phép ghép tối ưu (maximum matching), hoặc tính trung bình theo tỷ lệ giao cắt so với kích thước cụm gốc:
  $$
  \text{RecallOverlap}(U,V)=\frac{1}{k_u}\sum_i \max_j \frac{|U_i\cap V_j|}{|U_i|}.
  $$

- Diễn giải: giá trị càng gần $1$ cho thấy các cụm trong $U$ tương ứng tốt với các cụm trong $V$ (ít tách/ghép khác nhau). Khi so sánh, cần ghi rõ biến thể dùng (Jaccard, recall-based, hay theo matching tối ưu).
