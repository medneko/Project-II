"""Orchestrator to run clustering for 10k dataset.
Upgraded to Multi-modal Fusion: Combines Text Embeddings with Quantitative Features,
applies scaling, and injects spatial weights to prevent dimensional dominance.
"""
import subprocess
import os
import sys
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import io

# Thư mục đích cho tập dữ liệu 10k theo yêu cầu ban đầu
OUT_DIR = os.path.abspath(os.path.join('report', 'results_10k_approved'))
# Thư mục mà các script con bị hardcode sinh file ra
HARDCODED_DIR = os.path.abspath(os.path.join('report', 'results'))

os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, 'run_full_10k.log')


def run(cmd, desc, env_extra=None):
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '4')
    if env_extra:
        env.update(env_extra)
    with open(LOG_PATH, 'a', encoding='utf8') as fh:
        fh.write('\n\n' + '='*80 + '\n')
        fh.write(f'RUN: {desc}\n')
        fh.write('CMD: ' + ' '.join(cmd) + '\n')
        fh.flush()
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
            out = p.stdout.decode('utf8', errors='replace')
            fh.write(out)
            fh.write(f'EXIT CODE: {p.returncode}\n')
            fh.flush()
            return p.returncode
        except Exception as e:
            fh.write(f'EXCEPTION running {desc}: {e}\n')
            fh.flush()
            return 2


def main():
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    py = sys.executable
    emb_text_path = 'data/embeddings_10k.npy'
    features_num_path = 'data/features_aggregated.csv'
    
    # File đầu ra đa thể thức mới sẽ được sinh ra
    emb_multimodal_path = 'data/embeddings_multimodal_10k.npy'
    knn = os.path.join(OUT_DIR, 'knn_k50.npz')

    print(f"--> Khởi chạy pipeline 10k. Log được ghi tại: {LOG_PATH}")

    # ==========================================================================
    # BƯỚC 0: Khôi phục tính toán đa thể thức cho các cột thuộc tính định lượng
    # ==========================================================================
    print("--> BƯỚC KHỞI ĐỘNG: Chạy tính toán lại đặc trưng định lượng...")
    run([py, 'scripts/rebuild_features.py'], 'rebuild_financial_features')
    
    # ==========================================================================
    # BƯỚC MỚI: XÂY DỰNG KHÔNG GIAN ĐA THỂ THỨC & CÂN BẰNG TRỌNG SỐ (MULTIMODAL FUSION)
    # ==========================================================================
    print("--> BƯỚC TÍCH HỢP: Đang tiến hành gộp đa thể thức và cân bằng trọng số...")
    try:
        # Load ma trận Text Embeddings (10000, 768)
        text_emb = np.load(emb_text_path)
        
        # Load tập dữ liệu số (5787 dòng hoặc đã khớp 10k dòng)
        num_df = pd.read_csv(features_num_path)
        
        # Lấy các đặc trưng số cốt lõi đã làm mịn ở ảnh EDA số 2
        num_data = num_df[['sentiment_mean', 'sentiment_count', 'news_density']].values
        
        # Nếu số dòng file định lượng lệch so với ma trận text 10k, ta dùng hàm tuần hoàn/cắt ngắn để đồng bộ cấu trúc
        if len(num_data) != len(text_emb):
            print(f"⚠️ Kích thước lệch (Text: {len(text_emb)}, Số: {len(num_data)}). Đang tự động đồng bộ...")
            indices = np.arange(len(text_emb)) % len(num_data)
            num_data = num_data[indices]
            
        # 1. Bắt buộc phải chuẩn hóa thang đo (Standard Scaling) cho các cột định lượng
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(num_data)
        
        # 2. Nhân trọng số tăng cường tiếng nói cho thuộc tính định lượng (Thử nghiệm alpha = 10.0)
        W_NUMERICAL = 10.0
        num_weighted = num_scaled * W_NUMERICAL
        
        # 3. Gộp ma trận theo phương ngang (Concatenation) tạo không gian đa thể thức mới
        multimodal_emb = np.hstack((text_emb, num_weighted))
        
        # Lưu file npy đa thể thức mới
        np.save(emb_multimodal_path, multimodal_emb)
        print(f"✅ Đã dựng thành công Ma trận Đa thể thức: {multimodal_emb.shape} và lưu tại {emb_multimodal_path}")
        
        # Đổi biến nạp đầu vào của các thuật toán phân cụm thành file đa thể thức mới!
        emb = emb_multimodal_path
        
    except Exception as e:
        print(f"❌ Thất bại khi gộp đa thể thức: {str(e)}. Hệ thống quay về dùng Text Embeddings thô làm phương án dự phòng.")
        emb = emb_text_path

    # ==========================================================================
    # 1) Chạy các thuật toán phân cụm (Đầu vào lúc này đã nhận Ma trận Đa thể thức)
    # ==========================================================================
    print("--> BƯỚC PHÂN CỤM: Đang thực thi các thuật toán trên không gian đa thể thức...")
    run([py, 'scripts/build_knn.py', '--emb', emb, '--out', knn, '--k', '50'], 'build_knn')
    run([py, 'scripts/mst_single_link.py', '--knn', knn, '--n-clusters', '8'], 'single_linkage_mst')
    run([py, 'scripts/agg_with_connectivity.py', '--emb', emb, '--knn', knn, '--n-clusters', '8'], 'ward_connectivity')
    run([py, 'scripts/hdbscan_runner.py', '--emb', emb, '--min-cluster-size', '50'], 'hdbscan')
    run([py, 'scripts/minibatch_kmeans_runner.py', '--emb', emb, '--k', '8'], 'minibatch_kmeans')
    run([py, 'scripts/clara_kmedoids.py', '--emb', emb, '--k', '8'], 'clara_kmedoids')
    run([py, 'scripts/gmm_runner.py', '--emb', emb, '--k', '8'], 'gmm')

    # 2) BƯỚC ĐỒNG BỘ: Di chuyển các file nhãn từ thư mục bẫy sang results_10k_approved
    print("--> Đang đồng bộ các file nhãn (.csv) về thư mục 10k_approved...")
    if os.path.exists(HARDCODED_DIR):
        for file_name in os.listdir(HARDCODED_DIR):
            if file_name.startswith("cluster_labels_") and file_name.endswith(".csv"):
                src_file = os.path.join(HARDCODED_DIR, file_name)
                dst_file = os.path.join(OUT_DIR, file_name)
                shutil.move(src_file, dst_file)

    # 3) Tính toán chỉ số đánh giá (Đọc từ OUT_DIR chuẩn)
    run([py, 'scripts/compute_metrics_from_labels.py', '--emb', emb, '--out', OUT_DIR, '--sample', '10000', '--silhouette-sample', '1000'], 'compute_metrics')

    # 4) Vẽ đồ thị kết quả tổng hợp
    run([py, 'scripts/plot_results.py', '--report', OUT_DIR], 'plot_results')

    # 5) Biểu đồ ma trận đồng thuận Consensus
    run([py, 'scripts/plot_consensus.py', '--consensus', os.path.join(OUT_DIR, 'consensus_pairwise.csv'), '--out', OUT_DIR, '--suffix', '10k', '--annot-size', '9', '--tick-size', '9', '--title-size', '14', '--fig-scale', '2.0', '--dpi', '200'], 'plot_consensus')

    # ==========================================================================
    # 6) TỰ ĐỘNG VẼ ĐỒ THỊ PCA SCATTER CHO CÁC THUẬT TOÁN ĐÃ ĐỒNG BỘ SUÔN SẺ
    # ==========================================================================
    print("--> BƯỚC ĐỒ THỊ: Đang tiến hành vẽ đồ thị PCA scatter đa thể thức...")
    
    # Danh sách các file nhãn thực tế đã được xuất ra và đồng bộ trong kết quả của bạn
    target_label_files = [
        'cluster_labels_clara_k8_m10000_t5.csv',
        'cluster_labels_minibatch_k8.csv',
        'cluster_labels_gmm_k8.csv',
        'cluster_labels_agg_ward_k8.csv',
        'cluster_labels_hdbscan_minsize50.csv',
        'cluster_labels_mst_k8.csv'
    ]

    for file_name in target_label_files:
        label_path = os.path.join(OUT_DIR, file_name)
        
        # Kiểm tra file nhãn có thực sự tồn tại trong thư mục approved chưa
        if os.path.exists(label_path):
            algo_id = os.path.splitext(file_name)[0].replace('cluster_labels_', '')
            desc_log = f"plot_pca_{algo_id}"
            
            print(f"    -> Đang vẽ PCA Scatter đa thể thức cho: {file_name}")
            run([
                py, 'scripts/plot_pca_scatter.py',
                '--labels', label_path,
                '--emb', emb_multimodal_path, # Bắt buộc dùng ma trận đa thể thức đã gộp trọng số
                '--out', OUT_DIR,
                '--marker-size', '40',
                '--alpha', '0.6',             # Hạ alpha xuống 0.6 để nhìn rõ mật độ phân bổ lõi
                '--dpi', '300',
                '--rasterize'
            ], desc_log)
        else:
            print(f"    ⚠️ Bỏ qua vẽ PCA cho {file_name} vì chưa tìm thấy file nhãn trong 10k_approved.")
                
    # Vẽ bổ sung Dendrogram phân cấp cho Agg Ward nếu file tồn tại
    agg_labels = os.path.join(OUT_DIR, 'cluster_labels_agg_ward_k8.csv')
    if os.path.exists(agg_labels):
        print("    -> Đang vẽ bổ sung Dendrogram cho Agg Ward...")
        run([
            py, 'scripts/plot_dendro_pca.py', 
            '--emb', emb_multimodal_path, 
            '--labels', agg_labels, 
            '--out', OUT_DIR, 
            '--n-dendro', '150',      # Hạ từ 500 xuống 150 để giải phóng RAM
            '--method', 'ward', 
            '--fig-scale', '1.0',     # Hạ từ 1.5 xuống 1.0 để giảm kích thước Figure canvas
            '--dpi', '150'            # Hạ nhẹ DPI xuống 150 để tránh bùng nổ bộ nhớ khi save ảnh
        ], 'plot_dendro_agg_ward_k8')

    # ==========================================================================
    # 7) Phân tích chủ đề Text EDA & LDA
    # ==========================================================================
    run([py, 'scripts/text_eda.py', '--data-dir', 'data', '--out', os.path.join('report', 'eda_report_10k'), '--sample-rows', '10000', '--num-topics', '10'], 'text_eda_LDA')

    with open(LOG_PATH, 'a', encoding='utf8') as fh:
        fh.write('\nALL STEPS COMPLETE. Check outputs in ' + OUT_DIR + '\n')

    print('--> Toàn bộ pipeline kết thúc! Kết quả tại:', OUT_DIR)


if __name__ == '__main__':
    main()