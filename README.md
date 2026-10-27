# Phát hiện Clickbait Tiếng Việt (Vietnamese Clickbait Detection)

Hệ thống phân loại văn bản nhị phân (**Binary Text Classification**) phát hiện bài báo tiếng Việt giật tít, câu view (`clickbait`) so với tin tức chính thống (`non-clickbait`), khai thác đặc trưng ngôn ngữ từ tiêu đề (`title`) và đoạn mở đầu/sapo (`lead_paragraph`).

---

## 1. Cấu trúc Thư mục Dự án

```text
vi_clickbait_model/
│
├── data/
│   ├── raw/                        # Chứa dữ liệu gốc clickbait_dataset_vietnamese.csv
│   └── processed/                  # Chứa train.csv, val.csv, test.csv, full_processed.csv
├── notebooks/
│   ├── 01_EDA.ipynb                # Khám phá và trực quan hóa dữ liệu (EDA)
│   ├── 02_Preprocessing.ipynb      # Thực nghiệm tiền xử lý dữ liệu đặc thù tiếng Việt
│   └── 03_Error_Analysis.ipynb     # So sánh mô hình, Feature Importance & Phân tích lỗi PhoBERT
├── src/
│   ├── preprocess.py               # Làm sạch nhiễu đặc thù (VnExpress, Báo Mới), chuẩn hóa & tách từ PyVi
│   ├── models_ml.py                # Huấn luyện TF-IDF + Naive Bayes, Logistic Regression, Linear SVM, RF, XGBoost
│   ├── models_dl.py                # Huấn luyện mô hình học sâu BiLSTM & TextCNN (1D-CNN)
│   ├── models_bert.py              # Fine-tuning PhoBERT-base-v2 (Only Title & Title + Lead) + Error Analysis
│   └── evaluate.py                 # Tính toán metrics và xuất toàn bộ biểu đồ EDA / Đánh giá
├── outputs/
│   ├── figures/                    # Lưu trữ 9 bộ biểu đồ phân tích chất lượng cao (.png)
│   └── results/                    # Lưu bảng điểm (CSV/Markdown/JSON) và model checkpoints
├── run_pipeline.py                 # Script chạy tự động toàn bộ pipeline từ A đến Z
├── requirements.txt                # Danh sách thư viện phụ thuộc
├── FINAL_REPORT.md                 # Báo cáo bài tập lớn hoàn chỉnh kèm số liệu thực nghiệm
└── README.md                       # Tài liệu hướng dẫn cài đặt và sử dụng
```

---

## 2. Hướng dẫn Cài đặt & Chạy Thực nghiệm

### Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Chạy toàn bộ Pipeline từ A đến Z
```bash
python run_pipeline.py
```

Hoặc chạy riêng bước tiền xử lý dữ liệu:
```bash
python -m src.preprocess
```

---

## 3. Điểm nhấn Kỹ thuật trong Pipeline

1. **Khử nhiễu dữ liệu thực tế (Data-specific Noise Removal)**:
   - Tách địa danh bị dính liền vào từ đầu câu trong `lead_paragraph` của **VnExpress** (`165 mẫu`, ví dụ: `"Nghệ AnCảng hàng không..."` $\rightarrow$ `"Nghệ An Cảng hàng không..."`).
   - Cắt bỏ hậu tố tên báo ở cuối `title` của **Báo Mới** (`716 mẫu`, ví dụ: `"... - Tạp chí SaoStar"`).
2. **Chống Rò rỉ Dữ liệu & Học vẹt (Anti-Shortcut Learning)**:
   - Loại bỏ hoàn toàn cột `source` khỏi tập đặc trưng huấn luyện để buộc mô hình học ngữ nghĩa văn bản thay vì học vẹt tên báo.
3. **So sánh Đa chiều (Ablation Study)**:
   - So sánh 3 thế hệ mô hình: **Học máy truyền thống (`TF-IDF + NB / LR / Linear SVM / RF / XGBoost`)** vs **Học sâu (`BiLSTM`, `TextCNN`)** vs **Pre-trained Transformer (`vinai/phobert-base-v2`)**.
   - So sánh 2 phương án đầu vào: **Option A (`Only Title`)** vs **Option B (`Title + [SEP] + Lead Paragraph`)**.
