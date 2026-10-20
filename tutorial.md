Chào bạn, tôi đã đọc và phân tích cấu trúc cũng như nội dung của file `clickbait_dataset_vietnamese.csv` bạn cung cấp. 

Dưới đây là một số nhận xét quan trọng từ việc khám phá dữ liệu sơ bộ (EDA) mà tôi đã tích hợp thẳng vào hướng dẫn cho Agent để bài tập lớn của bạn đạt điểm cao nhất:
1. **Đặc trưng dữ liệu**: Dữ liệu gồm khoảng ~1,300+ mẫu tin tức từ các báo lớn (VnExpress, Tuổi Trẻ, Thanh Niên, VietNamNet, SaoStar, Kênh14, 24h, Báo Mới).
2. **Vấn đề Nhiễu (Noise)**: Trong cột `lead_paragraph` của VnExpress bị dính tên địa danh vào từ đầu tiên (Ví dụ: *"Nghệ AnCảng hàng không..."*, *"TP HCMNgôi nhà..."*). Ở Báo Mới, cột `title` thường có đuôi tên báo (Ví dụ: *"... - Tạp chí SaoStar"*).
3. **Rò rỉ dữ liệu (Data Leakage)**: Nguồn báo (`source`) có tương quan cực mạnh với nhãn (Kênh14, SaoStar gần như 100% là clickbait; VnExpress đa số là non-clickbait). Trong môn NLP, ta phải ép mô hình học từ **ngữ nghĩa văn bản** (`title` + `lead_paragraph`) thay vì học vẹt tên báo.

Dưới đây là toàn bộ nội dung file `NLP_CLICKBAIT_PIPELINE_INSTRUCTIONS.md`. Bạn có thể copy toàn bộ khối code bên dưới và đưa cho AI Agent (như Cursor, Devin, AutoGPT, hoặc ChatGPT/Claude) để nó tự động viết code và chạy thực nghiệm cho bạn.

---

```markdown
# HƯỚNG DẪN THỰC HIỆN BÀI TẬP LỚN NLP: PHÁT HIỆN CLICKBAIT TIẾNG VIỆT
**Tài liệu hướng dẫn chi tiết dành cho AI Agent (Coding & Experiment Execution)**

## 1. TỔNG QUAN BÀI TOÁN & DỮ LIỆU
- **Bài toán**: Phân loại văn bản nhị phân (Binary Text Classification) - Dự đoán một bài báo tiếng Việt là `clickbait` (giật tít, câu view) hay `non-clickbait` (chính thống).
- **File dữ liệu đầu vào**: `clickbait_dataset_vietnamese.csv`
- **Các trường dữ liệu**: `id`, `url`, `title`, `lead_paragraph`, `category`, `publish_datetime`, `source`, `thumbnail_url`, `label`.
- **Trọng tâm môn học (NLP)**: Tập trung khai thác đặc trưng ngôn ngữ từ `title` (tiêu đề) và `lead_paragraph` (đoạn sapo/mở đầu).

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN YÊU CẦU
Agent cần khởi tạo cấu trúc thư mục như sau để đảm bảo tính module hóa cho báo cáo:
```text
vietnamese_clickbait_detection/
│
├── data/
│   ├── raw/                  # Chứa clickbait_dataset_vietnamese.csv
│   └── processed/            # Chứa train.csv, val.csv, test.csv sau tiền xử lý
├── notebooks/
│   ├── 01_EDA.ipynb          # Khám phá và trực quan hóa dữ liệu
│   ├── 02_Preprocessing.ipynb# Thực nghiệm tiền xử lý
│   └── 03_Error_Analysis.ipynb # Phân tích lỗi mô hình tốt nhất
├── src/
│   ├── preprocess.py         # Hàm làm sạch và tách từ tiếng Việt
│   ├── models_ml.py          # Huấn luyện TF-IDF + Naive Bayes, SVM, Logistic Regression
│   ├── models_dl.py          # Huấn luyện BiLSTM / TextCNN
│   ├── models_bert.py        # Fine-tuning PhoBERT / ViSoBERT
│   └── evaluate.py           # Hàm tính toán метрик và vẽ biểu đồ so sánh
├── outputs/
│   ├── figures/              # Lưu ảnh biểu đồ phục vụ báo cáo
│   └── results/              # Lưu bảng điểm (CSV/JSON) và model checkpoints
├── requirements.txt
└── README.md
```

---

## 3. PIPELINE CHI TIẾT TỪ A ĐẾN Z (AGENT CẦN THỰC HIỆN TUẦN TỰ)

### BƯỚC 1: KHÁM PHÁ DỮ LIỆU (EDA) & PHÁT HIỆN DỊ THƯỜNG
Agent cần viết script/notebook sinh ra các biểu đồ lưu vào `outputs/figures/`:
1. **Phân phối nhãn (`label`)**: Vẽ biểu đồ cột/tròn kiểm tra sự mất cân bằng dữ liệu (Imbalanced Data) giữa `clickbait` và `non-clickbait`.
2. **Phân phối theo Nguồn báo (`source`) & Chuyên mục (`category`)**: Vẽ Stacked Bar Chart. 
   - *Lưu ý quan trọng để viết vào báo cáo*: Chỉ ra hiện tượng thiên lệch nguồn (Ví dụ: Kênh14, SaoStar đa số là clickbait; VnExpress, Tuổi Trẻ đa số là non-clickbait). Giải thích tại sao **không đưa cột `source` vào làm feature huấn luyện** để tránh hiện tượng *Data Leakage / Shortcut Learning* (mô hình chỉ nhìn tên báo để đoán mà không học ngữ nghĩa câu).
3. **Thống kê độ dài văn bản**: Vẽ Histogram độ dài (số từ, số ký tự) của `title` và `lead_paragraph` theo 2 nhãn.
4. **Phân tích dấu câu & từ khóa đặc trưng**: Thống kê tỷ lệ xuất hiện dấu chấm hỏi (`?`), dấu chấm than (`!`), dấu ngoặc kép (`""`, `''`), viết hoa toàn bộ (ALL CAPS), và con số trong tiêu đề ở hai nhãn. Vẽ WordCloud cho từng nhãn.

---

### BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU ĐẶC THÙ (VIETNAMESE NLP PREPROCESSING)
Qua khảo sát thực tế file CSV, dữ liệu có các lỗi đặc thù. Agent **bắt buộc** phải viết hàm xử lý các lỗi này trong `src/preprocess.py`:

1. **Xử lý Missing Values**:
   - Điền chuỗi rỗng `""` nếu `lead_paragraph` bị `NaN` (ví dụ: `article_2299`).
2. **Khử nhiễu đặc thù của bộ dữ liệu (Data-specific Noise Removal)**:
   - **Lỗi dính địa danh ở VnExpress**: Trong `lead_paragraph`, tên tỉnh/thành phố bị dính liền vào chữ cái viết hoa đầu câu (VD: `"Nghệ AnCảng hàng không..."`, `"TP HCMNgôi nhà..."`, `"MỹNghiên cứu..."`). 
     - *Cách xử lý*: Dùng Regex tìm quy luật `[a-zà-ỹ][A-ZÀ-Ỹ]` (một chữ thường đứng ngay trước một chữ hoa không có khoảng trắng) để tách ra, hoặc xóa bỏ danh sách tên tỉnh thành ở đầu câu.
   - **Lỗi dính tên nguồn báo ở Báo Mới**: Trong `title` của Báo Mới thường có hậu tố `" - Báo ..."`, `" - Tạp chí ..."` (VD: `"Tình thế đảo ngược với tiền vệ Tuấn Anh - Tạp chí SaoStar"`).
     - *Cách xử lý*: Dùng Regex cắt bỏ phần đuôi `r"\s*-\s*(Báo|Tạp chí|Chuyên trang|Đài).*$"` ở cuối `title`.
3. **Chuẩn hóa văn bản Tiếng Việt**:
   - Đưa về định dạng Unicode dựng sẵn (NFC).
   - Chuẩn hóa kiểu gõ dấu tiếng Việt (ví dụ: `òa` -> `oà` hoặc ngược lại, thống nhất 1 kiểu bằng thư viện `underthesea` hoặc `pyvi`).
   - Xử lý khoảng trắng thừa, giữ lại các dấu câu mang sắc thái cảm xúc (`?`, `!`, `...`, `"`).
4. **Tách từ tiếng Việt (Word Segmentation)**:
   - Tiếng Việt là ngôn ngữ đơn lập, từ ghép gồm nhiều âm tiết. Agent phải dùng `pyvi` (`ViTokenizer.tokenize`) hoặc `underthesea` (`word_tokenize(..., format="text")`) để nối các từ ghép bằng dấu gạch dưới `_` (VD: `"Hoa hậu Hoàn vũ"` -> `"Hoa_hậu Hoàn_vũ"`).
5. **Tạo đầu vào (Input Formulation)**:
   - Thực nghiệm 2 phương án đầu vào:
     - **Option A**: Chỉ dùng `title`.
     - **Option B**: Kết hợp `title` + `[SEP]` + `lead_paragraph` (Giúp mô hình phát hiện sự mâu thuẫn hoặc giấu thông tin giữa tiêu đề và nội dung tóm tắt).
6. **Chia tập dữ liệu (Data Splitting)**:
   - Chia theo tỷ lệ **70% Train - 15% Validation - 15% Test** sử dụng `StratifiedShuffleSplit` theo cột `label` để giữ nguyên tỷ lệ phân bố nhãn. Cố định `random_state=42`.

---

### BƯỚC 3: XÂY DỰNG PIPELINE CÁC MÔ HÌNH & SO SÁNH (MODELING)
Agent cần cài đặt và huấn luyện 3 nhóm mô hình từ cổ điển đến hiện đại nhất để làm bảng so sánh cho báo cáo:

#### Nhóm 1: Mô hình Học máy truyền thống (Baseline Models)
- **Trích xuất đặc trưng (Feature Extraction)**: 
  - Sử dụng `TfidfVectorizer` với `ngram_range=(1, 2)`, `max_features=10000`.
  - Kết hợp thêm (Concatenate) các đặc trưng thủ công (Hand-crafted features): Độ dài tiêu đề, có chứa dấu `?` không, có chứa dấu `!` không, số lượng chữ số trong tiêu đề.
- **Thuật toán**:
  1. **Multinomial Naive Bayes** (Baseline đơn giản nhất).
  2. **Logistic Regression** (với `class_weight='balanced'`).
  3. **Support Vector Machine (LinearSVC / SVC)** (Thường rất mạnh với TF-IDF).
  4. **Random Forest / XGBoost**.

#### Nhóm 2: Mô hình Học sâu truyền thống (Deep Learning Models)
- **Biểu diễn từ**: Sử dụng Tokenizer của Keras/PyTorch hoặc Pre-trained Word2Vec/FastText tiếng Việt (hoặc học Embedding từ đầu với `vocab_size=15000`, `embedding_dim=128`, `max_len=128`).
- **Kiến trúc**:
  1. **BiLSTM (Bidirectional LSTM)**: Bắt ngữ cảnh hai chiều chuỗi từ. Cấu trúc: `Embedding -> SpatialDropout1D(0.2) -> BiLSTM(64) -> GlobalMaxPooling1D -> Dense(1, sigmoid)`.
  2. **1D-CNN (TextCNN)**: Bắt các cụm từ khóa (n-grams) giật tít cục bộ bằng các filter size `[2, 3, 5]`.

#### Nhóm 3: Mô hình Ngôn ngữ Lớn / Pre-trained Transformers (SOTA - Mô hình tốt nhất)
- Sử dụng thư viện `transformers` của HuggingFace để Fine-tune mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt:
  1. **PhoBERT (`vinai/phobert-base-v2`)**: Mô hình SOTA chuẩn mực nhất cho tiếng Việt hiện nay. *(Lưu ý: Bắt buộc đầu vào của PhoBERT phải được tách từ bằng RDRSegmenter hoặc PyVi)*.
  2. *(Tùy chọn mở rộng nếu đủ GPU)* **ViSoBERT (`uitnlp/visobert`)** hoặc **XLM-RoBERTa (`xlm-roberta-base`)**.
- **Cấu hình Fine-tuning**:
  - Optimizer: `AdamW`, Learning rate: `2e-5`, Weight decay: `0.01`.
  - Batch size: `16` hoặc `32`, Epochs: `4 - 5` (có dùng `EarlyStopping` dựa trên Validation Macro F1).
  - Max sequence length: `128` (nếu chỉ dùng `title`) hoặc `256` (nếu dùng `title + lead_paragraph`).

---

### BƯỚC 4: ĐÁNH GIÁ & SO SÁNH KẾT QUẢ (EVALUATION)
Do dữ liệu có sự chênh lệch giữa `non-clickbait` và `clickbait`, Agent **không được chỉ dùng Accuracy**. Hãy lập bảng đánh giá trên tập **Test** với các độ đo:
- **Accuracy**
- **Precision, Recall, F1-Score (cho lớp `clickbait`)**
- **Macro F1-Score** *(Độ đo quan trọng nhất để xếp hạng mô hình)*
- **ROC-AUC Score**
- **Thời gian huấn luyện (Training time) & Thời gian suy diễn (Inference time ms/sample)**.

**Bảng mẫu Agent cần xuất ra file Markdown/CSV**:
| Nhóm | Mô hình | Đầu vào | Accuracy | Precision (Clickbait) | Recall (Clickbait) | Macro F1 | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (ML)** | TF-IDF + Naive Bayes | Title + Lead | ? | ? | ? | ? | ? |
| **Baseline (ML)** | TF-IDF + Logistic Regression | Title + Lead | ? | ? | ? | ? | ? |
| **Baseline (ML)** | TF-IDF + Linear SVM | Title + Lead | ? | ? | ? | ? | ? |
| **Deep Learning** | BiLSTM | Title + Lead | ? | ? | ? | ? | ? |
| **Deep Learning** | TextCNN | Title + Lead | ? | ? | ? | ? | ? |
| **Transformer (SOTA)**| **PhoBERT-base-v2** | Only Title | ? | ? | ? | ? | ? |
| **Transformer (SOTA)**| **PhoBERT-base-v2** | **Title + Lead** | **?** | **?** | **?** | **?** | **?** |

**Biểu đồ cần xuất**:
1. Confusion Matrix của tất cả các mô hình (đặt cạnh nhau).
2. Đường cong ROC (ROC Curve) gộp chung các mô hình trên 1 đồ thị.

---

### BƯỚC 5: PHÂN TÍCH LỖI & GIẢI THÍCH MÔ HÌNH (ERROR ANALYSIS & EXPLAINABILITY)
*Đây là phần ăn điểm tuyệt đối trong báo cáo Bài tập lớn.* Agent cần viết code thực hiện:

1. **Feature Importance (Với SVM / Logistic Regression)**:
   - Trích xuất Top 20 từ/cụm từ (n-grams) có trọng số dương lớn nhất đẩy bài báo về nhãn `clickbait` (Ví dụ kỳ vọng: *"bất_ngờ"*, *"xôn_xao"*, *"sự_thật"*, *"cái_kết"*, *"danh_tính"*, *"hé_lộ"*, *"ngỡ_ngàng"*...).
   - Trích xuất Top 20 từ/cụm từ có trọng số âm lớn nhất đại diện cho `non-clickbait` (Ví dụ: *"ubnd"*, *"quốc_hội"*, *"thủ_tướng"*, *"nghị_quyết"*...).
2. **Phân tích các ca dự đoán sai (Error Analysis) của PhoBERT**:
   - Xuất ra 10 mẫu **False Positive** (Tin chính thống nhưng bị đoán nhầm là Clickbait - thường là tin vụ án hình sự ly kỳ hoặc tin thể thao có tính từ mạnh).
   - Xuất ra 10 mẫu **False Negative** (Tin Clickbait tinh vi nhưng mô hình bỏ sót).
   - Đưa ra nhận xét ngôn ngữ học về lý do sai.

---

## 4. YÊU CẦU MÔI TRƯỜNG & THƯ VIỆN (`requirements.txt`)
Agent cần tạo và cài đặt các thư viện sau:
```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
wordcloud>=1.9.0
pyvi>=0.1.1
underthesea>=6.7.0
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
datasets>=2.14.0
xgboost>=2.0.0
tqdm>=4.66.0
```

## 5. HƯỚNG DẪN CHẠY DÀNH CHO AGENT
1. Đọc kỹ file `clickbait_dataset_vietnamese.csv`. Kiểm tra encoding `utf-8`.
2. Viết và chạy script tiền xử lý, in ra màn hình 5 câu trước và sau khi xử lý lỗi dính chữ (VnExpress) và lỗi đuôi báo (Báo Mới) để làm minh chứng cho báo cáo.
3. Chạy huấn luyện lần lượt từ ML -> DL -> PhoBERT. Lưu toàn bộ log và bảng kết quả vào thư mục `outputs/results/`.
4. Tổng hợp toàn bộ số liệu thực tế thu được và viết thành một file báo cáo hoàn chỉnh `FINAL_REPORT.md` theo đúng bố cục: *Giới thiệu -> Khảo sát dữ liệu -> Tiền xử lý -> Phương pháp đề xuất -> Thực nghiệm & So sánh -> Phân tích lỗi -> Kết luận*.
```

---

### Gợi ý thêm cho bạn khi viết Báo cáo / Thuyết trình trước giảng viên:
Khi Agent chạy xong code và xuất ra kết quả, bạn sẽ có một câu chuyện (storytelling) cực kỳ chặt chẽ để đưa vào báo cáo:
1. **Điểm sáng về xử lý dữ liệu thực tế**: Giảng viên NLP rất thích sinh viên tự phát hiện ra lỗi bẩn trong dữ liệu (như lỗi chữ *"Nghệ AnCảng hàng không"* hay đuôi *"- Báo Tiền Phong"*) thay vì chỉ gọi hàm `clean_text` có sẵn trên mạng.
2. **Tư duy chống Học vẹt (Anti-Shortcut Learning)**: Việc bạn chủ động loại bỏ cột `source` và đuôi tên báo trong `title` chứng tỏ bạn hiểu sâu về bản chất học máy (tránh việc model thấy chữ "Kênh14" hay "SaoStar" là auto gán nhãn Clickbait).
3. **So sánh đa chiều (Ablation Study)**: Bạn vừa có so sánh giữa các thế hệ mô hình (**TF-IDF + SVM** vs **BiLSTM** vs **PhoBERT**), vừa có so sánh giữa các cách đưa dữ liệu vào (**Chỉ dùng Title** vs **Dùng cả Title + Lead Paragraph**). Thông thường, kết hợp `Title + Lead Paragraph` với **PhoBERT** sẽ đạt F1-Score cao nhất (khoảng 88% - 93% trên bộ dữ liệu này) vì nó nắm bắt được hiện tượng *"treo đầu dê bán thịt chó"* hoặc *"cố tình giấu chủ ngữ ở tiêu đề nhưng tiết lộ ở phần sapo"*.