# Phát hiện Clickbait Tiếng Việt (Vietnamese Clickbait Detection)

Hệ thống phân loại văn bản nhị phân (**Binary Text Classification**) phát hiện bài báo tiếng Việt **giật tít (`clickbait`)** so với **tin không giật tít (`non-clickbait`)**, khai thác đặc trưng ngôn ngữ từ tiêu đề (`title`) và đoạn mở đầu/sapo (`lead_paragraph`). Đi kèm **Web Demo Gradio (XAI)** và **Chrome Extension 3 Tầng (Manifest V3)** phục vụ trực tiếp độc giả đọc báo điện tử hàng ngày.

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
│   ├── evaluate.py                 # Tính toán metrics và xuất toàn bộ biểu đồ EDA / Đánh giá
│   └── inference.py                # Bộ máy suy diễn hợp nhất (Ensemble 5 mô hình, XAI, De-clickbait, URL Scraper)
├── extension/                      # Chrome/Edge Extension (Manifest V3) — Giao diện 3 Tầng
│   ├── manifest.json               # Cấu hình Extension Manifest V3
│   ├── background.js               # Service Worker (Two-Tier Inference: Server Ensemble + Local ML Fallback)
│   ├── content.js                  # Tầng 1 (In-Page Ambient Badges & Tooltip) + Tầng 2 (Reading Assistant Pill)
│   ├── content.css                 # Stylesheet cô lập (.vn-acb-*) không xung đột với báo điện tử
│   ├── popup.html / .js / .css     # Tầng 3 (Action Popup: Thống kê phút tiết kiệm, Độ sạch trang, Bộ lọc)
│   ├── model_weights.json          # Trọng số n-gram xuất từ mô hình ML phục vụ suy diễn tức thì (<1ms)
│   └── demo_news.html              # Cổng báo điện tử giả lập kiểm thử trực quan Extension
├── outputs/
│   ├── figures/                    # Lưu trữ bộ biểu đồ phân tích chất lượng cao (.png)
│   └── results/                    # Lưu bảng điểm (CSV/Markdown/JSON) và model checkpoints
├── app.py                          # Khởi chạy đồng thời Demo Gradio + FastAPI Server cho Extension
├── run_pipeline.py                 # Script chạy tự động toàn bộ pipeline huấn luyện từ A đến Z
├── start_app.bat / stop_app.bat    # Script khởi chạy & dừng nhanh trên Windows
├── start_app.sh / stop_app.sh      # Script khởi chạy & dừng nhanh trên Linux / macOS
├── slides_demo.html                # Bộ Slide thuyết trình HTML tương tác toàn màn hình
├── requirements.txt                # Danh sách thư viện phụ thuộc
└── README.md                       # Tài liệu hướng dẫn cài đặt và sử dụng
```

---

## 2. Hướng dẫn Cài đặt & Chạy trên Windows và Linux

Yêu cầu hệ thống: **Python 3.10+** (khuyên dùng Python 3.10 – 3.12) và trình duyệt nhân Chromium (**Google Chrome, Microsoft Edge, Brave, Cốc Cốc, Chromium**).

### A. Trên Hệ điều hành Windows

#### 1. Tạo môi trường ảo & Cài đặt thư viện (Lần đầu)
Mở **PowerShell** hoặc **Command Prompt (CMD)** tại thư mục dự án:
```powershell
# 1. Tạo môi trường ảo venv (nếu chưa có)
python -m venv venv

# 2. Kích hoạt môi trường ảo
venv\Scripts\activate

# 3. Cài đặt đầy đủ thư viện phụ thuộc
pip install -r requirements.txt
```

#### 2. Khởi chạy Demo Gradio & Backend Server cho Extension
* **Cách 1 (1 Click chuột)**: Nhấp đúp chuột vào file `start_app.bat`.
* **Cách 2 (Bằng dòng lệnh)**:
  ```powershell
  venv\Scripts\activate
  python app.py
  ```
* **Cách dừng Server**: Nhấn `Ctrl + C` trong cửa sổ dòng lệnh hoặc nhấp đúp chuột vào file `stop_app.bat`.

#### 3. Chạy lại toàn bộ Pipeline Huấn luyện từ đầu (Tùy chọn)
```powershell
venv\Scripts\activate
python run_pipeline.py
```

---

### B. Trên Hệ điều hành Linux (Ubuntu / Debian / Fedora) & macOS

#### 1. Tạo môi trường ảo & Cài đặt thư viện (Lần đầu)
Mở **Terminal** tại thư mục dự án:
```bash
# 1. Cài đặt gói python3-venv (nếu dùng Ubuntu/Debian và chưa có venv)
sudo apt update && sudo apt install -y python3-venv python3-pip

# 2. Tạo môi trường ảo venv
python3 -m venv venv

# 3. Kích hoạt môi trường ảo
source venv/bin/activate

# 4. Nâng cấp pip và cài đặt đầy đủ thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Khởi chạy Demo Gradio & Backend Server cho Extension
* **Cách 1 (Dùng script tự động)**:
  ```bash
  chmod +x start_app.sh stop_app.sh
  ./start_app.sh
  ```
* **Cách 2 (Chạy trực tiếp trong Terminal)**:
  ```bash
  source venv/bin/activate
  python3 app.py
  ```
* **Cách dừng Server**: Nhấn `Ctrl + C` trong Terminal hoặc chạy `./stop_app.sh` (hoặc `fuser -k 7860/tcp`).

#### 3. Chạy lại toàn bộ Pipeline Huấn luyện từ đầu (Tùy chọn)
```bash
source venv/bin/activate
python3 run_pipeline.py
```

---

## 3. Truy cập Web Demo Gradio & Cài đặt Chrome Extension

Sau khi chạy `app.py` (trên Windows hoặc Linux), hệ thống mở các dịch vụ tại `http://127.0.0.1:7860`:
- **Dashboard Demo Gradio & Giải thích AI (XAI)**: `http://127.0.0.1:7860`
- **Trang Báo Điện Tử Giả Lập (Test Extension ngay trên trình duyệt)**: `http://127.0.0.1:7860/demo-news`
- **REST API cho Chrome Extension**: `POST /api/predict`, `POST /api/predict_batch`, `POST /api/analyze_url`, `POST /api/feedback`

### Các bước cài đặt Chrome / Edge Extension (Giống nhau trên Windows & Linux):
1. Mở trình duyệt Chrome, Chromium, Edge hoặc Brave và truy cập `chrome://extensions/` (hoặc `edge://extensions/`).
2. Bật công tắc **Developer mode (Chế độ dành cho nhà phát triển)** ở góc trên bên phải.
3. Nhấn nút **Load unpacked (Tải tiện ích đã giải nén)** và chọn thư mục `extension/` trong dự án này.
4. Mở trang thử nghiệm `http://127.0.0.1:7860/demo-news` hoặc truy cập trực tiếp các báo lớn (`vnexpress.net`, `dantri.com.vn`, `tuoitre.vn`, `kenh14.vn`, `24h.com.vn`, `baomoi.com`, `bongdaplus.vn`) để trải nghiệm 3 tầng bảo vệ:
   - **Tầng 1 (In-Page Ambient Badges)**: Gắn nhãn `🟢 Không giật tít`, `🟡 Chú ý`, `🔴 Giật tít` ngay cạnh tiêu đề; rê chuột vào nhãn để đọc **Tóm tắt Sapo** và **Tháo tít** mà không cần click vào bài câu view.
   - **Tầng 2 (Reading Assistant Floating Pill)**: Thanh nổi ở góc dưới phải khi đọc bài chi tiết, cảnh báo độ lệch ngữ nghĩa giữa Tiêu đề và Sapo, hỗ trợ đổi tiêu đề bài viết sang văn phong trung lập.
   - **Tầng 3 (Action Popup)**: Theo dõi số bài giật tít đã cảnh báo, số phút tiết kiệm được, bật/tắt chế độ **Tự động làm mờ (Blur) tin câu view** và điều chỉnh độ nhạy bộ lọc.

---

## 4. Điểm nhấn Kỹ thuật trong Pipeline

1. **Khử nhiễu dữ liệu thực tế (Data-specific Noise Removal)**:
   - Tách địa danh bị dính liền vào từ đầu câu trong `lead_paragraph` của **VnExpress** (`163 mẫu`, ví dụ: `"Nghệ AnCảng hàng không..."` $\rightarrow$ `"Nghệ An Cảng hàng không..."`).
   - Cắt bỏ hậu tố tên báo ở cuối `title` của **Báo Mới** (`716 mẫu`, ví dụ: `"... - Tạp chí SaoStar"`).
2. **Chống Rò rỉ Dữ liệu & Học vẹt (Anti-Shortcut Learning)**:
   - Loại bỏ hoàn toàn cột `source` khỏi tập đặc trưng huấn luyện để buộc mô hình học ngữ nghĩa văn bản (`giật tít` vs `không giật tít`) thay vì học vẹt tên báo.
3. **Suy diễn 2 Tầng (Two-Tier Inference) cho Extension**:
   - Kết hợp **Local ML Engine** (`model_weights.json` chạy trực tiếp trên trình duyệt trong `<1ms` khi offline) và **Server Hybrid Ensemble** (`Linear SVM + Logistic Regression + BiLSTM + TextCNN` qua FastAPI).
