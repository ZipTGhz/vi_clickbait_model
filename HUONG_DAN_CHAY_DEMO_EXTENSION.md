# HƯỚNG DẪN CÀI ĐẶT & CHẠY DEMO GRADIO + CHROME EXTENSION
### Hệ thống Phát hiện & Giải thích Tin Giật Tít Tiếng Việt (VN Anti-Clickbait Shield)

Tài liệu này hướng dẫn chi tiết cách khởi động **Web Demo Gradio**, sử dụng **REST API Backend** và cài đặt **Chrome / Edge Extension (Manifest V3)** để trải nghiệm trực tiếp trên các báo điện tử Việt Nam.

---

## 1. Yêu cầu Môi trường & Cài đặt (Chỉ cần làm lần đầu)

Dự án đã có sẵn môi trường ảo `venv` và file cấu hình `requirements.txt`. Nếu bạn chạy trên máy mới hoặc cần cài đặt lại:

```bash
# 1. Kích hoạt môi trường ảo venv (Windows)
venv\Scripts\activate

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 2. Cách Khởi động & Tắt Backend Server

Server backend điều phối đồng thời:
* **Giao diện Web Demo Gradio** (phục vụ thuyết trình, kiểm thử đa mô hình, xem biểu đồ EDA & giải thích XAI).
* **REST API** phục vụ Chrome Extension quét bài báo tự động (`/api/predict_batch`, `/api/predict`, `/api/analyze_url`).
* **Cổng báo điện tử giả lập** (`/demo-news`) để kiểm thử Extension.

### Cách khởi động:
* **Cách 1 (Nhanh nhất - 1 Click chuột)**:
  * Nhấp đúp chuột (Double click) vào file: **`start_app.bat`**
  * Cửa sổ dòng lệnh sẽ tự động kích hoạt `venv` và mở server tại cổng `7860`.
* **Cách 2 (Bằng Terminal / PowerShell)**:
  ```powershell
  cd D:\Work_Study\Coding\Github\vi_clickbait_model
  .\venv\Scripts\activate
  python app.py
  ```

Sau khi khởi chạy thành công, màn hình sẽ thông báo:
```text
🚀 KHỞI ĐỘNG HỆ THỐNG DEMO GRADIO & EXTENSION API TẠI: http://127.0.0.1:7860
📰 Trang Báo Điện Tử Giả Lập để test Extension:      http://127.0.0.1:7860/demo-news
🔌 Endpoint API cho Chrome Extension:                http://127.0.0.1:7860/api/predict_batch
```

### Cách tắt Server khi không dùng nữa:
* **Cách 1**: Nhấp vào cửa sổ dòng lệnh đang chạy và bấm tổ hợp phím **`Ctrl + C`** (hoặc bấm dấu **`X`** để đóng cửa sổ).
* **Cách 2 (1 Click chuột)**: Nhấp đúp chuột vào file **`stop_app.bat`**, script sẽ tự động tìm và ngắt tiến trình đang chạy trên cổng `7860`.

---

## 3. Hướng dẫn Sử dụng Giao diện Web Demo Gradio

Mở trình duyệt bất kỳ và truy cập địa chỉ: **`http://127.0.0.1:7860`**

Giao diện gồm **3 Tab chuyên sâu**:

### Tab 1: 🔍 Phân tích Bài báo & Giải thích AI (XAI)
1. **Đầu vào**:
   * Nhập thủ công **Tiêu đề (Title)** và **Đoạn mở đầu (Sapo)**.
   * Hoặc dán trực tiếp **URL báo điện tử** (*VnExpress, Dân Trí, Tuổi Trẻ, Kênh14, 24h, Báo Mới...*) và bấm **"Trích xuất URL"** để hệ thống tự động tải và bóc tách nội dung.
   * Hoặc nhấp chọn các bài báo mẫu thực tế ở bảng **Examples** bên dưới.
2. **Tùy chọn Mô hình**:
   * Chọn một trong 5 mô hình: `Ensemble (Khuyên dùng)`, `TF-IDF + Linear SVM`, `TF-IDF + Logistic Regression`, `BiLSTM`, `TextCNN`.
   * Kéo thanh trượt **Ngưỡng nhạy Clickbait** (mặc định 0.50).
3. **Kết quả hiển thị**:
   * **Thẻ Kết luận Trực quan**: Nhãn màu (🟢 Tin chuẩn / 🟡 Chú ý / 🔴 Giật tít), thanh đo xác suất Clickbait (%), lý do giải thích bằng ngôn ngữ tự nhiên, các từ khóa kích thích tò mò và **✨ Gợi ý Tháo tít** (tiêu đề trung lập chuẩn báo chí).
   * **Tô màu Từ khóa (Token-level XAI Attribution)**: Đánh dấu các từ ngữ theo 3 cấp độ: *Giật tít mạnh* (đỏ), *Gợi tò mò* (vàng), *Chính thống* (xanh).
   * **Chi tiết Kỹ thuật**: So sánh điểm số giữa 5 mô hình, bảng 12 đặc trưng thủ công (Hand-crafted features) và trạng thái khử nhiễu đặc thù (*khử lỗi dính địa danh VnExpress*, *cắt bỏ đuôi tên báo Báo Mới*).

### Tab 2: 🧩 Trải nghiệm Giao diện Chrome Extension (UX 3 Tầng)
* Cho phép bạn xem trước cách Extension hoạt động trên luồng tin tức báo mạng ngay trên web.
* Thử bật công tắc **"Tự động làm mờ (Blur) tin câu view"** và thay đổi độ nhạy bộ lọc để thấy các tiêu đề clickbait lập tức bị làm mờ, rê chuột vào sẽ hiển thị tooltip tóm tắt Sapo.

### Tab 3: 📊 Bảng điểm Thực nghiệm & Biểu đồ Phân tích Dữ liệu (EDA)
* Bảng tổng hợp metrics trên tập Test: **Accuracy**, **Precision**, **Recall**, **Macro F1**, **ROC-AUC** và **Tốc độ suy diễn (ms/mẫu)**.
* Gallery trình chiếu 6 bộ biểu đồ phân tích EDA chất lượng cao: phân phối nhãn, phân bố nguồn báo/chuyên mục, độ dài văn bản, đặc trưng dấu câu/con số, WordCloud và Top 20 Feature Importance.

---

## 4. Hướng dẫn Cài đặt & Sử dụng Chrome / Edge Extension

Extension hoạt động trên mọi trình duyệt nền tảng Chromium như **Google Chrome, Microsoft Edge, Brave, Cốc Cốc, Opera**.

### Bước 1: Cài đặt Extension vào trình duyệt
1. Mở trình duyệt và truy cập vào trang quản lý tiện ích:
   * **Chrome / Brave / Cốc Cốc**: Gõ `chrome://extensions/` vào thanh địa chỉ rồi nhấn Enter.
   * **Microsoft Edge**: Gõ `edge://extensions/` vào thanh địa chỉ rồi nhấn Enter.
2. Bật công tắc **Developer mode (Chế độ dành cho nhà phát triển)** ở góc trên bên phải màn hình.
3. Nhấn vào nút **Load unpacked (Tải tiện ích đã giải nén)** ở góc trên bên trái.
4. Điều hướng và chọn đúng thư mục sau trong máy tính của bạn:
   ```text
   D:\Work_Study\Coding\Github\vi_clickbait_model\extension
   ```
5. Tiện ích mang tên **VN Anti-Clickbait Shield — Lá chắn Tin Giật Tít** sẽ xuất hiện trong danh sách.
6. *(Khuyên dùng)* Nhấp vào biểu tượng mảnh ghép (Extensions) trên thanh công cụ của trình duyệt và nhấn vào **icon cái ghim 📌** để ghim Extension ra ngoài cho tiện theo dõi.

---

### Bước 2: Trải nghiệm 3 Tầng Giao diện Thực tế

Sau khi cài đặt xong, bạn có thể kiểm thử ngay theo 2 cách:

#### Cách A: Mở Trang Báo Giả Lập đi kèm
Truy cập: **`http://127.0.0.1:7860/demo-news`**
* Trang web này mô phỏng giao diện của một tờ báo điện tử với đầy đủ các thể loại tin (giải trí câu view, tài chính vĩ mô, sức khỏe giật gân, giao thông chính thống...).
* Bạn sẽ thấy:
  1. **Tầng 1 (In-Page Ambient Badges)**: Cạnh mỗi tiêu đề xuất hiện ngay nhãn **🟢 Tin chuẩn**, **🟡 Chú ý · xx%**, hoặc **🔴 Giật tít · xx%**.
     * Rê chuột vào nhãn đỏ: Một khung màu tối sang trọng hiện ra, hiển thị từ khóa kích động (*"bí ẩn"*, *"xôn xao"*, *"cái kết đắng"*...), lý do nhận diện và **💡 Tóm tắt Sapo** (giúp bạn nắm luôn nội dung thật mà **không cần bấm click vào bài báo**).
     * Bấm nút **"✨ Tháo tít trên trang"**: Tiêu đề giật gân trên bài viết sẽ lập tức đổi thành tiêu đề khách quan, chuẩn mực!
  2. **Tầng 2 (Reading Assistant Floating Pill)**: Nhấp vào bất kỳ bài báo nào để xem ở cột bên phải (Chế độ đọc chi tiết).
     * Ở góc dưới cùng bên phải màn hình sẽ xuất hiện một viên nang nổi (Floating Pill) mang tên **"🛡️ Trợ lý Đọc Báo"**.
     * Nó đánh giá độ lệch giữa Tiêu đề và Sapo của bài đang đọc. Nhấp vào nút **"✨ Tháo tít trên bài"** để đổi tiêu đề `<h1>` của bài viết sang tiêu đề trung lập; hoặc bấm **"🚩 Báo nhầm lẫn"** để gửi đóng góp dữ liệu cải thiện mô hình.

#### Cách B: Mở các Báo Điện Tử Thật
Mở các website báo chí tiếng Việt thông dụng:
* `https://vnexpress.net`
* `https://kenh14.vn`
* `https://24h.com.vn`
* `https://tuoitre.vn`
* `https://baomoi.com`
* `https://dantri.com.vn`

Extension sẽ tự động quét danh sách bài báo trên trang và gắn nhãn trực tiếp mà không làm chậm hay vỡ giao diện của báo.

---

### Bước 3: Sử dụng Bảng điều khiển Extension Popup (Tầng 3)
Nhấp vào icon **🛡️ VN Anti-Clickbait** trên thanh công cụ trình duyệt:
* **Xem hiệu quả bảo vệ**: Thống kê số lượng bài giật tít đã cảnh báo và số phút đọc tin bạn đã tiết kiệm được.
* **Độ sạch trang hiện tại**: Cho biết trang web bạn đang xem có bao nhiêu % tin chuẩn, bao nhiêu bài giật gân.
* **Chế độ Làm mờ (Blur)**: Bật công tắc *"Tự động làm mờ tít câu view"*, các tiêu đề giật tít trên trang báo sẽ lập tức bị mờ đi. Rê chuột vào tiêu đề nào thì tiêu đề đó mới hiện rõ lại.
* **Thanh trượt Độ nhạy**: Điều chỉnh giữa *Nghiêm ngặt (35%)* — *Cân bằng (50%)* — *Thoải mái (70%)*.
* **Tab Kiểm tra nhanh (Quick Check)**: Cho phép bạn dán nhanh bất kỳ câu tiêu đề nào vào để kiểm tra xác suất giật tít và nhận gợi ý tháo tít ngay lập tức.

---

## 5. Lưu ý Quan trọng về Chế độ Hoạt động

| Trạng thái Backend (`app.py`) | Cơ chế hoạt động của Chrome Extension |
| :--- | :--- |
| **Backend ĐANG BẬT** (`start_app.bat`) | Chạy chế độ **AI Server Online**: Sử dụng tổ hợp 4 mô hình học sâu (**Hybrid Ensemble**: Linear SVM + Logistic Regression + BiLSTM + TextCNN). Độ chính xác cao nhất, phân tích độ lệch ngữ nghĩa sâu và trích xuất Sapo tự động qua API. |
| **Backend TẮT / Khi tắt máy bật lại** | Chạy chế độ **Local ML Engine (Offline)**: Extension tự động sử dụng file trọng số `model_weights.json` nhúng sẵn. Vẫn quét bài, gắn nhãn Xanh/Vàng/Đỏ và làm mờ tít câu view trực tiếp bằng JavaScript trong **< 1 mili-giây** mà không cần mở server Python hay tốn RAM máy tính. |
