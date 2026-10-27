# GIẢI THÍCH KIẾN TRÚC KỸ THUẬT & CƠ CHẾ HOẠT ĐỘNG
### Chrome Extension Phát hiện Clickbait Tiếng Việt (VN Anti-Clickbait Shield)

Tài liệu này giải thích chi tiết toàn bộ cơ chế bên dưới của Extension: từ cách **trích xuất tiêu đề từ trang web**, **quy trình phân tích AI 2 tầng**, đến cách **nhúng nhãn cảnh báo (Xanh/Vàng/Đỏ), làm mờ và tháo tít trực tiếp trên trang báo** mà không làm ảnh hưởng đến bố cục gốc.

---

## 1. Sơ đồ Kiến trúc & Luồng Dữ liệu Tổng thể

Toàn bộ quy trình diễn ra khép kín trong trình duyệt theo mô hình dưới đây:

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TRÌNH DUYỆT NGƯỜI DÙNG (Browser Environment)                                                           │
│                                                                                                        │
│  [1. Trang Báo Mạng] (VnExpress, Dân Trí, Tuổi Trẻ, Kênh14, 24h, Báo Mới...)                          │
│          │                                                                                             │
│          ▼ (Tự động kích hoạt khi trang tải xong: document_idle)                                       │
│  [2. content.js - DOM Scraper]                                                                         │
│     • Dò tìm thẻ link tiêu đề: h2 a, h3 a, .title-news a...                                            │
│     • Lọc bỏ link menu/footer (yêu cầu >= 6 từ, >= 26 ký tự)                                           │
│     • Leo ngược lên thẻ Card cha để lấy đoạn tóm tắt Sapo (.description, .sapo...)                     │
│          │                                                                                             │
│          ▼ chrome.runtime.sendMessage({ type: "PREDICT_BATCH", items })                                │
│  [3. background.js - Service Worker (Điều phối 2 Tầng)]                                                │
│          │                                                                                             │
│     ┌────┴──────────────────────────────────────┐                                                      │
│     ▼ (Nếu Backend tắt)                         ▼ (Nếu Backend đang bật)                               │
│  [TẦNG 1: Local ML Engine]                 [TẦNG 2: Backend AI Server]                                  │
│  • Nạp model_weights.json nhúng sẵn       • Gọi POST http://127.0.0.1:7860/api/predict_batch           │
│  • Khớp n-grams từ SVM + Handcrafted      • Chạy Hybrid Ensemble (SVM + LR + BiLSTM + TextCNN)         │
│  • Chạy bằng JavaScript thuần             • Chuẩn hóa tiếng Việt & Tách từ PyVi                        │
│  • Tốc độ: < 1 mili-giây (Offline)        • Tốc độ: ~15 mili-giây                                      │
│     └────┬──────────────────────────────────────┘                                                      │
│          │ (Trả về mảng kết quả: BadgeLevel, Xác suất %, Lý do, Sapo takeaway, Declickbait)            │
│          ▼                                                                                             │
│  [4. content.js - DOM Injector]                                                                        │
│     • Tạo thẻ <span> Badge gắn cạnh tiêu đề (titleEl.insertAdjacentElement("afterend", badge))        │
│     • Áp dụng CSS làm mờ (filter: blur(3.5px)) nếu bật chế độ Focus                                    │
│     • Bắt sự kiện Hover: tính tọa độ getBoundingClientRect() để mở Hover Tooltip Card                 │
│     • Bấm "Tháo tít": thay thế trực tiếp titleEl.textContent bằng tiêu đề trung lập                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phần 1: Trích xuất các Đầu tiêu đề bằng cách nào?

Mã nguồn thực thi nằm tại hàm `collectArticleElements()` trong file [`extension/content.js`](file:///D:/Work_Study/Coding/Github/vi_clickbait_model/extension/content.js#L198-L255).

### 2.1. Nhận diện cấu trúc DOM báo chí Việt Nam
Các báo điện tử lớn tại Việt Nam đều sử dụng các thẻ tiêu đề chuẩn (`h1` đến `h4`) kết hợp thẻ liên kết `<a>`, hoặc các class mang ngữ nghĩa tin tức:
```javascript
const headlineLinks = document.querySelectorAll(
  "[data-acb-title], h1 a, h2 a, h3 a, h4 a, .title-news a, .article-title a, .story__heading a, .knswli-title a"
);
```

### 2.2. Thuật toán Lọc nhiễu (Deduplication & Noise Filtering)
Trên một trang web có hàng trăm thẻ `<a>` (link điều hướng menu như *"Thời sự"*, *"Kinh tế"*, link chân trang, điều khoản...). Script áp dụng bộ lọc:
1. **Loại bỏ trùng lặp**: Dùng cấu trúc `Set()` để mỗi phần tử tiêu đề chỉ được phân tích duy nhất 1 lần.
2. **Ngưỡng độ dài văn bản**: Tiêu đề bài báo tiếng Việt chuẩn có cấu trúc tối thiểu từ **6 từ đơn trở lên** và **chiều dài $\ge 26$ ký tự**. Bất kỳ chuỗi ngắn hơn (như *"Xem thêm"*, *"Tin mới nhất"*) đều bị bỏ qua.
3. **Giới hạn số lượng**: Chỉ quét tối đa 60 bài nổi bật nhất trên màn hình đầu tiên để đảm bảo trang web luôn phản hồi tức thì (`< 20ms`).

### 2.3. Kỹ thuật DOM Traversing để trích xuất cả đoạn mở đầu (Sapo)
Một bài báo trên danh mục thường được bọc trong một "thẻ cha" (Article Card). Script sử dụng phương thức `Element.closest()` để leo ngược từ tiêu đề lên thẻ cha của bài báo, sau đó truy vấn đoạn mô tả tóm tắt (`sapo`):
```javascript
// Leo ngược lên thẻ bao ngoài của bài viết (Card)
const card = linkEl.closest("article, .item-news, .box-category-item, .knswli, .news-item, .story, li");

// Tìm đoạn sapo tóm tắt nằm cùng bài viết đó
const descEl = card.querySelector(".description, .sapo, .article-excerpt, .knswli-sapo, p");
const leadText = descEl ? descEl.textContent.trim() : "";
```

Kết quả của Bước 1 là một mảng dữ liệu có cấu trúc:
```json
[
  {
    "id": "acb_art_0",
    "title": "Hé lộ danh tính người đàn ông bí ẩn xuất hiện cùng nữ diễn viên lúc nửa đêm khiến MXH xôn xao!",
    "lead_paragraph": "Tối 25/10, đoàn làm phim xác nhận người đi cùng nữ diễn viên là trợ lý sản xuất..."
  }
]
```

---

## 3. Phần 2: Phân tích & Đánh giá (Bộ máy AI 2 Tầng)

Content script gửi mảng bài báo về Service Worker [`extension/background.js`](file:///D:/Work_Study/Coding/Github/vi_clickbait_model/extension/background.js). Tại đây diễn ra cơ chế **Suy diễn 2 Tầng (Two-Tier Inference)**:

```
┌───────────────────────────────────────────────────────────────┐
│              Kiểm tra Backend Server (Cổng 7860)              │
├───────────────────────────────┬───────────────────────────────┤
│    Nếu Server ĐANG CHẠY       │     Nếu Server KHÔNG CHẠY     │
│   (Ưu tiên chất lượng SOTA)   │       (Ưu tiên tính sẵn sàng) │
│               ▼               │               ▼               │
│     TẦNG 2: SERVER ENSEMBLE   │      TẦNG 1: LOCAL ML ENGINE  │
│  FastAPI + PyTorch + PyVi     │  Pure JavaScript in Extension │
│  Tổ hợp 4 mô hình ML/DL       │  Trọng số n-gram từ SVM/LR    │
│  Độ trễ: ~15ms                │  Độ trễ: < 1ms (100% Offline) │
└───────────────────────────────┴───────────────────────────────┘
```

### 3.1. Tầng 1: Local ML Engine (Chạy thuần JavaScript trong Extension)
Để người dùng **vẫn được bảo vệ ngay cả khi không bật server Python hoặc tắt máy bật lại**, toàn bộ trọng số huấn luyện quan trọng nhất đã được xuất sẵn vào file [`extension/model_weights.json`](file:///D:/Work_Study/Coding/Github/vi_clickbait_model/extension/model_weights.json):
* **1,600 trọng số n-gram thực tế** trích xuất từ mô hình `Linear SVM` và `Logistic Regression` đã huấn luyện trên tập dữ liệu tiếng Việt.
* **Tập cụm từ kích động cảm xúc**: *"bất ngờ"*, *"xôn xao"*, *"sự thật"*, *"cái kết"*, *"danh tính"*, *"hé lộ"*, *"ngỡ ngàng"*, *"khiến ai cũng"*...
* **Tập cụm từ báo chí chính thống**: *"ubnd"*, *"quốc hội"*, *"chính phủ"*, *"nghị định"*, *"điều tra"*, *"khởi tố"*...

**Thuật toán chấm điểm trên Client:**
1. Khởi tạo điểm số cơ sở (Bias score): $\text{Score} = -0.35$ (do tỷ lệ clickbait trong tập dữ liệu thực tế là $\sim 31.2\%$).
2. Khớp các cụm từ giật gân $\rightarrow$ tăng điểm $\text{Score} += w \times 0.55$.
3. Khớp các thuật ngữ chính luận $\rightarrow$ giảm điểm $\text{Score} -= 0.45$.
4. Tính điểm từ **12 đặc trưng thủ công (Handcrafted Features)**:
   * Có dấu `?` tu từ: $+0.55$
   * Có dấu `!` cảm thán: $+0.65$
   * Có dấu ngoặc kép trích dẫn: $+0.25$
   * Có dấu ba chấm bỏ lửng `...`: $+0.35$
   * Dùng đại từ giấu thông tin (*"thứ này"*, *"người này"*, *"cô gái này"*): $+0.50$
5. Chuyển đổi sang xác suất bằng hàm Sigmoid:
   $$\text{Probability} = \frac{1}{1 + e^{-\text{Score}}}$$

### 3.2. Tầng 2: Server Hybrid Ensemble (Khi bật `app.py`)
Khi server hoạt động, `background.js` gọi API `POST /api/predict_batch`. Server chạy quy trình xử lý chuyên sâu trong [`src/inference.py`](file:///D:/Work_Study/Coding/Github/vi_clickbait_model/src/inference.py):
1. **Khử nhiễu đặc thù tiếng Việt**:
   * Khử lỗi dính địa danh ở đầu Sapo của VnExpress (ví dụ: *"Nghệ AnCảng hàng không..."* $\rightarrow$ *"Nghệ An Cảng hàng không..."*).
   * Cắt bỏ hậu tố tên báo ở cuối Tiêu đề của Báo Mới (ví dụ: *"... - Tạp chí SaoStar"*).
2. **Chuẩn hóa & Tách từ**: Đưa về Unicode dựng sẵn (NFC), chuẩn hóa vị trí dấu thanh tiếng Việt, tách từ ghép bằng PyVi (`ViTokenizer`).
3. **Dự đoán từ Tổ hợp 4 Mô hình (Hybrid Ensemble)**:
   $$\text{Score}_{\text{Ensemble}} = 0.35 \times P_{\text{SVM}} + 0.20 \times P_{\text{Logistic}} + 0.22 \times P_{\text{BiLSTM}} + 0.23 \times P_{\text{TextCNN}}$$
4. **Phân tích Độ lệch Ngữ nghĩa (Semantic Gap)**: So sánh từ vựng giữa Tiêu đề và Sapo để phát hiện bài viết "treo đầu dê bán thịt chó" (Tiêu đề nói quá nhưng Sapo lại bình thường).
5. **Sinh Tiêu đề Trung lập (De-clickbait)**: Trích xuất mệnh đề thông tin cốt lõi từ Sapo hoặc làm sạch các từ ngữ kích động ở tiêu đề.

### 3.3. Quy tắc Phân cấp Nhãn Tín hiệu (Traffic-Light Badge)
* $\text{Xác suất} \ge 62\%$: 🔴 **Giật tít cao** (`badge_level: "red"`) — Kèm tỷ lệ % và gợi ý tháo tít.
* $40\% \le \text{Xác suất} < 62\%$: 🟡 **Chú ý** (`badge_level: "yellow"`) — Tiêu đề có yếu tố gây tò mò / giấu thông tin nhẹ.
* $\text{Xác suất} < 40\%$: 🟢 **Tin chuẩn** (`badge_level: "green"`) — Bài viết chính thống, khách quan.

---

## 4. Phần 3: Nhúng ngược lại vào Web như thế nào? (DOM Injection)

Sau khi nhận kết quả phân tích về, [`extension/content.js`](file:///D:/Work_Study/Coding/Github/vi_clickbait_model/extension/content.js#L295-L330) tiến hành can thiệp giao diện theo các bước:

### 4.1. Nhúng thẻ Badge cạnh tiêu đề
Dùng phương thức DOM chuẩn `insertAdjacentElement("afterend", badgeElement)`. Thao tác này chèn một thẻ `<span>` ngay phía sau thẻ `<a>` tiêu đề:

```javascript
const badge = document.createElement("span");
badge.className = `vn-acb-badge vn-acb-badge-${pred.badge_level}`;
badge.innerHTML = `<span>${dot}</span><span>${pred.badge_text}</span>`;

// Gắn trực tiếp vào sau tiêu đề bài báo:
item.titleEl.insertAdjacentElement("afterend", badge);
```

### 4.2. Kỹ thuật Cô lập CSS (Scoped CSS)
Toàn bộ stylesheet trong [`extension/content.css`](file:///D:/Work_Study/Coding/Github/vi_clickbait_model/extension/content.css) đều sử dụng class có tiền tố `.vn-acb-*` và thuộc tính `!important`:
* Không sử dụng các class chung chung như `.badge`, `.card` để tránh ghi đè giao diện gốc của báo.
* Định dạng badge dạng viên thuốc bo tròn (`border-radius: 999px`), chữ nhỏ gọn (`11.5px`), hiển thị dạng `inline-flex` thẳng hàng với dòng chữ tiêu đề.

### 4.3. Cơ chế Tự động Làm mờ (Focus Blur Mode)
Nếu người dùng kích hoạt công tắc *"Tự động làm mờ tít câu view"* trong Popup, Extension sẽ gán thêm class `.vn-acb-blurred-title` vào thẻ tiêu đề:
```css
.vn-acb-blurred-title {
  filter: blur(3.5px) !important;
  opacity: 0.52 !important;
  transition: filter 0.25s ease, opacity 0.25s ease !important;
}

/* Khi người đọc rê chuột vào tiêu đề bị mờ, CSS tự động mở rõ ra */
.vn-acb-blurred-title:hover {
  filter: none !important;
  opacity: 1 !important;
}
```

### 4.4. Cơ chế Hover Tooltip Card (Hiển thị Tóm tắt Sapo & Tháo tít)
Khi người dùng rê chuột vào chiếc nhãn màu:
1. Script tính toán tọa độ chính xác của chiếc nhãn trên màn hình bằng `anchorEl.getBoundingClientRect()`.
2. Khung tooltip nổi (`.vn-acb-tooltip`) mang giao diện Dark Mode cao cấp hiển thị ngay phía dưới nhãn.
3. **Đọc tóm tắt không cần click**: Hiển thị đoạn Sapo cốt lõi (`pred.quick_summary`). Người đọc nắm được sự thật ngay lập tức mà không cần bấm vào bài câu view của báo.
4. **Nút "✨ Tháo tít trên trang"**: Khi bấm nút này, JavaScript thực hiện can thiệp DOM trực tiếp:
   ```javascript
   item.titleEl.textContent = `✨ ${pred.declickbait_title}`;
   item.titleEl.classList.remove("vn-acb-blurred-title");
   ```
   Tiêu đề giật gân lập tức biến mất và được thay thế bằng tiêu đề khách quan chuẩn mực ngay trước mắt người đọc.

---

## 5. Phần 4: Trợ lý Đọc bài Chi tiết (Reading Assistant Floating Pill)

Khi người dùng click vào đọc một bài báo chi tiết, Extension tự động kích hoạt **Tầng 2**:

1. **Nhận diện bài chi tiết**: Dò tìm thẻ tiêu đề chính `h1.title-detail`, `h1.article-title`, `h1.the-article-title` cùng đoạn sapo `p.description`, `h2.sapo`.
2. **Đánh giá cặp câu (Title + Sapo)**: Gửi cặp văn bản về mô hình để kiểm tra độ lệch ngữ nghĩa.
3. **Hiển thị viên nang nổi (Floating Pill)** ở góc dưới bên phải màn hình:
   * Cho biết mức độ trung thực của bài viết.
   * Cung cấp nút **"✨ Tháo tít trên bài"** để đổi tiêu đề `h1` của bài báo sang tiêu đề trung lập (và cho phép bấm đổi ngược lại tiêu đề gốc bất cứ lúc nào).
   * Cung cấp nút **"🚩 Báo nhầm lẫn"** để người đọc đóng góp phản hồi (Active Learning), dữ liệu này được lưu tự động vào `outputs/results/user_feedback.jsonl` để tái huấn luyện mô hình sau này.

---

## 6. Tổng kết Giá trị Kỹ thuật của Kiến trúc này

| Tiêu chí | Cách làm truyền thống | Giải pháp của Dự án này |
| :--- | :--- | :--- |
| **Thao tác của Người dùng** | Phải copy tiêu đề, mở popup, paste vào rồi bấm "Kiểm tra". | **Tự động 100% (Ambient)**: Nhãn cảnh báo xuất hiện sẵn ngay trên trang web. |
| **Độ trễ (Latency)** | 500ms - 2s (phải gọi server liên tục). | **< 1ms với Local ML Engine** nhúng sẵn trong trình duyệt. |
| **Tính khả dụng khi Offline** | Bị lỗi hoàn toàn nếu không bật server Python. | **Hoạt động bình thường** nhờ bộ trọng số `model_weights.json`. |
| **Giá trị cho Độc giả** | Chỉ biết một con số % khô khan. | **Bảo vệ thời gian**: Cho đọc luôn tóm tắt Sapo và hỗ trợ **Tháo tít câu view** thành tít chuẩn. |
