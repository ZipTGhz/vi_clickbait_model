# TỔNG HỢP TOÀN BỘ KIẾN THỨC HỌC THUẬT & KỸ THUẬT CỦA DỰ ÁN
## Đề tài: Hệ thống Phát hiện & Giải thích Tin Giật Tít Tiếng Việt (Vietnamese Clickbait Detection)

Tài liệu này tổng hợp **toàn bộ kiến thức nền tảng môn Xử lý Ngôn ngữ Tự nhiên (NLP)** kết hợp với **cách áp dụng và số liệu thực tế 100% trong mã nguồn dự án** để bạn nắm vững bản chất, tự tin thuyết trình từng slide và giải thích bất kỳ đoạn code nào trong đồ án.

---

## PHẦN 1: BẢN CHẤT BÀI TOÁN & NGÔN NGỮ HỌC CỦA TIN GIẬT TÍT

### 1.1. Phân biệt Tin giật tít (`Clickbait`) và Tin giả (`Fake News`)
* **Tin giả (`Fake News`):** Là bài viết chứa thông tin **sai sự thật**, bịa đặt hoàn toàn hoặc bóp méo sự kiện không có thật (thuộc bài toán *Fact-Checking / Rumor Detection*).
* **Tin giật tít (`Clickbait`):** Nội dung bên trong bài báo có thể hoàn toàn có thật và được đăng trên các báo điện tử hợp pháp, nhưng **cách đặt tiêu đề (`title`) cố tình thao túng tâm lý tò mò** của người đọc để câu lượt nhấp chuột (*click*).
* **Lý thuyết Khoảng trống Tò mò (*Loewenstein’s Information Gap Theory / Curiosity Gap*):**
  * Tiêu đề giật tít cố tình tạo ra một khoảng trống thông tin giữa *"những gì người đọc biết khi đọc tiêu đề"* và *"những gì thực sự diễn ra trong bài báo"*.
  * **Các thủ pháp ngôn ngữ đặc trưng trong tiếng Việt:**
    1. **Đại từ chỉ định bỏ lửng (*Cataphoric / Deictic Reference*):** *"Người đàn ông **này**..."*, *"Làm ngay **thứ này**..."*, *"Cô gái nhận **kết đắng** vì..."* (giấu chủ thể thật để buộc người đọc phải bấm vào bài).
    2. **Từ vựng kích động cảm xúc mạnh (*Hyperbolic Words*):** *"bất ngờ"*, *"ngỡ ngàng"*, *"xôn xao"*, *"sốc"*, *"tiết lộ"*, *"bí mật"*, *"chưa từng có"*.
    3. **Lạm dụng dấu câu cảm thán & trích dẫn nửa vời:** Dấu chấm hỏi tu từ (`?`), dấu chấm than (`!`), dấu ba chấm bỏ lửng (`...`), hoặc nhặt một cụm từ kịch tính cho vào ngoặc kép (`"..."`).
    4. **Độ lệch ngữ nghĩa giữa Tiêu đề và Sapo (*Title–Lead Incongruence*):** Tiêu đề viết cực kỳ nghiêm trọng/gay cấn nhưng đoạn mở đầu (*Sapo / Lead paragraph*) lại nói về một chuyện bình thường (ví dụ: hậu trường đóng phim, mẹo vặt đời sống).

### 1.2. Tại sao phải kết hợp `Title + Sapo` thay vì chỉ dùng `Only Title`?
* Nếu mô hình **chỉ nhìn Tiêu đề (`Only Title`)**:
  * **Dễ bị Dương tính giả (*False Positive*):** Một tin thời sự/tai nạn/pháp luật nghiêm trọng (*"Cháy lớn tại xưởng gỗ..."*, *"Bắt khẩn cấp đối tượng..."*) có chứa từ ngữ mạnh nhưng là sự kiện thật 100%, tiêu đề phản ánh đúng nội dung Sapo $\rightarrow$ Đây là **Tin không giật tít (`non-clickbait`)**.
  * **Dễ bị Âm tính giả (*False Negative*):** Một tiêu đề viết ngắn gọn không có dấu `?` hay `!`, nhưng giấu chủ thể so với Sapo.
* Khi kết hợp **`Title + [SEP] + Lead Paragraph`**:
  * Mô hình có thêm **ngữ cảnh đối chứng** từ đoạn Sapo (vốn là đoạn tóm tắt chuẩn 5W1H: *Ai, Cái gì, Ở đâu, Khi nào, Tại sao* nằm ngay dưới tiêu đề bài báo) để kiểm tra xem tiêu đề có đang phóng đại hay giấu thông tin hay không.

---

## PHẦN 2: KIẾN THỨC VỀ TẬP DỮ LIỆU, DATA LEAKAGE & CHIA TẬP (EDA)

### 2.1. Quy mô và Cấu trúc Tập dữ liệu (`data/raw/clickbait_dataset_vietnamese.csv`)
* **Tổng số mẫu:** `3,414` bài báo tiếng Việt từ **8 nguồn tin điện tử**: `VnExpress`, `Báo Mới`, `Thanh Niên`, `24h`, `SaoStar`, `VietNamNet`, `Tuổi Trẻ`, `Kênh14` trải dài trên **13 chuyên mục**.
* **Phân bố nhãn (Imbalanced Dataset mức độ vừa):**
  * `Không giật tít (non-clickbait, label_id = 0)`: **2,349 mẫu (68.80%)**
  * `Giật tít (clickbait, label_id = 1)`: **1,065 mẫu (31.20%)**
  * Tỷ lệ mất cân bằng xấp xỉ **2.2 : 1**.
* **Các thống kê quan trọng từ EDA (`outputs/results/eda_statistics.json`):**
  * **Độ dài từ trung bình:**
    * Tiêu đề (`title`): Giật tít trung bình **13.98 từ** ($\pm 4.44$), Không giật tít **13.68 từ** ($\pm 3.64$).
    * Đoạn mở đầu (`lead_paragraph`): Giật tít trung bình **32.98 từ** ($\pm 12.73$), Không giật tít **36.72 từ** ($\pm 13.24$) $\rightarrow$ Tin không giật tít có Sapo dài và đầy đủ thông tin hơn.
  * **Tỷ lệ dấu câu trên tiêu đề:**
    * Dấu chấm hỏi (`?`): **20.38%** (Giật tít) vs **3.41%** (Không giật tít) $\rightarrow$ **Chênh lệch gấp 6 lần**.
    * Dấu ngoặc kép/đơn (`" / '`): **32.11%** (Giật tít) vs **11.66%** (Không giật tít) $\rightarrow$ **Chênh lệch gấp ~2.8 lần**.
    * Dấu chấm than (`!`): **1.78%** (Giật tít) vs **0.30%** (Không giật tít) $\rightarrow$ **Chênh lệch gấp 6 lần**.
    * Dấu ba chấm (`...`): **0.94%** (Giật tít) vs **0.21%** (Không giật tít).
    * Chứa con số (`0-9`): **31.36%** (Giật tít) vs **40.44%** (Không giật tít) $\rightarrow$ Tin chính thống hay chứa số liệu thống kê, ngày tháng, nghị định.

### 2.2. Hiện tượng Học tắt (*Shortcut Learning*), Thiên lệch Nguồn tin (*Source Bias*) & Rò rỉ Dữ liệu (*Data Leakage*)
* **Phát hiện từ EDA:** Tỷ lệ giật tít giữa các nguồn báo chênh lệch cực lớn:
  * **Nhóm báo giải trí/tổng hợp:** `Kênh14` (**76.92%** clickbait), `SaoStar` (**65.03%**), `24h` (**43.97%**).
  * **Nhóm báo thời sự chính luận:** `VnExpress` (**14.47%** clickbait), `Báo Mới` (**20.25%**), `Tuổi Trẻ` (**29.05%**), `Thanh Niên` (**29.86%**).
  * **Theo chuyên mục:** *Giải trí & Showbiz* (**70.59%** CB), *Sức khỏe & Đời sống* (**57.02%** CB) so với *Chính trị & An ninh* (**16.76%** CB).
* **Shortcut Learning là gì?**
  * Nếu ta đưa cột `source` (hoặc để nguyên chữ `"- Tạp chí SaoStar"` ở cuối tiêu đề) vào mô hình, mô hình sẽ đi đường tắt: cứ thấy chữ `Kênh14` hay `SaoStar` là phán `Giật tít`, cứ thấy `VnExpress` là phán `Không giật tít` mà không thèm học ngữ nghĩa tiếng Việt. Khi đem mô hình đó áp dụng sang một tờ báo mới (ví dụ `Bongdaplus` hay `Dân Trí`), mô hình sẽ thất bại.
* **Cách dự án xử lý và góc nhìn học thuật chuẩn xác:**
  1. **Loại bỏ hoàn toàn cột `source` và `category`** khỏi đầu vào huấn luyện.
  2. **Dùng Regex cắt sạch tên báo/tạp chí** bị dính ở cuối tiêu đề (`716 mẫu` của Báo Mới) và tiền tố viết tắt ở đầu Sapo (`TTO -`, `VOV.VN -`).
  3. **Kiểm tra trùng lặp văn bản (*Duplicate & Near-duplicate Check*):** Trong `3,414` bài báo, có **39 bài trùng tiêu đề (`1.14%`)**, trong đó có **13 tiêu đề xuất hiện ở cả tập Train và Test** do hiện tượng các báo điện tử đăng lại (*syndicate*) cùng một bản tin từ thông tấn xã. Ngoài ra, ngay cả khi đã xóa cột `source`, văn phong và chủ đề đặc trưng của từng báo vẫn tiềm ẩn trong câu chữ (*Implicit Source Leakage*). Đây là điểm bạn cần nêu rõ để hội đồng thấy tư duy nghiên cứu trung thực, sắc bén.

### 2.3. Kỹ thuật Chia Tập dữ liệu (`StratifiedShuffleSplit`)
* Chia theo tỷ lệ **70% Train (`2,389` mẫu) – 15% Validation (`512` mẫu) – 15% Test (`513` mẫu)** với `random_state = 42`.
* **Tại sao phải dùng `Stratified` (Phân tầng theo nhãn)?** Vì dữ liệu lệch nhãn (`68.8%` vs `31.2%`), chia phân tầng đảm bảo cả 3 tập Train, Validation và Test đều giữ nguyên tỷ lệ `68.8% Không giật tít : 31.2% Giật tít`:
  * **Train (`2,389`):** `1,644` Non-CB / `745` CB.
  * **Validation (`512`):** `352` Non-CB / `160` CB *(dùng để chọn siêu tham số, tìm ngưỡng cắt threshold tối ưu và dừng sớm Early Stopping)*.
  * **Test (`513`):** `353` Non-CB / `160` CB *(chỉ dùng 1 lần cuối cùng để chấm điểm khách quan)*.

---

## PHẦN 3: TIỀN XỬ LÝ VĂN BẢN ĐẶC THÙ TIẾNG VIỆT (`src/preprocess.py`)

Trong NLP thực tế, chất lượng dữ liệu (*Data-centric AI*) quyết định phần lớn độ chính xác. File `src/preprocess.py` giải quyết 4 bài toán kỹ thuật:

### 3.1. Khử lỗi dính địa danh đầu Sapo của VnExpress (`163 mẫu`)
* **Nguyên nhân lỗi:** Khi trình thu thập dữ liệu (*Web Scraper*) cào các bài báo của VnExpress, thẻ HTML `<span class="location-stamp">Nghệ An</span>` nằm sát nội dung đoạn mở đầu mà không có dấu cách, tạo ra chuỗi bị dính chữ: `"Nghệ AnCảng hàng không quốc tế Vinh..."`, `"Hà NộiBị cáo Nguyễn Văn A..."`, `"TP HCMCông an quận..."`.
* **Hậu quả nếu không sửa:** Bộ tách từ `PyVi` sẽ coi `"AnCảng"` hoặc `"NộiBị"` là một từ lạ (`<UNK> - Out-Of-Vocabulary`), làm hỏng hoàn toàn cấu trúc câu đầu tiên của Sapo.
* **Giải pháp kỹ thuật (`fix_vnexpress_glued_location`):**
  * Xây dựng biểu thức chính quy `VNEXPRESS_GLUED_LOCATION_PATTERN` nhận diện cụm địa danh viết hoa ở đầu câu bị dính liền vào một chữ cái in hoa hoặc dấu ngoặc kép/chữ số ngay phía sau, tự động chèn dấu cách tách đôi.
  * **Danh sách trắng bảo vệ thương hiệu (`CAMEL_BRANDS` gồm 35 từ):** Các tên thương hiệu công nghệ/ngân hàng có chữ hoa ở giữa từ (*CamelCase*) như `TikTok`, `ChatGPT`, `OpenAI`, `VinFast`, `iPhone`, `YouTube`, `Vietcombank`... nếu dùng Regex tách chữ hoa thông thường sẽ bị tách nhầm thành `"Tik Tok"` hoặc `"Vin Fast"`. Hàm tiền xử lý kiểm tra `CAMEL_BRANDS` trước để bảo toàn 100% các thương hiệu này.

### 3.2. Khử hậu tố nguồn báo ở cuối Tiêu đề Báo Mới (`716 mẫu`)
* **Nguyên nhân lỗi:** Báo Mới là trang tổng hợp tin tức tự động gắn hậu tố `"- Báo..."`, `"- Tạp chí SaoStar"`, `"- Chuyên trang..."`, `"- VOV"` vào cuối tiêu đề.
* **Giải pháp (`strip_baomoi_source_suffix`):** Dùng Regex cắt bỏ toàn bộ phần hậu tố này trên `716` mẫu tiêu đề, vừa làm sạch câu vừa ngăn chặn mô hình học vẹt tên báo.

### 3.3. Phân biệt Chuẩn hóa Unicode (`NFC`) và Chuẩn hóa Vị trí Dấu thanh Tiếng Việt
Hai khái niệm này hoàn toàn khác nhau về mặt kỹ thuật máy tính (rất hay được hỏi khi bảo vệ):
1. **Chuẩn hóa bảng mã Unicode (`unicodedata.normalize("NFC", text)`):**
   * Trong Unicode, một chữ cái tiếng Việt có dấu (ví dụ chữ `ệ`) có thể được lưu dưới 2 dạng mã byte:
     * **NFC (*Normalization Form C - Dựng sẵn*):** `ệ` là **1 ký tự duy nhất** (`U+1EC7`).
     * **NFD (*Normalization Form D - Tổ hợp*):** `ệ` bị tách thành **2 ký tự rời** gồm chữ `ê` (`U+00EA`) + dấu nặng rời (`U+0323`).
   * Nếu hai bài báo dùng 2 chuẩn khác nhau, máy tính sẽ coi `"nghệ_thuật"` (NFC) và `"nghệ_thuật"` (NFD) là 2 từ hoàn toàn khác nhau! Đưa toàn bộ về **NFC** giúp đồng nhất mã byte.
2. **Chuẩn hóa vị trí đặt dấu thanh (`TONE_NORM_MAP`):**
   * Tiếng Việt có 2 quy tắc đặt dấu thanh trên các nguyên âm đôi `oa, oe, uy`: kiểu cũ (`oà, oá, uỳ, uý`) và kiểu mới (`òa, óa, ùy, úy`). Unicode `NFC` **không** tự chuyển đổi giữa `oà` và `òa` vì cả hai đều là ký tự NFC hợp lệ.
   * Do từ điển huấn luyện của `PyVi` và `PhoBERT` sử dụng chuẩn `òa, óa, ùy, úy`, dự án dùng bảng ánh xạ `TONE_NORM_MAP` để đồng nhất toàn bộ về một chuẩn duy nhất trước khi tách từ.

### 3.4. Tách từ Tiếng Việt (*Word Segmentation*) bằng `PyVi`
* **Tại sao tiếng Việt phải tách từ (Word Segmentation) còn tiếng Anh thì không?**
  * Trong tiếng Anh, khoảng trắng (`space`) chính là ranh giới từ (*word boundary*).
  * Trong tiếng Việt, khoảng trắng chỉ là ranh giới giữa các **âm tiết (*syllables*)**, không phải ranh giới **từ (*words*)**. Ví dụ: `"bất ngờ"` gồm 2 âm tiết nhưng là 1 từ ghép; nếu cắt theo khoảng trắng thành `"bất"` và `"ngờ"` rời nhau thì nghĩa bị biến đổi (vì `"bất"` còn nằm trong *"bất động sản"*, *"bất cập"*).
* **Cơ chế của `PyVi (ViTokenizer.tokenize)`:** Sử dụng mô hình tuyến tính dựa trên *Conditional Random Fields (CRF)* để nối các âm tiết thuộc cùng một từ ghép bằng dấu gạch dưới `_`:
  * `"Tiết lộ bí mật bất ngờ"` $\rightarrow$ `"Tiết_lộ bí_mật bất_ngờ"`.

---

## PHẦN 4: KỸ THUẬT ĐẶC TRƯNG (FEATURE ENGINEERING) & ABLATION STUDY

### 4.1. Biểu diễn Văn bản bằng Không gian TF-IDF Kép (`Dual-Space TF-IDF`, 16,000 chiều)
* **Công thức TF-IDF (*Term Frequency – Inverse Document Frequency*) với `sublinear_tf=True`:**
  $$\text{tf}(t, d) = 1 + \ln(\text{count}(t, d)), \quad \text{idf}(t) = \ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
  * Dùng `sublinear_tf=True` (hàm logarit) để một từ xuất hiện 5 lần trong đoạn Sapo dài không bị phóng đại trọng số gấp 5 lần so với xuất hiện 1 lần trên tiêu đề ngắn.
  * Dùng `ngram_range=(1, 2)` để bắt cả từ đơn (*unigram*: `bất_ngờ`, `khiến`) lẫn cụm 2 từ (*bigram*: `chưa từng`, `thứ này`, `người này`, `gây sốt`).
  * Giữ lại các dấu câu cảm thán (`?`, `!`, `"`, `'`) trong `token_pattern=r"(?u)\b\w[\w_]*\b|[?!\"']"` vì dấu câu là tín hiệu vàng của tin giật tít.
* **Tại sao phải thiết kế `Dual-Space TF-IDF` (Không gian TF-IDF Kép) trong `src/models_ml.py`?**
  * Nếu gộp chung `Title + Sapo` vào 1 vector TF-IDF duy nhất: Vì đoạn Sapo dài trung bình `~35 từ` còn Tiêu đề chỉ dài `~14 từ`, khi chuẩn hóa L2 (*L2-normalization*) của TF-IDF, các từ xuất hiện trong Sapo sẽ làm **loãng (dilute)** trọng số của các từ khóa giật tít nằm trên Tiêu đề.
  * Do đó, dự án tách thành 2 không gian ghép nối với nhau (`scipy.sparse.hstack`):
    1. `X_tfidf_combined` (`10,000` chiều): TF-IDF trên chuỗi `Title + [SEP] + Sapo`.
    2. `X_tfidf_title * 1.25` (`6,000` chiều): TF-IDF tính riêng trên `Title` và **nhân hệ số nhấn mạnh `1.25`**.

### 4.2. Nhóm 12 Đặc trưng Thủ công (`HANDCRAFTED_FEATURE_COLS`)
Được trích xuất trực tiếp từ văn bản gốc trước khi hạ chữ thường (để bắt được chữ IN HOA) và chuẩn hóa về đoạn `[0, 1]` bằng `MinMaxScaler`:
1. `title_word_count`: Số từ của tiêu đề.
2. `title_char_len`: Độ dài ký tự của tiêu đề.
3. `lead_word_count`: Số từ của đoạn Sapo.
4. `lead_char_len`: Độ dài ký tự của đoạn Sapo.
5. `title_lead_len_ratio`: Tỷ lệ độ dài tiêu đề trên độ dài Sapo.
6. `num_all_caps_words`: Số lượng từ viết IN HOA toàn bộ (ví dụ: `SỐC`, `NÓNG`, `KHẨN`).
7. `upper_char_ratio`: Tỷ lệ chữ cái viết hoa trên tổng số chữ cái của tiêu đề.
8. `has_question`: Có chứa dấu chấm hỏi `?` hay không (`0` hoặc `1`).
9. `has_exclamation`: Có chứa dấu chấm than `!` hay không (`0` hoặc `1`).
10. `has_quotes`: Có chứa dấu ngoặc kép/đơn trích dẫn `"..."` hay không (`0` hoặc `1`).
11. `has_ellipsis`: Có chứa dấu ba chấm lửng `...` hay không (`0` hoặc `1`).
12. `has_number`: Có chứa chữ số `0–9` hay không (`0` hoặc `1`).

$\Rightarrow$ Tổng số chiều đặc trưng đầu vào cho mô hình ML là: **$10,000 + 6,000 + 12 = 16,012$ chiều**.

### 4.3. Kết quả Ablation Study (Nghiên cứu Loại bỏ Thành phần)
Bảng thực nghiệm dưới đây (lưu tại `outputs/results/ablation_study.json` trên tập Test $N = 513$) chứng minh sự đóng góp của từng nhóm đặc trưng:

| Cấu hình Thực nghiệm | LR Accuracy | LR F1 (CB) | LR Macro-F1 | LR ROC-AUC | SVM Accuracy | SVM Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Only Title** *(Chỉ TF-IDF 10k)* | 78.95% | 69.14% | 76.58% | 85.64% | 79.92% | 76.41% |
| **1b. Only Title + 12 Hand-crafted** | 79.73% | 69.59% | 77.19% | 85.00% | 80.90% | 77.27% |
| **2. Title + Sapo** *(Dual TF-IDF 16k)* | 79.34% | 69.89% | 77.08% | **86.54%** | 80.12% | 76.43% |
| **3. Title + Sapo + 12 Hand-crafted (Full)** | **81.09%** | **71.72%** | **78.76%** | 86.34% | **81.29%** | **77.97%** |

* **Nhận xét cốt lõi:**
  * Thêm `Sapo` (Cấu hình 2) giúp tăng ngay **ROC-AUC** lên mức đỉnh `86.54%` (nhờ có ngữ cảnh Sapo để phân biệt tin thật kịch tính với tin câu view).
  * Kết hợp cả `Title + Sapo + 12 Hand-crafted` (Cấu hình 3) giúp Logistic Regression tăng **+2.18% Macro-F1** (`76.58% → 78.76%`) và Linear SVM tăng **+1.56% Macro-F1** (`76.41% → 77.97%`).

---

## PHẦN 5: KIẾN TRÚC CÁC MÔ HÌNH & KẾT QUẢ THỰC NGHIỆM

### 5.1. Nhóm 1 — Các Mô hình Học máy Truyền thống (`src/models_ml.py`)
1. **Multinomial Naive Bayes (`alpha=0.5`):**
   * Dựa trên định lý Bayes với giả định các từ độc lập có điều kiện và làm mượt Laplace/Lidstone ($\alpha = 0.5$). Rất nhanh nhưng giả định độc lập từ làm hạn chế khả năng bắt tương tác giữa các đặc trưng.
2. **Logistic Regression (`C=1.5, class_weight="balanced"`):**
   * Tính xác suất qua hàm Sigmoid: $P(y=1 \mid x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$.
   * Dùng `class_weight="balanced"` để tự động phạt nặng gấp $\sim 2.2$ lần nếu dự đoán sai lớp thiểu số (`Giật tít`), giúp đạt **Recall trên lớp Giật tít cao nhất toàn dự án (76.88%)** và **Macro-F1 cao nhất (78.76%)**.
3. **Linear SVM (`LinearSVC(C=0.8, class_weight="balanced")` + `CalibratedClassifierCV(cv=5)`):**
   * Tìm siêu phẳng (*hyperplane*) có lề cực đại (*Maximum Margin*) phân tách 2 lớp trong không gian `16,012` chiều.
   * Vì `LinearSVC` chỉ xuất ra khoảng cách tới siêu phẳng chứ không xuất ra xác suất $[0, 1]$, dự án bọc ngoài bằng `CalibratedClassifierCV(cv=5)` (hiệu chuẩn Platt Scaling trên 5-fold cross-validation) để thu được xác suất chuẩn phục vụ Ensemble.
   * Đạt **Accuracy cao nhất toàn dự án (81.29%)** và **F1 trên lớp Không giật tít cao nhất (86.52%)**.
4. **Random Forest (`300 cây, class_weight="balanced_subsample"`):**
   * Mô hình cây quyết định tổ hợp đạt **Precision trên lớp Giật tít cao nhất (71.43%)**, nhưng do ma trận TF-IDF `16,012` chiều rất thưa (*sparse*), cây quyết định cắt theo từng trục đơn lẻ nên Recall thấp hơn (`62.50%`).

### 5.2. Nhóm 2 — Các Mô hình Học sâu PyTorch (`src/models_dl.py`)
Cả hai mô hình đều sử dụng bộ từ điển `Vocabulary` (`15,000` từ, `max_len=128`), tầng nhúng từ `nn.Embedding(15000, 128)` đi qua `SpatialDropout1D(0.2)` (loại bỏ ngẫu nhiên toàn bộ kênh đặc trưng dọc theo chuỗi từ để chống overfitting), và đều có một nhánh phụ `nn.Linear(12, 16) + ReLU` để tích hợp 12 đặc trưng thủ công trước khi đưa vào tầng phân loại:
1. **BiLSTM (`BiLSTMClassifier`):**
   * Đưa chuỗi vector qua mạng `LSTM(input=128, hidden=64, bidirectional=True)` đọc xuôi và đọc ngược câu văn để tạo vector ngữ cảnh `128` chiều tại mỗi vị trí từ.
   * Áp dụng **Masked GlobalMaxPooling1D** (che các vị trí `<PAD>` bằng $-10^9$ rồi lấy giá trị cực đại dọc theo chiều thời gian) ghép với 16 chiều đặc trưng thủ công $\rightarrow$ `Linear(144 → 64 → 1)`.
   * Đạt **Recall lớp Giật tít rất tốt (73.75%)** và **ROC-AUC 85.19%**.
2. **TextCNN (`TextCNNClassifier`):**
   * Áp dụng 3 lớp tích chập 1 chiều song song `nn.Conv1d` với kích thước cửa sổ trượt `filter_sizes = (2, 3, 5)` (tương ứng bắt các cụm 2 từ, 3 từ và 5 từ liên tiếp), mỗi bộ lọc có `64` kênh $\rightarrow$ `GlobalMaxPooling1D` thu được $64 \times 3 = 192$ chiều ghép với 16 chiều đặc trưng thủ công $\rightarrow$ `Linear(208 → 64 → 1)`.
   * Đạt **Accuracy 78.56%** và **Macro-F1 75.51%**.
* **Hàm mất mát & Tối ưu hóa cho DL:**
  * Dùng `BCEWithLogitsLoss(pos_weight=2.206)` để xử lý lệch nhãn, tối ưu bằng `AdamW (lr=1.5e-3, weight_decay=1e-4)`, cắt gradient `clip_grad_norm_(max_norm=1.0)`, giảm tốc độ học tự động `ReduceLROnPlateau` và quét ngưỡng quyết định tối ưu (`best_threshold` trong đoạn `[0.35, 0.65]`) trên tập Validation.

### 5.3. Nhóm 3 — Pre-trained Transformer (`PhoBERT-base-v2` trong `src/models_bert.py`)
* Sử dụng mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt `vinai/phobert-base-v2` (kiến trúc RoBERTa 12 tầng Transformer Encoder, 135 triệu tham số).
* **Chiến lược Layer-wise Freezing:** Đóng băng tầng `Embeddings` và **6 tầng Encoder đầu tiên** (giữ nguyên tri thức từ vựng - cú pháp tiếng Việt nền tảng), chỉ tinh chỉnh **6 tầng Encoder trên cùng + Classification Head** với `Dynamic Padding` theo từng batch, `AdamW (lr=4e-5)`, `Linear Warmup 10%` và trọng số lớp cân bằng căn bậc hai $\sqrt{N_{\text{neg}} / N_{\text{pos}}}$.
* **Tại sao không đưa PhoBERT vào hệ thống Ensemble chạy thời gian thực?**
  * Trọng số của PhoBERT nặng `~540 MB` và khi chạy suy diễn trên CPU phổ thông tốn hàng trăm mili-giây cho mỗi bài báo. Trong khi đó, tổ hợp 4 mô hình `SVM + LR + BiLSTM + TextCNN` chỉ tốn **0.523 ms/mẫu** và đã đạt **ROC-AUC 86.41%**, hoàn toàn đáp ứng yêu cầu quét hàng chục tiêu đề cùng lúc trên trình duyệt.

### 5.4. Nhóm 4 — Tổ hợp Lai (`Hybrid Ensemble` trong `src/inference.py`)
* Kết hợp xác suất dự đoán của cả 2 trường phái (Không gian вектор thưa TF-IDF của ML + Biểu diễn chuỗi liên tục của DL) theo công thức **Weighted Soft-Voting**:
  $$P_{\text{ensemble}} = 0.35 \cdot P_{\text{LinearSVM}} + 0.20 \cdot P_{\text{LogReg}} + 0.22 \cdot P_{\text{BiLSTM}} + 0.23 \cdot P_{\text{TextCNN}}$$
* **Tổng trọng số:** Nhánh ML chiếm `55%` ($0.35 + 0.20$), nhánh DL chiếm `45%` ($0.22 + 0.23$).
* **Kết quả:** Đạt **ROC-AUC cao nhất toàn cục: 86.41%** (vượt qua mọi mô hình đơn lẻ).

---

## PHẦN 6: GIẢI THÍCH AI (XAI) & TÁI CẤU TRÚC TIÊU ĐỀ (DE-CLICKBAIT)

### 6.1. Giải thích Mô hình Toàn cục (*Global Feature Importance*) vs Cục bộ (*Local Explanation*)
1. **Giải thích Toàn cục (*Global*):** Nhìn vào bộ trọng số $\mathbf{w}$ đã học của Logistic Regression và Linear SVM trên toàn bộ tập huấn luyện (`outputs/results/feature_importance.json`):
   * **Top đặc trưng đẩy mạnh nhất về nhãn `Giật tít` ($w > 0$):**
     * Từ vựng: `bất_ngờ (+2.724)`, `khiến (+2.176)`, `gây (+1.836)`, `là (+1.683)`, `tiết_lộ (+1.569)`, `cũng (+1.513)`, `bạn (+1.484)`, `mạng (+1.415)`, `bí_mật (+1.257)`, `chưa từng (+1.178)`, `bí_quyết (+1.173)`.
     * Đặc trưng thủ công: `[HC]_has_question (+2.156)`, `[HC]_has_ellipsis (+1.680)`, `[HC]_has_quotes (+1.522)`, `[HC]_has_exclamation (+1.496)`.
   * **Top đặc trưng đẩy mạnh nhất về nhãn `Không giật tít` ($w < 0$):**
     * Từ vựng: `báo_chí (-1.198)`, `đà_nẵng (-1.125)`, `việt_nam (-1.112)`, `nguy_kịch (-1.076)`, `tp (-1.025)`, `theo (-0.974)`, `vụ (-0.963)`, `thủ_tướng (-0.863)`, `tai_nạn (-0.824)`.
2. **Giải thích Cục bộ cho Từng Bài báo (*Local Token Attribution* trong `src/inference.py`):**
   * Khi người dùng đưa vào một bài báo cụ thể, hàm `highlight_tokens()` tách từ bằng `PyVi`, tính giá trị $\text{TF-IDF}(t_i)$ của từng từ trong bài đó rồi nhân với trọng số mô hình $w_{t_i}$:
     $$\text{Contribution}(t_i) = \text{TF-IDF}(t_i) \times \left(0.5 \cdot w_{t_i}^{\text{LR}} + 0.5 \cdot w_{t_i}^{\text{SVM}}\right)$$
   * Nhờ vậy, hệ thống chỉ tô màu những từ thực sự xuất hiện trong bài báo đó và đóng góp trực tiếp vào quyết định của mô hình.

### 6.2. Cơ chế "Tháo tít" (`De-clickbait` — `rewrite_neutral_title`)
* **Nguyên tắc cốt lõi:** De-clickbait là một bước **hậu xử lý ngôn ngữ (*Post-processing*)**, không phải XAI, và **tuyệt đối không được bịa thêm thông tin ngoài bài báo**.
* **Thuật toán 2 bước trong `src/inference.py`:**
  * **Bước 1 (Khi bài báo có đoạn Sapo):** Trích xuất câu sự kiện đầu tiên từ đoạn Sapo (cắt bỏ tiền tố địa danh/tên báo, giữ độ dài `15–145` ký tự). Vì câu đầu tiên của Sapo trong báo chí luôn chứa thông tin thật trả lời trực tiếp cho tiêu đề, cách trích xuất nguyên văn (*Extractive*) này đảm bảo **0% ảo giác (*Zero Hallucination*)**.
  * **Bước 2 (Khi chỉ có Tiêu đề, không có Sapo):** Dùng bộ quy tắc Regex cắt bỏ các cụm từ mồi cảm thán ở đầu câu (*"Bất ngờ trước..."*, *"Hé lộ..."*, *"Sự thật đằng sau..."*) và ở cuối câu (*"...khiến ai cũng ngỡ ngàng"*, *"...gây xôn xao mạng xã hội"*), đồng thời thay dấu `?!...` ở cuối câu bằng dấu chấm trung lập.

---

## PHẦN 7: KIẾN TRÚC TRIỂN KHAI WEB GRADIO & CHROME EXTENSION 3 TẦNG

### 7.1. Kiến trúc Giao diện 3 Tầng theo Triết lý *Ambient Computing*
* **Tầng 1 — In-Page Ambient Badges (`extension/content.js`):**
  * Tự động quét trang chủ báo điện tử, gắn nhãn ngay cạnh tiêu đề: `🟢 Không giật tít` ($<45\%$), `🟡 Chú ý` ($45\%–64\%$), `🔴 Giật tít` ($\ge 65\%$).
  * Khi rê chuột (*hover*) vào nhãn: Hiển thị khung **Tooltip** chứa đoạn mở đầu Sapo và tiêu đề đã tháo tít để người đọc biết ngay nội dung thật mà không cần click vào bài câu view.
* **Tầng 2 — Reading Assistant Floating Pill (`extension/content.js`):**
  * Xuất hiện ở góc dưới bên phải khi người dùng đang đọc một bài báo chi tiết, hiển thị độ lệch ngữ nghĩa giữa `Title` và `Sapo`, nút `✨ Tháo tít trên bài` (ghi đè tiêu đề trung lập trực tiếp lên DOM trang web) và nút `🚩 Báo nhầm lẫn`.
* **Tầng 3 — Action Popup (`extension/popup.html` & `popup.js`):**
  * Bảng điều khiển khi bấm vào icon Extension: Thống kê số tiêu đề đã quét, số bài giật tít đã cảnh báo, số phút tiết kiệm được (`số bài CB × 1.5 phút`), chỉ số **Độ sạch trang báo (%)**, công tắc **Tự động làm mờ (Blur)** bài giật tít và thanh trượt chỉnh độ nhạy bộ lọc.

### 7.2. Cơ chế Suy diễn 2 Tầng (*Two-Tier Inference*) trong `extension/background.js`
1. **Tầng 1 — Server Hybrid Ensemble (`Online Mode`):**
   * Khi bạn bật `app.py` (`http://127.0.0.1:7860`), Service Worker `background.js` gửi lô tối đa `25` bài báo qua endpoint `POST /api/predict_batch` (timeout `2.8 giây`) để chạy mô hình **Hybrid Ensemble 4 models** đầy đủ bằng Python/PyTorch/Scikit-learn.
2. **Tầng 2 — Local ML Engine (`Offline Fallback Mode`):**
   * Nếu bạn tắt máy bật lại mà không chạy Python (hoặc server không phản hồi), `background.js` tự động chuyển sang chạy hàm `predictItemLocally()` trực tiếp trên trình duyệt bằng JavaScript.
   * Hàm này nạp file `extension/model_weights.json` (được xuất tự động từ mô hình `Logistic Regression` trong `src/inference.py`), bên trong chứa **800 n-gram trọng số dương lớn nhất + 800 n-gram trọng số âm lớn nhất (tổng cộng 1,600 n-gram)**, kết hợp với 5 luật cộng điểm dấu câu (`?`, `!`, `...`, `"..."`, `ALL CAPS`) và kiểm tra độ trùng lặp từ khóa giữa `Title` và `Sapo`.
   * Nhờ vậy, Extension **không bao giờ bị chết** khi tắt server, với tốc độ suy diễn nội bộ `< 1 ms/bài`.
