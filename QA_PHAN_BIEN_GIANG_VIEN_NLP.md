# BỘ CÂU HỎI & TRẢ LỜI PHẢN BIỆN GIẢ LẬP (Q&A VỚI GIẢNG VIÊN BỘ MÔN NLP)
## Đề tài: Hệ thống Phát hiện & Giải thích Tin Giật Tít Tiếng Việt (Vietnamese Clickbait Detection)

Tài liệu này mô phỏng **20 câu hỏi phản biện chuyên sâu và sắc bén nhất** mà các giảng viên hướng dẫn hoặc hội đồng chấm đồ án Xử lý Ngôn ngữ Tự nhiên (NLP) thường đặt ra, đi kèm **câu trả lời mẫu chuẩn học thuật, bám sát 100% số liệu và mã nguồn trong project của bạn**.

---

## CHỦ ĐỀ 1: ĐỊNH NGHĨA BÀI TOÁN, DỮ LIỆU & DATA LEAKAGE

### Câu 1: "Em hãy phân biệt rõ bài toán Phát hiện Tin giật tít (Clickbait Detection) của nhóm với bài toán Phát hiện Tin giả (Fake News Detection). Mô hình của em có phát hiện được tin giả không?"
> **Trả lời mẫu:**
> Thưa thầy/cô, hai bài toán này khác nhau hoàn toàn về bản chất ngôn ngữ học và mục tiêu kiểm chứng:
> 1. **Phát hiện Tin giả (*Fake News Detection*)** kiểm tra tính đúng/sai sự thật (*factuality*) của thông tin so với thực tế khách quan.
> 2. **Phát hiện Tin giật tít (*Clickbait Detection*)** đánh giá **hình thức truyền đạt ngôn ngữ** và **khoảng trống thông tin (*Curiosity Gap*)** giữa Tiêu đề (`title`) và Đoạn mở đầu (`lead_paragraph`). Một bài báo trên nguồn tin chính thống có nội dung hoàn toàn thật nhưng nếu đặt tiêu đề bỏ lửng đại từ (*"Người đàn ông này..."*), cường điệu hóa hoặc lệch ngữ nghĩa so với đoạn Sapo thì vẫn là **Tin giật tít (`clickbait`)**.
>
> Vì vậy, hệ thống của nhóm tập trung phân loại `Giật tít` vs `Không giật tít` dựa trên cấu trúc ngôn ngữ và sự tương hợp `Title – Sapo`, chứ không kết luận bài báo đó là tin giả hay tin thật. Đây cũng là lý do nhóm chuẩn hóa toàn bộ nhãn hiển thị trên giao diện thành **`Không giật tít`** và **`Giật tít`**.

---

### Câu 2: "Trong Slide 3, nhóm nói đã loại bỏ cột `source` để chống Data Leakage. Vậy chỉ cần xóa cột `source` trong file CSV là đã hết hoàn toàn Data Leakage chưa?"
> **Trả lời mẫu:**
> Dạ thưa thầy/cô, **chỉ xóa cột `source` là chưa đủ để loại bỏ hoàn toàn Data Leakage**, mà mới chỉ giải quyết được phần **rò rỉ nhãn trực tiếp (*Explicit Source Shortcut*)**. Trong quá trình thực nghiệm, nhóm đã phát hiện và xử lý **3 tầng rò rỉ dữ liệu**:
> 1. **Tầng 1 — Rò rỉ trực tiếp trong văn bản:** Trong dữ liệu gốc có tới **716 mẫu của Báo Mới** chứa hậu tố tên báo ngay cuối tiêu đề (ví dụ: `"... - Tạp chí SaoStar"`, mà SaoStar có tới `65.03%` là clickbait) và các tiền tố `TTO -`, `VOV.VN -` ở đầu Sapo. Nhóm đã viết Regex trong `src/preprocess.py` để cắt bỏ triệt để toàn bộ các dấu hiệu này.
> 2. **Tầng 2 — Trùng lặp bản tin đăng lại (*Syndicated News Duplicates*):** Nhóm đã kiểm tra trùng lặp trên toàn bộ `3,414` bài và phát hiện có **39 bài trùng tiêu đề (`1.14%`)**, trong đó có **13 tiêu đề xuất hiện giao thoa giữa tập Train và tập Test** do các báo đăng lại cùng một bản tin.
> 3. **Tầng 3 — Thiên lệch văn phong và chuyên mục ngầm (*Implicit Source & Topic Bias*):** Kênh14 (`76.92%` CB) chủ yếu viết về Giải trí (`70.59%` CB), trong khi VnExpress (`14.47%` CB) nhiều tin Thời sự/Chính trị (`16.76%` CB). Dù đã xóa tên báo, từ vựng chủ đề vẫn mang dấu ấn của từng tòa soạn. Do đó, nhóm đã đề xuất trong phần hướng phát triển phương pháp đánh giá **Source Holdout (Group Split theo nguồn báo)** để kiểm chứng khả năng tổng quát hóa sang nguồn báo hoàn toàn mới.

---

### Câu 3: "Tập dữ liệu của nhóm có tỷ lệ 68.8% Không giật tít và 31.2% Giật tít. Nhóm xử lý mất cân bằng dữ liệu (Class Imbalance) như thế nào ở cả bước chia dữ liệu, huấn luyện và đánh giá?"
> **Trả lời mẫu:**
> Dạ nhóm xử lý đồng bộ ở cả 3 khâu:
> 1. **Ở khâu chia dữ liệu:** Sử dụng `StratifiedShuffleSplit (70/15/15, random_state=42)` để bảo toàn chính xác tỷ lệ `68.8% : 31.2%` trên cả 3 tập Train (`2,389` mẫu), Validation (`512` mẫu) và Test (`513` mẫu: `353` Non-CB / `160` CB).
> 2. **Ở khâu hàm mất mát khi huấn luyện:**
>    * Với `Logistic Regression` và `LinearSVC`: Thiết lập `class_weight="balanced"` (tự động nhân trọng số phạt lớp `Giật tít` cao gấp $\sim 2.21$ lần lớp `Không giật tít`).
>    * Với `BiLSTM` và `TextCNN`: Sử dụng `nn.BCEWithLogitsLoss(pos_weight=2.2067)` kết hợp tối ưu ngưỡng quyết định (`threshold tuning` trên lưới `[0.35, 0.65]`) theo Validation Macro-F1.
> 3. **Ở khâu chỉ số đánh giá:** Không chỉ nhìn vào `Accuracy` mà lấy **Macro-F1** (trung bình cộng không trọng số giữa `F1(Clickbait)` và `F1(Non-Clickbait)`), **Precision/Recall riêng trên lớp Clickbait** và **ROC-AUC** làm thước đo chính.

---

## CHỦ ĐỀ 2: TIỀN XỬ LÝ TIẾNG VIỆT & KỸ THUẬT ĐẶC TRƯNG

### Câu 4: "Ở Slide 4, lỗi dính địa danh của VnExpress ('Nghệ AnCảng hàng không...') nếu em dùng Regex thấy chữ thường đứng trước chữ hoa là tách ra, vậy gặp từ 'iPhone', 'TikTok', 'VinFast', 'ChatGPT' thì nó có bị tách nhầm thành 'i Phone', 'Tik Tok' không?"
> **Trả lời mẫu:**
> Dạ thầy/cô hỏi đúng điểm khó nhất khi nhóm viết hàm `fix_vnexpress_glued_location()` trong `src/preprocess.py` ạ!
> Nếu chỉ dùng Regex tách giữa chữ thường và chữ hoa thì chắc chắn các từ *CamelCase* sẽ bị hỏng. Để giải quyết triệt để:
> 1. Nhóm chỉ giới hạn Regex ở **vị trí đầu chuỗi đoạn Sapo (`^`)** (nơi VnExpress đặt thẻ địa danh), không tách bừa bãi ở giữa câu.
> 2. Nhóm xây dựng danh sách trắng **`CAMEL_BRANDS` gồm 35 thương hiệu phổ biến** trong tập dữ liệu (`TikTok`, `ChatGPT`, `OpenAI`, `VietNamNet`, `VnExpress`, `VinFast`, `iPhone`, `iPad`, `iOS`, `MacBook`, `YouTube`, `Vietcombank`, `Techcombank`...). Trước khi tách, thuật toán kiểm tra nếu tiền tố khớp với danh sách trắng này thì sẽ giữ nguyên 100%. Kết quả là sửa chính xác **163 mẫu lỗi dính địa danh của VnExpress** mà không làm hỏng bất kỳ tên thương hiệu nào.

---

### Câu 5: "Chuẩn hóa Unicode NFC và chuẩn hóa kiểu gõ dấu tiếng Việt ('oà' -> 'òa') khác gì nhau? Tại sao phải làm cả hai trước khi đưa vào PyVi hoặc PhoBERT?"
> **Trả lời mẫu:**
> Dạ hai bước này giải quyết hai tầng biểu diễn khác nhau của văn bản tiếng Việt:
> 1. **Chuẩn hóa `Unicode NFC` (`unicodedata.normalize('NFC')`):** Giải quyết sự khác biệt ở **tầng mã hóa ký tự (byte-level)**. Một chữ tiếng Việt có dấu nếu copy từ nguồn dùng bảng mã tổ hợp `NFD` sẽ bị tách thành 2 ký tự (chữ cái gốc + dấu thanh rời), còn chuẩn dựng sẵn `NFC` gộp thành 1 ký tự duy nhất. Nếu không đưa về `NFC`, cùng một từ `"bất_ngờ"` nhưng ở 2 chuẩn NFC và NFD sẽ bị `TfidfVectorizer` coi là 2 từ hoàn toàn khác nhau.
> 2. **Chuẩn hóa vị trí đặt dấu thanh (`TONE_NORM_MAP`):** Giải quyết sự khác biệt ở **tầng chính tả tiếng Việt** trên các nguyên âm đôi `oa, oe, uy` (`oà → òa`, `oá → óa`, `uỳ → ùy`, `uý → úy`). Cả `"hoà"` và `"hòa"` đều là ký tự `NFC` hợp lệ nên hàm `unicodedata.normalize` không tự đổi cho nhau được. Vì từ điển của `PyVi` và `PhoBERT` được huấn luyện theo chuẩn `"hòa, thúy"`, nhóm phải chuẩn hóa thống nhất dấu thanh trước khi tách từ để tránh sinh ra từ ngoài từ điển (*Out-Of-Vocabulary*).

---

### Câu 6: "Tại sao trong `src/models_ml.py`, khi dùng đầu vào `Title + Sapo`, nhóm không đưa chung vào 1 bộ `TfidfVectorizer` mà lại tách thành 2 bộ TF-IDF rồi nhân hệ số `1.25` cho nhánh Title?"
> **Trả lời mẫu:**
> Dạ lý do xuất phát từ toán học của phép chuẩn hóa độ dài L2 (*L2-normalization*) trong TF-IDF và thống kê độ dài văn bản ở bước EDA:
> * Trên tập dữ liệu của nhóm, `Title` chỉ dài trung bình **13.98 từ**, trong khi đoạn `Sapo` dài trung bình **32.98 – 36.72 từ** (dài gấp ~2.5 lần).
> * Nếu gộp chung `Title + Sapo` vào một vector TF-IDF duy nhất, tổng bình phương trọng số của các từ trong đoạn Sapo dài sẽ áp đảo và làm **loãng (*dilute*)** giá trị TF-IDF của các từ khóa giật tít quan trọng nằm trên tiêu đề.
> * Vì vậy, nhóm thiết kế **Dual-Space TF-IDF (`16,000` chiều)**: ghép nối `TF-IDF(Title + [SEP] + Sapo)` (`10,000` chiều) với một không gian riêng `TF-IDF(Title)` (`6,000` chiều) được nhân hệ số khuếch đại **`× 1.25`**. Thực nghiệm Ablation Study chứng minh thiết kế này kết hợp với 12 đặc trưng thủ công giúp tăng Macro-F1 của Logistic Regression từ `76.58%` lên **`78.76%` (`+2.18%`)**.

---

### Câu 7: "Tại sao nhóm lại giữ nguyên dấu chấm hỏi `?`, dấu chấm than `!` và dấu ngoặc kép `\"` trong biểu thức `token_pattern` của TF-IDF thay vì xóa hết dấu câu như các bài toán phân loại chủ đề thông thường?"
> **Trả lời mẫu:**
> Dạ trong bài toán phân loại chủ đề (*Topic Classification* như Thể thao vs Kinh tế), dấu câu không mang nghĩa chủ đề nên thường bị xóa. Nhưng trong bài toán **Phát hiện Giật tít (*Clickbait Detection*)**, dấu câu lại là **đặc trưng cú pháp và cảm xúc cực kỳ quan trọng**:
> * Số liệu EDA của nhóm chỉ ra tiêu đề giật tít chứa dấu `?` cao gấp **6 lần** (`20.38%` vs `3.41%`) và chứa dấu ngoặc kép `"..."` cao gấp **2.8 lần** (`32.11%` vs `11.66%`) so với tin không giật tít.
> * Khi giữ lại `?`, `!`, `"` trong `token_pattern=r"(?u)\b\w[\w_]*\b|[?!\"']"` kết hợp với `ngram_range=(1, 2)`, mô hình học được cả các bigram kết thúc bằng dấu hỏi hoặc cụm từ nằm sát dấu ngoặc kép. Thực tế trong bảng `feature_importance.json`, đặc trưng `[HC]_has_question` đứng **Top 3 toàn mô hình** với trọng số **`+2.156`**.

---

## CHỦ ĐỀ 3: MÔ HÌNH HỌC MÁY, HỌC SÂU, PHOBERT & KẾT QUẢ THỰC NGHIỆM

### Câu 8: "Nhìn vào bảng kết quả ở Slide 7, tại sao các mô hình Học máy truyền thống (`Linear SVM`: 81.29% Acc, `Logistic Regression`: 78.76% Macro-F1) lại vượt trội hơn cả hai mô hình Học sâu (`BiLSTM`: 74.74% Macro-F1, `TextCNN`: 75.51% Macro-F1)?"
> **Trả lời mẫu:**
> Dạ đây là một kết quả thực nghiệm rất thú vị và hoàn toàn phù hợp với lý thuyết học máy trên dữ liệu văn bản cỡ vừa:
> 1. **Về quy mô dữ liệu và tầng Embedding:** Tập huấn luyện có `2,389` mẫu. Với `BiLSTM` và `TextCNN`, ma trận nhúng từ `nn.Embedding(15000, 128)` (gần 2 triệu tham số) được khởi tạo ngẫu nhiên và huấn luyện từ đầu (*trained from scratch*). Với cỡ mẫu `2,389` bài, nhiều từ ghép chỉ xuất hiện 2–3 lần nên vector dense embedding chưa hội tụ tốt bằng trọng số thống kê TF-IDF toàn cục, dẫn đến hiện tượng dễ overfitting hơn trên tập Test.
> 2. **Về đặc thù không gian đặc trưng:** Bài toán giật tít phụ thuộc rất mạnh vào các tín hiệu n-gram cụ thể (`bất_ngờ`, `khiến`, `tiết_lộ`, `chưa từng`) và dấu câu. Trong không gian thưa chiều cao (`16,012` chiều của Dual TF-IDF + 12 HC), dữ liệu gần như khả phân tuyến tính (*linearly separable*), nơi `Linear SVM` và `Logistic Regression` có ưu thế toán học tuyệt đối về việc tìm siêu phẳng lề cực đại mà không bị rơi vào cực tiểu địa phương.

---

### Câu 9: "Nếu `Linear SVM` (81.29% Acc) và `Logistic Regression` (78.76% Macro-F1) đã cao như vậy, tại sao nhóm vẫn kết hợp thêm `BiLSTM` và `TextCNN` vào `Hybrid Ensemble` (có Accuracy 79.53%)?"
> **Trả lời mẫu:**
> Dạ có 2 lý do thực tiễn và học thuật rất rõ ràng:
> 1. **Tối ưu hóa chỉ số ROC-AUC (Đạt đỉnh `86.41%`):** Nếu xét tại một ngưỡng cắt cứng `threshold = 0.5`, `Linear SVM` có Accuracy cao nhất (`81.29%`) nhưng **Recall trên lớp Giật tít chỉ đạt `68.13%`** (bỏ sót 51/160 bài giật tít). Ngược lại, `BiLSTM` bắt được cấu trúc chuỗi dài nên có **Recall đạt `73.75%`**. Khi kết hợp bằng Soft-Voting (`0.35·SVM + 0.20·LR + 0.22·BiLSTM + 0.23·TextCNN`), mô hình `Hybrid Ensemble` nâng Recall lên **`73.13%`** và đạt **ROC-AUC cao nhất toàn cục (`86.41%`)**.
> 2. **Phục vụ thanh trượt Độ nhạy trên Chrome Extension:** Trên giao diện Extension, người dùng có thể kéo thanh trượt thay đổi ngưỡng cắt (*Strict / Balanced / Relaxed*). Một mô hình có **ROC-AUC cao nhất (`86.41%`)** đảm bảo chất lượng xếp hạng xác suất (*probability ranking*) tốt nhất trên toàn dải ngưỡng cắt chứ không chỉ tại điểm `0.5`.

---

### Câu 10: "Tại sao `LinearSVC` trong Scikit-learn không có hàm `predict_proba()`, làm thế nào nhóm lấy được xác suất $P_{\text{SVM}}$ để đưa vào công thức Soft-Voting Ensemble?"
> **Trả lời mẫu:**
> Dạ đúng vậy ạ, `LinearSVC` tối ưu hàm mất mát *Squared Hinge Loss* nên đầu ra `decision_function(x)` là khoảng cách đại số từ điểm dữ liệu đến siêu phẳng phân lớp $(-\infty, +\infty)$ chứ không phải xác suất $[0, 1]$.
> Để lấy được xác suất chuẩn xác, trong `src/models_ml.py` (dòng 166), nhóm đã bọc `LinearSVC(class_weight="balanced", C=0.8)` bên trong **`CalibratedClassifierCV(cv=5)`**. Kỹ thuật này sử dụng **Platt Scaling** (khớp một hàm hồi quy Logistic Sigmoid trên đầu ra của SVM bằng kiểm chứng chéo 5-fold) để hiệu chuẩn khoảng cách siêu phẳng thành phân phối xác suất thực sự trước khi đưa vào tính trung bình có trọng số trong `Hybrid Ensemble`.

---

### Câu 11: "Trong `src/models_dl.py`, tại sao ở tầng Pooling của BiLSTM nhóm lại viết `lstm_out.masked_fill(~mask, -1e9)` trước khi gọi `torch.max()`?"
> **Trả lời mẫu:**
> Dạ đây là kỹ thuật **Masked Global Max Pooling** để xử lý các token đệm (`<PAD>`, có chỉ số `index = 0`).
> Khi đưa một câu ngắn (ví dụ 40 từ) vào chuỗi cố định `max_len = 128`, 88 vị trí cuối là token `<PAD>`. Khi đi qua tầng `BiLSTM`, các vị trí `<PAD>` này vẫn sinh ra các giá trị ẩn khác 0 (có thể mang số dương nếu các phần tử ẩn còn lại trong câu mang số âm). Nếu gọi `torch.max(lstm_out, dim=1)` trực tiếp mà không che mặt nạ (`mask = (input_ids != 0)`), phép Max Pooling sẽ lấy nhầm giá trị tại vị trí `<PAD>` thay vì từ thực sự trong câu. Gán `-1e9` cho các vị trí `<PAD>` đảm bảo `torch.max` chỉ lấy cực đại trên các từ thật.

---

### Câu 12: "Về mô hình `PhoBERT-base-v2` trong `src/models_bert.py`, tại sao nhóm lại đóng băng (freeze) tầng Embedding và 6 tầng Encoder đầu tiên mà chỉ fine-tune 6 tầng trên?"
> **Trả lời mẫu:**
> Dạ `vinai/phobert-base-v2` có 12 tầng Transformer Encoder với khoảng 135 triệu tham số. Theo các nghiên cứu về *BERTology*, tầng Embedding và 6 tầng Encoder dưới cùng nắm giữ các đặc trưng hình thái từ vựng và cú pháp tiếng Việt phổ quát, trong khi 6 tầng trên cùng đặc trưng cho ngữ nghĩa cấp cao của tác vụ đích (*task-specific semantics*).
> Với tập dữ liệu huấn luyện có quy mô `2,389` mẫu, nếu mở khóa toàn bộ 135 triệu tham số sẽ rất dễ dẫn đến hiện tượng **Catastrophic Forgetting** (quên tri thức ngôn ngữ gốc và overfitting cực nhanh sau 1–2 epoch), đồng thời tốn gấp đôi bộ nhớ và thời gian tính toán backward. Đóng băng 6 tầng đầu giúp mô hình hội tụ ổn định và giữ nguyên biểu diễn từ vựng tiếng Việt của PhoBERT.

---

## CHỦ ĐỀ 4: EXPLAINABLE AI (XAI) & TÍNH NĂNG DE-CLICKBAIT

### Câu 13: "Em hãy giải thích công thức tính điểm tô màu từ khóa (Local Explanation) trên giao diện Gradio. Nó khác gì với việc chỉ tra từ điển xem từ nào có trong danh sách từ giật tít?"
> **Trả lời mẫu:**
> Dạ nếu chỉ tra từ điển cố định thì mọi bài báo có cùng một từ sẽ bị gán điểm y hệt nhau bất kể độ dài câu. Trong `src/inference.py` (hàm `highlight_tokens`), nhóm tính điểm đóng góp cục bộ của từng token $t_i$ trong bài báo $x$ bằng tích số giữa **trọng số TF-IDF thực tế của từ đó trong bài báo đang xét** và **trọng số tuyến tính đã học của mô hình**:
> $$\text{Score}(t_i, x) = \text{TF-IDF}(t_i, x) \times w_{t_i}$$
> Trong đó $w_{t_i}$ là trung bình trọng số đã học của `Logistic Regression` và `Linear SVM`. Cách tính này chính là **Linear Feature Attribution** (tương đương giá trị SHAP tuyến tính đối với mô hình cộng tính): một từ chỉ được tô màu đỏ đậm khi nó vừa xuất hiện nổi bật trong bài báo đó ($\text{TF-IDF}$ cao), vừa mang trọng số đẩy mạnh về phía nhãn `Giật tít` ($w_{t_i} > 0$).

---

### Câu 14: "Tính năng 'Tháo tít' (De-clickbait) của nhóm hoạt động như thế nào? Có phải nhóm dùng ChatGPT hay mô hình sinh văn bản để viết lại tiêu đề không? Nếu nó tự bịa ra thông tin sai thì sao?"
> **Trả lời mẫu:**
> Dạ nhóm **hoàn toàn không sử dụng mô hình sinh văn bản tự do (Generative LLM)** cho tính năng Tháo tít, chính là để **triệt tiêu 100% rủi ro ảo giác (*Hallucination* — tự bịa đặt sự kiện báo chí)**.
> Hàm `rewrite_neutral_title()` trong `src/inference.py` được thiết kế theo phương pháp **Trích xuất kết hợp Luật ngôn ngữ (*Extractive & Rule-based Post-processing*)**:
> 1. **Khi bài báo có đoạn Sapo:** Trong cấu trúc báo chí chuẩn (hình tháp ngược), câu đầu tiên của đoạn Sapo luôn chứa thông tin sự kiện cốt lõi giải đáp cho khoảng trống tò mò trên tiêu đề. Thuật toán sẽ làm sạch tiền tố địa danh/tên báo và trích xuất nguyên văn mệnh đề sự kiện đầu tiên từ Sapo (độ dài `15–145` ký tự).
> 2. **Khi chỉ có Tiêu đề:** Thuật toán dùng bộ quy tắc Regex cắt bỏ các cụm hư từ cảm thán ở đầu câu (*"Bất ngờ trước..."*, *"Hé lộ..."*) và cuối câu (*"...khiến dân mạng xôn xao"*).
> Nhờ chỉ trích xuất từ chính văn bản gốc của nhà báo nên tiêu đề mới đảm bảo **0% bịa đặt thông tin**.

---

## CHỦ ĐỀ 5: KIẾN TRÚC CHROME EXTENSION & TRIỂN KHAI THỰC TẾ

### Câu 15: "Khi người dùng tắt máy tính rồi bật lại mà không mở code Python (`app.py`), Chrome Extension của nhóm lấy mô hình ở đâu để phân loại? Chất lượng của nó so với bản Server như thế nào?"
> **Trả lời mẫu:**
> Dạ nhóm thiết kế kiến trúc **Suy diễn 2 Tầng (*Two-Tier Inference*)** trong `extension/background.js`:
> * **Khi bật Backend Python (`app.py`):** Extension gửi request `POST /api/predict_batch` về FastAPI Server để chạy đầy đủ **Hybrid Ensemble 4 mô hình** (`SVM + LR + BiLSTM + TextCNN`).
> * **Khi tắt Backend Python (chế độ Offline):** Sau timeout `2.8 giây` không thấy server, `background.js` tự động chuyển sang **Local ML Fallback Engine** chạy hoàn toàn bằng JavaScript trong trình duyệt. Engine này đọc file `extension/model_weights.json` (chứa **800 n-gram trọng số dương lớn nhất + 800 n-gram trọng số âm lớn nhất** được trích xuất trực tiếp từ mô hình `Logistic Regression` của nhóm), kết hợp với 5 luật dấu câu cảm thán và độ trùng lặp từ khóa `Title – Sapo`.
> * **Về chất lượng:** Nhóm không tuyên bố Local Engine tương đương bản Server Ensemble (vì Local Engine không có bộ tách từ PyVi đầy đủ và không có 2 mạng học sâu BiLSTM/TextCNN), nhưng nó đạt tốc độ `< 1 ms/bài` và đóng vai trò cơ chế dự phòng (*graceful degradation*) giúp Extension luôn hoạt động 100% ngay cả khi người dùng không bật server Python.

---

### Câu 16: "Trên các trang chủ báo điện tử, có những bài chỉ hiện mỗi cái tiêu đề chứ không hiện đoạn Sapo bên dưới. Lúc đó mô hình của nhóm xử lý thế nào?"
> **Trả lời mẫu:**
> Dạ đây chính là một điểm thực tế mà nhóm đã phân tích rõ trong cả phần Thiết kế kỹ thuật lẫn phần Hạn chế của đồ án:
> * Khi `content.js` quét DOM qua hàm `extractLeadNearLink()`, nếu thẻ bài viết trên trang chủ có kèm đoạn tóm tắt (`.description`, `.sapo`, `.summary`), hệ thống sẽ gửi cả `Title + Sapo` về mô hình (đạt hiệu năng tối đa `81.29%` Acc / `86.41%` AUC).
> * Nếu vị trí đó trên trang chủ chỉ hiển thị duy nhất dòng tiêu đề (`lead_paragraph = ""`), hàm tiền xử lý sẽ tự động điền chuỗi rỗng và mô hình vẫn phân loại dựa trên các đặc trưng của `Title` và `12 đặc trưng thủ công` (tương ứng với cấu hình `Only Title + 12 HC` trong Ablation Study, đạt khoảng **79.73% – 80.90% Accuracy**).
> * Khi người dùng bấm vào đọc bài chi tiết, Tầng 2 (*Reading Assistant Pill*) sẽ lấy được toàn bộ đoạn Sapo thật của bài viết để đánh giá lại với độ chính xác cao nhất.

---

### Câu 17: "Làm sao nhóm đảm bảo khi chèn các nhãn Xanh/Vàng/Đỏ và khung Tooltip vào hàng chục trang báo khác nhau (VnExpress, Kênh14, Bongdaplus...) thì không làm vỡ giao diện của các trang báo đó?"
> **Trả lời mẫu:**
> Dạ nhóm áp dụng 3 kỹ thuật an toàn trên DOM trong `extension/content.js` và `content.css`:
> 1. **Chèn phần tử ngang hàng (*Sibling Insertion*):** Sử dụng `titleEl.insertAdjacentElement("afterend", badge)` với `display: inline-flex` thay vì ghi đè `innerHTML` của thẻ `<a>`, giúp giữ nguyên toàn bộ sự kiện click và cấu trúc thẻ của trang báo.
> 2. **Cô lập CSS (*Scoped Namespace `.vn-acb-*`):** Mọi class CSS của Extension đều có tiền tố `.vn-acb-` với `all: initial` hoặc định nghĩa tường minh `font-family, line-height, box-sizing` để không bị CSS của trang báo ghi đè và ngược lại.
> 3. **Dịch chuyển khung Tooltip ra `document.body`:** Khi người dùng rê chuột vào nhãn, khung Tooltip được tính toán tọa độ bằng `getBoundingClientRect()` và gắn trực tiếp vào `document.body` với `position: fixed; z-index: 2147483647`, giúp Tooltip không bao giờ bị cắt khuất bởi thuộc tính `overflow: hidden` của các thẻ bài viết trên trang báo.

---

### Câu 18: "Nút '🚩 Báo nhầm lẫn' trên Extension hoạt động như thế nào? Có phải cứ người dùng bấm báo nhầm là hệ thống tự động nạp vào tập Train để huấn luyện lại ngay không?"
> **Trả lời mẫu:**
> Dạ **tuyệt đối không tự động nạp thẳng vào tập Train mà không qua kiểm duyệt** ạ!
> Trong kỹ thuật học máy thực tế, nếu tự động đưa mọi phản hồi của người dùng vào huấn luyện lại ngay sẽ dẫn đến nguy cơ **Đầu độc dữ liệu (*Data Poisoning*)** hoặc nhiễu nhãn do người dùng bấm nhầm.
> Quy trình **Active Learning có kiểm duyệt (*Human-in-the-loop*)** của nhóm hoạt động như sau:
> 1. Khi người dùng bấm nút `🚩 Báo nhầm lẫn` (trên Tooltip hoặc trên thanh Reading Pill), Extension gửi `title`, `lead_paragraph`, `url`, `predicted_label`, `predicted_prob` và `suggested_label` về endpoint `POST /api/feedback`.
> 2. Server ghi nhận mẫu này vào file nhật ký cách ly `outputs/results/user_feedback.jsonl`.
> 3. Định kỳ, nhóm phát triển (hoặc người quản trị dữ liệu) sẽ rà soát, kiểm tra chéo các mẫu trong `user_feedback.jsonl` trước khi gộp các mẫu đạt chuẩn vào tập dữ liệu huấn luyện cho phiên bản mô hình tiếp theo.

---

### Câu 19: "Thầy/cô thấy trong mã nguồn của nhóm có xây dựng mô hình `PhoBERT-base-v2` (`src/models_bert.py`), nhưng tại sao trong Chrome Extension và Backend suy diễn thời gian thực (`src/inference.py`) lại không sử dụng PhoBERT?"
> **Trả lời mẫu:**
> Dạ thưa thầy/cô, đây là quyết định thiết kế có chủ đích của nhóm dựa trên bài toán **Đánh đổi giữa Độ chính xác và Độ trễ thời gian thực (*Accuracy–Latency Trade-off*)** khi đưa mô hình NLP từ pha Nghiên cứu sang pha Sản phẩm (*Production/MLOps*):
> 1. **Bài toán Độ trễ khi Quét hàng loạt trên Trang chủ Báo (*Batch Latency*):**
>    * Khi người dùng mở một trang báo điện tử (`vnexpress.net`, `kenh14.vn`, `bongdaplus.vn`...), Chrome Extension không phải chỉ phân tích 1 câu đơn lẻ, mà phải quét đồng loạt **25 – 60 tiêu đề bài báo** ngay lập tức và tiếp tục quét liên tục mỗi khi người dùng cuộn chuột.
>    * Tổ hợp **Hybrid Ensemble 4 mô hình (`Linear SVM + Logistic Regression + BiLSTM + TextCNN`)** của nhóm chỉ mất **0.523 ms / bài** trên CPU $\rightarrow$ Quét 1 lô 25 bài báo chỉ tốn **~13 ms**, giúp nhãn Xanh/Vàng/Đỏ hiển thị gần như tức thì.
>    * Trong khi đó, `PhoBERT-base-v2` (135 triệu tham số, 12 tầng Transformer) khi suy diễn trên CPU phổ thông tốn khoảng **80 – 150 ms / bài** (chậm hơn gấp **~200 lần**) $\rightarrow$ Quét 25 bài trên trang chủ sẽ gây trễ **2.5 – 4 giây** mỗi lần cuộn trang.
> 2. **Giới hạn Bộ nhớ của Trình duyệt (*Browser Sandbox & RAM*):**
>    * Ở phía trình duyệt (Client-side Extension), không thể nhúng trực tiếp cục trọng số nặng **~540 MB** của PhoBERT vào gói cài đặt Extension. Do đó, nhóm đã chắt lọc **1,600 trọng số n-gram quan trọng nhất** ra file `extension/model_weights.json` chỉ nặng **~45 KB** (nhẹ hơn PhoBERT **12,000 lần**) để chạy chế độ *Offline Fallback* trong `< 1 ms`.
>    * Ở phía Backend (`app.py`), không nạp PhoBERT mặc định giúp server khởi động nhanh trong 2–3 giây và tiết kiệm hơn **1.2 GB RAM** cho máy cá nhân.
> 3. **Hiệu quả thực nghiệm:**
>    * Trên tập dữ liệu `3,414` bài báo, tổ hợp `Hybrid Ensemble (SVM + LR + BiLSTM + TextCNN)` trên không gian `16,012` chiều đã đạt **ROC-AUC 86.41%** và `Linear SVM` đạt **Accuracy 81.29%**, đảm bảo độ chính xác rất cao với chi phí tính toán chỉ bằng **1/200** so với Transformer.

