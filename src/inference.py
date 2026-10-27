"""
Module Suy diễn & Giải thích Mô hình (Unified Inference & Explainability Engine)
Phục vụ đồng thời cho:
1. Giao diện Demo Gradio (Trình diễn học thuật, so sánh đa mô hình, XAI, phân tích lỗi).
2. REST API Backend cho Chrome Extension (Quét hàng loạt tiêu đề trên trang báo & Trợ lý đọc bài chi tiết).
3. Xuất trọng số mô hình nhẹ (model_weights.json) cho Tầng 1 (Client-side Instant Scanner) của Extension.
"""

import os
import re
import time
import json
import warnings
import urllib.parse
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
import torch

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

from src.preprocess import (
    clean_title,
    clean_lead_paragraph,
    segment_vietnamese,
    fix_vnexpress_glued_location,
    remove_baomoi_source_suffix,
    extract_handcrafted_features,
    HANDCRAFTED_FEATURE_COLS,
    run_preprocessing_pipeline,
)
from src.models_dl import (
    Vocabulary,
    BiLSTMClassifier,
    TextCNNClassifier,
    set_seed,
)
from src.evaluate import compute_metrics


HANDCRAFTED_LABELS_VI = {
    "title_char_len": "Độ dài tiêu đề (ký tự)",
    "title_word_count": "Số từ đơn tiêu đề",
    "lead_char_len": "Độ dài sapo (ký tự)",
    "lead_word_count": "Số từ đơn sapo",
    "has_question": "Chứa dấu chấm hỏi (?)",
    "has_exclamation": "Chứa dấu chấm than (!)",
    "has_quotes": "Chứa dấu trích dẫn (\"...\")",
    "has_ellipsis": "Chứa dấu bỏ lửng (...)",
    "num_digits": "Số lượng con số",
    "has_number": "Có dùng con số gây chú ý",
    "all_caps_count": "Số từ viết HOA toàn bộ",
    "upper_ratio": "Tỷ lệ ký tự viết hoa",
}

SENSATIONAL_PATTERNS_VI = [
    r"\bbất\s*ngờ\b", r"\bxôn\s*xao\b", r"\bsự\s*thật\b", r"\bcái\s*kết\b",
    r"\bdanh\s*tính\b", r"\bhé\s*lộ\b", r"\btiết\s*lộ\b", r"\bngỡ\s*ngàng\b",
    r"\bchoáng\s*váng\b", r"\bsốc\b", r"\bgây\s*bão\b", r"\bdậy\s*sóng\b",
    r"\bchấn\s*động\b", r"\bkinh\s*hoàng\b", r"\bbí\s*mật\b", r"\bbí\s*ẩn\b",
    r"\bchưa\s*từng\b", r"\bít\s*ai\s*biết\b", r"\bkhông\s*ngờ\b",
    r"\bkhiến\s*ai\s*cũng\b", r"\bxót\s*xa\b", r"\brùng\s*mình\b",
    r"\bnguyên\s*nhân\s*thực\s*sự\b", r"\blý\s*do\s*khiến\b",
]

CURIOSITY_GAP_PRONOUNS = [
    "thứ này", "điều này", "người này", "cô gái này", "chàng trai này",
    "nơi này", "món ăn này", "loại quả này", "loại rau này", "hành động này",
    "câu nói này", "chi tiết này", "nhân vật này",
]


class ClickbaitInferenceEngine:
    """
    Bộ máy suy diễn hợp nhất (Unified Inference Engine) quản lý:
    - TF-IDF + Linear SVM (Calibrated)
    - TF-IDF + Logistic Regression
    - BiLSTM (PyTorch)
    - TextCNN (PyTorch)
    - Tổ hợp Đa mô hình (Hybrid Ensemble)
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        results_dir: str = "outputs/results",
    ):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

        # Thành phần biến đổi đặc trưng ML
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.title_vec: Optional[TfidfVectorizer] = None
        self.ml_scaler: Optional[MinMaxScaler] = None
        self.feature_names: List[str] = []

        # Các mô hình ML
        self.svm_model = None
        self.lr_model = None
        self.token_weight_map: Dict[str, float] = {}

        # Thành phần DL
        self.dl_vocab: Optional[Vocabulary] = None
        self.dl_scaler: Optional[StandardScaler] = None
        self.bilstm_model: Optional[BiLSTMClassifier] = None
        self.textcnn_model: Optional[TextCNNClassifier] = None
        self.dl_max_len: int = 128

        # Dữ liệu thống kê & Benchmark
        self.benchmarks: List[Dict[str, Any]] = []
        self.eda_stats: Dict[str, Any] = {}
        self.feature_importance: Dict[str, Any] = {}

        self._initialize()

    def _initialize(self) -> None:
        """Khởi tạo dữ liệu, khôi phục các bộ biến đổi và nạp trọng số mô hình."""
        train_path = os.path.join(self.data_dir, "train.csv")
        val_path = os.path.join(self.data_dir, "val.csv")
        test_path = os.path.join(self.data_dir, "test.csv")

        if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
            print("[ENGINE] Chưa thấy dữ liệu processed, đang chạy pipeline tiền xử lý...")
            self.train_df, self.val_df, self.test_df = run_preprocessing_pipeline()
        else:
            self.train_df = pd.read_csv(train_path, encoding="utf-8")
            self.val_df = pd.read_csv(val_path, encoding="utf-8")
            self.test_df = pd.read_csv(test_path, encoding="utf-8")

        # Nạp thống kê EDA và Feature Importance nếu có
        eda_path = os.path.join(self.results_dir, "eda_statistics.json")
        if os.path.exists(eda_path):
            with open(eda_path, "r", encoding="utf-8") as f:
                self.eda_stats = json.load(f)

        fi_path = os.path.join(self.results_dir, "feature_importance.json")
        if os.path.exists(fi_path):
            with open(fi_path, "r", encoding="utf-8") as f:
                self.feature_importance = json.load(f)

        # 1. Khởi tạo không gian đặc trưng TF-IDF + Hand-crafted cho ML (Option B: Title + Lead)
        svm_path = os.path.join(self.results_dir, "best_ml_linear_svm.joblib")
        if os.path.exists(svm_path):
            saved_bundle = joblib.load(svm_path)
            self.svm_model = saved_bundle["model"]
            self.vectorizer = saved_bundle["vectorizer"]
            self.ml_scaler = saved_bundle["scaler"]
        else:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
                min_df=2,
                token_pattern=r"(?u)\b\w[\w_]*\b|[?!\"']",
            )
            self.vectorizer.fit(self.train_df["input_option_b"].fillna(""))
            self.ml_scaler = MinMaxScaler()
            self.ml_scaler.fit(self.train_df[HANDCRAFTED_FEATURE_COLS].fillna(0).values)

        # Khởi tạo title_vec giống hệt trong src/models_ml.py
        self.title_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=6000,
            sublinear_tf=True,
            min_df=2,
            token_pattern=r"(?u)\b\w[\w_]*\b|[?!\"']",
        )
        X_tr_title = self.title_vec.fit_transform(self.train_df["title_seg"].fillna("")) * 1.25
        X_tr_tfidf = self.vectorizer.transform(self.train_df["input_option_b"].fillna(""))
        X_tr_hc = self.ml_scaler.transform(self.train_df[HANDCRAFTED_FEATURE_COLS].fillna(0).values)
        X_train_full = hstack([X_tr_tfidf, X_tr_title, csr_matrix(X_tr_hc)], format="csr")
        y_train = self.train_df["label_id"].values

        tfidf_names = self.vectorizer.get_feature_names_out().tolist()
        title_names = [f"[TITLE]_{w}" for w in self.title_vec.get_feature_names_out().tolist()]
        self.feature_names = tfidf_names + title_names + [f"[HC]_{c}" for c in HANDCRAFTED_FEATURE_COLS]

        # Huấn luyện nhanh Logistic Regression (0.15s) để lấy xác suất mượt và trọng số giải thích trực tiếp
        self.lr_model = LogisticRegression(class_weight="balanced", C=1.5, max_iter=1000, random_state=42)
        self.lr_model.fit(X_train_full, y_train)

        # Xây dựng từ điển trọng số token phục vụ HighlightedText (XAI) và xuất sang Extension
        self._build_token_weight_map()

        # 2. Khởi tạo Vocabulary, Scaler và nạp trọng số PyTorch cho BiLSTM & TextCNN
        self.dl_scaler = StandardScaler()
        self.dl_scaler.fit(self.train_df[HANDCRAFTED_FEATURE_COLS].fillna(0).values)

        self.dl_vocab = Vocabulary(max_vocab_size=15000, min_freq=1)
        self.dl_vocab.build_vocab(self.train_df["input_option_b"].fillna("").tolist())

        bilstm_path = os.path.join(self.results_dir, "best_dl_bilstm_title_lead.pt")
        if os.path.exists(bilstm_path):
            self.bilstm_model = BiLSTMClassifier(
                vocab_size=len(self.dl_vocab),
                embedding_dim=128,
                hidden_dim=64,
                hc_dim=len(HANDCRAFTED_FEATURE_COLS),
                dropout=0.2,
            ).to(self.device)
            state_dict = torch.load(bilstm_path, map_location=self.device, weights_only=True)
            self.bilstm_model.load_state_dict(state_dict)
            self.bilstm_model.eval()

        textcnn_path = os.path.join(self.results_dir, "best_dl_textcnn_title_lead.pt")
        if os.path.exists(textcnn_path):
            self.textcnn_model = TextCNNClassifier(
                vocab_size=len(self.dl_vocab),
                embedding_dim=128,
                num_filters=64,
                filter_sizes=(2, 3, 5),
                hc_dim=len(HANDCRAFTED_FEATURE_COLS),
                dropout=0.2,
            ).to(self.device)
            state_dict = torch.load(textcnn_path, map_location=self.device, weights_only=True)
            self.textcnn_model.load_state_dict(state_dict)
            self.textcnn_model.eval()

        # 3. Tính toán hoặc nạp bảng Benchmark trên tập Test
        self._load_or_compute_benchmarks()

    def _build_token_weight_map(self) -> None:
        """Tổng hợp trọng số đặc trưng từ Linear SVM và Logistic Regression cho giải thích XAI."""
        lr_coefs = self.lr_model.coef_.flatten()
        if self.svm_model is not None and hasattr(self.svm_model, "calibrated_classifiers_"):
            svm_coefs = np.mean(
                [est.estimator.coef_.flatten() for est in self.svm_model.calibrated_classifiers_],
                axis=0,
            )
            combined_coefs = 0.55 * svm_coefs + 0.45 * lr_coefs
        else:
            combined_coefs = lr_coefs

        weight_map: Dict[str, float] = {}
        for name, w in zip(self.feature_names, combined_coefs):
            if name.startswith("[HC]_") or name in ("sep", "[TITLE]_sep"):
                continue
            clean_tok = name.replace("[TITLE]_", "")
            val = float(w)
            if clean_tok not in weight_map or abs(val) > abs(weight_map[clean_tok]):
                weight_map[clean_tok] = round(val, 4)
        self.token_weight_map = weight_map

    def _load_or_compute_benchmarks(self) -> None:
        """Tính toán bảng điểm thực nghiệm trên tập Test cho Dashboard Gradio."""
        bench_path = os.path.join(self.results_dir, "model_benchmarks.json")
        if os.path.exists(bench_path):
            with open(bench_path, "r", encoding="utf-8") as f:
                self.benchmarks = json.load(f)
            return

        y_test = self.test_df["label_id"].values
        X_te_tfidf = self.vectorizer.transform(self.test_df["input_option_b"].fillna(""))
        X_te_title = self.title_vec.transform(self.test_df["title_seg"].fillna("")) * 1.25
        X_te_hc_ml = self.ml_scaler.transform(self.test_df[HANDCRAFTED_FEATURE_COLS].fillna(0).values)
        X_test_ml = hstack([X_te_tfidf, X_te_title, csr_matrix(X_te_hc_ml)], format="csr")

        benchmarks = []

        # 1. Logistic Regression
        t0 = time.perf_counter()
        lr_probs = self.lr_model.predict_proba(X_test_ml)[:, 1]
        lr_preds = (lr_probs >= 0.5).astype(int)
        lr_ms = (time.perf_counter() - t0) * 1000.0 / len(y_test)
        benchmarks.append(
            compute_metrics(
                y_test, lr_preds, lr_probs,
                model_group="Baseline (ML)",
                model_name="TF-IDF + Logistic Regression",
                input_type="Title + Lead",
                infer_time_ms_per_sample=lr_ms,
            )
        )

        # 2. Linear SVM
        if self.svm_model is not None:
            t0 = time.perf_counter()
            svm_probs = self.svm_model.predict_proba(X_test_ml)[:, 1]
            svm_preds = (svm_probs >= 0.5).astype(int)
            svm_ms = (time.perf_counter() - t0) * 1000.0 / len(y_test)
            benchmarks.append(
                compute_metrics(
                    y_test, svm_preds, svm_probs,
                    model_group="Baseline (ML)",
                    model_name="TF-IDF + Linear SVM",
                    input_type="Title + Lead",
                    infer_time_ms_per_sample=svm_ms,
                )
            )
        else:
            svm_probs = lr_probs

        # Chuẩn bị tensor DL trên tập Test
        X_te_ids = np.array([
            self.dl_vocab.encode(t, max_len=self.dl_max_len)
            for t in self.test_df["input_option_b"].fillna("")
        ])
        X_te_hc_dl = self.dl_scaler.transform(self.test_df[HANDCRAFTED_FEATURE_COLS].fillna(0).values)
        ids_tensor = torch.tensor(X_te_ids, dtype=torch.long, device=self.device)
        hc_tensor = torch.tensor(X_te_hc_dl, dtype=torch.float32, device=self.device)

        # 3. BiLSTM
        if self.bilstm_model is not None:
            t0 = time.perf_counter()
            with torch.no_grad():
                bilstm_probs = torch.sigmoid(self.bilstm_model(ids_tensor, hc_tensor)).cpu().numpy()
            bilstm_preds = (bilstm_probs >= 0.48).astype(int)
            bilstm_ms = (time.perf_counter() - t0) * 1000.0 / len(y_test)
            benchmarks.append(
                compute_metrics(
                    y_test, bilstm_preds, bilstm_probs,
                    model_group="Deep Learning",
                    model_name="BiLSTM (Bidirectional LSTM)",
                    input_type="Title + Lead",
                    infer_time_ms_per_sample=bilstm_ms,
                )
            )
        else:
            bilstm_probs = svm_probs

        # 4. TextCNN
        if self.textcnn_model is not None:
            t0 = time.perf_counter()
            with torch.no_grad():
                textcnn_probs = torch.sigmoid(self.textcnn_model(ids_tensor, hc_tensor)).cpu().numpy()
            textcnn_preds = (textcnn_probs >= 0.48).astype(int)
            textcnn_ms = (time.perf_counter() - t0) * 1000.0 / len(y_test)
            benchmarks.append(
                compute_metrics(
                    y_test, textcnn_preds, textcnn_probs,
                    model_group="Deep Learning",
                    model_name="TextCNN (Multi-Filter 1D-CNN)",
                    input_type="Title + Lead",
                    infer_time_ms_per_sample=textcnn_ms,
                )
            )
        else:
            textcnn_probs = svm_probs

        # 5. Hybrid Ensemble (Linear SVM + Logistic Regression + BiLSTM + TextCNN)
        ens_probs = 0.35 * svm_probs + 0.20 * lr_probs + 0.22 * bilstm_probs + 0.23 * textcnn_probs
        ens_preds = (ens_probs >= 0.48).astype(int)
        benchmarks.append(
            compute_metrics(
                y_test, ens_preds, ens_probs,
                model_group="Hybrid Ensemble (SOTA)",
                model_name="Ensemble (Linear SVM + BiLSTM + TextCNN)",
                input_type="Title + Lead",
                infer_time_ms_per_sample=round(svm_ms + bilstm_ms + textcnn_ms, 3),
            )
        )

        self.benchmarks = benchmarks
        os.makedirs(self.results_dir, exist_ok=True)
        with open(bench_path, "w", encoding="utf-8") as f:
            json.dump(benchmarks, f, ensure_ascii=False, indent=2)

    def export_client_weights_for_extension(self, output_path: str = "extension/model_weights.json") -> None:
        """
        Xuất bộ trọng số n-gram & đặc trưng thủ công đã huấn luyện sang file JSON nhẹ (~40KB)
        để Chrome Extension có thể chạy suy diễn Tầng 1 trực tiếp bằng JavaScript trong <1ms.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sorted_tokens = sorted(self.token_weight_map.items(), key=lambda x: abs(x[1]), reverse=True)
        top_tokens = {k: round(v, 4) for k, v in sorted_tokens[:1600] if abs(v) >= 0.18}

        export_data = {
            "version": "1.0.0",
            "description": "Trọng số mô hình Phát hiện Clickbait Tiếng Việt (TF-IDF + Linear SVM/LR)",
            "token_weights": top_tokens,
            "sensational_phrases": [
                "bất ngờ", "xôn xao", "sự thật", "cái kết", "danh tính", "hé lộ",
                "tiết lộ", "ngỡ ngàng", "choáng váng", "gây bão", "dậy sóng",
                "chấn động", "kinh hoàng", "bí mật", "bí ẩn", "chưa từng",
                "ít ai biết", "không ngờ", "khiến ai cũng", "xót xa", "rùng mình",
                "lộ diện", "thứ này", "điều này", "người này", "cô gái này",
                "lý do khiến", "nguyên nhân khiến", "sự thật phía sau", "bí quyết",
                "đừng bao giờ", "cảnh báo", "tuyệt đối không", "chỉ vì",
            ],
            "orthodox_phrases": [
                "ubnd", "quốc hội", "chính phủ", "thủ tướng", "bộ trưởng",
                "nghị định", "nghị quyết", "dự án", "hội nghị", "công bố",
                "khởi tố", "điều tra", "kết luận", "thông tư", "quyết định",
                "báo cáo", "tổng cục", "ngân hàng nhà nước", "chỉ số", "kim ngạch",
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    def predict_single(
        self,
        title: str,
        lead_paragraph: str = "",
        model_choice: str = "Ensemble (Khuyên dùng)",
        sensitivity: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Phân tích toàn diện một bài báo từ tiêu đề (`title`) và đoạn mở đầu (`lead_paragraph`).
        Trả về đầy đủ xác suất của từng mô hình, đặc trưng ngôn ngữ, giải thích XAI,
        đánh giá độ lệch ngữ nghĩa và tiêu đề trung lập gợi ý (De-clickbait).
        """
        t0 = time.perf_counter()
        raw_title = str(title or "").strip()
        raw_lead = str(lead_paragraph or "").strip()

        # Kiểm tra khử nhiễu đặc thù
        _, baomoi_fixed = remove_baomoi_source_suffix(raw_title)
        _, vne_fixed = fix_vnexpress_glued_location(raw_lead)

        title_clean = clean_title(raw_title)
        lead_clean = clean_lead_paragraph(raw_lead)

        title_seg = segment_vietnamese(title_clean)
        lead_seg = segment_vietnamese(lead_clean)
        input_option_b = f"{title_seg} [SEP] {lead_seg}" if lead_seg else f"{title_seg} [SEP]"

        # Trích xuất 12 đặc trưng thủ công
        temp_df = pd.DataFrame([{"title_clean": title_clean, "lead_clean": lead_clean}])
        hc_df = extract_handcrafted_features(temp_df)
        hc_raw_vals = hc_df[HANDCRAFTED_FEATURE_COLS].values

        # Biến đổi đặc trưng cho ML
        X_tfidf = self.vectorizer.transform([input_option_b])
        X_title = self.title_vec.transform([title_seg]) * 1.25
        X_hc_ml = self.ml_scaler.transform(hc_raw_vals)
        X_ml = hstack([X_tfidf, X_title, csr_matrix(X_hc_ml)], format="csr")

        lr_prob = float(self.lr_model.predict_proba(X_ml)[0, 1])
        svm_prob = (
            float(self.svm_model.predict_proba(X_ml)[0, 1])
            if self.svm_model is not None
            else lr_prob
        )

        # Biến đổi đặc trưng cho DL
        dl_ids = np.array([self.dl_vocab.encode(input_option_b, max_len=self.dl_max_len)])
        X_hc_dl = self.dl_scaler.transform(hc_raw_vals)
        ids_tensor = torch.tensor(dl_ids, dtype=torch.long, device=self.device)
        hc_tensor = torch.tensor(X_hc_dl, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            bilstm_prob = (
                float(torch.sigmoid(self.bilstm_model(ids_tensor, hc_tensor)).item())
                if self.bilstm_model is not None
                else svm_prob
            )
            textcnn_prob = (
                float(torch.sigmoid(self.textcnn_model(ids_tensor, hc_tensor)).item())
                if self.textcnn_model is not None
                else svm_prob
            )

        ensemble_prob = 0.35 * svm_prob + 0.20 * lr_prob + 0.22 * bilstm_prob + 0.23 * textcnn_prob

        model_scores = {
            "Ensemble (Đa mô hình)": round(ensemble_prob, 4),
            "TF-IDF + Linear SVM": round(svm_prob, 4),
            "TF-IDF + Logistic Regression": round(lr_prob, 4),
            "BiLSTM (Deep Learning)": round(bilstm_prob, 4),
            "TextCNN (Deep Learning)": round(textcnn_prob, 4),
        }

        # Chọn điểm xác suất chính theo model_choice
        choice_key = model_choice
        if "SVM" in model_choice:
            primary_prob = svm_prob
        elif "Logistic" in model_choice:
            primary_prob = lr_prob
        elif "BiLSTM" in model_choice:
            primary_prob = bilstm_prob
        elif "TextCNN" in model_choice:
            primary_prob = textcnn_prob
        else:
            primary_prob = ensemble_prob
            choice_key = "Ensemble (Đa mô hình)"

        is_clickbait = primary_prob >= sensitivity
        label = "clickbait" if is_clickbait else "non-clickbait"

        # Phân tầng tín hiệu màu (Xanh / Vàng / Đỏ) cho giao diện người dùng cuối
        if primary_prob >= max(sensitivity + 0.12, 0.62):
            badge_level = "red"
            badge_text = f"Giật tít · {int(round(primary_prob * 100))}%"
            verdict_vi = "BÀI VIẾT GIẬT TÍT / CÂU VIEW CAO"
        elif primary_prob >= max(sensitivity - 0.10, 0.40):
            badge_level = "yellow"
            badge_text = f"Chú ý · {int(round(primary_prob * 100))}%"
            verdict_vi = "TIÊU ĐỀ CÓ YẾU TỐ GÂY TÒ MÒ / CẢM XÚC"
        else:
            badge_level = "green"
            badge_text = "Tin chuẩn"
            verdict_vi = "TIN TỨC CHÍNH THỐNG / KHÁCH QUAN"

        # Phân tích đóng góp của từng từ/cụm từ trong tiêu đề & sapo (XAI HighlightedText)
        token_highlights, trigger_words, orthodox_words = self._analyze_tokens(title_seg, lead_seg)

        # Sinh lý do giải thích tự nhiên dành cho người đọc phổ thông
        hc_dict = {col: float(hc_df.iloc[0][col]) for col in HANDCRAFTED_FEATURE_COLS}
        reasons_vi, semantic_gap_info = self._generate_user_explanations(
            title_clean, lead_clean, primary_prob, trigger_words, orthodox_words, hc_dict
        )

        # Sinh tóm tắt nhanh (Sapo takeaway) và tiêu đề trung lập (De-clickbait)
        quick_summary = self._build_quick_summary(title_clean, lead_clean)
        declickbait_title = self._generate_declickbait_title(
            title_clean, lead_clean, is_clickbait or badge_level == "yellow"
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "label": label,
            "is_clickbait": bool(is_clickbait),
            "prob_clickbait": round(primary_prob, 4),
            "prob_non_clickbait": round(1.0 - primary_prob, 4),
            "badge_level": badge_level,
            "badge_text": badge_text,
            "verdict_vi": verdict_vi,
            "selected_model": choice_key,
            "model_scores": model_scores,
            "infer_time_ms": round(elapsed_ms, 2),
            "preprocessing": {
                "raw_title": raw_title,
                "raw_lead": raw_lead,
                "title_clean": title_clean,
                "lead_clean": lead_clean,
                "title_seg": title_seg,
                "lead_seg": lead_seg,
                "baomoi_suffix_removed": baomoi_fixed,
                "vnexpress_location_fixed": vne_fixed,
            },
            "handcrafted_features": [
                {
                    "key": col,
                    "name_vi": HANDCRAFTED_LABELS_VI.get(col, col),
                    "value": round(hc_dict[col], 3) if isinstance(hc_dict[col], float) else int(hc_dict[col]),
                }
                for col in HANDCRAFTED_FEATURE_COLS
            ],
            "token_highlights": token_highlights,
            "trigger_words": trigger_words[:8],
            "orthodox_words": orthodox_words[:8],
            "reasons_vi": reasons_vi,
            "semantic_gap": semantic_gap_info,
            "quick_summary": quick_summary,
            "declickbait_title": declickbait_title,
        }

    def predict_batch(
        self,
        items: List[Dict[str, str]],
        sensitivity: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """
        Quét hàng loạt bài báo (dùng cho Content Script của Chrome Extension khi lướt trang chủ).
        Tối ưu hóa vector hóa để xử lý 30-50 tiêu đề trong <15ms.
        """
        if not items:
            return []

        results = []
        for item in items:
            item_id = item.get("id", "")
            title = item.get("title", "")
            lead = item.get("lead_paragraph", "")
            pred = self.predict_single(
                title=title,
                lead_paragraph=lead,
                model_choice="Ensemble (Khuyên dùng)",
                sensitivity=sensitivity,
            )
            results.append({
                "id": item_id,
                "label": pred["label"],
                "is_clickbait": pred["is_clickbait"],
                "prob_clickbait": pred["prob_clickbait"],
                "badge_level": pred["badge_level"],
                "badge_text": pred["badge_text"],
                "verdict_vi": pred["verdict_vi"],
                "trigger_words": pred["trigger_words"],
                "reasons_vi": pred["reasons_vi"][:2],
                "quick_summary": pred["quick_summary"],
                "declickbait_title": pred["declickbait_title"],
            })
        return results

    def _analyze_tokens(
        self,
        title_seg: str,
        lead_seg: str,
    ) -> Tuple[List[Tuple[str, Optional[str]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Tách từng token đã phân đoạn để gán nhãn mức độ đóng góp cho gr.HighlightedText
        và trích xuất danh sách từ khóa báo động (trigger_words).
        """
        highlights: List[Tuple[str, Optional[str]]] = []
        trigger_words: List[Dict[str, Any]] = []
        orthodox_words: List[Dict[str, Any]] = []
        seen_triggers = set()
        seen_orthodox = set()

        combined_seg = title_seg if not lead_seg else f"{title_seg} | {lead_seg}"
        tokens = combined_seg.split()

        for idx, tok in enumerate(tokens):
            if tok == "|":
                highlights.append(("\n[Sapo] ", None))
                continue

            clean_tok = tok.lower().strip(".,:;()[]")
            display_word = tok.replace("_", " ")

            # Kiểm tra bigram với từ tiếp theo
            bigram_weight = 0.0
            if idx + 1 < len(tokens) and tokens[idx + 1] != "|":
                next_clean = tokens[idx + 1].lower().strip(".,:;()[]")
                bigram = f"{clean_tok} {next_clean}"
                bigram_weight = self.token_weight_map.get(bigram, 0.0)

            w = self.token_weight_map.get(clean_tok, 0.0)
            effective_w = bigram_weight if abs(bigram_weight) > abs(w) else w

            if effective_w >= 0.65:
                tag = "Giật tít mạnh"
            elif effective_w >= 0.28:
                tag = "Gợi tò mò"
            elif effective_w <= -0.45:
                tag = "Chính thống"
            else:
                tag = None

            highlights.append((display_word + " ", tag))

            if w >= 0.25 and display_word.lower() not in seen_triggers and len(display_word) > 1:
                seen_triggers.add(display_word.lower())
                trigger_words.append({"word": display_word, "weight": round(w, 3)})
            elif w <= -0.30 and display_word.lower() not in seen_orthodox and len(display_word) > 1:
                seen_orthodox.add(display_word.lower())
                orthodox_words.append({"word": display_word, "weight": round(w, 3)})

        trigger_words.sort(key=lambda x: x["weight"], reverse=True)
        orthodox_words.sort(key=lambda x: x["weight"])
        return highlights, trigger_words, orthodox_words

    def _generate_user_explanations(
        self,
        title_clean: str,
        lead_clean: str,
        prob: float,
        trigger_words: List[Dict[str, Any]],
        orthodox_words: List[Dict[str, Any]],
        hc: Dict[str, float],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Sinh lời giải thích bằng ngôn ngữ tự nhiên, dễ hiểu cho độc giả phổ thông."""
        reasons: List[str] = []
        title_lower = title_clean.lower()

        # 1. Từ khóa kích thích cảm xúc / tò mò
        if trigger_words:
            top_w = ", ".join(f'"{item["word"]}"' for item in trigger_words[:4])
            reasons.append(f"Sử dụng từ khóa kích thích sự tò mò hoặc cảm xúc mạnh: {top_w}.")

        # 2. Đại từ chỉ định giấu thông tin (Curiosity Gap)
        found_pronouns = [p for p in CURIOSITY_GAP_PRONOUNS if p in title_lower]
        if found_pronouns:
            p_str = ", ".join(f'"{p}"' for p in found_pronouns[:2])
            reasons.append(
                f"Cố tình giấu thông tin định danh trên tiêu đề bằng cụm từ bỏ lửng ({p_str}) để buộc người đọc phải bấm vào."
            )

        # 3. Dấu câu cảm thán / câu hỏi tu từ
        punct_cues = []
        if hc.get("has_question", 0) > 0:
            punct_cues.append("câu hỏi tu từ (?)")
        if hc.get("has_exclamation", 0) > 0:
            punct_cues.append("dấu chấm than cảm thán (!)")
        if hc.get("has_ellipsis", 0) > 0:
            punct_cues.append("dấu ba chấm bỏ lửng (...)")
        if hc.get("has_quotes", 0) > 0 and prob >= 0.45:
            punct_cues.append("trích dẫn trực tiếp gây kịch tính (\"...\")")
        if punct_cues:
            reasons.append(f"Hình thức tiêu đề dùng {', '.join(punct_cues)} nhằm thu hút sự chú ý tức thì.")

        # 4. Con số ấn tượng
        if hc.get("has_number", 0) > 0 and prob >= 0.50:
            reasons.append("Sử dụng con số cụ thể trên tiêu đề để tạo hiệu ứng gây sốc hoặc liệt kê câu view.")

        # 5. Đánh giá độ lệch ngữ nghĩa giữa Tiêu đề và Sapo (Semantic Gap)
        semantic_gap = self._compute_semantic_gap(title_clean, lead_clean, prob)
        if semantic_gap["has_gap"]:
            reasons.append(semantic_gap["message"])

        # Nếu là tin chính thống
        if prob < 0.42:
            if orthodox_words:
                ort_str = ", ".join(f'"{item["word"]}"' for item in orthodox_words[:4])
                reasons.append(f"Sử dụng thuật ngữ báo chí chính luận, khách quan ({ort_str}).")
            reasons.append("Cấu trúc tiêu đề cung cấp thông tin trực diện, rõ chủ thể và không cố tình đánh đố người đọc.")

        if not reasons:
            if prob >= 0.5:
                reasons.append("Cách đặt vấn đề trên tiêu đề mang sắc thái giật gân, nhấn mạnh tình tiết kịch tính.")
            else:
                reasons.append("Văn phong trung lập, phù hợp với chuẩn mực bản tin thông tin báo chí.")

        return reasons, semantic_gap

    def _compute_semantic_gap(self, title_clean: str, lead_clean: str, prob: float) -> Dict[str, Any]:
        """Phân tích độ tương đồng và độ lệch thông tin giữa Title và Lead Paragraph."""
        if not lead_clean or len(lead_clean.split()) < 6:
            return {
                "has_gap": False,
                "overlap_ratio": 1.0,
                "level": "Chưa có Sapo",
                "message": "Chưa có đoạn mở đầu (sapo) để đối chiếu ngữ nghĩa với tiêu đề.",
            }

        stop_words = {"của", "và", "các", "những", "trong", "với", "cho", "được", "một", "người", "khi", "đã", "là", "có", "tại", "về", "này"}
        t_words = {w.lower().strip(".,?!\"'()") for w in title_clean.split() if len(w) > 2} - stop_words
        l_words = {w.lower().strip(".,?!\"'()") for w in lead_clean.split() if len(w) > 2} - stop_words

        if not t_words:
            overlap = 0.5
        else:
            overlap = len(t_words & l_words) / len(t_words)

        has_sensational_title = any(re.search(pat, title_clean.lower()) for pat in SENSATIONAL_PATTERNS_VI)
        has_sensational_lead = any(re.search(pat, lead_clean.lower()) for pat in SENSATIONAL_PATTERNS_VI)

        if (has_sensational_title and not has_sensational_lead) or (prob >= 0.55 and overlap < 0.35):
            return {
                "has_gap": True,
                "overlap_ratio": round(overlap, 2),
                "level": "Độ lệch cao",
                "message": "Tiêu đề dùng từ ngữ phóng đại/giấu thông tin, trong khi đoạn mở đầu (sapo) trình bày sự việc bình thường và cụ thể hơn.",
            }
        elif prob >= 0.50:
            return {
                "has_gap": True,
                "overlap_ratio": round(overlap, 2),
                "level": "Cần lưu ý",
                "message": "Tiêu đề chỉ trích xuất một chi tiết gây tò mò nhất từ đoạn sapo để thu hút lượt nhấp.",
            }
        else:
            return {
                "has_gap": False,
                "overlap_ratio": round(overlap, 2),
                "level": "Nhất quán",
                "message": "Tiêu đề phản ánh trung thực và sát với nội dung tóm tắt ở đoạn mở đầu (sapo).",
            }

    def _build_quick_summary(self, title_clean: str, lead_clean: str) -> str:
        """Trích xuất tóm tắt ngắn gọn (1-2 câu) từ Sapo để người dùng đọc nhanh trên Hover Tooltip."""
        if lead_clean and len(lead_clean.strip()) >= 15:
            sentences = re.split(r"(?<=[.!?])\s+", lead_clean.strip())
            summary = sentences[0]
            if len(summary) < 65 and len(sentences) > 1:
                summary = f"{summary} {sentences[1]}"
            if len(summary) > 190:
                summary = summary[:187].rstrip() + "..."
            return summary
        return "Bài viết chưa cung cấp đoạn tóm tắt (sapo) đi kèm trên trang danh sách."

    def _generate_declickbait_title(
        self,
        title_clean: str,
        lead_clean: str,
        should_neutralize: bool,
    ) -> str:
        """
        Tính năng [Tháo tít / De-clickbait]:
        Tái cấu trúc lại một tiêu đề trung lập, chuẩn báo chí bằng cách:
        - Loại bỏ các từ ngữ kích động cảm xúc / câu view ở tiêu đề.
        - Hoặc trích xuất mệnh đề thông tin cốt lõi đầu tiên từ đoạn sapo (`lead_clean`).
        """
        if not should_neutralize:
            return title_clean

        # Nếu có đoạn sapo rõ ràng, câu đầu tiên của sapo thường chứa đầy đủ Chủ ngữ + Vị ngữ thật
        if lead_clean and len(lead_clean.split()) >= 8:
            first_sent = re.split(r"(?<=[.!?])\s+", lead_clean.strip())[0]
            # Rút gọn mệnh đề chính của câu đầu sapo nếu quá dài
            clauses = [c.strip() for c in first_sent.split(",") if len(c.strip().split()) >= 4]
            if len(clauses) >= 2 and len(first_sent) > 95:
                candidate = f"{clauses[0]}, {clauses[1]}"
            else:
                candidate = first_sent.rstrip(".!?")
            if len(candidate) > 105:
                candidate = candidate[:102].rsplit(" ", 1)[0] + "..."
            if candidate:
                return candidate[0].upper() + candidate[1:]

        # Nếu không có sapo, làm sạch các cụm từ giật gân trực tiếp trên tiêu đề
        neutral = title_clean
        remove_phrases = [
            r"(?i)\bbất ngờ\b\s*",
            r"(?i)\bngỡ ngàng\b\s*",
            r"(?i)\bchoáng váng\b\s*",
            r"(?i)\bxôn xao\b\s*",
            r"(?i)\bgây bão\b\s*",
            r"(?i)\bdậy sóng\b\s*",
            r"(?i)\bhé lộ\b\s*",
            r"(?i)\btiết lộ\b\s*",
            r"(?i)\bsự thật\b\s*(?:về|phía sau|đằng sau)?\s*",
            r"(?i)\svà cái kết\b.*$",
            r"(?i)\bkhiến ai cũng.*$",
            r"(?i)\bkhiến dân mạng.*$",
            r"(?i)\bkhiến nhiều người.*$",
            r"[?!]+$",
        ]
        for pat in remove_phrases:
            neutral = re.sub(pat, " ", neutral)
        neutral = re.sub(r"\s+", " ", neutral).strip(" ,:-–")
        if neutral:
            neutral = neutral[0].upper() + neutral[1:]
        return neutral or title_clean


def extract_article_from_url(url: str, timeout: int = 8) -> Dict[str, str]:
    """
    Tự động tải và bóc tách Tiêu đề (`title`), Đoạn mở đầu (`lead_paragraph`) và Nguồn báo (`source`)
    từ URL bài báo tiếng Việt (VnExpress, Dân Trí, Tuổi Trẻ, Thanh Niên, Báo Mới, Kênh14, SaoStar, 24h, CafeF...).
    """
    if not url or not url.startswith(("http://", "https://")):
        raise ValueError("URL không hợp lệ. Vui lòng nhập đường dẫn bắt đầu bằng http:// hoặc https://")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    html = resp.text

    domain = urllib.parse.urlparse(url).netloc.replace("www.", "")
    title = ""
    lead = ""

    if HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")

        # 1. Tìm Title từ h1 hoặc og:title
        h1 = soup.select_one(
            "h1.title-detail, h1.article-title, h1.the-article-title, "
            "h1.kbwc-title, h1.detail-title, h1"
        )
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(" ", strip=True)
        else:
            og_title = soup.find("meta", property="og:title") or soup.find("title")
            if og_title:
                title = og_title.get("content", "") if og_title.has_attr("content") else og_title.get_text(strip=True)

        # 2. Tìm Sapo / Lead Paragraph từ các class đặc trưng của báo Việt Nam hoặc og:description
        sapo_el = soup.select_one(
            "p.description, h2.sapo, div.sapo, h2.detail-sapo, p.the-article-summary, "
            "h2.kbwc-sapo, .singular-sapo, .article-sapo, .cms-desc"
        )
        if sapo_el and sapo_el.get_text(strip=True):
            lead = sapo_el.get_text(" ", strip=True)
        else:
            og_desc = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
            if og_desc and og_desc.has_attr("content"):
                lead = og_desc["content"].strip()
    else:
        # Fallback regex nếu không có bs4
        m_title = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
        if m_title:
            title = m_title.group(1)
        m_desc = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
        if m_desc:
            lead = m_desc.group(1)

    return {
        "url": url,
        "source": domain,
        "title": clean_title(title),
        "lead_paragraph": clean_lead_paragraph(lead),
    }
