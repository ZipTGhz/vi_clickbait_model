"""
Module huấn luyện và đánh giá Nhóm 1: Các mô hình Học máy truyền thống (Baseline ML Models).
Bao gồm:
- Trích xuất đặc trưng: TF-IDF (ngram_range=(1, 2), max_features=10000) kết hợp (Concatenate) với Hand-crafted features (độ dài tiêu đề, dấu ?, dấu !, dấu ngoặc kép, số lượng chữ số, ALL CAPS...).
- Các thuật toán:
  1. Multinomial Naive Bayes
  2. Logistic Regression (class_weight='balanced')
  3. Linear Support Vector Machine (LinearSVC + CalibratedClassifierCV)
  4. Random Forest & XGBoost
- Trích xuất Feature Importance (Top 20 n-grams hướng về Clickbait và Top 20 n-grams hướng về Non-Clickbait).
"""

import os
import time
import json
import joblib
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.preprocess import HANDCRAFTED_FEATURE_COLS
from src.evaluate import compute_metrics, plot_feature_importance


def build_feature_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    max_features: int = 10000,
) -> Tuple[csr_matrix, csr_matrix, csr_matrix, TfidfVectorizer, MinMaxScaler, List[str]]:
    """
    Xây dựng ma trận đặc trưng kết hợp (TF-IDF 1-2 grams + Hand-crafted features đã chuẩn hóa MinMax [0, 1]).
    Khi đầu vào là Title + Lead (input_option_b), tách riêng không gian TF-IDF của Title và Lead
    để tránh hiện tượng đoạn sapo dài làm loãng trọng số TF-IDF của các từ khóa giật tít trên tiêu đề.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
        token_pattern=r"(?u)\b\w[\w_]*\b|[?!\"']",
    )

    X_train_tfidf = vectorizer.fit_transform(train_df[text_col].fillna(""))
    X_val_tfidf = vectorizer.transform(val_df[text_col].fillna(""))
    X_test_tfidf = vectorizer.transform(test_df[text_col].fillna(""))

    scaler = MinMaxScaler()
    X_train_hc = scaler.fit_transform(train_df[HANDCRAFTED_FEATURE_COLS].fillna(0).values)
    X_val_hc = scaler.transform(val_df[HANDCRAFTED_FEATURE_COLS].fillna(0).values)
    X_test_hc = scaler.transform(test_df[HANDCRAFTED_FEATURE_COLS].fillna(0).values)

    tfidf_names = vectorizer.get_feature_names_out().tolist()

    if text_col == "input_option_b":
        title_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=6000,
            sublinear_tf=True,
            min_df=2,
            token_pattern=r"(?u)\b\w[\w_]*\b|[?!\"']",
        )
        X_tr_title = title_vec.fit_transform(train_df["title_seg"].fillna("")) * 1.25
        X_va_title = title_vec.transform(val_df["title_seg"].fillna("")) * 1.25
        X_te_title = title_vec.transform(test_df["title_seg"].fillna("")) * 1.25

        X_train = hstack([X_train_tfidf, X_tr_title, csr_matrix(X_train_hc)], format="csr")
        X_val = hstack([X_val_tfidf, X_va_title, csr_matrix(X_val_hc)], format="csr")
        X_test = hstack([X_test_tfidf, X_te_title, csr_matrix(X_test_hc)], format="csr")
        title_names = [f"[TITLE]_{w}" for w in title_vec.get_feature_names_out().tolist()]
        all_feature_names = tfidf_names + title_names + [f"[HC]_{c}" for c in HANDCRAFTED_FEATURE_COLS]
    else:
        X_train = hstack([X_train_tfidf, csr_matrix(X_train_hc)], format="csr")
        X_val = hstack([X_val_tfidf, csr_matrix(X_val_hc)], format="csr")
        X_test = hstack([X_test_tfidf, csr_matrix(X_test_hc)], format="csr")
        all_feature_names = tfidf_names + [f"[HC]_{c}" for c in HANDCRAFTED_FEATURE_COLS]

    return X_train, X_val, X_test, vectorizer, scaler, all_feature_names


def extract_top_features(
    coef: np.ndarray,
    feature_names: List[str],
    top_k: int = 20,
    only_ngrams: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Trích xuất Top K đặc trưng có trọng số dương lớn nhất (Clickbait)
    và Top K đặc trưng có trọng số âm lớn nhất (Non-Clickbait).
    """
    weights = coef.flatten()
    best_by_token: Dict[str, float] = {}
    for name, w in zip(feature_names, weights):
        if only_ngrams and (name.startswith("[HC]_") or name in ("sep", "[TITLE]_sep")):
            continue
        clean_name = name.replace("[TITLE]_", "")
        if clean_name not in best_by_token or abs(float(w)) > abs(best_by_token[clean_name]):
            best_by_token[clean_name] = float(w)

    pairs = list(best_by_token.items())
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    top_pos = [{"feature": k, "weight": v} for k, v in pairs_sorted[:top_k]]
    top_neg = [{"feature": k, "weight": v} for k, v in sorted(pairs, key=lambda x: x[1])[:top_k]]
    return top_pos, top_neg


def run_ml_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results_dir: str = "outputs/results",
    figures_dir: str = "outputs/figures",
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Huấn luyện và đánh giá các mô hình Học máy truyền thống trên cả Option A (Only Title) và Option B (Title + Lead).
    Trả về danh sách kết quả metrics và từ điển chứa dự đoán trên tập Test.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    y_train = train_df["label_id"].values
    y_val = val_df["label_id"].values
    y_test = test_df["label_id"].values

    results_list: List[Dict[str, Any]] = []
    predictions_dict: Dict[str, Dict[str, Any]] = {}
    feature_importance_data: Dict[str, Any] = {}

    input_configs = [
        ("Only Title", "input_option_a"),
        ("Title + Lead", "input_option_b"),
    ]

    for input_label, text_col in input_configs:
        print(f"\n[ML MODELS] Đang trích xuất đặc trưng TF-IDF + Hand-crafted cho đầu vào: {input_label}...")
        X_train, X_val, X_test, vectorizer, scaler, feature_names = build_feature_matrices(
            train_df, val_df, test_df, text_col=text_col, max_features=10000
        )

        # Khởi tạo danh sách mô hình học máy
        pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        models_to_run = [
            ("TF-IDF + Naive Bayes", MultinomialNB(alpha=0.5)),
            (
                "TF-IDF + Logistic Regression",
                LogisticRegression(class_weight="balanced", C=1.5, max_iter=1000, random_state=42),
            ),
            (
                "TF-IDF + Linear SVM",
                CalibratedClassifierCV(
                    LinearSVC(class_weight="balanced", C=0.8, random_state=42, max_iter=3000),
                    cv=5,
                ),
            ),
            (
                "TF-IDF + Random Forest",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
        if HAS_XGB:
            models_to_run.append(
                (
                    "TF-IDF + XGBoost",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=6,
                        learning_rate=0.08,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        scale_pos_weight=pos_weight,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            )

        for model_name, clf in models_to_run:
            # Để bảng kết quả trọng tâm đúng chuẩn tutorial.md:
            # Chạy tất cả mô hình trên Title + Lead, và chạy Logistic Regression + Linear SVM trên Only Title để làm Ablation Study
            if input_label == "Only Title" and model_name not in ("TF-IDF + Logistic Regression", "TF-IDF + Linear SVM"):
                continue

            print(f"  -> Đang huấn luyện {model_name} ({input_label})...")
            t0 = time.perf_counter()
            clf.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            t_inf0 = time.perf_counter()
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            infer_time_ms = (time.perf_counter() - t_inf0) * 1000.0 / len(y_test)

            metrics = compute_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                model_group="Baseline (ML)",
                model_name=model_name,
                input_type=input_label,
                train_time_sec=train_time,
                infer_time_ms_per_sample=infer_time_ms,
            )
            results_list.append(metrics)

            key_name = f"{model_name} ({input_label})"
            predictions_dict[key_name] = {
                "y_pred": y_pred,
                "y_prob": y_prob,
                "macro_f1": metrics["macro_f1"],
            }

            print(
                f"     [KẾT QUẢ TEST] Acc={metrics['accuracy']:.4f} | "
                f"Prec(CB)={metrics['precision_clickbait']:.4f} | "
                f"Rec(CB)={metrics['recall_clickbait']:.4f} | "
                f"Macro-F1={metrics['macro_f1']:.4f} | AUC={metrics['roc_auc']:.4f}"
            )

            # Trích xuất Feature Importance trên Title + Lead
            if input_label == "Title + Lead" and model_name == "TF-IDF + Logistic Regression":
                top_pos_lr, top_neg_lr = extract_top_features(clf.coef_, feature_names, top_k=20, only_ngrams=True)
                top_pos_all, top_neg_all = extract_top_features(clf.coef_, feature_names, top_k=25, only_ngrams=False)
                feature_importance_data["logistic_regression"] = {
                    "top_20_clickbait_ngrams": top_pos_lr,
                    "top_20_non_clickbait_ngrams": top_neg_lr,
                    "top_25_including_handcrafted": top_pos_all,
                }
                plot_feature_importance(top_pos_lr, top_neg_lr, figures_dir=figures_dir)

            if input_label == "Title + Lead" and model_name == "TF-IDF + Linear SVM":
                # Lấy trung bình trọng số từ các base estimators của CalibratedClassifierCV
                coefs = [est.estimator.coef_ for est in clf.calibrated_classifiers_]
                avg_coef = np.mean(coefs, axis=0)
                top_pos_svm, top_neg_svm = extract_top_features(avg_coef, feature_names, top_k=20, only_ngrams=True)
                feature_importance_data["linear_svm"] = {
                    "top_20_clickbait_ngrams": top_pos_svm,
                    "top_20_non_clickbait_ngrams": top_neg_svm,
                }
                # Lưu checkpoint mô hình ML tốt nhất
                joblib.dump(
                    {"model": clf, "vectorizer": vectorizer, "scaler": scaler},
                    os.path.join(results_dir, "best_ml_linear_svm.joblib"),
                )

    with open(os.path.join(results_dir, "feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump(feature_importance_data, f, ensure_ascii=False, indent=2)

    return results_list, predictions_dict
