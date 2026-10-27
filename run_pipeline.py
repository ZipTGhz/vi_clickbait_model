"""
Script điều phối toàn bộ Pipeline từ A đến Z cho Bài tập lớn NLP: Phát hiện Clickbait Tiếng Việt.
Thực hiện tuần tự:
1. Tiền xử lý dữ liệu đặc thù (Missing values, lỗi dính địa danh VnExpress, lỗi đuôi báo Báo Mới, chuẩn hóa Unicode/dấu thanh, tách từ PyVi, chia tập 70/15/15).
2. Khám phá dữ liệu (EDA) và sinh 5 bộ biểu đồ phân tích lưu vào outputs/figures/.
3. Huấn luyện & đánh giá Nhóm 1 (Baseline ML: Naive Bayes, Logistic Regression, Linear SVM, Random Forest, XGBoost) + Feature Importance.
4. Huấn luyện & đánh giá Nhóm 2 (Deep Learning: BiLSTM, TextCNN).
5. Fine-tuning & đánh giá Nhóm 3 (Transformer SOTA: PhoBERT-base-v2 với Only Title & Title + Lead) + Error Analysis.
6. Tổng hợp bảng so sánh (CSV/Markdown) và vẽ Confusion Matrices, ROC Curves, Model Comparison Bar Chart.
"""

import os
import sys
import json
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from src.preprocess import run_preprocessing_pipeline
from src.evaluate import generate_eda_figures, plot_evaluation_charts
from src.models_ml import run_ml_experiments
from src.models_dl import run_dl_experiments
from src.models_bert import run_bert_experiments


def main():
    print("=" * 90)
    print("BẮT ĐẦU PIPELINE PHÁT HIỆN CLICKBAIT TIẾNG VIỆT (NLP CLICKBAIT DETECTION)")
    print("=" * 90)

    # Bước 1 & 2: Tiền xử lý dữ liệu và Khám phá dữ liệu (EDA)
    train_df, val_df, test_df = run_preprocessing_pipeline(
        raw_csv_path="clickbait_dataset_vietnamese.csv",
        output_dir="data/processed",
        raw_copy_dir="data/raw",
        results_dir="outputs/results",
        random_state=42,
    )

    full_df = pd.read_csv("data/processed/full_processed.csv", encoding="utf-8")
    generate_eda_figures(full_df, figures_dir="outputs/figures", results_dir="outputs/results")

    # Bước 3.1: Huấn luyện Nhóm 1 - Học máy truyền thống (ML Baselines)
    ml_results, ml_preds = run_ml_experiments(
        train_df, val_df, test_df, results_dir="outputs/results", figures_dir="outputs/figures"
    )

    # Bước 3.2: Huấn luyện Nhóm 2 - Học sâu truyền thống (BiLSTM, TextCNN)
    dl_results, dl_preds = run_dl_experiments(
        train_df, val_df, test_df, results_dir="outputs/results"
    )

    # Bước 3.3: Huấn luyện Nhóm 3 - Pre-trained Transformer (PhoBERT-base-v2) & Phân tích lỗi
    bert_results, bert_preds, error_summary = run_bert_experiments(
        train_df, val_df, test_df, results_dir="outputs/results"
    )

    # Bước 4: Tổng hợp kết quả và vẽ biểu đồ đánh giá
    all_results = ml_results + dl_results + bert_results
    all_preds = {}
    all_preds.update(ml_preds)
    all_preds.update(dl_preds)
    all_preds.update(bert_preds)

    # Chọn các mô hình tiêu biểu nhất để vẽ Confusion Matrix & ROC Curve rõ nét, trực quan
    selected_keys_for_plots = [
        "TF-IDF + Naive Bayes (Title + Lead)",
        "TF-IDF + Logistic Regression (Title + Lead)",
        "TF-IDF + Linear SVM (Title + Lead)",
        "TF-IDF + XGBoost (Title + Lead)",
        "BiLSTM (Title + Lead)",
        "TextCNN (Title + Lead)",
        "PhoBERT-base-v2 (Only Title)",
        "PhoBERT-base-v2 (Title + Lead)",
    ]
    filtered_preds = {k: all_preds[k] for k in selected_keys_for_plots if k in all_preds}

    res_df = plot_evaluation_charts(
        results_list=all_results,
        predictions_dict=filtered_preds,
        y_true=test_df["label_id"].values,
        figures_dir="outputs/figures",
        results_dir="outputs/results",
    )

    with open("outputs/results/all_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM TRÊN TẬP TEST:")
    print("=" * 90)
    with open("outputs/results/model_comparison.md", "r", encoding="utf-8") as f:
        print(f.read())
    print("=" * 90)
    print("HOÀN TẤT PIPELINE THÀNH CÔNG!")


if __name__ == "__main__":
    main()
