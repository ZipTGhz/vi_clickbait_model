"""
Module đánh giá mô hình (Evaluation), trực quan hóa dữ liệu (EDA) và phân tích lỗi (Error Analysis).
Cung cấp:
- Tính toán đầy đủ các độ đo: Accuracy, Precision (Clickbait), Recall (Clickbait), F1 (Clickbait), Macro F1, ROC-AUC, Training Time (s), Inference Time (ms/sample).
- Sinh các biểu đồ EDA (Phân phối nhãn, Nguồn báo & Chuyên mục, Độ dài văn bản, Dấu câu & Đặc trưng, WordCloud).
- Sinh các biểu đồ đánh giá mô hình (Confusion Matrices đặt cạnh nhau, Đường cong ROC gộp chung, Biểu đồ so sánh Macro F1 & Ablation Study, Top 20 Feature Importance).
"""

import os
import json
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# Cấu hình font tiếng Việt chuẩn trên Windows (Arial / Segoe UI / Tahoma)
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Tahoma", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Segoe UI")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_group: str,
    model_name: str,
    input_type: str,
    train_time_sec: float = 0.0,
    infer_time_ms_per_sample: float = 0.0,
) -> Dict[str, Any]:
    """
    Tính toán toàn bộ các độ đo đánh giá trên tập dữ liệu (Validation/Test).
    """
    acc = accuracy_score(y_true, y_pred)
    prec_cb = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_cb = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_cb = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_non_cb = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "group": model_group,
        "model": model_name,
        "input_type": input_type,
        "accuracy": float(acc),
        "precision_clickbait": float(prec_cb),
        "recall_clickbait": float(rec_cb),
        "f1_clickbait": float(f1_cb),
        "f1_non_clickbait": float(f1_non_cb),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc),
        "train_time_sec": float(train_time_sec),
        "infer_ms_per_sample": float(infer_time_ms_per_sample),
        "confusion_matrix": cm,
    }


def generate_eda_figures(
    df: pd.DataFrame,
    figures_dir: str = "outputs/figures",
    results_dir: str = "outputs/results",
) -> Dict[str, Any]:
    """
    Thực hiện Bước 1 (EDA) và xuất toàn bộ biểu đồ chất lượng cao vào outputs/figures/.
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Phân phối nhãn (Label Distribution)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=180)
    label_counts = df["label"].value_counts()
    colors = ["#2b5c8f", "#d95f02"]

    bars = axes[0].bar(
        label_counts.index,
        label_counts.values,
        color=colors,
        width=0.55,
        edgecolor="black",
        alpha=0.9,
    )
    axes[0].set_title("Số lượng mẫu theo Nhãn (Label Count)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Nhãn (Label)", fontsize=11)
    axes[0].set_ylabel("Số lượng bài báo", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            h + 25,
            f"{int(h):,}\n({h/len(df)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    axes[0].set_ylim(0, max(label_counts.values) * 1.18)

    axes[1].pie(
        label_counts.values,
        labels=label_counts.index,
        autopct="%1.2f%%",
        startangle=140,
        colors=colors,
        explode=(0.03, 0.03),
        textprops={"fontsize": 11, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[1].set_title("Tỷ lệ phân bố Nhãn (Imbalanced Ratio ~ 68.8% : 31.2%)", fontsize=13, fontweight="bold")
    plt.suptitle("BƯỚC 1 - PHÂN PHỐI NHÃN DỮ LIỆU (CLICKBAIT VS NON-CLICKBAIT)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "01_label_distribution.png"), bbox_inches="tight")
    plt.close(fig)

    # 2. Phân phối theo Nguồn báo (Source) & Chuyên mục (Category) - Stacked Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=180)

    source_ct = pd.crosstab(df["source"], df["label"])
    source_pct = source_ct.div(source_ct.sum(axis=1), axis=0) * 100
    source_pct = source_pct.sort_values(by="clickbait", ascending=True)

    source_pct[["non-clickbait", "clickbait"]].plot(
        kind="barh",
        stacked=True,
        color=["#2b5c8f", "#d95f02"],
        ax=axes[0],
        edgecolor="black",
        width=0.72,
    )
    axes[0].set_title("Tỷ lệ Clickbait theo Nguồn báo (Source Bias / Leakage Risk)", fontsize=12.5, fontweight="bold")
    axes[0].set_xlabel("Tỷ lệ phần trăm (%)", fontsize=11)
    axes[0].set_ylabel("Nguồn báo (Source)", fontsize=11)
    axes[0].legend(title="Nhãn", loc="lower right")
    for i, (idx, row) in enumerate(source_pct.iterrows()):
        cb_val = row["clickbait"]
        total_cnt = source_ct.loc[idx].sum()
        axes[0].text(
            101,
            i,
            f"{cb_val:.1f}% CB (n={total_cnt})",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
    axes[0].set_xlim(0, 125)

    cat_ct = pd.crosstab(df["category"], df["label"])
    cat_pct = cat_ct.div(cat_ct.sum(axis=1), axis=0) * 100
    cat_pct = cat_pct.sort_values(by="clickbait", ascending=True)

    cat_pct[["non-clickbait", "clickbait"]].plot(
        kind="barh",
        stacked=True,
        color=["#2b5c8f", "#d95f02"],
        ax=axes[1],
        edgecolor="black",
        width=0.75,
    )
    axes[1].set_title("Tỷ lệ Clickbait theo Chuyên mục (Category)", fontsize=12.5, fontweight="bold")
    axes[1].set_xlabel("Tỷ lệ phần trăm (%)", fontsize=11)
    axes[1].set_ylabel("Chuyên mục", fontsize=11)
    axes[1].legend(title="Nhãn", loc="lower right")
    for i, (idx, row) in enumerate(cat_pct.iterrows()):
        cb_val = row["clickbait"]
        total_cnt = cat_ct.loc[idx].sum()
        axes[1].text(
            101,
            i,
            f"{cb_val:.1f}% (n={total_cnt})",
            va="center",
            fontsize=9,
        )
    axes[1].set_xlim(0, 125)

    plt.suptitle("PHÂN TÍCH THIÊN LỆCH NGUỒN BÁO (SOURCE) VÀ CHUYÊN MỤC (CATEGORY)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "02_source_category_distribution.png"), bbox_inches="tight")
    plt.close(fig)

    # 3. Thống kê độ dài văn bản (Title & Lead Paragraph: Word Count & Character Length)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=180)
    palette = {"non-clickbait": "#2b5c8f", "clickbait": "#d95f02"}

    sns.histplot(
        data=df, x="title_word_count", hue="label", kde=True, stat="density",
        common_norm=False, palette=palette, ax=axes[0, 0], bins=25, alpha=0.45
    )
    axes[0, 0].set_title("Phân phối Số từ (Âm tiết) của Tiêu đề (Title)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Số âm tiết trong Title")

    sns.histplot(
        data=df, x="title_char_len", hue="label", kde=True, stat="density",
        common_norm=False, palette=palette, ax=axes[0, 1], bins=30, alpha=0.45
    )
    axes[0, 1].set_title("Phân phối Số ký tự của Tiêu đề (Title)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Số ký tự trong Title")

    sns.histplot(
        data=df[df["lead_word_count"] < 120], x="lead_word_count", hue="label", kde=True, stat="density",
        common_norm=False, palette=palette, ax=axes[1, 0], bins=30, alpha=0.45
    )
    axes[1, 0].set_title("Phân phối Số từ (Âm tiết) của Đoạn Sapo (Lead Paragraph)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Số âm tiết trong Lead Paragraph")

    sns.histplot(
        data=df[df["lead_char_len"] < 600], x="lead_char_len", hue="label", kde=True, stat="density",
        common_norm=False, palette=palette, ax=axes[1, 1], bins=30, alpha=0.45
    )
    axes[1, 1].set_title("Phân phối Số ký tự của Đoạn Sapo (Lead Paragraph)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Số ký tự trong Lead Paragraph")

    plt.suptitle("THỐNG KÊ ĐỘ DÀI VĂN BẢN (TITLE & LEAD PARAGRAPH) THEO NHÃN", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "03_text_length_distribution.png"), bbox_inches="tight")
    plt.close(fig)

    # 4. Phân tích dấu câu & từ khóa đặc trưng trong Tiêu đề
    feature_rates = []
    feature_defs = [
        ("has_question", "Dấu chấm hỏi (?)"),
        ("has_exclamation", "Dấu chấm than (!)"),
        ("has_quotes", "Dấu ngoặc kép/đơn (\" / ')"),
        ("has_number", "Chứa con số (0-9)"),
        ("has_ellipsis", "Dấu ba chấm (...)"),
    ]
    df["has_all_caps"] = (df["all_caps_count"] > 0).astype(int)
    feature_defs.append(("has_all_caps", "Từ viết hoa (ALL CAPS)"))

    for col, label_vn in feature_defs:
        for lbl in ["non-clickbait", "clickbait"]:
            sub = df[df["label"] == lbl]
            rate = sub[col].mean() * 100
            feature_rates.append({"Feature": label_vn, "Label": lbl, "Rate (%)": rate})

    rate_df = pd.DataFrame(feature_rates)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    sns.barplot(
        data=rate_df,
        x="Feature",
        y="Rate (%)",
        hue="Label",
        palette=palette,
        edgecolor="black",
        ax=ax,
    )
    ax.set_title("Tỷ lệ xuất hiện Dấu câu & Đặc trưng hình thái trong Tiêu đề (%)", fontsize=13.5, fontweight="bold")
    ax.set_xlabel("Đặc trưng ngôn ngữ / Dấu câu", fontsize=11)
    ax.set_ylabel("Tỷ lệ bài báo có chứa đặc trưng (%)", fontsize=11)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, max(rate_df["Rate (%)"]) * 1.18)
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "04_punctuation_features.png"), bbox_inches="tight")
    plt.close(fig)

    # 5. WordCloud cho từng nhãn (Clickbait vs Non-Clickbait)
    if HAS_WORDCLOUD:
        font_path = "C:/Windows/Fonts/arial.ttf"
        if not os.path.exists(font_path):
            font_path = None

        cb_text = " ".join(df[df["label"] == "clickbait"]["title_seg"].dropna().tolist())
        ncb_text = " ".join(df[df["label"] == "non-clickbait"]["title_seg"].dropna().tolist())

        stopwords_vn = {
            "của", "và", "các", "có", "được", "trong", "cho", "với", "không",
            "những", "một", "là", "đến", "từ", "này", "khi", "về", "đã", "sẽ",
            "tại", "sau", "trên", "để", "người", "bị", "ra", "theo", "nhiều",
            "cùng", "như", "cũng", "vào", "lại", "ở", "do", "vì", "hơn", "năm"
        }

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=180)
        wc_cb = WordCloud(
            width=900, height=550, background_color="white", colormap="OrRd",
            stopwords=stopwords_vn, max_words=100, font_path=font_path, random_state=42
        ).generate(cb_text)
        axes[0].imshow(wc_cb, interpolation="bilinear")
        axes[0].set_title("WordCloud Tiêu đề Nhãn CLICKBAIT", fontsize=14, fontweight="bold", color="#b33000")
        axes[0].axis("off")

        wc_ncb = WordCloud(
            width=900, height=550, background_color="white", colormap="Blues",
            stopwords=stopwords_vn, max_words=100, font_path=font_path, random_state=42
        ).generate(ncb_text)
        axes[1].imshow(wc_ncb, interpolation="bilinear")
        axes[1].set_title("WordCloud Tiêu đề Nhãn NON-CLICKBAIT", fontsize=14, fontweight="bold", color="#1b4f72")
        axes[1].axis("off")

        plt.suptitle("SO SÁNH TỪ VỰNG ĐẶC TRƯNG (WORDCLOUD) TRONG TIÊU ĐỀ HAI NHÃN", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "05_wordcloud_comparison.png"), bbox_inches="tight")
        plt.close(fig)

    # Lưu thống kê chi tiết EDA ra JSON
    eda_stats = {
        "label_counts": label_counts.to_dict(),
        "source_vs_label": source_ct.to_dict(),
        "source_clickbait_pct": source_pct["clickbait"].round(2).to_dict(),
        "category_vs_label": cat_ct.to_dict(),
        "category_clickbait_pct": cat_pct["clickbait"].round(2).to_dict(),
        "length_stats": df.groupby("label")[
            ["title_word_count", "title_char_len", "lead_word_count", "lead_char_len"]
        ].agg(["mean", "std", "median"]).round(2).to_dict(),
        "punctuation_rates": rate_df.to_dict(orient="records"),
    }
    with open(os.path.join(results_dir, "eda_statistics.json"), "w", encoding="utf-8") as f:
        # Chuyển tuple keys trong length_stats sang string
        serializable_stats = dict(eda_stats)
        serializable_stats["length_stats"] = {
            f"{k[0]}_{k[1]}": v for k, v in eda_stats["length_stats"].items()
        }
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2)

    print("[EDA] Đã sinh thành công 5 bộ biểu đồ EDA vào thư mục outputs/figures/.")
    return eda_stats


def plot_evaluation_charts(
    results_list: List[Dict[str, Any]],
    predictions_dict: Dict[str, Dict[str, Any]],
    y_true: np.ndarray,
    figures_dir: str = "outputs/figures",
    results_dir: str = "outputs/results",
) -> pd.DataFrame:
    """
    Xuất bảng tổng hợp kết quả (CSV & Markdown) và vẽ:
    1. Confusion Matrices của tất cả mô hình đặt cạnh nhau.
    2. Đường cong ROC (ROC Curve) gộp chung các mô hình trên 1 đồ thị.
    3. Biểu đồ cột so sánh Macro F1, F1 Clickbait và ROC-AUC.
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    res_df = pd.DataFrame(results_list)
    res_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False, encoding="utf-8")

    # Tạo bảng Markdown theo chuẩn tutorial.md
    md_lines = [
        "| Nhóm | Mô hình | Đầu vào | Accuracy | Precision (Clickbait) | Recall (Clickbait) | F1 (Clickbait) | Macro F1 | ROC-AUC | Train Time (s) | Infer (ms/mẫu) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    best_macro_f1 = res_df["macro_f1"].max()
    for _, r in res_df.iterrows():
        is_best = abs(r["macro_f1"] - best_macro_f1) < 1e-6
        b = "**" if is_best else ""
        md_lines.append(
            f"| {b}{r['group']}{b} | {b}{r['model']}{b} | {b}{r['input_type']}{b} | "
            f"{b}{r['accuracy']:.4f}{b} | {b}{r['precision_clickbait']:.4f}{b} | "
            f"{b}{r['recall_clickbait']:.4f}{b} | {b}{r['f1_clickbait']:.4f}{b} | "
            f"{b}{r['macro_f1']:.4f}{b} | {b}{r['roc_auc']:.4f}{b} | "
            f"{r['train_time_sec']:.2f}s | {r['infer_ms_per_sample']:.3f} |"
        )

    with open(os.path.join(results_dir, "model_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # 1. Vẽ Confusion Matrices đặt cạnh nhau (cho các mô hình chính)
    main_keys = list(predictions_dict.keys())
    n_models = len(main_keys)
    ncols = 4
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows), dpi=180)
    axes = np.array(axes).reshape(-1)

    for idx, key in enumerate(main_keys):
        info = predictions_dict[key]
        cm = confusion_matrix(y_true, info["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Non-CB", "Clickbait"],
            yticklabels=["Non-CB", "Clickbait"],
            annot_kws={"size": 12, "weight": "bold"},
            ax=axes[idx],
        )
        axes[idx].set_title(f"{key}\n(Macro F1: {info['macro_f1']:.4f})", fontsize=10.5, fontweight="bold")
        axes[idx].set_xlabel("Dự đoán (Predicted)", fontsize=9.5)
        axes[idx].set_ylabel("Thực tế (Actual)", fontsize=9.5)

    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("MA TRẬN NHẦM LẪN (CONFUSION MATRIX) CỦA CÁC MÔ HÌNH TRÊN TẬP TEST", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "06_confusion_matrices.png"), bbox_inches="tight")
    plt.close(fig)

    # 2. Vẽ Đường cong ROC (ROC Curve) gộp chung trên 1 đồ thị
    fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=180)
    cmap = plt.cm.get_cmap("tab10", len(main_keys))

    for idx, key in enumerate(main_keys):
        info = predictions_dict[key]
        fpr, tpr, _ = roc_curve(y_true, info["y_prob"])
        auc_val = roc_auc_score(y_true, info["y_prob"])
        lw = 2.6 if "PhoBERT" in key or "SVM" in key else 1.8
        ls = "-" if "Title + Lead" in key else "--"
        ax.plot(fpr, tpr, lw=lw, linestyle=ls, label=f"{key} (AUC = {auc_val:.4f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle=":", label="Random Guess (AUC = 0.5000)")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    ax.set_title("ĐƯỜNG CONG ROC (ROC CURVE) SO SÁNH TẤT CẢ MÔ HÌNH TRÊN TẬP TEST", fontsize=13.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "07_roc_curves.png"), bbox_inches="tight")
    plt.close(fig)

    # 3. Biểu đồ cột ngang so sánh Macro F1 & F1 Clickbait
    fig, ax = plt.subplots(figsize=(12.5, 7), dpi=180)
    plot_df = res_df.copy()
    plot_df["label_name"] = plot_df["model"] + " (" + plot_df["input_type"] + ")"
    plot_df = plot_df.sort_values(by="macro_f1", ascending=True)

    y_pos = np.arange(len(plot_df))
    height = 0.38

    bars1 = ax.barh(y_pos + height / 2, plot_df["macro_f1"], height, label="Macro F1-Score", color="#1f77b4", edgecolor="black")
    bars2 = ax.barh(y_pos - height / 2, plot_df["f1_clickbait"], height, label="F1-Score (Clickbait)", color="#ff7f0e", edgecolor="black")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label_name"], fontsize=10, fontweight="bold")
    ax.set_xlabel("Điểm số (Score)", fontsize=11)
    ax.set_title("SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH TRÊN TẬP TEST (MACRO F1 & CLICKBAIT F1)", fontsize=13, fontweight="bold")
    ax.set_xlim(0.5, 1.0)
    ax.legend(loc="lower right", fontsize=10.5)

    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=8.5, fontweight="bold", color="#1f77b4")
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=8.5, color="#d95f02")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "08_model_comparison_bars.png"), bbox_inches="tight")
    plt.close(fig)

    print("[EVALUATE] Đã lưu bảng kết quả và biểu đồ đánh giá vào outputs/figures/ & outputs/results/.")
    return res_df


def plot_feature_importance(
    top_pos: List[Dict[str, Any]],
    top_neg: List[Dict[str, Any]],
    figures_dir: str = "outputs/figures",
) -> None:
    """
    Vẽ biểu đồ Top 20 đặc trưng có trọng số dương lớn nhất (Clickbait)
    và Top 20 đặc trưng có trọng số âm lớn nhất (Non-Clickbait).
    """
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=180)

    pos_df = pd.DataFrame(top_pos).sort_values(by="weight", ascending=True)
    neg_df = pd.DataFrame(top_neg).sort_values(by="weight", ascending=False)

    axes[0].barh(pos_df["feature"], pos_df["weight"], color="#d95f02", edgecolor="black", alpha=0.9)
    axes[0].set_title("Top 20 Đặc trưng / Từ khóa đẩy về nhãn CLICKBAIT (+)", fontsize=12.5, fontweight="bold", color="#b33000")
    axes[0].set_xlabel("Trọng số mô hình (Coefficient Weight)")
    for i, (feat, w) in enumerate(zip(pos_df["feature"], pos_df["weight"])):
        axes[0].text(w + 0.02, i, f"+{w:.3f}", va="center", fontsize=9, fontweight="bold")
    axes[0].set_xlim(0, max(pos_df["weight"]) * 1.18)

    axes[1].barh(neg_df["feature"], neg_df["weight"], color="#2b5c8f", edgecolor="black", alpha=0.9)
    axes[1].set_title("Top 20 Đặc trưng / Từ khóa đẩy về nhãn NON-CLICKBAIT (-)", fontsize=12.5, fontweight="bold", color="#1b4f72")
    axes[1].set_xlabel("Trọng số mô hình (Coefficient Weight)")
    for i, (feat, w) in enumerate(zip(neg_df["feature"], neg_df["weight"])):
        axes[1].text(w - 0.02, i, f"{w:.3f}", va="center", ha="right", fontsize=9, fontweight="bold")
    axes[1].set_xlim(min(neg_df["weight"]) * 1.18, 0)

    plt.suptitle("PHÂN TÍCH MỨC ĐỘ ĐÓNG GÓP ĐẶC TRƯNG (FEATURE IMPORTANCE - LINEAR SVM / LOGISTIC REGRESSION)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "09_feature_importance_top20.png"), bbox_inches="tight")
    plt.close(fig)
