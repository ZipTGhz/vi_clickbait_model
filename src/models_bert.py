"""
Module Fine-tuning Nhóm 3: Mô hình Ngôn ngữ Lớn / Pre-trained Transformers (PhoBERT-base-v2)
và Phân tích lỗi (Error Analysis - False Positives & False Negatives).
Thực hiện:
- Fine-tune `vinai/phobert-base-v2` cho bài toán phân loại nhị phân trên 2 cấu hình:
  1. Option A: Only Title (max_length=96)
  2. Option B: Title + Lead Paragraph (max_length=192, truyền cặp câu title_seg & lead_seg)
- Áp dụng Dynamic Padding theo từng mini-batch và Layer-wise Fine-tuning (đóng băng tầng Embedding & 6 tầng Encoder đầu, tinh chỉnh 6 tầng Transformer trên cùng + Classification Head) giúp chống Overfitting trên tập dữ liệu vừa và tăng tốc độ hội tụ.
- Xuất 10 mẫu False Positive và 10 mẫu False Negative kèm phân tích ngôn ngữ học phục vụ báo cáo.
"""

import os
import re
import time
import json
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from src.evaluate import compute_metrics

torch.set_num_threads(os.cpu_count() or 8)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PhoBERTDynamicDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[List[int]]], labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {k: self.encodings[k][idx] for k in self.encodings}
        item["labels"] = int(self.labels[idx])
        return item


def encode_texts(
    tokenizer,
    df: pd.DataFrame,
    mode: str = "title_lead",
    max_length: int = 192,
) -> Dict[str, List[List[int]]]:
    titles = df["title_seg"].fillna("").tolist()
    if mode == "only_title":
        return tokenizer(
            titles,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
    else:
        leads = df["lead_seg"].fillna("").tolist()
        return tokenizer(
            titles,
            leads,
            truncation=True,
            padding=False,
            max_length=max_length,
        )


def finetune_and_evaluate_phobert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_checkpoint: str = "vinai/phobert-base-v2",
    mode: str = "title_lead",
    max_length: int = 160,
    batch_size: int = 24,
    epochs: int = 3,
    lr: float = 4e-5,
    weight_decay: float = 0.01,
    device: torch.device = torch.device("cpu"),
    save_dir: str = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Fine-tune PhoBERT trên cấu hình đầu vào chỉ định (only_title hoặc title_lead)."""
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_enc = encode_texts(tokenizer, train_df, mode=mode, max_length=max_length)
    val_enc = encode_texts(tokenizer, val_df, mode=mode, max_length=max_length)
    test_enc = encode_texts(tokenizer, test_df, mode=mode, max_length=max_length)

    y_train = train_df["label_id"].values
    y_val = val_df["label_id"].values
    y_test = test_df["label_id"].values

    train_dataset = PhoBERTDynamicDataset(train_enc, y_train)
    val_dataset = PhoBERTDynamicDataset(val_enc, y_val)
    test_dataset = PhoBERTDynamicDataset(test_enc, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collator)

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    # Đóng băng tầng Embedding và 6 tầng Encoder đầu (giữ nguyên biểu diễn từ vựng - cú pháp nền tảng của PhoBERT),
    # tập trung tinh chỉnh 6 tầng Transformer cấp cao (ngữ nghĩa) + Classification Head.
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    for layer in model.roberta.encoder.layer[:6]:
        for param in layer.parameters():
            param.requires_grad = False

    model.to(device)

    n_neg = float((y_train == 0).sum())
    n_pos = float((y_train == 1).sum())
    class_weights = torch.tensor([1.0, np.sqrt(n_neg / max(n_pos, 1.0))], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    no_decay = ["bias", "LayerNorm.weight"]
    trainable_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in trainable_named_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in trainable_named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_f1 = -1.0
    best_threshold = 0.5
    best_state = None

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        # Đánh giá trên tập Validation
        model.eval()
        val_probs_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                val_probs_list.append(probs.cpu().numpy())

        val_probs = np.concatenate(val_probs_list)
        cur_best_f1 = -1.0
        cur_best_th = 0.5
        for th in np.linspace(0.35, 0.65, 13):
            preds = (val_probs >= th).astype(int)
            mf1 = f1_score(y_val, preds, average="macro", zero_division=0)
            if mf1 > cur_best_f1:
                cur_best_f1 = mf1
                cur_best_th = float(th)

        print(
            f"     Epoch {epoch}/{epochs} ({time.perf_counter()-ep_t0:.1f}s) | "
            f"Train Loss: {total_loss/len(train_loader):.4f} | "
            f"Val Macro-F1: {cur_best_f1:.4f} (th={cur_best_th:.2f})",
            flush=True,
        )

        if cur_best_f1 > best_val_f1:
            best_val_f1 = cur_best_f1
            best_threshold = cur_best_th
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    model.eval()
    t_inf0 = time.perf_counter()
    test_probs_list = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            test_probs_list.append(probs.cpu().numpy())

    test_probs = np.concatenate(test_probs_list)
    test_preds = (test_probs >= best_threshold).astype(int)
    infer_ms = (time.perf_counter() - t_inf0) * 1000.0 / len(y_test)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_preds, test_probs, train_time, infer_ms


def explain_error_sample(row: pd.Series, error_type: str) -> str:
    """Đưa ra nhận xét ngôn ngữ học tự động dựa trên đặc điểm văn bản của ca dự đoán sai."""
    title = str(row.get("title_clean", ""))
    cat = str(row.get("category", ""))
    if error_type == "FP":
        reasons = []
        if "?" in title or "!" in title:
            reasons.append("tiêu đề dùng câu hỏi tu từ/dấu cảm thán")
        if '"' in title or "'" in title:
            reasons.append("chứa cụm từ trích dẫn trực tiếp tạo cảm giác kịch tính")
        if any(w in title.lower() for w in ["bất ngờ", "xôn xao", "kinh hoàng", "bí mật", "hé lộ", "sốc", "lạ", "hiếm"]):
            reasons.append("chứa tính từ/động từ mạnh thường gặp ở tin giật tít")
        if re.search(r"\d+", title):
            reasons.append("sử dụng con số gây ấn tượng mạnh")
        if not reasons:
            reasons.append(f"nội dung thuộc chủ đề '{cat}' có tình tiết vụ việc kịch tính dù cách đưa tin chính thống")
        return "False Positive (Nhầm Non-Clickbait thành Clickbait) do " + ", ".join(reasons) + "."
    else:
        reasons = []
        if "?" not in title and "!" not in title:
            reasons.append("không dùng dấu câu cảm thán (? / !)")
        if not any(w in title.lower() for w in ["sốc", "ngỡ ngàng", "bất ngờ", "xôn xao"]):
            reasons.append("không chứa từ khóa giật tít lộ liễu mà dùng lối viết ẩn ý/bỏ lửng chủ ngữ tinh vi (curiosity gap)")
        if len(title.split()) <= 12:
            reasons.append("cấu trúc tiêu đề ngắn gọn giống tin tức thông thường")
        return "False Negative (Bỏ sót Clickbait tinh vi) do " + ", ".join(reasons) + "."


def perform_error_analysis(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    results_dir: str = "outputs/results",
) -> Dict[str, Any]:
    """
    Thực hiện Bước 5: Phân tích các ca dự đoán sai của PhoBERT (10 mẫu False Positive & 10 mẫu False Negative).
    """
    os.makedirs(results_dir, exist_ok=True)
    analysis_df = test_df.copy()
    analysis_df["pred_label_id"] = y_pred
    analysis_df["pred_prob_clickbait"] = np.round(y_prob, 4)
    analysis_df["pred_label"] = np.where(y_pred == 1, "clickbait", "non-clickbait")
    analysis_df["is_correct"] = analysis_df["label_id"] == analysis_df["pred_label_id"]

    analysis_df.to_csv(os.path.join(results_dir, "phobert_test_predictions.csv"), index=False, encoding="utf-8")

    fp_df = (
        analysis_df[(analysis_df["label_id"] == 0) & (analysis_df["pred_label_id"] == 1)]
        .sort_values(by="pred_prob_clickbait", ascending=False)
        .head(10)
    )
    fn_df = (
        analysis_df[(analysis_df["label_id"] == 1) & (analysis_df["pred_label_id"] == 0)]
        .sort_values(by="pred_prob_clickbait", ascending=True)
        .head(10)
    )

    fp_samples = []
    for _, row in fp_df.iterrows():
        fp_samples.append({
            "id": row["id"],
            "source": row["source"],
            "category": row["category"],
            "title_clean": row["title_clean"],
            "lead_clean": str(row["lead_clean"])[:180],
            "true_label": "non-clickbait",
            "pred_label": "clickbait",
            "prob_clickbait": float(row["pred_prob_clickbait"]),
            "linguistic_comment": explain_error_sample(row, "FP"),
        })

    fn_samples = []
    for _, row in fn_df.iterrows():
        fn_samples.append({
            "id": row["id"],
            "source": row["source"],
            "category": row["category"],
            "title_clean": row["title_clean"],
            "lead_clean": str(row["lead_clean"])[:180],
            "true_label": "clickbait",
            "pred_label": "non-clickbait",
            "prob_clickbait": float(row["pred_prob_clickbait"]),
            "linguistic_comment": explain_error_sample(row, "FN"),
        })

    total_errors = int((~analysis_df["is_correct"]).sum())
    total_fp = int(((analysis_df["label_id"] == 0) & (analysis_df["pred_label_id"] == 1)).sum())
    total_fn = int(((analysis_df["label_id"] == 1) & (analysis_df["pred_label_id"] == 0)).sum())

    error_summary = {
        "total_test_samples": int(len(analysis_df)),
        "total_errors": total_errors,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "false_positives_top10": fp_samples,
        "false_negatives_top10": fn_samples,
    }

    with open(os.path.join(results_dir, "phobert_error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(error_summary, f, ensure_ascii=False, indent=2)

    print(
        f"[ERROR ANALYSIS] Đã xuất phân tích lỗi của PhoBERT: "
        f"Tổng lỗi={total_errors}/{len(analysis_df)} (FP={total_fp}, FN={total_fn})."
    )
    return error_summary


def run_bert_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results_dir: str = "outputs/results",
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Fine-tune PhoBERT-base-v2 trên cả 2 cấu hình:
    1. Only Title (Option A)
    2. Title + Lead (Option B)
    """
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[TRANSFORMER SOTA] Thiết bị Fine-tuning PhoBERT: {device}", flush=True)

    y_test = test_df["label_id"].values
    results_list: List[Dict[str, Any]] = []
    predictions_dict: Dict[str, Dict[str, Any]] = {}
    error_summary: Dict[str, Any] = {}

    configs = [
        ("Only Title", "only_title", 64, None),
        ("Title + Lead", "title_lead", 160, os.path.join(results_dir, "phobert_best_checkpoint")),
    ]

    for input_label, mode, max_len, save_ckpt_dir in configs:
        print(f"  -> Đang Fine-tune PhoBERT-base-v2 ({input_label}, max_length={max_len})...", flush=True)
        y_pred, y_prob, train_time, infer_ms = finetune_and_evaluate_phobert(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_checkpoint="vinai/phobert-base-v2",
            mode=mode,
            max_length=max_len,
            batch_size=24,
            epochs=3,
            lr=4e-5,
            weight_decay=0.01,
            device=device,
            save_dir=save_ckpt_dir,
        )

        metrics = compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            model_group="Transformer (SOTA)",
            model_name="PhoBERT-base-v2",
            input_type=input_label,
            train_time_sec=train_time,
            infer_time_ms_per_sample=infer_ms,
        )
        results_list.append(metrics)

        key_name = f"PhoBERT-base-v2 ({input_label})"
        predictions_dict[key_name] = {
            "y_pred": y_pred,
            "y_prob": y_prob,
            "macro_f1": metrics["macro_f1"],
        }

        print(
            f"     [KẾT QUẢ TEST] Acc={metrics['accuracy']:.4f} | "
            f"Prec(CB)={metrics['precision_clickbait']:.4f} | "
            f"Rec(CB)={metrics['recall_clickbait']:.4f} | "
            f"Macro-F1={metrics['macro_f1']:.4f} | AUC={metrics['roc_auc']:.4f}",
            flush=True,
        )

        if mode == "title_lead":
            error_summary = perform_error_analysis(test_df, y_pred, y_prob, results_dir=results_dir)

    return results_list, predictions_dict, error_summary
