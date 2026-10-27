"""
Module huấn luyện và đánh giá Nhóm 2: Các mô hình Học sâu truyền thống (Deep Learning Models) bằng PyTorch.
Bao gồm:
1. BiLSTM (Bidirectional LSTM):
   Embedding(vocab_size=15000, embedding_dim=128) -> SpatialDropout1D(0.2) -> BiLSTM(64) -> GlobalMaxPooling1D -> Linear(1, Sigmoid).
2. TextCNN (1D-CNN Multi-filter):
   Embedding(vocab_size=15000, embedding_dim=128) -> SpatialDropout1D(0.2) -> Conv1d với filter sizes [2, 3, 5] -> GlobalMaxPooling1D -> Linear(1, Sigmoid).
Kết hợp thêm nhánh đặc trưng thủ công (Hand-crafted features) chuẩn hóa để tăng cường khả năng nhận diện dấu câu cảm thán.
"""

import os
import re
import time
import copy
from collections import Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from src.preprocess import HANDCRAFTED_FEATURE_COLS
from src.evaluate import compute_metrics


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Vocabulary:
    """Bộ từ điển ánh xạ token tiếng Việt đã tách từ sang chỉ số nguyên (index)."""

    def __init__(self, max_vocab_size: int = 15000, min_freq: int = 1):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        if not isinstance(text, str) or not text:
            return []
        return re.findall(r"\w[\w_]*|[?!\"']|\.\.\.", text.lower())

    def build_vocab(self, texts: List[str]) -> None:
        counter = Counter()
        for t in texts:
            counter.update(self.tokenize(t))
        for word, freq in counter.most_common(self.max_vocab_size - 2):
            if freq >= self.min_freq:
                self.word2idx[word] = len(self.word2idx)

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        tokens = self.tokenize(text)[:max_len]
        ids = [self.word2idx.get(tok, 1) for tok in tokens]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.word2idx)


class ClickbaitDLDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, hc_feats: np.ndarray, labels: np.ndarray):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.hc_feats = torch.tensor(hc_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.hc_feats[idx], self.labels[idx]


class SpatialDropout1D(nn.Module):
    """Spatial Dropout 1D: Dropout toàn bộ kênh đặc trưng dọc theo chiều chuỗi."""

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.dropout2d = nn.Dropout2d(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, emb_dim)
        x = x.permute(0, 2, 1).unsqueeze(3)  # (batch, emb_dim, seq_len, 1)
        x = self.dropout2d(x)
        return x.squeeze(3).permute(0, 2, 1)


class BiLSTMClassifier(nn.Module):
    """
    Kiến trúc BiLSTM theo đặc tả:
    Embedding(15000, 128) -> SpatialDropout1D(0.2) -> BiLSTM(64) -> GlobalMaxPooling1D -> Dense(1, sigmoid)
    """

    def __init__(
        self,
        vocab_size: int = 15000,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        hc_dim: int = 12,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.spatial_dropout = SpatialDropout1D(dropout)
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.hc_proj = nn.Sequential(
            nn.Linear(hc_dim, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids: torch.Tensor, hc_feats: torch.Tensor) -> torch.Tensor:
        emb = self.spatial_dropout(self.embedding(input_ids))
        lstm_out, _ = self.bilstm(emb)  # (batch, seq_len, hidden_dim * 2)
        # GlobalMaxPooling1D (bỏ qua các vị trí PAD = 0 nếu không toàn PAD)
        mask = (input_ids != 0).unsqueeze(-1)
        lstm_masked = lstm_out.masked_fill(~mask, -1e9)
        pooled, _ = torch.max(lstm_masked, dim=1)
        hc_out = self.hc_proj(hc_feats)
        combined = torch.cat([pooled, hc_out], dim=1)
        logits = self.classifier(combined).squeeze(-1)
        return logits


class TextCNNClassifier(nn.Module):
    """
    Kiến trúc 1D-CNN (TextCNN) với các filter sizes [2, 3, 5] bắt cụm từ khóa giật tít cục bộ.
    """

    def __init__(
        self,
        vocab_size: int = 15000,
        embedding_dim: int = 128,
        num_filters: int = 64,
        filter_sizes: Tuple[int, ...] = (2, 3, 5),
        hc_dim: int = 12,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.spatial_dropout = SpatialDropout1D(dropout)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs, padding=fs - 1)
            for fs in filter_sizes
        ])
        self.hc_proj = nn.Sequential(
            nn.Linear(hc_dim, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(num_filters * len(filter_sizes) + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids: torch.Tensor, hc_feats: torch.Tensor) -> torch.Tensor:
        emb = self.spatial_dropout(self.embedding(input_ids))  # (batch, seq_len, emb_dim)
        emb_t = emb.permute(0, 2, 1)  # (batch, emb_dim, seq_len)
        conv_outs = [torch.relu(conv(emb_t)) for conv in self.convs]
        pooled_outs = [torch.max(c, dim=2)[0] for c in conv_outs]
        cnn_feat = torch.cat(pooled_outs, dim=1)
        hc_out = self.hc_proj(hc_feats)
        combined = torch.cat([cnn_feat, hc_out], dim=1)
        logits = self.classifier(combined).squeeze(-1)
        return logits


def train_and_evaluate_dl_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    y_val: np.ndarray,
    y_test: np.ndarray,
    pos_weight: float,
    device: torch.device,
    epochs: int = 18,
    lr: float = 1.5e-3,
    patience: int = 5,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Huấn luyện mô hình DL với EarlyStopping theo Validation Macro F1."""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_threshold = 0.5
    no_improve = 0

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_ids, batch_hc, batch_y in train_loader:
            batch_ids = batch_ids.to(device)
            batch_hc = batch_hc.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_ids, batch_hc)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Đánh giá trên tập Validation
        model.eval()
        val_probs_list = []
        with torch.no_grad():
            for batch_ids, batch_hc, _ in val_loader:
                batch_ids = batch_ids.to(device)
                batch_hc = batch_hc.to(device)
                probs = torch.sigmoid(model(batch_ids, batch_hc))
                val_probs_list.append(probs.cpu().numpy())
        val_probs = np.concatenate(val_probs_list)

        # Tìm ngưỡng tối ưu trên tập Validation theo Macro F1
        cur_best_th = 0.5
        cur_best_f1 = -1.0
        for th in np.linspace(0.35, 0.65, 13):
            preds = (val_probs >= th).astype(int)
            mf1 = f1_score(y_val, preds, average="macro", zero_division=0)
            if mf1 > cur_best_f1:
                cur_best_f1 = mf1
                cur_best_th = float(th)

        scheduler.step(cur_best_f1)
        if cur_best_f1 > best_val_f1:
            best_val_f1 = cur_best_f1
            best_threshold = cur_best_th
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    train_time = time.perf_counter() - t0

    # Suy diễn trên tập Test với trọng số tốt nhất
    model.load_state_dict(best_state)
    model.eval()
    t_inf0 = time.perf_counter()
    test_probs_list = []
    with torch.no_grad():
        for batch_ids, batch_hc, _ in test_loader:
            batch_ids = batch_ids.to(device)
            batch_hc = batch_hc.to(device)
            probs = torch.sigmoid(model(batch_ids, batch_hc))
            test_probs_list.append(probs.cpu().numpy())
    test_probs = np.concatenate(test_probs_list)
    test_preds = (test_probs >= best_threshold).astype(int)
    infer_ms = (time.perf_counter() - t_inf0) * 1000.0 / len(y_test)

    return test_preds, test_probs, train_time, infer_ms


def run_dl_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results_dir: str = "outputs/results",
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Thực thi huấn luyện và đánh giá các mô hình Học sâu truyền thống (BiLSTM và TextCNN).
    """
    os.makedirs(results_dir, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DL MODELS] Thiết bị huấn luyện Deep Learning: {device}")

    y_train = train_df["label_id"].values
    y_val = val_df["label_id"].values
    y_test = test_df["label_id"].values
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    scaler = StandardScaler()
    X_train_hc = scaler.fit_transform(train_df[HANDCRAFTED_FEATURE_COLS].fillna(0).values)
    X_val_hc = scaler.transform(val_df[HANDCRAFTED_FEATURE_COLS].fillna(0).values)
    X_test_hc = scaler.transform(test_df[HANDCRAFTED_FEATURE_COLS].fillna(0).values)

    results_list: List[Dict[str, Any]] = []
    predictions_dict: Dict[str, Dict[str, Any]] = {}

    input_configs = [
        ("Only Title", "input_option_a", 64),
        ("Title + Lead", "input_option_b", 128),
    ]

    for input_label, text_col, max_len in input_configs:
        vocab = Vocabulary(max_vocab_size=15000, min_freq=1)
        vocab.build_vocab(train_df[text_col].fillna("").tolist())

        X_train_ids = np.array([vocab.encode(t, max_len=max_len) for t in train_df[text_col].fillna("")])
        X_val_ids = np.array([vocab.encode(t, max_len=max_len) for t in val_df[text_col].fillna("")])
        X_test_ids = np.array([vocab.encode(t, max_len=max_len) for t in test_df[text_col].fillna("")])

        train_loader = DataLoader(
            ClickbaitDLDataset(X_train_ids, X_train_hc, y_train),
            batch_size=32,
            shuffle=True,
        )
        val_loader = DataLoader(
            ClickbaitDLDataset(X_val_ids, X_val_hc, y_val),
            batch_size=64,
            shuffle=False,
        )
        test_loader = DataLoader(
            ClickbaitDLDataset(X_test_ids, X_test_hc, y_test),
            batch_size=64,
            shuffle=False,
        )

        dl_Readiness = [
            (
                "BiLSTM",
                lambda: BiLSTMClassifier(
                    vocab_size=len(vocab),
                    embedding_dim=128,
                    hidden_dim=64,
                    hc_dim=len(HANDCRAFTED_FEATURE_COLS),
                    dropout=0.2,
                ),
            ),
            (
                "TextCNN",
                lambda: TextCNNClassifier(
                    vocab_size=len(vocab),
                    embedding_dim=128,
                    num_filters=64,
                    filter_sizes=(2, 3, 5),
                    hc_dim=len(HANDCRAFTED_FEATURE_COLS),
                    dropout=0.2,
                ),
            ),
        ]

        for model_name, model_fn in dl_Readiness:
            set_seed(42)
            model = model_fn()
            print(f"  -> Đang huấn luyện {model_name} ({input_label}, vocab={len(vocab)}, max_len={max_len})...")
            y_pred, y_prob, train_time, infer_ms = train_and_evaluate_dl_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                y_val=y_val,
                y_test=y_test,
                pos_weight=pos_weight,
                device=device,
                epochs=18,
                lr=1.5e-3,
                patience=5,
            )

            metrics = compute_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                model_group="Deep Learning",
                model_name=model_name,
                input_type=input_label,
                train_time_sec=train_time,
                infer_time_ms_per_sample=infer_ms,
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

            if input_label == "Title + Lead":
                torch.save(
                    model.state_dict(),
                    os.path.join(results_dir, f"best_dl_{model_name.lower()}_title_lead.pt"),
                )

    return results_list, predictions_dict
