"""
Module tiền xử lý dữ liệu văn bản tiếng Việt đặc thù cho bài toán Phát hiện Clickbait.
Thực hiện:
1. Xử lý Missing Values (điền chuỗi rỗng cho lead_paragraph bị NaN).
2. Khử nhiễu đặc thù:
   - Tách địa danh bị dính liền ở đầu lead_paragraph của VnExpress (VD: "Nghệ AnCảng..." -> "Nghệ An Cảng...").
   - Cắt bỏ hậu tố tên nguồn báo ở cuối title của Báo Mới (VD: "... - Tạp chí SaoStar").
3. Chuẩn hóa văn bản tiếng Việt:
   - Chuẩn hóa bảng mã Unicode dựng sẵn (NFC).
   - Chuẩn hóa vị trí dấu thanh tiếng Việt (thống nhất chuẩn òa, óa, ùy, úy... tương thích với PyVi & PhoBERT).
   - Chuẩn hóa khoảng trắng và bảo toàn các dấu câu mang sắc thái cảm xúc (?, !, ..., ").
4. Tách từ tiếng Việt (Word Segmentation) bằng PyVi (ViTokenizer).
5. Tạo đầu vào Option A (Only Title) và Option B (Title + [SEP] + Lead Paragraph), kèm các đặc trưng thủ công (Hand-crafted features).
6. Chia tập dữ liệu 70% Train - 15% Validation - 15% Test bằng StratifiedShuffleSplit (random_state=42).
"""

import os
import re
import sys
import json
import shutil
import unicodedata
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

try:
    from underthesea import word_tokenize as uts_word_tokenize
    HAS_UTS = True
except ImportError:
    HAS_UTS = False


# Tập ký tự tiếng Việt viết thường và viết hoa đầy đủ (tránh lỗi dải Unicode [a-zà-ỹ] ăn nhầm chữ hoa Đ, Ă, Â...)
LOWER_VN = "a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
UPPER_VN = "A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ"

# Các thương hiệu có viết hoa giữa từ (CamelCase) không phải là lỗi dính địa danh
CAMEL_BRANDS = (
    "TikTok", "ChatGPT", "OpenAI", "VietNamNet", "VnExpress", "VinFast",
    "YouTuber", "McLaren", "LaLiga", "PhuQuoc", "SeaStar", "MasterD",
    "RadarOnline", "GameVerse", "CellphoneS", "InterContinental",
    "iPhone", "iPad", "iOS", "MacBook", "xAI", "eSports", "YouTube",
    "WeChat", "WhatsApp", "LinkedIn", "PlayStation", "SmartTV", "VietJet",
    "Vietcombank", "Techcombank", "VPBank", "MBBank", "TPBank", "HDBank"
)

# Regex bắt địa danh ở đầu lead_paragraph bị dính vào câu tiếp theo (đặc thù của VnExpress)
VNEXPRESS_GLUED_LOCATION_PATTERN = re.compile(
    rf"^((?:TP\s+HCM|TP\s+Huế|Bà\s+Rịa\s*-\s*Vũng\s+Tàu|Thừa\s+Thiên\s*-\s*Huế|"
    rf"[{UPPER_VN}][{LOWER_VN}]+(?:\s+[{UPPER_VN}][{LOWER_VN}]+){{0,3}}))"
    rf"(?=[0-9\"\'“‘]|[{UPPER_VN}])"
)

# Regex cắt bỏ đuôi tên báo ở cuối tiêu đề (đặc thù của Báo Mới)
BAOMOI_TITLE_SUFFIX_PATTERN = re.compile(
    r"\s*-\s*(?:Báo|Tạp\s*chí|Chuyên\s*trang|Đài|Thời\s*báo|Bnews|VietTimes|"
    r"TheLEADER|Giác\s*ngộ|Công\s*An\s*TP\.HCM|Đại\s*Biểu\s*Nhân\s*Dân|"
    r"PLO|VOV|VTV|TTXVN).*$",
    re.IGNORECASE
)

# Regex cắt bỏ tiền tố viết tắt tên báo ở đầu sapo (VD: "TTO - ", "HNN.VN - ")
LEAD_SOURCE_PREFIX_PATTERN = re.compile(
    r"^(?:TTO|HNN\.VN|VOV\.VN|PLO|Dân\s+trí|NLĐO)\s*[-–]\s*"
)

# Bảng chuẩn hóa dấu thanh tiếng Việt (thống nhất về chuẩn òa, óa, ỏa, õa, ọa, òe, óe, ùy, úy... tương thích với PyVi & PhoBERT)
TONE_NORM_MAP = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "Oà": "Òa", "Oá": "Óa", "Oả": "Ỏa", "Oã": "Õa", "Oạ": "Ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "Oè": "Òe", "Oé": "Óe", "Oẻ": "Ỏe", "Oẽ": "Õe", "Oẹ": "Ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
    "Uỳ": "Ùy", "Uý": "Úy", "Uỷ": "Ủy", "Uỹ": "Ũy", "Uỵ": "Ụy",
}
TONE_NORM_REGEX = re.compile("|".join(re.escape(k) for k in TONE_NORM_MAP.keys()))


def normalize_unicode_and_tone(text: str) -> str:
    """Đưa văn bản về chuẩn Unicode dựng sẵn (NFC) và chuẩn hóa kiểu gõ dấu tiếng Việt."""
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = TONE_NORM_REGEX.sub(lambda m: TONE_NORM_MAP[m.group(0)], text)
    return text


def fix_vnexpress_glued_location(text: str) -> Tuple[str, bool]:
    """
    Khử lỗi dính tên địa danh ở đầu lead_paragraph của VnExpress.
    Ví dụ:
      "Nghệ AnCảng hàng không Vinh..." -> "Nghệ An Cảng hàng không Vinh..."
      "TP HCMNgôi nhà hơn 50 năm..."   -> "TP HCM Ngôi nhà hơn 50 năm..."
    Trả về (văn_bản_đã_sửa, có_sửa_hay_không).
    """
    if not isinstance(text, str) or not text:
        return "", False
    text = unicodedata.normalize("NFC", text).strip()
    # Kiểm tra trong 35 ký tự đầu có chứa từ khóa thương hiệu CamelCase hay không
    prefix_window = text[:35]
    if any(brand in prefix_window for brand in CAMEL_BRANDS):
        return text, False

    match = VNEXPRESS_GLUED_LOCATION_PATTERN.match(text)
    if match:
        loc = match.group(1)
        rest = text[len(loc):]
        return f"{loc} {rest}", True
    return text, False


def remove_baomoi_source_suffix(title: str) -> Tuple[str, bool]:
    """
    Cắt bỏ hậu tố tên báo ở cuối title (đặc thù của Báo Mới).
    Ví dụ:
      "Tình thế đảo ngược với tiền vệ Tuấn Anh - Tạp chí SaoStar"
      -> "Tình thế đảo ngược với tiền vệ Tuấn Anh"
    Trả về (tiêu_đề_đã_sửa, có_sửa_hay_không).
    """
    if not isinstance(title, str) or not title:
        return "", False
    title = unicodedata.normalize("NFC", title).strip()
    if BAOMOI_TITLE_SUFFIX_PATTERN.search(title):
        cleaned = BAOMOI_TITLE_SUFFIX_PATTERN.sub("", title).strip()
        return cleaned, True
    return title, False


def clean_whitespace_and_punctuation(text: str) -> str:
    """
    Làm sạch khoảng trắng thừa nhưng giữ nguyên các dấu câu mang sắc thái cảm xúc (?, !, ..., ").
    """
    if not isinstance(text, str) or not text:
        return ""
    # Thay thế các ký tự khoảng trắng đặc biệt (non-breaking space, tab, newline) bằng dấu cách chuẩn
    text = re.sub(r"[\r\n\t\u00a0\u200b]+", " ", text)
    # Chuẩn hóa dấu ba chấm unicode (…) về ...
    text = text.replace("…", "...")
    # Chuẩn hóa dấu ngoặc kép cong về ngoặc kép chuẩn để đồng nhất token
    text = text.replace("“", '"').replace("”", '"')
    # Xóa khoảng trắng thừa liên tiếp
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_title(title: str) -> str:
    """Pipeline làm sạch hoàn chỉnh cho trường title (trước khi tách từ)."""
    if pd.isna(title):
        return ""
    text = str(title)
    text, _ = remove_baomoi_source_suffix(text)
    text = normalize_unicode_and_tone(text)
    text = clean_whitespace_and_punctuation(text)
    return text


def clean_lead_paragraph(lead: str) -> str:
    """Pipeline làm sạch hoàn chỉnh cho trường lead_paragraph (trước khi tách từ)."""
    if pd.isna(lead):
        return ""
    text = str(lead)
    text, _ = fix_vnexpress_glued_location(text)
    text = LEAD_SOURCE_PREFIX_PATTERN.sub("", text)
    text = normalize_unicode_and_tone(text)
    text = clean_whitespace_and_punctuation(text)
    return text


def segment_vietnamese(text: str) -> str:
    """
    Tách từ tiếng Việt (Word Segmentation) nối từ ghép bằng dấu gạch dưới '_'.
    Ưu tiên sử dụng PyVi (ViTokenizer.tokenize), fallback sang Underthesea nếu cần.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    if HAS_PYVI:
        return ViTokenizer.tokenize(text)
    elif HAS_UTS:
        return uts_word_tokenize(text, format="text")
    return text


def extract_handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trích xuất các đặc trưng thủ công (Hand-crafted features) từ title_clean và lead_clean:
    - title_char_len: Số ký tự trong tiêu đề
    - title_word_count: Số âm tiết (từ đơn) trong tiêu đề
    - lead_char_len: Số ký tự trong đoạn sapo
    - lead_word_count: Số âm tiết trong đoạn sapo
    - has_question: Tiêu đề chứa dấu '?' (0/1)
    - has_exclamation: Tiêu đề chứa dấu '!' (0/1)
    - has_quotes: Tiêu đề chứa dấu trích dẫn " hoặc ' (0/1)
    - has_ellipsis: Tiêu đề chứa dấu '...' (0/1)
    - num_digits: Số lượng cụm chữ số xuất hiện trong tiêu đề
    - has_number: Tiêu đề có chứa con số hay không (0/1)
    - all_caps_count: Số từ viết hoa toàn bộ (độ dài >= 2) trong tiêu đề
    - upper_ratio: Tỷ lệ chữ cái viết hoa trên tổng số ký tự trong tiêu đề
    """
    titles = df["title_clean"].fillna("")
    leads = df["lead_clean"].fillna("")

    features = pd.DataFrame(index=df.index)
    features["title_char_len"] = titles.str.len()
    features["title_word_count"] = titles.str.split().str.len()
    features["lead_char_len"] = leads.str.len()
    features["lead_word_count"] = leads.str.split().str.len()

    features["has_question"] = titles.str.contains(r"\?", regex=True).astype(int)
    features["has_exclamation"] = titles.str.contains(r"!", regex=True).astype(int)
    features["has_quotes"] = titles.str.contains(r"[\"\'‘’“”]", regex=True).astype(int)
    features["has_ellipsis"] = titles.str.contains(r"\.\.\.|…", regex=True).astype(int)

    features["num_digits"] = titles.apply(lambda s: len(re.findall(r"\d+", s)))
    features["has_number"] = (features["num_digits"] > 0).astype(int)

    upper_word_re = re.compile(rf"\b[{UPPER_VN}]{{2,}}\b")
    features["all_caps_count"] = titles.apply(lambda s: len(upper_word_re.findall(s)))

    upper_char_re = re.compile(rf"[{UPPER_VN}]")
    features["upper_ratio"] = titles.apply(
        lambda s: len(upper_char_re.findall(s)) / max(len(s), 1)
    )
    return features


HANDCRAFTED_FEATURE_COLS = [
    "title_char_len",
    "title_word_count",
    "lead_char_len",
    "lead_word_count",
    "has_question",
    "has_exclamation",
    "has_quotes",
    "has_ellipsis",
    "num_digits",
    "has_number",
    "all_caps_count",
    "upper_ratio",
]


def run_preprocessing_pipeline(
    raw_csv_path: str = "clickbait_dataset_vietnamese.csv",
    output_dir: str = "data/processed",
    raw_copy_dir: str = "data/raw",
    results_dir: str = "outputs/results",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Thực thi toàn bộ quy trình tiền xử lý dữ liệu:
    - Sao lưu file gốc vào data/raw/
    - In ra 5 ví dụ minh chứng trước & sau khi xử lý lỗi dính địa danh (VnExpress) và lỗi đuôi báo (Báo Mới)
    - Làm sạch, chuẩn hóa Unicode/dấu thanh, tách từ tiếng Việt bằng PyVi
    - Tạo cột đầu vào Option A (input_title_seg) và Option B (input_title_lead_seg)
    - Chia tập dữ liệu 70% Train - 15% Validation - 15% Test (Stratified theo label, random_state=42)
    - Lưu các file train.csv, val.csv, test.csv vào data/processed/
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_copy_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    raw_dest = os.path.join(raw_copy_dir, "clickbait_dataset_vietnamese.csv")
    if os.path.abspath(raw_csv_path) != os.path.abspath(raw_dest) and os.path.exists(raw_csv_path):
        shutil.copy2(raw_csv_path, raw_dest)

    df = pd.read_csv(raw_csv_path, encoding="utf-8")
    print(f"[PREPROCESS] Đã đọc dữ liệu gốc: {df.shape[0]} mẫu, {df.shape[1]} cột.")

    # 1. Kiểm tra & xử lý Missing Values
    missing_lead_ids = df[df["lead_paragraph"].isna()]["id"].tolist()
    print(f"[PREPROCESS] Số mẫu thiếu lead_paragraph: {len(missing_lead_ids)} -> IDs: {missing_lead_ids}")
    df["title"] = df["title"].fillna("")
    df["lead_paragraph"] = df["lead_paragraph"].fillna("")

    # 2. Thu thập thống kê và 5 mẫu minh chứng trước & sau khi khử nhiễu đặc thù
    vne_examples = []
    vne_fixed_count = 0
    for idx, row in df.iterrows():
        fixed_text, was_fixed = fix_vnexpress_glued_location(row["lead_paragraph"])
        if was_fixed:
            vne_fixed_count += 1
            if len(vne_examples) < 5:
                vne_examples.append({
                    "id": row["id"],
                    "source": row["source"],
                    "before": row["lead_paragraph"],
                    "after": fixed_text,
                })

    bm_examples = []
    bm_fixed_count = 0
    for idx, row in df.iterrows():
        fixed_title, was_fixed = remove_baomoi_source_suffix(row["title"])
        if was_fixed:
            bm_fixed_count += 1
            if len(bm_examples) < 5:
                bm_examples.append({
                    "id": row["id"],
                    "source": row["source"],
                    "before": row["title"],
                    "after": fixed_title,
                })

    print(f"\n[PREPROCESS] Khử lỗi dính địa danh VnExpress: Đã sửa {vne_fixed_count} mẫu.")
    print("=" * 80)
    print("5 VÍ DỤ MINH CHỨNG XỬ LÝ LỖI DÍNH ĐỊA DANH (VNEXPRESS):")
    for i, ex in enumerate(vne_examples, 1):
        print(f"  [{i}] ID: {ex['id']}")
        print(f"      TRƯỚC: {ex['before'][:110]}...")
        print(f"      SAU  : {ex['after'][:110]}...")

    print(f"\n[PREPROCESS] Khử lỗi dính đuôi tên báo (Báo Mới): Đã sửa {bm_fixed_count} mẫu.")
    print("=" * 80)
    print("5 VÍ DỤ MINH CHỨNG XỬ LÝ LỖI ĐUÔI TÊN BÁO (BÁO MỚI):")
    for i, ex in enumerate(bm_examples, 1):
        print(f"  [{i}] ID: {ex['id']}")
        print(f"      TRƯỚC: {ex['before']}")
        print(f"      SAU  : {ex['after']}")
    print("=" * 80)

    # 3. Áp dụng làm sạch văn bản
    df["title_clean"] = df["title"].apply(clean_title)
    df["lead_clean"] = df["lead_paragraph"].apply(clean_lead_paragraph)

    # 4. Tách từ tiếng Việt (Word Segmentation)
    print("[PREPROCESS] Đang tách từ tiếng Việt (Word Segmentation bằng PyVi)...")
    df["title_seg"] = df["title_clean"].apply(segment_vietnamese)
    df["lead_seg"] = df["lead_clean"].apply(segment_vietnamese)

    # 5. Tạo đầu vào Option A và Option B
    # Option A: Chỉ dùng Title
    df["input_option_a"] = df["title_seg"]
    # Option B: Kết hợp Title + [SEP] + Lead Paragraph (nếu lead_seg rỗng thì chỉ lấy title_seg)
    df["input_option_b"] = df.apply(
        lambda r: f"{r['title_seg']} [SEP] {r['lead_seg']}" if r["lead_seg"].strip() else r["title_seg"],
        axis=1,
    )

    # 6. Mã hóa nhãn nhị phân: clickbait = 1, non-clickbait = 0
    df["label_id"] = (df["label"].str.strip().str.lower() == "clickbait").astype(int)

    # 7. Trích xuất các đặc trưng thủ công (Hand-crafted features)
    hc_features = extract_handcrafted_features(df)
    for col in hc_features.columns:
        df[col] = hc_features[col]

    # Lưu 5 mẫu minh chứng tách từ tiếng Việt
    seg_examples = []
    for idx, row in df.head(5).iterrows():
        seg_examples.append({
            "id": row["id"],
            "title_clean": row["title_clean"],
            "title_seg": row["title_seg"],
            "lead_clean": row["lead_clean"][:120],
            "lead_seg": row["lead_seg"][:120],
        })

    # 8. Chia tập dữ liệu 70% Train - 15% Validation - 15% Test (StratifiedShuffleSplit, random_state=42)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(sss1.split(df, df["label_id"]))

    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    val_rel_idx, test_rel_idx = next(sss2.split(temp_df, temp_df["label_id"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = temp_df.iloc[val_rel_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_rel_idx].reset_index(drop=True)

    # Lưu ra đĩa
    df.to_csv(os.path.join(output_dir, "full_processed.csv"), index=False, encoding="utf-8")
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding="utf-8")
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False, encoding="utf-8")

    split_summary = {
        "total_samples": int(len(df)),
        "missing_lead_ids": missing_lead_ids,
        "vnexpress_glued_fixed_count": int(vne_fixed_count),
        "baomoi_suffix_fixed_count": int(bm_fixed_count),
        "splits": {
            "train": {
                "total": int(len(train_df)),
                "clickbait": int(train_df["label_id"].sum()),
                "non_clickbait": int((train_df["label_id"] == 0).sum()),
                "clickbait_ratio": float(train_df["label_id"].mean()),
            },
            "val": {
                "total": int(len(val_df)),
                "clickbait": int(val_df["label_id"].sum()),
                "non_clickbait": int((val_df["label_id"] == 0).sum()),
                "clickbait_ratio": float(val_df["label_id"].mean()),
            },
            "test": {
                "total": int(len(test_df)),
                "clickbait": int(test_df["label_id"].sum()),
                "non_clickbait": int((test_df["label_id"] == 0).sum()),
                "clickbait_ratio": float(test_df["label_id"].mean()),
            },
        },
        "vnexpress_examples": vne_examples,
        "baomoi_examples": bm_examples,
        "segmentation_examples": seg_examples,
    }

    with open(os.path.join(results_dir, "preprocessing_summary.json"), "w", encoding="utf-8") as f:
        json.dump(split_summary, f, ensure_ascii=False, indent=2)

    print("\n[PREPROCESS] Thống kê phân chia dữ liệu (70/15/15 Stratified):")
    for split_name, info in split_summary["splits"].items():
        print(
            f"  - {split_name.upper():5s}: {info['total']:4d} mẫu | "
            f"Clickbait: {info['clickbait']:4d} ({info['clickbait_ratio']*100:.2f}%) | "
            f"Non-clickbait: {info['non_clickbait']:4d} ({(1-info['clickbait_ratio'])*100:.2f}%)"
        )

    return train_df, val_df, test_df


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run_preprocessing_pipeline()
