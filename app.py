"""
Ứng dụng Web Demo Gradio & Backend API cho Hệ thống Phát hiện Clickbait Tiếng Việt.
Chạy đồng thời trên cùng một cổng (mặc định http://127.0.0.1:7860):
1. Giao diện Gradio (Phục vụ thuyết trình, kiểm thử đa mô hình, XAI Highlight, xem báo cáo EDA & Benchmark).
2. REST API (FastAPI + CORS) phục vụ trực tiếp cho Chrome Extension (Tầng 1 Quét trang báo, Tầng 2 Trợ lý đọc bài, Tầng 3 Popup).
3. Trang báo điện tử giả lập (/demo-news) để kiểm thử trực quan Chrome Extension ngay lập tức.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

sys.stdout.reconfigure(encoding="utf-8")

from src.inference import ClickbaitInferenceEngine, extract_article_from_url


# ==============================================================================
# 1. KHỞI TẠO ENGINE SUY DIỄN & XUẤT TRỌNG SỐ CHO EXTENSION
# ==============================================================================
print("[SERVER] Đang khởi tạo ClickbaitInferenceEngine...")
ENGINE = ClickbaitInferenceEngine()
ENGINE.export_client_weights_for_extension("extension/model_weights.json")
print("[SERVER] Đã nạp thành công các mô hình ML/DL và xuất extension/model_weights.json!")


# ==============================================================================
# 2. CÁC MẪU BÀI BÁO THỰC TẾ ĐỂ DEMO NHANH (EXAMPLES)
# ==============================================================================
DEMO_EXAMPLES = [
    [
        "Hé lộ danh tính người đàn ông bí ẩn xuất hiện cùng nữ diễn viên lúc nửa đêm khiến MXH xôn xao!",
        "Tối 25/10, đoàn làm phim xác nhận người đi cùng nữ diễn viên tại sân bay Tân Sơn Nhất là trợ lý sản xuất hỗ trợ lịch trình quay quảng cáo.",
        "Ensemble (Khuyên dùng)",
        0.50,
    ],
    [
        "Bất ngờ với thứ quả mọc dại đầy ở quê Việt Nam nhưng được bán giá tiền triệu tại nước ngoài, biết công dụng ai cũng ngỡ ngàng",
        "Quả tầm bóp (hay còn gọi là thù lù) từng mọc hoang ở nhiều vùng nông thôn Việt Nam hiện được xuất khẩu sang thị trường Nhật Bản và châu Âu nhờ giàu vitamin C.",
        "Ensemble (Khuyên dùng)",
        0.50,
    ],
    [
        "Nghệ AnCảng hàng không Vinh tạm dừng khai thác từ ngày 1/7 để nâng cấp đường băng",
        "Nghệ AnCảng hàng không quốc tế Vinh sẽ đóng cửa tạm thời trong 6 tháng nhằm sửa chữa đường cất hạ cánh và mở rộng sân đỗ tàu bay theo quy hoạch của Bộ Giao thông Vận tải.",
        "Ensemble (Khuyên dùng)",
        0.50,
    ],
    [
        "Ngân hàng Nhà nước giữ nguyên lãi suất điều hành nhằm ổn định thị trường tiền tệ quý IV - Báo Điện tử Chính phủ",
        "TTO - Ngày 26/10, Ngân hàng Nhà nước Việt Nam công bố báo cáo điều hành chính sách tiền tệ, tiếp tục duy trì mặt bằng lãi suất hỗ trợ doanh nghiệp phục hồi sản xuất.",
        "Ensemble (Khuyên dùng)",
        0.50,
    ],
    [
        "Chỉ vì một thói quen vào buổi sáng mà 90% người Việt mắc phải, người đàn ông 35 tuổi nhận cái kết đắng!",
        "Các bác sĩ Bệnh viện Bạch Mai vừa tiếp nhận một bệnh nhân nam 35 tuổi nhập viện do viêm loét dạ dày cấp tính vì thường xuyên bỏ bữa sáng và uống cà phê đặc khi bụng đói.",
        "Ensemble (Khuyên dùng)",
        0.50,
    ],
]


# ==============================================================================
# 3. HÀM RENDER HTML TRỰC QUAN CHO GIAO DIỆN GRADIO
# ==============================================================================
def build_verdict_card_html(res: Dict[str, Any]) -> str:
    prob_pct = round(res["prob_clickbait"] * 100, 1)
    level = res["badge_level"]

    if level == "red":
        bg_grad = "linear-gradient(135deg, #fff5f5 0%, #ffe3e3 100%)"
        border_col = "#e03131"
        badge_bg = "#e03131"
        icon = "🔴"
        bar_col = "linear-gradient(90deg, #ff6b6b, #c92a2a)"
    elif level == "yellow":
        bg_grad = "linear-gradient(135deg, #fff9db 0%, #ffec99 100%)"
        border_col = "#f08c00"
        badge_bg = "#f08c00"
        icon = "🟡"
        bar_col = "linear-gradient(90deg, #fcc419, #e67700)"
    else:
        bg_grad = "linear-gradient(135deg, #ebfbee 0%, #d3f9d8 100%)"
        border_col = "#2f9e44"
        badge_bg = "#2f9e44"
        icon = "🟢"
        bar_col = "linear-gradient(90deg, #51cf66, #2b8a3e)"

    reasons_html = "".join(
        f"<li style='margin-bottom:6px;line-height:1.5;color:#1f2937;'>{r}</li>"
        for r in res.get("reasons_vi", [])
    )

    triggers = res.get("trigger_words", [])
    if triggers:
        trigger_chips = "".join(
            f"<span style='display:inline-block;background:#ffe3e3;color:#c92a2a;border:1px solid #ffa8a8;"
            f"border-radius:999px;padding:3px 10px;font-size:12.5px;font-weight:600;margin:3px 5px 3px 0;'>"
            f"⚡ {t['word']} (+{t['weight']:.2f})</span>"
            for t in triggers
        )
    else:
        trigger_chips = "<span style='color:#6b7280;font-size:13px;'>Không phát hiện từ khóa kích động mạnh.</span>"

    declickbait_block = ""
    if res["is_clickbait"] or level == "yellow":
        declickbait_block = f"""
        <div style="margin-top:14px;padding:12px 15px;background:#ffffff;border-left:4px solid #2563eb;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
            <div style="font-size:12px;font-weight:700;color:#2563eb;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">
                ✨ Gợi ý Tháo tít (Tiêu đề trung lập chuẩn báo chí)
            </div>
            <div style="font-size:15px;font-weight:600;color:#111827;line-height:1.45;">
                “{res['declickbait_title']}”
            </div>
        </div>
        """

    gap = res.get("semantic_gap", {})
    gap_badge_color = "#dc2626" if gap.get("has_gap") else "#16a34a"

    return f"""
    <div style="background:{bg_grad};border:2px solid {border_col};border-radius:14px;padding:20px;font-family:'Segoe UI',system-ui,sans-serif;box-shadow:0 4px 14px rgba(0,0,0,0.05);">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:14px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="background:{badge_bg};color:#fff;padding:6px 14px;border-radius:999px;font-weight:700;font-size:14px;letter-spacing:0.3px;">
                    {icon} {res['badge_text']}
                </span>
                <span style="font-size:17px;font-weight:800;color:#111827;">
                    {res['verdict_vi']}
                </span>
            </div>
            <div style="font-size:12.5px;color:#4b5563;background:rgba(255,255,255,0.75);padding:4px 10px;border-radius:6px;font-weight:600;">
                ⏱️ Suy diễn: {res['infer_time_ms']} ms · Mô hình: {res['selected_model']}
            </div>
        </div>

        <!-- Thanh đo xác suất Clickbait -->
        <div style="margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;font-size:13px;font-weight:700;color:#374151;margin-bottom:5px;">
                <span>Chỉ số Giật tít (Clickbait Score)</span>
                <span style="color:{border_col};font-size:15px;">{prob_pct}%</span>
            </div>
            <div style="width:100%;height:12px;background:#e5e7eb;border-radius:999px;overflow:hidden;">
                <div style="width:{min(max(prob_pct, 2), 100)}%;height:100%;background:{bar_col};border-radius:999px;transition:width 0.3s ease;"></div>
            </div>
        </div>

        <!-- Lý do giải thích cho người đọc -->
        <div style="background:rgba(255,255,255,0.88);border-radius:10px;padding:13px 16px;margin-bottom:12px;">
            <div style="font-size:13px;font-weight:700;color:#111827;margin-bottom:6px;">
                🔎 Giải thích tại sao hệ thống đưa ra đánh giá này:
            </div>
            <ul style="margin:0;padding-left:18px;font-size:14px;">
                {reasons_html}
            </ul>
        </div>

        <!-- Từ khóa báo động & Độ lệch ngữ nghĩa -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            <div style="background:rgba(255,255,255,0.88);border-radius:10px;padding:11px 14px;">
                <div style="font-size:12px;font-weight:700;color:#4b5563;margin-bottom:6px;">
                    Từ khóa kích thích tò mò phát hiện được:
                </div>
                <div>{trigger_chips}</div>
            </div>
            <div style="background:rgba(255,255,255,0.88);border-radius:10px;padding:11px 14px;">
                <div style="font-size:12px;font-weight:700;color:#4b5563;margin-bottom:4px;">
                    Đối chiếu Tiêu đề vs Đoạn mở đầu (Sapo):
                    <span style="color:{gap_badge_color};font-weight:800;">[{gap.get('level', 'Nhất quán')}]</span>
                </div>
                <div style="font-size:13px;color:#1f2937;line-height:1.4;">
                    💡 <b>Tóm tắt Sapo:</b> {res['quick_summary']}
                </div>
            </div>
        </div>

        {declickbait_block}
    </div>
    """


def build_preprocessing_html(prep: Dict[str, Any]) -> str:
    badges = []
    if prep.get("vnexpress_location_fixed"):
        badges.append(
            "<span style='background:#dbeafe;color:#1d4ed8;padding:3px 9px;border-radius:6px;font-size:12px;font-weight:700;'>"
            "🛠️ Đã tách lỗi dính địa danh đầu câu (VnExpress Noise Fix)</span>"
        )
    if prep.get("baomoi_suffix_removed"):
        badges.append(
            "<span style='background:#fef3c7;color:#b45309;padding:3px 9px;border-radius:6px;font-size:12px;font-weight:700;'>"
            "✂️ Đã cắt bỏ hậu tố tên báo cuối tiêu đề (Báo Mới Noise Fix)</span>"
        )
    if not badges:
        badges.append(
            "<span style='background:#f3f4f6;color:#374151;padding:3px 9px;border-radius:6px;font-size:12px;font-weight:600;'>"
            "✓ Chuẩn hóa Unicode NFC, dấu thanh tiếng Việt & Tách từ PyVi</span>"
        )
    badges_html = " ".join(badges)

    return f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px;font-family:'Segoe UI',system-ui,sans-serif;font-size:13.5px;">
        <div style="margin-bottom:10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
            <b style="color:#0f172a;">Trạng thái Tiền xử lý Đặc thù:</b> {badges_html}
        </div>
        <div style="margin-bottom:6px;">
            <span style="color:#475569;font-weight:600;">• Tiêu đề sau làm sạch & tách từ (PyVi):</span>
            <code style="background:#e2e8f0;padding:2px 6px;border-radius:4px;color:#0f172a;">{prep.get('title_seg', '')}</code>
        </div>
        <div>
            <span style="color:#475569;font-weight:600;">• Sapo sau làm sạch & tách từ (PyVi):</span>
            <code style="background:#e2e8f0;padding:2px 6px;border-radius:4px;color:#0f172a;">{prep.get('lead_seg', '') or '(Trống)'}</code>
        </div>
    </div>
    """


def run_gradio_analysis(
    title: str,
    lead_paragraph: str,
    model_choice: str,
    sensitivity: float,
):
    if not title or not title.strip():
        empty_html = (
            "<div style='padding:18px;border:1px dashed #cbd5e1;border-radius:10px;color:#64748b;text-align:center;'>"
            "Vui lòng nhập Tiêu đề bài báo hoặc chọn một mẫu thử bên dưới để bắt đầu phân tích.</div>"
        )
        return empty_html, [], {}, pd.DataFrame(), ""

    res = ENGINE.predict_single(
        title=title,
        lead_paragraph=lead_paragraph,
        model_choice=model_choice,
        sensitivity=sensitivity,
    )

    verdict_html = build_verdict_card_html(res)
    highlights = res["token_highlights"]
    model_scores = {f"{k} (Clickbait)": float(v) for k, v in res["model_scores"].items()}
    hc_df = pd.DataFrame(res["handcrafted_features"])[["name_vi", "value"]].rename(
        columns={"name_vi": "Đặc trưng ngôn ngữ (Hand-crafted Feature)", "value": "Giá trị"}
    )
    prep_html = build_preprocessing_html(res["preprocessing"])

    return verdict_html, highlights, model_scores, hc_df, prep_html


def handle_url_fetch(url: str, model_choice: str, sensitivity: float):
    if not url or not url.strip():
        raise gr.Error("Vui lòng nhập đường dẫn URL bài báo (VD: https://vnexpress.net/...)")
    try:
        scraped = extract_article_from_url(url.strip())
    except Exception as e:
        raise gr.Error(f"Không thể trích xuất tự động từ URL này: {str(e)}")

    title = scraped["title"]
    lead = scraped["lead_paragraph"]
    v_html, hl, scores, hc_df, p_html = run_gradio_analysis(title, lead, model_choice, sensitivity)
    return title, lead, v_html, hl, scores, hc_df, p_html


def get_benchmark_dataframe() -> pd.DataFrame:
    rows = []
    for b in ENGINE.benchmarks:
        rows.append({
            "Nhóm mô hình": b["group"],
            "Tên mô hình": b["model"],
            "Đầu vào": b["input_type"],
            "Accuracy": f"{b['accuracy']*100:.2f}%",
            "Precision (Clickbait)": f"{b['precision_clickbait']*100:.2f}%",
            "Recall (Clickbait)": f"{b['recall_clickbait']*100:.2f}%",
            "F1 (Clickbait)": f"{b['f1_clickbait']*100:.2f}%",
            "Macro F1 (Quan trọng nhất)": f"{b['macro_f1']*100:.2f}%",
            "ROC-AUC": f"{b['roc_auc']*100:.2f}%",
            "Tốc độ suy diễn (ms/mẫu)": f"{b['infer_ms_per_sample']:.3f} ms",
        })
    return pd.DataFrame(rows)


def build_extension_simulator_html(blur_mode: bool = False, sensitivity: float = 0.50) -> str:
    """
    Mô phỏng trực quan Giao diện 3 Tầng của Chrome Extension ngay bên trong Gradio
    để người dùng và hội đồng chấm điểm thấy rõ cách sản phẩm vận hành trên báo điện tử.
    """
    sample_feed = [
        {
            "category": "Giải trí · Showbiz",
            "time": "15 phút trước",
            "title": "Hé lộ danh tính người đàn ông bí ẩn xuất hiện cùng nữ diễn viên lúc nửa đêm khiến MXH xôn xao!",
            "lead": "Tối 25/10, đoàn làm phim xác nhận người đi cùng nữ diễn viên tại sân bay Tân Sơn Nhất là trợ lý sản xuất hỗ trợ lịch trình quay quảng cáo.",
        },
        {
            "category": "Kinh tế · Vĩ mô",
            "time": "32 phút trước",
            "title": "Ngân hàng Nhà nước giữ nguyên lãi suất điều hành nhằm ổn định thị trường tiền tệ quý IV",
            "lead": "Ngày 26/10, Ngân hàng Nhà nước Việt Nam công bố báo cáo điều hành chính sách tiền tệ, tiếp tục duy trì mặt bằng lãi suất hỗ trợ doanh nghiệp phục hồi sản xuất.",
        },
        {
            "category": "Sức khỏe · Đời sống",
            "time": "1 giờ trước",
            "title": "Chỉ vì một thói quen vào buổi sáng mà 90% người Việt mắc phải, người đàn ông 35 tuổi nhận cái kết đắng!",
            "lead": "Các bác sĩ Bệnh viện Bạch Mai vừa tiếp nhận một bệnh nhân nam 35 tuổi nhập viện do viêm loét dạ dày cấp tính vì thường xuyên bỏ bữa sáng và uống cà phê đặc khi bụng đói.",
        },
        {
            "category": "Hạ tầng · Giao thông",
            "time": "2 giờ trước",
            "title": "Cảng hàng không Vinh tạm dừng khai thác từ ngày 1/7 để nâng cấp đường băng",
            "lead": "Cảng hàng không quốc tế Vinh sẽ đóng cửa tạm thời trong 6 tháng nhằm sửa chữa đường cất hạ cánh và mở rộng sân đỗ tàu bay theo quy hoạch của Bộ Giao thông Vận tải.",
        },
    ]

    cards_html = ""
    cb_count = 0
    for art in sample_feed:
        pred = ENGINE.predict_single(art["title"], art["lead"], sensitivity=sensitivity)
        lvl = pred["badge_level"]
        if pred["is_clickbait"]:
            cb_count += 1

        if lvl == "red":
            b_bg, b_col, b_border = "#fee2e2", "#b91c1c", "#fca5a5"
            dot = "🔴"
        elif lvl == "yellow":
            b_bg, b_col, b_border = "#fef3c7", "#b45309", "#fcd34d"
            dot = "🟡"
        else:
            b_bg, b_col, b_border = "#dcfce7", "#15803d", "#86efac"
            dot = "🟢"

        blur_style = (
            "filter: blur(3.5px); opacity: 0.55; transition: all 0.25s ease;"
            if (blur_mode and pred["is_clickbait"])
            else "transition: all 0.25s ease;"
        )
        blur_hint = (
            "<div style='font-size:11.5px;color:#dc2626;font-weight:700;margin-bottom:4px;'>"
            "🙈 Đã tự động làm mờ tin giật tít (Rê chuột vào để xem hoặc đọc tóm tắt bên dưới)</div>"
            if (blur_mode and pred["is_clickbait"])
            else ""
        )

        trig_str = ", ".join(f'"{t["word"]}"' for t in pred["trigger_words"][:3]) or "Không có"

        cards_html += f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:6px;">
                <span style="font-size:12px;color:#64748b;font-weight:600;">{art['category']} · {art['time']}</span>
                <span style="background:{b_bg};color:{b_col};border:1px solid {b_border};padding:3px 10px;border-radius:999px;font-size:12px;font-weight:700;">
                    {dot} {pred['badge_text']}
                </span>
            </div>
            {blur_hint}
            <div style="{blur_style}" onmouseover="this.style.filter='none';this.style.opacity='1';" onmouseout="this.style.filter='{'blur(3.5px)' if (blur_mode and pred['is_clickbait']) else 'none'}';">
                <div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:6px;line-height:1.4;">
                    {art['title']}
                </div>
            </div>
            <!-- Khung Hover Tooltip / Tóm tắt thông minh của Extension -->
            <div style="margin-top:10px;background:#f8fafc;border-left:3px solid {b_col};padding:9px 12px;border-radius:6px;font-size:12.5px;color:#334155;">
                <div><b>⚡ Từ khóa báo động:</b> {trig_str}</div>
                <div style="margin-top:3px;"><b>💡 Tóm tắt Sapo (Đọc ngay không cần click):</b> {pred['quick_summary']}</div>
                {f"<div style='margin-top:3px;color:#1d4ed8;'><b>✨ Tiêu đề trung lập:</b> “{pred['declickbait_title']}”</div>" if pred['is_clickbait'] else ""}
            </div>
        </div>
        """

    return f"""
    <div style="display:grid;grid-template-columns:2fr 1fr;gap:18px;font-family:'Segoe UI',system-ui,sans-serif;">
        <!-- Cột trái: Luồng báo điện tử có nhúng Tầng 1 Badge -->
        <div style="background:#f1f5f9;padding:18px;border-radius:14px;border:1px solid #cbd5e1;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
                <div>
                    <span style="font-size:16px;font-weight:800;color:#0f172a;">📰 Trang chủ Báo điện tử (Tầng 1: In-Page Ambient Badges)</span>
                    <div style="font-size:12.5px;color:#475569;">Nhãn Xanh/Vàng/Đỏ và tóm tắt Sapo được gắn tự động ngay cạnh tiêu đề</div>
                </div>
                <a href="/demo-news" target="_blank" style="background:#2563eb;color:#fff;text-decoration:none;padding:7px 14px;border-radius:8px;font-size:12.5px;font-weight:700;">
                    ↗ Mở Trang Báo Giả Lập (/demo-news)
                </a>
            </div>
            {cards_html}
        </div>

        <!-- Cột phải: Tầng 3 Action Popup của Chrome Extension -->
        <div style="background:#0f172a;color:#f8fafc;padding:20px;border-radius:16px;box-shadow:0 10px 25px rgba(15,23,42,0.25);height:fit-content;">
            <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #334155;padding-bottom:12px;margin-bottom:14px;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-size:20px;">🛡️</span>
                    <div>
                        <div style="font-weight:800;font-size:15px;">VN Anti-Clickbait Shield</div>
                        <div style="font-size:11px;color:#94a3b8;">Tầng 3: Giao diện Extension Popup</div>
                    </div>
                </div>
                <span style="background:#22c55e;color:#052e16;font-size:11px;font-weight:800;padding:2px 8px;border-radius:999px;">ĐANG BẬT</span>
            </div>

            <div style="font-size:11.5px;color:#94a3b8;font-weight:700;text-transform:uppercase;margin-bottom:8px;">
                Hiệu quả bảo vệ của bạn
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px;">
                <div style="background:#1e293b;padding:12px;border-radius:10px;text-align:center;border:1px solid #334155;">
                    <div style="font-size:22px;font-weight:800;color:#f87171;">{cb_count} bài</div>
                    <div style="font-size:11.5px;color:#cbd5e1;">Tin giật tít phát hiện</div>
                </div>
                <div style="background:#1e293b;padding:12px;border-radius:10px;text-align:center;border:1px solid #334155;">
                    <div style="font-size:22px;font-weight:800;color:#38bdf8;">~{cb_count * 2} phút</div>
                    <div style="font-size:11.5px;color:#cbd5e1;">Thời gian tiết kiệm</div>
                </div>
            </div>

            <div style="background:#1e293b;padding:12px 14px;border-radius:10px;border:1px solid #334155;margin-bottom:14px;font-size:13px;">
                <div style="font-weight:700;margin-bottom:6px;color:#e2e8f0;">📊 Độ sạch trang hiện tại</div>
                <div style="display:flex;justify-content:space-between;color:#cbd5e1;font-size:12.5px;">
                    <span>Tỷ lệ tin không giật tít:</span>
                    <b style="color:#4ade80;">{int((4 - cb_count) / 4 * 100)}% ({4 - cb_count}/4 bài)</b>
                </div>
                <div style="display:flex;justify-content:space-between;color:#cbd5e1;font-size:12.5px;margin-top:4px;">
                    <span>Ngưỡng nhạy bộ lọc:</span>
                    <b style="color:#fbbf24;">{int(sensitivity * 100)}%</b>
                </div>
            </div>

            <div style="font-size:12px;color:#94a3b8;line-height:1.5;">
                💡 <b>Cách cài Extension thật vào Chrome/Edge:</b><br>
                1. Mở <code>chrome://extensions</code><br>
                2. Bật <b>Developer mode (Chế độ dành cho nhà phát triển)</b><br>
                3. Nhấn <b>Load unpacked (Tải tiện ích đã giải nén)</b> và chọn thư mục <code>extension/</code> trong dự án này.
            </div>
        </div>
    </div>
    """


# ==============================================================================
# 4. XÂY DỰNG GIAO DIỆN GRADIO BLOCKS
# ==============================================================================
CUSTOM_CSS = """
.gradio-container { max-width: 1280px !important; font-family: 'Segoe UI', system-ui, sans-serif !important; }
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: white;
    padding: 22px 28px;
    border-radius: 16px;
    margin-bottom: 16px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.15);
}
"""

with gr.Blocks(title="VN Anti-Clickbait AI — Demo & Extension Hub") as gradio_app:
    gr.HTML(
        """
        <div class="header-banner">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
                <div>
                    <div style="display:inline-block;background:rgba(56,189,248,0.2);color:#7dd3fc;padding:4px 12px;border-radius:999px;font-size:12px;font-weight:700;margin-bottom:8px;">
                        🛡️ VIETNAMESE CLICKBAIT DETECTION · NLP & BROWSER EXTENSION
                    </div>
                    <h1 style="margin:0;font-size:26px;font-weight:800;color:#ffffff;">
                        Hệ thống Phát hiện & Giải thích Tin Giật Tít Tiếng Việt
                    </h1>
                    <p style="margin:6px 0 0 0;color:#cbd5e1;font-size:14.5px;">
                        Kết hợp đặc trưng ngôn ngữ từ <b>Tiêu đề (Title)</b> và <b>Đoạn mở đầu (Sapo)</b> · Hỗ trợ 5 mô hình ML/DL & API trực tiếp cho Chrome Extension
                    </p>
                </div>
                <div style="display:flex;gap:10px;">
                    <a href="/demo-news" target="_blank" style="background:#38bdf8;color:#0f172a;text-decoration:none;padding:10px 16px;border-radius:10px;font-weight:800;font-size:13.5px;">
                        📰 Mở Trang Báo Thử Nghiệm Extension
                    </a>
                </div>
            </div>
        </div>
        """
    )

    with gr.Tabs():
        # ----------------------------------------------------------------------
        # TAB 1: PHÂN TÍCH & GIẢI THÍCH MÔ HÌNH (CHUYÊN SÂU & TRỰC QUAN)
        # ----------------------------------------------------------------------
        with gr.TabItem("🔍 1. Phân tích Bài báo & Giải thích AI (XAI)"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("### 📥 Đầu vào Bài báo (Nhập trực tiếp hoặc Dán URL)")
                    with gr.Row():
                        url_input = gr.Textbox(
                            label="🌐 Tự động trích xuất từ URL báo điện tử (VnExpress, Dân Trí, Tuổi Trẻ, Kênh14...)",
                            placeholder="https://vnexpress.net/...",
                            scale=4,
                        )
                        fetch_btn = gr.Button("Trích xuất URL", variant="secondary", scale=1)

                    title_input = gr.Textbox(
                        label="📌 Tiêu đề bài báo (Title)",
                        placeholder="Nhập tiêu đề bài báo tiếng Việt cần kiểm tra...",
                        lines=2,
                        value=DEMO_EXAMPLES[0][0],
                    )
                    lead_input = gr.Textbox(
                        label="📝 Đoạn mở đầu / Sapo (Lead Paragraph — Giúp đối chiếu độ lệch ngữ nghĩa)",
                        placeholder="Nhập đoạn sapo tóm tắt ở đầu bài báo (nếu có)...",
                        lines=3,
                        value=DEMO_EXAMPLES[0][1],
                    )

                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            label="🤖 Chọn Mô hình Suy diễn",
                            choices=[
                                "Ensemble (Khuyên dùng)",
                                "TF-IDF + Linear SVM",
                                "TF-IDF + Logistic Regression",
                                "BiLSTM (Deep Learning)",
                                "TextCNN (Deep Learning)",
                            ],
                            value="Ensemble (Khuyên dùng)",
                            scale=3,
                        )
                        sensitivity_slider = gr.Slider(
                            label="🎚️ Ngưỡng nhạy Clickbait",
                            minimum=0.35,
                            maximum=0.75,
                            step=0.05,
                            value=0.50,
                            scale=2,
                        )

                    analyze_btn = gr.Button("🚀 Phân tích & Giải thích Ngay", variant="primary", size="lg")

                    gr.Examples(
                        label="💡 Nhấp chọn nhanh các mẫu bài báo thực tế (Giật tít vs Không giật tít & Khử nhiễu VnExpress/Báo Mới):",
                        examples=DEMO_EXAMPLES,
                        inputs=[title_input, lead_input, model_dropdown, sensitivity_slider],
                    )

                with gr.Column(scale=6):
                    gr.Markdown("### 🎯 Kết quả Đánh giá & Giải thích cho Độc giả")
                    verdict_output = gr.HTML(label="Kết luận")

                    highlighted_output = gr.HighlightedText(
                        label="🖍️ Giải thích Trọng số Từ khóa (Token-level XAI Attribution)",
                        combine_adjacent=False,
                        show_legend=True,
                        color_map={
                            "Giật tít mạnh": "#ef4444",
                            "Gợi tò mò": "#f59e0b",
                            "Không giật tít": "#10b981",
                        },
                    )

                    with gr.Accordion("🔬 Chi tiết Kỹ thuật: Điểm số 5 Mô hình, Tiền xử lý & 12 Đặc trưng Thủ công", open=False):
                        model_scores_output = gr.Label(
                            label="So sánh Xác suất Clickbait giữa các Mô hình",
                            num_top_classes=5,
                        )
                        prep_output = gr.HTML(label="Chi tiết Tiền xử lý Đặc thù")
                        hc_table_output = gr.Dataframe(
                            label="12 Đặc trưng Ngôn ngữ Thủ công (Hand-crafted Features)",
                            interactive=False,
                        )

        # ----------------------------------------------------------------------
        # TAB 2: MÔ PHỎNG GIAO DIỆN EXTENSION 3 TẦNG CHO NGƯỜI DÙNG CUỐI
        # ----------------------------------------------------------------------
        with gr.TabItem("🧩 2. Trải nghiệm Giao diện Chrome Extension (UX 3 Tầng)"):
            gr.Markdown(
                """
                ### Thiết kế Giao diện hướng tới Tệp khách hàng Độc giả Phổ thông & Dân văn phòng
                Không bắt người dùng phải copy-paste thủ công từng tiêu đề: Extension tự động gắn **Nhãn tín hiệu (Xanh/Vàng/Đỏ)**,
                cung cấp **Tóm tắt Sapo khi rê chuột** để không cần bấm vào tin rác, và hỗ trợ **Làm mờ (Blur) tin câu view**.
                """
            )
            with gr.Row():
                sim_blur_toggle = gr.Checkbox(
                    label="🙈 Bật chế độ Tự động làm mờ (Blur) các tiêu đề Giật tít trên trang báo",
                    value=False,
                )
                sim_sens_slider = gr.Slider(
                    label="🎚️ Độ nhạy bộ lọc của Extension",
                    minimum=0.35,
                    maximum=0.75,
                    step=0.05,
                    value=0.50,
                )
            sim_html_output = gr.HTML(value=build_extension_simulator_html(False, 0.50))

            sim_blur_toggle.change(
                fn=build_extension_simulator_html,
                inputs=[sim_blur_toggle, sim_sens_slider],
                outputs=sim_html_output,
            )
            sim_sens_slider.change(
                fn=build_extension_simulator_html,
                inputs=[sim_blur_toggle, sim_sens_slider],
                outputs=sim_html_output,
            )

        # ----------------------------------------------------------------------
        # TAB 3: BẢNG ĐIỂM THỰC NGHIỆM & BIỂU ĐỒ PHÂN TÍCH DỮ LIỆU (EDA)
        # ----------------------------------------------------------------------
        with gr.TabItem("📊 3. Bảng điểm Thực nghiệm (Benchmark) & Biểu đồ EDA"):
            gr.Markdown("### 🏆 Bảng tổng hợp Kết quả Thực nghiệm trên Tập Kiểm tra (Test Set)")
            gr.Dataframe(
                value=get_benchmark_dataframe(),
                interactive=False,
            )

            gr.Markdown("### 📈 Bộ Biểu đồ Khám phá Dữ liệu (EDA) & Feature Importance")
            figure_files = [
                ("outputs/figures/01_label_distribution.png", "1. Phân phối nhãn Clickbait vs Non-Clickbait"),
                ("outputs/figures/02_source_category_distribution.png", "2. Phân bố theo Nguồn báo & Chuyên mục (Lý do chống Data Leakage)"),
                ("outputs/figures/03_text_length_distribution.png", "3. Phân phối độ dài Tiêu đề và Đoạn mở đầu (Sapo)"),
                ("outputs/figures/04_punctuation_features.png", "4. Tỷ lệ Dấu câu cảm thán (?, !, \"...\") và Con số"),
                ("outputs/figures/05_wordcloud_comparison.png", "5. WordCloud đối sánh từ vựng hai nhãn"),
                ("outputs/figures/09_feature_importance_top20.png", "6. Top 20 n-grams đặc trưng cho Clickbait & Non-Clickbait"),
            ]
            existing_figs = [(p, cap) for p, cap in figure_files if os.path.exists(p)]
            if existing_figs:
                gr.Gallery(
                    value=existing_figs,
                    label="Biểu đồ Phân tích từ Tập dữ liệu Báo chí Tiếng Việt",
                    columns=2,
                    height=520,
                )

    # Sự kiện tương tác Tab 1
    analyze_btn.click(
        fn=run_gradio_analysis,
        inputs=[title_input, lead_input, model_dropdown, sensitivity_slider],
        outputs=[verdict_output, highlighted_output, model_scores_output, hc_table_output, prep_output],
    )
    title_input.submit(
        fn=run_gradio_analysis,
        inputs=[title_input, lead_input, model_dropdown, sensitivity_slider],
        outputs=[verdict_output, highlighted_output, model_scores_output, hc_table_output, prep_output],
    )
    fetch_btn.click(
        fn=handle_url_fetch,
        inputs=[url_input, model_dropdown, sensitivity_slider],
        outputs=[title_input, lead_input, verdict_output, highlighted_output, model_scores_output, hc_table_output, prep_output],
    )

    gradio_app.load(
        fn=run_gradio_analysis,
        inputs=[title_input, lead_input, model_dropdown, sensitivity_slider],
        outputs=[verdict_output, highlighted_output, model_scores_output, hc_table_output, prep_output],
    )


# ==============================================================================
# 5. FASTAPI BACKEND SERVER PHỤC VỤ CHROME EXTENSION (MANIFEST V3)
# ==============================================================================
api_app = FastAPI(
    title="VN Anti-Clickbait API & Gradio Hub",
    description="REST API phục vụ Chrome Extension Phát hiện Clickbait Tiếng Việt",
    version="1.0.0",
)

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SinglePredictRequest(BaseModel):
    title: str = Field(..., description="Tiêu đề bài báo")
    lead_paragraph: str = Field("", description="Đoạn mở đầu / sapo bài báo")
    model_choice: str = Field("Ensemble (Khuyên dùng)", description="Mô hình lựa chọn")
    sensitivity: float = Field(0.50, ge=0.2, le=0.9)


class BatchArticleItem(BaseModel):
    id: str
    title: str
    lead_paragraph: str = ""


class BatchPredictRequest(BaseModel):
    items: List[BatchArticleItem]
    sensitivity: float = Field(0.50, ge=0.2, le=0.9)


class UrlExtractRequest(BaseModel):
    url: str
    sensitivity: float = Field(0.50, ge=0.2, le=0.9)


class FeedbackRequest(BaseModel):
    url: str = ""
    title: str
    lead_paragraph: str = ""
    predicted_label: str
    user_suggested_label: str
    comment: str = ""


@api_app.get("/api/health")
def api_health():
    return {
        "status": "ok",
        "engine": "ClickbaitInferenceEngine (Ensemble SVM + LR + BiLSTM + TextCNN)",
        "models_loaded": ["Linear SVM", "Logistic Regression", "BiLSTM", "TextCNN"],
    }


@api_app.post("/api/predict")
def api_predict_single(req: SinglePredictRequest):
    return ENGINE.predict_single(
        title=req.title,
        lead_paragraph=req.lead_paragraph,
        model_choice=req.model_choice,
        sensitivity=req.sensitivity,
    )


@api_app.post("/api/predict_batch")
def api_predict_batch(req: BatchPredictRequest):
    raw_items = [item.model_dump() for item in req.items]
    results = ENGINE.predict_batch(raw_items, sensitivity=req.sensitivity)
    return {"count": len(results), "results": results}


@api_app.post("/api/analyze_url")
def api_analyze_url(req: UrlExtractRequest):
    try:
        scraped = extract_article_from_url(req.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    pred = ENGINE.predict_single(
        title=scraped["title"],
        lead_paragraph=scraped["lead_paragraph"],
        sensitivity=req.sensitivity,
    )
    pred["scraped_source"] = scraped["source"]
    pred["scraped_url"] = scraped["url"]
    return pred


@api_app.post("/api/feedback")
def api_submit_feedback(req: FeedbackRequest):
    os.makedirs("outputs/results", exist_ok=True)
    feedback_file = "outputs/results/user_feedback.jsonl"
    record = req.model_dump()
    record["timestamp"] = datetime.now().isoformat()
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "saved", "message": "Cảm ơn bạn đã đóng góp dữ liệu giúp cải thiện mô hình!"}


@api_app.get("/demo-news", response_class=HTMLResponse)
def serve_demo_news():
    demo_html_path = os.path.join("extension", "demo_news.html")
    if os.path.exists(demo_html_path):
        with open(demo_html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h3>Chưa tìm thấy file extension/demo_news.html</h3>")


@api_app.get("/content.css")
def serve_extension_css():
    css_path = os.path.join("extension", "content.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="Not found")


# Gắn Gradio vào gốc "/" của FastAPI
app = gr.mount_gradio_app(api_app, gradio_app, path="/", css=CUSTOM_CSS)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print("=" * 80)
    print(f"🚀 KHỞI ĐỘNG HỆ THỐNG DEMO GRADIO & EXTENSION API TẠI: http://127.0.0.1:{port}")
    print(f"📰 Trang Báo Điện Tử Giả Lập để test Extension:      http://127.0.0.1:{port}/demo-news")
    print(f"🔌 Endpoint API cho Chrome Extension:                http://127.0.0.1:{port}/api/predict_batch")
    print("=" * 80)
    uvicorn.run(app, host="127.0.0.1", port=port)
