#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "====================================================================="
echo " ĐANG KHỞI ĐỘNG VN ANTI-CLICKBAIT AI SERVER (LINUX / MACOS)..."
echo "====================================================================="

if [ ! -d "venv" ]; then
    echo "[THÔNG BÁO] Chưa tìm thấy thư mục venv, đang khởi tạo môi trường ảo..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "Đang khởi chạy app.py tại http://127.0.0.1:7860 ..."
python3 app.py
