@echo off
chcp 65001 > nul
title VN Anti-Clickbait AI Server (Gradio & Extension Hub)
cd /d "%~dp0"

echo =====================================================================
echo  ĐANG KHỞI ĐỘNG VN ANTI-CLICKBAIT AI SERVER...
echo =====================================================================

if not exist "venv\Scripts\python.exe" (
    echo [LỖI] Không tìm thấy môi trường ảo venv!
    echo Vui lòng đảm bảo thư mục venv đã được tạo.
    pause
    exit /b
)

echo Đang kích hoạt môi trường ảo venv...
call venv\Scripts\activate

echo Đang khởi chạy app.py...
python app.py

pause
