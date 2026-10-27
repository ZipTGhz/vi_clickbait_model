@echo off
chcp 65001 > nul
title Dừng VN Anti-Clickbait Server
cd /d "%~dp0"

echo =====================================================================
echo  ĐANG DỪNG TIẾN TRÌNH SERVER TRÊN CỔNG 7860...
echo =====================================================================

for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7860') do (
    echo Đang dừng tiến trình PID: %%a...
    taskkill /f /pid %%a > nul 2>&1
)

echo.
echo [XONG] Đã dừng toàn bộ tiến trình trên cổng 7860.
timeout /t 3 > nul
