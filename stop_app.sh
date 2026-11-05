#!/usr/bin/env bash

echo "====================================================================="
echo " ĐANG DỪNG TIẾN TRÌNH SERVER TRÊN CỔNG 7860 (LINUX / MACOS)..."
echo "====================================================================="

if command -v fuser >/dev/null 2>&1; then
    fuser -k 7860/tcp >/dev/null 2>&1 || true
elif command -v lsof >/dev/null 2>&1; then
    PID=$(lsof -ti:7860)
    if [ -n "$PID" ]; then
        kill -9 $PID >/dev/null 2>&1 || true
    fi
else
    pkill -f "python.*app.py" >/dev/null 2>&1 || true
fi

echo "[XONG] Đã dừng toàn bộ tiến trình trên cổng 7860."
