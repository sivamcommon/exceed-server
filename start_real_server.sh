#!/bin/bash

echo "============================================"
echo "  Real Server Startup"
echo "============================================"

# Kill any process running on port 8000 (dummy server or old instance)
echo "[*] Stopping any server on port 8000..."
fuser -k 8000/tcp 2>/dev/null && echo "[*] Killed process on port 8000" || echo "[*] No process on port 8000"

# Also kill by process name to be thorough
pkill -f "uvicorn server:app" 2>/dev/null && sleep 0.5

sleep 1
echo "[*] Port cleared. Starting Real Server..."
echo "============================================"

cd /home/gomicro/GoMicro_Common/Server2_Backup_Live_Video_Stable_V2

# Clear old log file so tail starts fresh
> frontend_logs.log

# Tail the Python app log file in background so it prints to this terminal too
tail -f frontend_logs.log &
TAIL_PID=$!

# Run uvicorn in foreground (its own startup/request logs print to terminal directly)
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1

# Clean up tail when server exits
kill $TAIL_PID 2>/dev/null
