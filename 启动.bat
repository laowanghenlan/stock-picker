@echo off
cd /d "%~dp0"
echo ============================================
echo   A-share Quant Analyzer
echo ============================================
echo Starting, browser will open in about 8 seconds...
echo Keep this window open. Close it to stop the server.
echo.

start "" /min powershell -WindowStyle Hidden -Command "Start-Sleep -Seconds 8; Start-Process 'http://localhost:8501'"

"C:\Users\Administrator\Documents\Trae-CN\temp_20261109_03\stock_analyzer\.venv\Scripts\python.exe" -m streamlit run app.py --server.headless true

echo.
echo Server stopped.
pause
