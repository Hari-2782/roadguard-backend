@echo off
cd /d "%~dp0"
".\.venv\Scripts\python.exe" -c "from backend.api.api_server import app; app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)"
