@echo off
cd /d "%~dp0"
python run.py %*
if errorlevel 9009 (
    py run.py %*
)
pause
