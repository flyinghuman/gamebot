@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Konfiguration
set "CONFIG=config.yaml"
set "PYTHON=python"

rem Falls venv existiert, diese Python-Exe nehmen
if exist "..\.venv\Scripts\python.exe" (
  set "PYTHON=..\.venv\Scripts\python.exe"
)

echo [INFO] Verwende Python: %PYTHON%
echo [INFO] Config: %CONFIG%
echo.

rem 1) run.py bevorzugen
if exist "bot.py" (
  echo [INFO] Starte: %PYTHON% bot.py --config "%CONFIG%"
  "%PYTHON%" "bot.py" --config "%CONFIG%"
  goto :end
)

rem 2) Fallback: Python-Modul "bot"
echo [INFO] Starte: %PYTHON% -m bot --config "%CONFIG%"
"%PYTHON%" -m bot --config "%CONFIG%"

:end
echo.
if errorlevel 1 (
  echo [FEHLER] Bot konnte nicht gestartet werden. Bitte Pfade/Abhaengigkeiten pruefen.
) else (
  echo [OK] Bot beendet.
)
endlocal