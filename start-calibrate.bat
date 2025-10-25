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

if exist "calibrate_rois.py" (
  echo [INFO] Starte: %PYTHON% calibrate_rois.py
  "%PYTHON%" "calibrate_rois.py"
  goto :end
)


:end
echo.
if errorlevel 1 (
  echo [FEHLER] Konnte nicht gestartet werden. Bitte Pfade/Abhaengigkeiten pruefen.
  pause
) else (
  echo [OK] calibrate_rois.py beendet.
)
endlocal