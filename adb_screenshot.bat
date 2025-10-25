@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem ADB-Konfiguration anpassen, falls noetig
set "ADB=adb"
set "DEVICE_ID=emulator-5554"  rem z.B.: emulator-5554 oder leer lassen

if defined DEVICE_ID (
  set "ADB_TARGET=-s %DEVICE_ID%"
) else (
  set "ADB_TARGET="
)

rem ADB starten (Daemon)
"%ADB%" start-server >nul 2>&1

rem Screenshot-Verzeichnis
set "OUTDIR=screenshots"
if not exist "%OUTDIR%" mkdir "%OUTDIR%" 2>nul

rem Timestamp erzeugen: yyyy-MM-dd_HH-mm-ss (datei-sicher)
for /f %%t in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd_HH-mm-ss')"') do set "TS=%%t"

set "OUTFILE=%OUTDIR%\adb_screenshot_!TS!.png"

echo [INFO] Geraet: %DEVICE_ID%
echo [INFO] Datei:  %OUTFILE%
echo [INFO] Erzeuge Screenshot...

rem Bevorzugt exec-out (keine CR/LF-Konvertierung)
"%ADB%" %ADB_TARGET% exec-out screencap -p > "!OUTFILE!"
if errorlevel 1 (
  echo [WARN] exec-out fehlgeschlagen. Versuche Fallback via shell...
  "%ADB%" %ADB_TARGET% shell screencap -p > "!OUTDIR!\screenshot_!TS!_fallback.png"
  if errorlevel 1 (
    echo [FEHLER] Screenshot fehlgeschlagen. Ist ein Geraet verbunden? (adb devices)
    exit /b 1
  ) else (
    echo [OK] Screenshot erstellt: !OUTDIR!\screenshot_!TS!_fallback.png
    exit /b 0
  )
) else (
  echo [OK] Screenshot erstellt: !OUTFILE!
  exit /b 0
)

endlocal