@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem ADB-Konfiguration anpassen, falls noetig
set "ADB=adb"
set "DEVICE_ID=emulator-5554"  rem z.B.: emulator-5554 oder leer lassen

set "MODE=SINGLE"
set "LOOP_MINUTES="
set "LOOP_INTERVAL_SECONDS=60"

rem Argumente verarbeiten
if /i "%~1"=="--help" (
  call :Usage
  exit /b 0
)

if /i "%~1"=="--loop" (
  if "%~2"=="" (
    echo [FEHLER] Dauer fuer --loop fehlt.
    call :Usage
    exit /b 1
  )
  set "LOOP_MINUTES=%~2"
  for /f "delims=0123456789" %%a in ("%LOOP_MINUTES%") do (
    echo [FEHLER] Loop-Dauer muss eine ganze Zahl ^(in Minuten^) sein.
    exit /b 1
  )
  set /a LOOP_DURATION_SECONDS=LOOP_MINUTES*60
  if !LOOP_DURATION_SECONDS! LEQ 0 (
    echo [FEHLER] Loop-Dauer muss groesser als 0 sein.
    exit /b 1
  )
  set /a LOOP_WAIT_COUNT=LOOP_DURATION_SECONDS/LOOP_INTERVAL_SECONDS
  if !LOOP_WAIT_COUNT! LSS 0 set /a LOOP_WAIT_COUNT=0
  set /a LOOP_TOTAL_SHOTS=LOOP_WAIT_COUNT+1
  set "MODE=LOOP"
)

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

if /i "%MODE%"=="LOOP" (
  echo [INFO] Loop-Modus aktiv: !LOOP_MINUTES! Minuten, Intervall !LOOP_INTERVAL_SECONDS! Sekunden.
  echo [INFO] Es werden !LOOP_TOTAL_SHOTS! Screenshot^(s^) erstellt.
  set /a SHOT_INDEX=1
  echo [INFO] Screenshot !SHOT_INDEX! von !LOOP_TOTAL_SHOTS!
  call :TakeScreenshot
  if errorlevel 1 (
    echo [FEHLER] Abbruch nach Fehler beim Screenshot.
    exit /b 1
  )
  if !LOOP_WAIT_COUNT! LEQ 0 (
    echo [OK] Loop abgeschlossen.
    exit /b 0
  )
  for /l %%i in (1,1,!LOOP_WAIT_COUNT!) do (
    echo [INFO] Warte !LOOP_INTERVAL_SECONDS! Sekunden... ^(%%i von !LOOP_WAIT_COUNT!^)
    timeout /t !LOOP_INTERVAL_SECONDS! /nobreak >nul
    set /a SHOT_INDEX+=1
    echo [INFO] Screenshot !SHOT_INDEX! von !LOOP_TOTAL_SHOTS!
    call :TakeScreenshot
    if errorlevel 1 (
      echo [FEHLER] Abbruch nach Fehler beim Screenshot.
      exit /b 1
    )
  )
  echo [OK] Loop abgeschlossen.
  exit /b 0
) else (
  call :TakeScreenshot
  exit /b !errorlevel!
)

endlocal
exit /b 0

:TakeScreenshot
rem Timestamp erzeugen: yyyy-MM-dd_HH-mm-ss (datei-sicher)
for /f %%t in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd_HH-mm-ss')"') do set "TS=%%t"
set "OUTFILE=%OUTDIR%\adb_screenshot_!TS!.png"

echo [INFO] Geraet: %DEVICE_ID%
echo [INFO] Datei:  !OUTFILE!
echo [INFO] Erzeuge Screenshot...

rem Bevorzugt exec-out (keine CR/LF-Konvertierung)
"%ADB%" %ADB_TARGET% exec-out screencap -p > "!OUTFILE!"
if errorlevel 1 (
  echo [WARN] exec-out fehlgeschlagen. Versuche Fallback via shell...
  "%ADB%" %ADB_TARGET% shell screencap -p > "!OUTDIR!\screenshot_!TS!_fallback.png"
  if errorlevel 1 (
    echo [FEHLER] Screenshot fehlgeschlagen. Ist ein Geraet verbunden? ^(adb devices^)
    exit /b 1
  ) else (
    echo [OK] Screenshot erstellt: !OUTDIR!\screenshot_!TS!_fallback.png
    exit /b 0
  )
) else (
  echo [OK] Screenshot erstellt: !OUTFILE!
  exit /b 0
)

:Usage
echo.
echo Nutzung:
echo   adb_screenshot.bat          Erstellt einen Screenshot.
echo   adb_screenshot.bat --loop ^<Minuten^>  Erstellt alle 1 Min. einen Screenshot fuer die angegebene Dauer.
echo.
echo Beispiele:
echo   adb_screenshot.bat --loop 30
echo      Nimmt ueber 30 Minuten hinweg Screenshots im 1-Minuten-Intervall auf.
echo.
exit /b 0
