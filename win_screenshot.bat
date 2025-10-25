@echo off

:: Stellt sicher, dass die Konsole deutsche Zeichen korrekt anzeigt
chcp 65001 > nul

:: Titel für das Konsolenfenster setzen
TITLE Screenshot Helper

REM ====================================================================================
REM ==                                                                                ==
REM ==  Batch-Skript zum Ausführen des Python-Skripts zur Screenshot-Erstellung     ==
REM ==                                                                                ==
REM ====================================================================================


REM --- ANLEITUNG ---
REM 1. Speichern Sie das Python-Skript als `win_screenshot.py` im selben Ordner wie diese Batch-Datei.
REM 2. Stellen Sie sicher, dass Python auf Ihrem System installiert und zum PATH hinzugefügt ist.
REM 3. Führen Sie diese Batch-Datei per Doppelklick aus.


REM --- Name der Python-Skriptdatei ---
SET PYTHON_SCRIPT=win_screenshot.py

rem Konfiguration
set "PYTHON=python"

rem Falls venv existiert, diese Python-Exe nehmen
if exist "..\.venv\Scripts\python.exe" (
  set "PYTHON=..\.venv\Scripts\python.exe"
)


cls
echo ======================================
echo      Screenshot Skript Starter
echo ======================================
echo.

REM Prüfen, ob das Python-Skript existiert
if not exist "%PYTHON_SCRIPT%" (
    echo.
    echo FEHLER: Die Python-Skriptdatei '%PYTHON_SCRIPT%' wurde nicht gefunden.
    echo.
    echo Bitte stellen Sie sicher, dass sich beide Dateien im selben Ordner befinden.
    echo.
    goto :end
)


echo Das Spiel-Fenster muss aktiv sein, um einen Screenshot zu erstellen.

echo.
echo Starte das Python-Skript '%PYTHON_SCRIPT%'...
echo --------------------------------------------------

REM Führe das Python-Skript aus
"%PYTHON%" "%PYTHON_SCRIPT%"


echo --------------------------------------------------
echo Skript-Ausführung beendet.

:end
echo.
echo Drücken Sie eine beliebige Taste, um dieses Fenster zu schließen...
pause > nul