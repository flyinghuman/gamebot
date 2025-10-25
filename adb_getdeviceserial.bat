@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem ADB-Konfiguration anpassen, falls noetig
set "ADB=adb"

rem ADB starten (Daemon)
"%ADB%" start-server >nul 2>&1

rem ADB Device List ausgeben
"%ADB%" devices -l

pause