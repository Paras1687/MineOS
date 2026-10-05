@echo off
powershell.exe -NoProfile -File "%~dp0run.ps1"
if errorlevel 1 pause
