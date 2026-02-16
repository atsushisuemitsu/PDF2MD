@echo off
REM Launch PowerShell installer (UTF-8 safe)
powershell -ExecutionPolicy Bypass -File "%~dp0install_context_menu.ps1"
pause
