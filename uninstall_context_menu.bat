@echo off
REM Launch PowerShell uninstaller (UTF-8 safe)
powershell -ExecutionPolicy Bypass -File "%~dp0uninstall_context_menu.ps1"
pause
