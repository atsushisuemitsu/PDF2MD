@echo off
chcp 65001 >nul 2>&1
setlocal

REM ============================================================
REM PDF2MD - Windows Explorer Context Menu Uninstaller
REM PDFファイルの右クリックメニューから「PDF → MD 変換」を削除
REM ============================================================

REM --- Admin check ---
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 管理者権限が必要です。
    echo 右クリック →「管理者として実行」で再実行してください。
    pause
    exit /b 1
)

echo ============================================================
echo  PDF2MD コンテキストメニュー アンインストーラー
echo ============================================================
echo.

REM --- Remove registry entries ---
reg delete "HKEY_CLASSES_ROOT\SystemFileAssociations\.pdf\shell\PDF2MD" /f >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] コンテキストメニューを削除しました。
) else (
    echo [INFO] コンテキストメニューは登録されていませんでした。
)

echo.
pause
