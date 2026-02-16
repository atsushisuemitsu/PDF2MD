@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

REM ============================================================
REM PDF2MD - Windows Explorer Context Menu Installer
REM PDFファイルの右クリックメニューに「PDF → MD 変換」を追加
REM ============================================================

REM --- Admin check ---
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 管理者権限が必要です。
    echo 右クリック →「管理者として実行」で再実行してください。
    pause
    exit /b 1
)

REM --- Locate PDF2MD.exe ---
set "EXE_PATH=%~dp0dist\PDF2MD.exe"
if not exist "%EXE_PATH%" (
    echo [ERROR] dist\PDF2MD.exe が見つかりません。
    echo 先に build.bat でビルドしてください。
    pause
    exit /b 1
)

echo ============================================================
echo  PDF2MD コンテキストメニュー インストーラー
echo ============================================================
echo.
echo 登録するEXE: %EXE_PATH%
echo.

REM --- Register context menu ---
reg add "HKEY_CLASSES_ROOT\SystemFileAssociations\.pdf\shell\PDF2MD" /ve /d "PDF → MD 変換" /f >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] レジストリの書き込みに失敗しました。
    pause
    exit /b 1
)

reg add "HKEY_CLASSES_ROOT\SystemFileAssociations\.pdf\shell\PDF2MD" /v "Icon" /d "\"%EXE_PATH%\",0" /f >nul 2>&1

reg add "HKEY_CLASSES_ROOT\SystemFileAssociations\.pdf\shell\PDF2MD\command" /ve /d "\"%EXE_PATH%\" \"%%1\"" /f >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] コマンド登録に失敗しました。
    pause
    exit /b 1
)

echo [OK] コンテキストメニューを登録しました。
echo.
echo 使い方:
echo   1. エクスプローラーでPDFファイルを右クリック
echo   2. 「PDF → MD 変換」を選択
echo      (Windows 11: 「その他のオプションを確認」から表示)
echo   3. 変換結果はPDFと同じフォルダに出力されます
echo.
echo 削除するには uninstall_context_menu.bat を管理者として実行してください。
echo.
pause
