@echo off
REM PDF to Markdown Converter - EXE Build Script
REM
REM 事前準備:
REM   pip install -r requirements.txt
REM
REM 使用方法:
REM   build.bat
REM

echo ================================
echo PDF2MD EXE Build Script
echo ================================

REM Python仮想環境があれば有効化
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM 依存関係インストール確認
echo Checking dependencies...
pip install -r requirements.txt -q

REM PyInstallerでビルド (specファイル使用でmagikaモデル含む)
echo Building EXE...
pyinstaller --clean PDF2MD.spec

echo.
echo ================================
echo Build completed!
echo EXE location: dist\PDF2MD.exe
echo ================================

pause
