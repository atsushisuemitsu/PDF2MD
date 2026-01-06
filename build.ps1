# PDF to Markdown Converter - EXE Build Script (PowerShell)
#
# 事前準備:
#   pip install -r requirements.txt
#
# 使用方法:
#   .\build.ps1
#

Write-Host "================================" -ForegroundColor Cyan
Write-Host "PDF2MD EXE Build Script" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Python仮想環境があれば有効化
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    . .\venv\Scripts\Activate.ps1
}

# 依存関係インストール確認
Write-Host "Checking dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt -q

# PyInstallerでビルド
Write-Host "Building EXE..." -ForegroundColor Yellow

$iconArg = @()
if (Test-Path "icon.ico") {
    $iconArg = @("--icon", "icon.ico")
}

$buildArgs = @(
    "--onefile",
    "--windowed",
    "--name", "PDF2MD",
    "--clean"
)

if ($iconArg.Count -gt 0) {
    $buildArgs += $iconArg
}

$buildArgs += "pdf2md.py"

pyinstaller @buildArgs

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "Build completed!" -ForegroundColor Green
Write-Host "EXE location: dist\PDF2MD.exe" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
