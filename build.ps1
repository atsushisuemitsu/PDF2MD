# PDF to Markdown Converter - EXE Build Script (PowerShell)

Write-Host "================================" -ForegroundColor Cyan
Write-Host "PDF2MD EXE Build Script" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Check dependencies
Write-Host "Checking dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Build EXE using spec file (includes magika model files)
Write-Host "Building EXE..." -ForegroundColor Yellow
pyinstaller --clean PDF2MD.spec

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "Build completed!" -ForegroundColor Green
Write-Host "EXE location: dist\PDF2MD.exe" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
