# ============================================================
# PDF2MD - Windows Explorer Context Menu Uninstaller
# ============================================================

# --- Admin check ---
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[ERROR] Administrator privileges required." -ForegroundColor Red
    Write-Host "Right-click -> 'Run as administrator'" -ForegroundColor Red
    exit 1
}

Write-Host "============================================================"
Write-Host " PDF2MD Context Menu Uninstaller"
Write-Host "============================================================"
Write-Host ""

# Mount HKCR if not available
if (-not (Test-Path "HKCR:\")) {
    New-PSDrive -Name HKCR -PSProvider Registry -Root HKEY_CLASSES_ROOT | Out-Null
}

$extensions = @('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.xlsm', '.pptx', '.csv', '.epub', '.ipynb', '.html', '.htm', '.msg', '.jpg', '.jpeg', '.png')
$removed = 0

foreach ($ext in $extensions) {
    $shellKey = "HKCR:\SystemFileAssociations\$ext\shell\PDF2MD"
    if (Test-Path $shellKey) {
        Remove-Item -Path $shellKey -Recurse -Force
        Write-Host "[OK] Removed for $ext" -ForegroundColor Green
        $removed++
    }
}

if ($removed -eq 0) {
    Write-Host "[INFO] Context menu was not registered." -ForegroundColor Yellow
}
else {
    Write-Host ""
    Write-Host "[OK] Context menu removed ($removed extensions)." -ForegroundColor Green
}
