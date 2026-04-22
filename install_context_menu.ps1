# ============================================================
# PDF2MD - Windows Explorer Context Menu Installer
# ============================================================

# --- Admin check ---
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[ERROR] Administrator privileges required." -ForegroundColor Red
    Write-Host "Right-click -> 'Run as administrator'" -ForegroundColor Red
    exit 1
}

# --- Locate PDF2MD.exe ---
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$exePath = Join-Path $scriptDir "dist\PDF2MD.exe"
if (-not (Test-Path $exePath)) {
    Write-Host "[ERROR] dist\PDF2MD.exe not found." -ForegroundColor Red
    Write-Host "Run build.bat first." -ForegroundColor Red
    exit 1
}

Write-Host "============================================================"
Write-Host " PDF2MD Context Menu Installer"
Write-Host "============================================================"
Write-Host ""
Write-Host "EXE path: $exePath"
Write-Host ""

# Mount HKCR if not available
if (-not (Test-Path "HKCR:\")) {
    New-PSDrive -Name HKCR -PSProvider Registry -Root HKEY_CLASSES_ROOT | Out-Null
}

$extensions = @('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.xlsm', '.pptx', '.csv', '.epub', '.ipynb', '.html', '.htm', '.msg', '.jpg', '.jpeg', '.png')

try {
    foreach ($ext in $extensions) {
        $shellKey = "HKCR:\SystemFileAssociations\$ext\shell\PDF2MD"
        $commandKey = "$shellKey\command"

        # Create shell key with display name
        New-Item -Path $shellKey -Force | Out-Null
        Set-ItemProperty -Path $shellKey -Name "(Default)" -Value "PDF2MD -> MD" -Force
        Set-ItemProperty -Path $shellKey -Name "Icon" -Value "`"$exePath`",0" -Force

        # Create command key
        New-Item -Path $commandKey -Force | Out-Null
        Set-ItemProperty -Path $commandKey -Name "(Default)" -Value "`"$exePath`" `"%1`"" -Force

        Write-Host "[OK] Registered for $ext" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  1. Right-click a supported file in Explorer"
    Write-Host "     (.pdf, .doc, .docx, .xls, .xlsx, .xlsm, .pptx,"
    Write-Host "      .csv, .epub, .ipynb, .html, .htm, .msg, .jpg, .jpeg, .png)"
    Write-Host "  2. Select 'PDF2MD -> MD'"
    Write-Host "     (Windows 11: under 'Show more options')"
    Write-Host "  3. Output is saved next to the input file"
    Write-Host ""
    Write-Host "To remove, run uninstall_context_menu.bat as administrator."
}
catch {
    Write-Host "[ERROR] Registry write failed: $_" -ForegroundColor Red
    exit 1
}
