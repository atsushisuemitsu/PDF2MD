# -*- mode: python ; coding: utf-8 -*-
# PDF2MD PyInstaller spec file
# PyMuPDF + EasyOCR version

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# EasyOCRのモデルファイルを収集（初回実行時にダウンロードされる）
easyocr_datas = []
try:
    easyocr_datas = collect_data_files('easyocr')
except Exception:
    pass

# PyMuPDFのデータファイル
fitz_datas = []
try:
    fitz_datas = collect_data_files('fitz')
except Exception:
    pass

a = Analysis(
    ['pdf2md.py'],
    pathex=[],
    binaries=[],
    datas=easyocr_datas + fitz_datas,
    hiddenimports=[
        'fitz',
        'PIL',
        'PIL.Image',
        'easyocr',
        'torch',
        'torchvision',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PDF2MD',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
