# -*- mode: python ; coding: utf-8 -*-
# PDF2MD PyInstaller spec file
# PyMuPDF + EasyOCR version

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

block_cipher = None

# PyMuPDFを完全に収集（データ、バイナリ、hiddenimports全て）
fitz_datas = []
fitz_binaries = []
fitz_hiddenimports = []
try:
    fitz_datas, fitz_binaries, fitz_hiddenimports = collect_all('fitz')
except Exception:
    pass

# pymupdfも収集（fitz の新しいパッケージ名）
pymupdf_datas = []
pymupdf_binaries = []
pymupdf_hiddenimports = []
try:
    pymupdf_datas, pymupdf_binaries, pymupdf_hiddenimports = collect_all('pymupdf')
except Exception:
    pass

# EasyOCRのデータファイルを収集
easyocr_datas = []
try:
    easyocr_datas = collect_data_files('easyocr')
except Exception:
    pass

a = Analysis(
    ['pdf2md.py'],
    pathex=[],
    binaries=fitz_binaries + pymupdf_binaries,
    datas=fitz_datas + pymupdf_datas + easyocr_datas,
    hiddenimports=[
        'fitz',
        'pymupdf',
        'PIL',
        'PIL.Image',
        'easyocr',
        'torch',
        'torchvision',
    ] + fitz_hiddenimports + pymupdf_hiddenimports,
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
    console=True,  # 変換進捗を表示するため（False にするとコンソール非表示）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
