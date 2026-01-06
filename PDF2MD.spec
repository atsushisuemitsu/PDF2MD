# -*- mode: python ; coding: utf-8 -*-
# PDF2MD PyInstaller spec file
# magikaモデルファイルを含めるための設定

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# magikaのデータファイルを収集
magika_datas = collect_data_files('magika')

# markitdownの追加データがあれば収集
markitdown_datas = collect_data_files('markitdown')

a = Analysis(
    ['pdf2md.py'],
    pathex=[],
    binaries=[],
    datas=magika_datas + markitdown_datas,
    hiddenimports=[
        'magika',
        'magika.types',
        'markitdown',
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
