# -*- mode: python ; coding: utf-8 -*-
# PDF2MD PyInstaller spec file
# PyMuPDF + PyMuPDF4LLM + EasyOCR version

import sys
import os
import importlib
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

# pymupdf/fitzのバイナリを明示的に収集（collect_allで漏れる場合がある）
for pkg_name in ['pymupdf', 'fitz']:
    try:
        pkg_spec = importlib.util.find_spec(pkg_name)
        if pkg_spec and pkg_spec.submodule_search_locations:
            pkg_dir = pkg_spec.submodule_search_locations[0]
            for fname in os.listdir(pkg_dir):
                fpath = os.path.join(pkg_dir, fname)
                if fname.endswith(('.pyd', '.dll', '.so')):
                    pymupdf_binaries.append((fpath, pkg_name))
                elif fname.endswith('.py') and fname != '__init__.py':
                    pymupdf_datas.append((fpath, pkg_name))
        elif pkg_spec and pkg_spec.origin:
            # 単一ファイルモジュールの場合
            origin = pkg_spec.origin
            if origin.endswith(('.pyd', '.dll', '.so')):
                pymupdf_binaries.append((origin, '.'))
    except Exception:
        pass

# pymupdf内のmupdf-develディレクトリも収集
try:
    pymupdf_spec = importlib.util.find_spec('pymupdf')
    if pymupdf_spec and pymupdf_spec.submodule_search_locations:
        pymupdf_dir = pymupdf_spec.submodule_search_locations[0]
        mupdf_devel = os.path.join(pymupdf_dir, 'mupdf-devel')
        if os.path.isdir(mupdf_devel):
            for root, dirs, files in os.walk(mupdf_devel):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    rel_dir = os.path.join('pymupdf', os.path.relpath(root, pymupdf_dir))
                    if fname.endswith(('.dll', '.so', '.pyd')):
                        pymupdf_binaries.append((fpath, rel_dir))
                    else:
                        pymupdf_datas.append((fpath, rel_dir))
except Exception:
    pass

# pymupdf4llmのデータファイルを収集
pymupdf4llm_datas = []
pymupdf4llm_hiddenimports = []
try:
    pymupdf4llm_datas, _, pymupdf4llm_hiddenimports = collect_all('pymupdf4llm')
except Exception:
    pass

# EasyOCRのデータファイルを収集
easyocr_datas = []
try:
    easyocr_datas = collect_data_files('easyocr')
except Exception:
    pass

# Anthropic SDK（Claude API図表解析）
anthropic_datas = []
try:
    anthropic_datas = collect_data_files('anthropic')
except Exception:
    pass

# MarkItDown（Microsoft製PDF/Office変換）
markitdown_datas = []
markitdown_binaries = []
markitdown_hiddenimports = []
try:
    markitdown_datas, markitdown_binaries, markitdown_hiddenimports = collect_all('markitdown')
except Exception:
    pass

# pdfplumber / pdfminer（MarkItDown依存 - PDF変換）
pdfplumber_datas = []
pdfminer_datas = []
pdfminer_hiddenimports = []
try:
    pdfplumber_datas = collect_data_files('pdfplumber')
except Exception:
    pass
try:
    pdfminer_datas, _, pdfminer_hiddenimports = collect_all('pdfminer')
except Exception:
    pass

# magika（MarkItDown依存 - ファイルタイプ検出）
magika_datas = []
magika_hiddenimports = []
try:
    magika_datas, _, magika_hiddenimports = collect_all('magika')
except Exception:
    pass

# Office 変換用 MarkItDown 依存 (v4.5)
# mammoth: docx, openpyxl/xlrd: xlsx/xls, pptx: pptx
# charset_normalizer は mypyc コンパイル版でハッシュ名モジュールを含むため collect_all 必須
office_datas = []
office_binaries = []
office_hiddenimports = []
for pkg in ('mammoth', 'openpyxl', 'xlrd', 'pptx', 'puremagic', 'markdownify',
            'bs4', 'beautifulsoup4', 'defusedxml', 'lxml', 'cobble',
            'charset_normalizer', 'pathvalidate', 'isodate',
            'tabulate', 'docstring_parser', 'soupsieve'):
    try:
        datas_, binaries_, hidden_ = collect_all(pkg)
        office_datas.extend(datas_)
        office_binaries.extend(binaries_)
        office_hiddenimports.extend(hidden_)
    except Exception:
        pass

a = Analysis(
    ['pdf2md.py'],
    pathex=[],
    binaries=fitz_binaries + pymupdf_binaries + markitdown_binaries + office_binaries,
    datas=fitz_datas + pymupdf_datas + pymupdf4llm_datas + easyocr_datas + anthropic_datas + markitdown_datas + pdfplumber_datas + pdfminer_datas + magika_datas + office_datas,
    hiddenimports=[
        'fitz',
        'fitz.table',
        'fitz.utils',
        'pymupdf',
        'pymupdf.extra',
        'pymupdf.mupdf',
        'pymupdf.pymupdf',
        'pymupdf.table',
        'pymupdf.utils',
        'pymupdf._apply_pages',
        'pymupdf._build',
        'pymupdf._extra',
        'pymupdf._mupdf',
        'pymupdf._wxcolors',
        'pymupdf4llm',
        'PIL',
        'PIL.Image',
        'easyocr',
        'torch',
        'torchvision',
        'anthropic',
        'markitdown',
        'markitdown._markitdown',
        'pdfplumber',
        'pdfminer',
        'pdfminer.high_level',
        'pdfminer.layout',
        'pdfminer.pdfpage',
        'magika',
        'markdownify',
        'bs4',
        'defusedxml',
        'charset_normalizer',
        # Office 変換 (v4.5)
        'mammoth',
        'openpyxl',
        'xlrd',
        'pptx',
        'puremagic',
        'lxml',
        'lxml.etree',
        'cobble',
        'tabulate',
    ] + fitz_hiddenimports + pymupdf_hiddenimports + pymupdf4llm_hiddenimports + markitdown_hiddenimports + magika_hiddenimports + pdfminer_hiddenimports + office_hiddenimports,
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
