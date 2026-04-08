---
title: Build System
type: infrastructure
tags: [build, pyinstaller, exe]
---

# Build System

PyInstallerを使用したWindows EXEビルド。

## ビルド手順

```bash
# build.bat を実行（venv自動検出、依存インストール含む）
build.bat

# または PowerShell
powershell -ExecutionPolicy Bypass -File build.ps1
```

出力: `dist/PDF2MD.exe`

## spec ファイル (`PDF2MD.spec`)

PyInstallerの設定ファイル。以下のパッケージのデータ/バイナリを収集:

| パッケージ | 収集方法 |
|-----------|---------|
| fitz (PyMuPDF) | `collect_all('fitz')` — datas, binaries, hiddenimports 全て |
| pymupdf | `collect_all('pymupdf')` + 明示的 `.pyd`/`.dll` 収集 |
| pymupdf4llm | `collect_all('pymupdf4llm')` |
| easyocr | `collect_data_files('easyocr')` |
| anthropic | `collect_data_files('anthropic')` |

### 注意点

- `console=True` — 変換進捗をコンソールに表示するため（`False`にするとコンソール非表示）
- pymupdf の `.pyd` バイナリは `collect_all` で漏れる場合があるため、明示的にディレクトリ走査で収集
- hiddenimports に `torch`, `torchvision` を含む（EasyOCR依存）

## build.bat の動作

1. venv があれば自動で activate
2. `pip install -r requirements.txt -q` で依存インストール確認
3. `pyinstaller --clean PDF2MD.spec` でビルド

## 関連ページ

- [[dependencies]] — ビルドに含まれるパッケージ
- [[optional-dependencies]] — 各パッケージの任意性
