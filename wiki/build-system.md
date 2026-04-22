---
title: Build System
type: infrastructure
tags: [build, pyinstaller, exe]
---

# Build System

PyInstaller を使って Windows 用 `PDF2MD.exe` を生成する。

## 基本コマンド

```powershell
build.bat
# または
powershell -ExecutionPolicy Bypass -File build.ps1
```

標準出力先は `dist/PDF2MD.exe`。

## 代替ビルド

`build\PDF2MD` がロックされる環境では、作業ディレクトリと出力先を分けて実行する。

```powershell
pyinstaller PDF2MD.spec --distpath dist_build --workpath build_build
```

## spec ファイル

`PDF2MD.spec` は次をまとめて収集する。

- PyMuPDF / pymupdf
- pymupdf4llm
- MarkItDown
- pdfplumber / pdfminer
- magika
- EasyOCR / torch 系 hidden import
- anthropic

## 配布物

配布フォルダに含める最小構成:

- `dist/PDF2MD.exe`
- `README.md`
- `.env.sample`
- `install_context_menu.bat`
- `install_context_menu.ps1`
- `uninstall_context_menu.bat`
- `uninstall_context_menu.ps1`
- `INSTALL.txt`

## 関連

- [[dependencies]]
- [[optional-dependencies]]
