---
title: "Source: README.md"
type: source
tags: [source, documentation]
ingested: 2026-04-08
---

# Source: README.md

## 基本情報

- **ファイル**: `README.md`
- **言語**: 日本語
- **バージョン**: v3.3（※コード自体はv4.0に進んでいるがREADMEは未更新）

## 主要コンテンツ

### 機能説明
テキスト抽出、画像抽出（ラスター+ベクター）、PyMuPDF4LLM統合、OCR対応、高度な構造認識（表、見出し、リスト、キャプション）、マルチカラム対応、GUI/CLI/バッチ処理。

### インストール方法
EXEダウンロード（推奨）またはPythonから実行。GitHub Releasesからの配布。

### 右クリックメニュー
`install_context_menu.bat` で登録。Windows 11では「その他のオプションを確認」から表示。

### 技術仕様
Python 3.9+, PyMuPDF, PyMuPDF4LLM, EasyOCR, Pillow, tkinter, PyInstaller。

### 更新履歴
v1.0.0 (2025-01-07) から v3.3.0 (2026-02-16) まで記載。

## 注意: READMEとコードのバージョン差異

READMEは v3.3 までの記載だが、`pdf2md.py` は v4.0。v4.0の新機能（レイアウトモード切替、段組み検出、Obsidian互換出力、ページ画像モード）はREADMEに未記載。

## 関連ページ

- [[overview]] — プロジェクト全体像
- [[layout-modes]] — v4.0で追加されたがREADME未反映の機能
