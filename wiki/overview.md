---
title: PDF2MD Overview
type: overview
version: "4.0"
tags: [project, overview]
---

# PDF2MD Overview

PDF2MDは、PDFファイルをMarkdown形式に変換するWindowsデスクトップアプリケーション。

## 目的

日本語PDFドキュメントを、文書構造（見出し階層、表、リスト、図表キャプション、段組み）を保持したままMarkdownに変換する。

## 対象ユーザー

- 日本語PDFを扱うユーザー（英語も対応）
- ObsidianなどのMarkdownエディタで文書を管理したい人
- バッチ処理で大量のPDFを変換する必要がある人

## 技術スタック

- **言語**: Python 3.13（Windows環境）
- **PDF解析**: [[dependencies#PyMuPDF|PyMuPDF (fitz)]]
- **レイアウト変換**: [[dependencies#pymupdf4llm|pymupdf4llm]]
- **OCR**: [[ocr-system|EasyOCR / pytesseract]]
- **AI解析**: [[ClaudeDiagramAnalyzer|Claude API Vision]]
- **GUI**: tkinter + tkinterdnd2
- **ビルド**: [[build-system|PyInstaller]]

## アーキテクチャ概要

単一ファイル構成（`pdf2md.py` 約3100行）。全クラスが1モジュールに収まっている。

主要な変換フローは [[conversion-pipeline]] を参照。4つの [[layout-modes]] を切り替え可能。

## バージョン履歴

| バージョン | 主要な追加機能 |
|-----------|---------------|
| v1.0 | 初回リリース、GUI/CLI、バッチ変換 |
| v2.0 | PyMuPDF変換エンジン、画像抽出、OCR |
| v3.0 | 高度な文書構造分析、罫線なし表、リスト、キャプション |
| v3.1 | ベクター描画検出・抽出 |
| v3.2 | PyMuPDF4LLM統合、マルチカラム対応 |
| v3.3 | 右クリックメニュー対応 |
| v4.0 | レイアウトモード切替、段組み(2/3段)、Obsidian互換HTML/CSS、ページ画像モード |

## 関連ページ

- [[conversion-pipeline]] — 変換の全体フロー
- [[AdvancedPDFConverter]] — メインコンバータクラス
- [[PDF2MDGUI]] — GUIアプリケーション
- [[cli-interface]] — コマンドライン利用
