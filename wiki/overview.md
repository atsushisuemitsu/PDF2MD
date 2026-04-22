---
title: PDF2MD Overview
type: overview
version: "4.5"
tags: [project, overview]
---

# PDF2MD Overview

PDF2MDは、PDFおよび Microsoft Office ファイル (doc/docx/xls/xlsx/xlsm/pptx) をMarkdown形式に変換するWindowsデスクトップアプリケーション。

## 目的

日本語PDFドキュメントおよび Office ファイルを、文書構造（見出し階層、表、リスト、図表キャプション、段組み）を保持したままMarkdownに変換する。

## 対象ファイル形式

- **PDF**: `.pdf`
- **Microsoft Office**: `.doc`, `.docx`, `.xls`, `.xlsx`, `.xlsm`, `.pptx`
  - 新形式 (docx/xlsx/xlsm/pptx) は埋め込み画像を `{name}_images/` に抽出
  - 旧バイナリ形式 (.doc/.xls) はテキストのみ

## 対象ユーザー

- 日本語PDFやOfficeドキュメントを扱うユーザー（英語も対応）
- ObsidianなどのMarkdownエディタで文書を管理したい人
- バッチ処理で大量のドキュメントを変換する必要がある人

## 技術スタック

- **言語**: Python 3.13（Windows環境）
- **PDF解析**: [[dependencies#PyMuPDF|PyMuPDF (fitz)]]
- **レイアウト変換**: [[dependencies#pymupdf4llm|pymupdf4llm]]
- **OCR**: [[ocr-system|ndlocr_cli / EasyOCR / pytesseract]]
- **AI解析**: [[ClaudeDiagramAnalyzer|Claude API Vision]]
- **GUI**: tkinter + tkinterdnd2
- **ビルド**: [[build-system|PyInstaller]]

## アーキテクチャ概要

単一ファイル構成（`pdf2md.py` 約3300行）。全クラスが1モジュールに収まっている。

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
| v4.1 | Claude API図表・フローチャートAI解析機能 |
| v4.2 | MarkItDown統合、WaveDromタイミングチャート対応、.env対応 |
| v4.3 | ndlocr_cli（国立国会図書館OCR）統合、レイアウト抽出による領域検出・切り出し・Claude API構造化変換 |
| v4.5 | Microsoft Office ファイル対応 (doc/docx/xls/xlsx/xlsm/pptx)、MarkItDown統合、ZIP構造からの埋め込み画像抽出 |

## 関連ページ

- [[conversion-pipeline]] — 変換の全体フロー
- [[AdvancedPDFConverter]] — メインコンバータクラス
- [[PDF2MDGUI]] — GUIアプリケーション
- [[cli-interface]] — コマンドライン利用
