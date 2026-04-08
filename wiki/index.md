# PDF2MD Wiki Index

PDF2MD プロジェクトのナレッジベース。ソースコード (`pdf2md.py` v4.0) から抽出・構造化した情報を管理する。

## Overview

- [[overview]] — プロジェクト全体像、目的、対象ユーザー

## Architecture

- [[conversion-pipeline]] — 変換パイプラインの全体フロー（4つのレイアウトモード分岐）
- [[layout-modes]] — auto / precise / page_image / legacy の各モード詳細
- [[optional-dependencies]] — try/except importパターンとフィーチャーフラグ一覧

## Core Classes

- [[DocumentAnalyzer]] — フォントサイズ分析による見出し階層推定（H1-H6）
- [[AdvancedTableExtractor]] — PyMuPDF find_tables + テキスト位置ベースの罫線なし表検出
- [[LayoutAnalyzer]] — 段組み検出（2/3段）、ヘッダー/フッター領域判定
- [[ListDetector]] — 箇条書き・番号付きリストの正規表現パターン認識
- [[CaptionDetector]] — 図表キャプションの自動検出・紐付け
- [[ClaudeDiagramAnalyzer]] — Claude API Visionによる図表・フローチャート構造化
- [[AdvancedPDFConverter]] — メインコンバータ（全変換パスの統合、画像抽出、OCR）
- [[PDF2MDGUI]] — tkinter GUI（Treeview、スレッド変換、D&D、レイアウトモード選択）

## Concepts

- [[image-extraction]] — ラスター画像とベクター描画の2段階抽出プロセス
- [[font-encoding-detection]] — フォントエンコーディング問題の自動検出とOCRフォールバック
- [[ocr-system]] — EasyOCR / pytesseract によるOCR処理
- [[obsidian-layout]] — Obsidian互換マルチカラムHTML/CSS出力
- [[vector-drawing-supplement]] — pymupdf4llm変換後のベクター描画補完・重複排除

## Infrastructure

- [[build-system]] — PyInstaller EXEビルド、specファイル構成
- [[cli-interface]] — argparse CLI、右クリックメニュー統合
- [[dependencies]] — 依存パッケージ一覧と役割

## Sources

- [[source-pdf2md-py]] — pdf2md.py v4.0 ソースコード分析サマリー
- [[source-readme]] — README.md の要約
