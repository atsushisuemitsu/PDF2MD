---
title: Dependencies
type: infrastructure
tags: [dependencies, packages]
---

# Dependencies

`requirements.txt` で管理。一部はオプショナル。

## 必須

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| PyMuPDF | >=1.23.0 | PDF解析、テキスト/表/画像抽出、ベクター描画レンダリング |
| Pillow | >=10.0.0 | 画像処理（リサイズ、フォーマット変換） |

## 推奨

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| pymupdf4llm | >=0.2.0 | レイアウト認識Markdown変換（マルチカラム対応） |
| pymupdf-layout | >=1.0.0 | ML レイアウト検出（pymupdf4llm内OCR有効化） |

## オプショナル

| パッケージ | バージョン | 用途 | 備考 |
|-----------|-----------|------|------|
| easyocr | >=1.7.0 | 日英OCR | requirements.txt ではコメントアウト。64-bit Python + PyTorch 必須。手動インストール: `pip install easyocr` |
| anthropic | >=0.40.0 | Claude API 図表解析 | `ANTHROPIC_API_KEY` 環境変数が必要 |
| markitdown | — | Microsoft製PDF/ドキュメント変換 | markitdownモード利用不可 |
| python-dotenv | — | .envファイルから環境変数読み込み | 手動設定が必要 |
| tkinterdnd2 | >=0.3.0 | GUI ドラッグ&ドロップ | なくても基本機能は動作 |
| pyinstaller | >=6.0.0 | EXEビルド | 開発時のみ |

## 標準ライブラリ依存

tkinter (GUI), argparse (CLI), threading, pathlib, dataclasses, collections, json, base64, io, re

## 関連ページ

- [[optional-dependencies]] — try/except importパターン
- [[build-system]] — PyInstaller EXEビルド
- [[ocr-system]] — EasyOCR詳細
- [[ClaudeDiagramAnalyzer]] — Claude API統合
