---
title: Optional Dependencies Pattern
type: concept
tags: [architecture, imports, feature-flags]
---

# Optional Dependencies Pattern

PDF2MD は try/except import と可用性フラグで機能を段階的に有効化する。

## 主なフラグ

| フラグ | 依存 | 用途 |
| --- | --- | --- |
| `PYMUPDF_AVAILABLE` | PyMuPDF | PDF 読み込み |
| `PYMUPDF4LLM_AVAILABLE` | pymupdf4llm | レイアウト変換 |
| `LAYOUT_AVAILABLE` | pymupdf.layout | pymupdf4llm 補助 |
| `NDLOCR_AVAILABLE` | ndlocr_cli + cv2 | 国立国会図書館OCR（レイアウト抽出付き） |
| `OCR_ENGINE` | ndlocr / easyocr / pytesseract | OCR エンジン名 |
| `CLAUDE_API_AVAILABLE` | anthropic + key | Claude 図表解析 |
| `MARKITDOWN_AVAILABLE` | markitdown | MarkItDown 変換 (PDF + Office) |
| `OPENAI_CLIENT_AVAILABLE` | openai | OpenAI クライアント生成 |
| `MARKITDOWN_OCR_PLUGIN_AVAILABLE` | markitdown_ocr | MarkItDown OCR プラグイン登録 |
| `DND_AVAILABLE` | tkinterdnd2 | GUI ドラッグ&ドロップ |

## .env ファイル対応（v4.2）

`python-dotenv` がインストール済みの場合、スクリプトと同ディレクトリの `.env` ファイルから環境変数を自動読み込みする。`.env.sample` がテンプレートとして提供されている。

## Claude

`anthropic` の import 成功に加えて `ANTHROPIC_API_KEY` が必要。未設定時は GUI 側で無効表示になる。

## MarkItDown OCR

`layout=markitdown` 実行時、次の条件がそろうと OpenAI OCR を自動有効化する。

- `OPENAI_API_KEY` が設定済み
- `openai` パッケージが利用可能
- `markitdown_ocr` が import 可能

モデルは `MARKITDOWN_LLM_MODEL` を使い、未設定時は `gpt-4o`。

## Office ファイル変換 (v4.5)

`MARKITDOWN_AVAILABLE` が `True` で、入力ファイル拡張子が `.doc/.docx/.xls/.xlsx/.xlsm/.pptx` の場合、`_convert_office_file()` に自動ルーティングされる。`MARKITDOWN_AVAILABLE` が `False` の場合は Office ファイル入力時にエラーを返す (PDF 変換は既存経路で可)。

## 方針

- 依存がなくてもアプリは起動する
- GUI では使えない機能を明示する
- 利用可能な経路へ自動フォールバックする

## 関連

- [[dependencies]]
- [[ClaudeDiagramAnalyzer]]
- [[ocr-system]]
