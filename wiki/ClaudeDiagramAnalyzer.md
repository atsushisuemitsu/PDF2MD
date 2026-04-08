---
title: ClaudeDiagramAnalyzer
type: class
location: "pdf2md.py:773"
tags: [class, claude-api, ai, diagram]
---

# ClaudeDiagramAnalyzer

Claude API Vision を使用して図表・フローチャートを構造化テキスト（Markdownテーブル / PlantUML）に変換するクラス。

## 位置

`pdf2md.py` line ~773

## 前提条件

- `anthropic` パッケージがインストール済み
- `ANTHROPIC_API_KEY` 環境変数が設定済み
- [[optional-dependencies]] の `CLAUDE_API_AVAILABLE` が `True`

## 初期化

```python
self.client = anthropic.Anthropic()
self.model = "claude-sonnet-4-20250514"
self.disabled = False
```

初期化時にテスト呼び出しでAPIキーを検証。`AuthenticationError` なら `self.disabled = True` で自動無効化し、以降の呼び出しをスキップ。

## 主要メソッド

### `analyze_page(page_image_bytes, page_height, page_width) -> list`

ページ画像全体から図表・フローチャートを検出し、構造化データのリストを返す。OCR変換パス（`_convert_with_page_ocr`）から呼ばれる。

### `analyze_single_image(image_bytes) -> Optional[dict]`

個別に抽出された画像を解析。標準パイプラインの画像抽出時に呼ばれる。

## エラーハンドリング

- `AuthenticationError` → 自動無効化（以降全スキップ）
- その他のエラー → ログ出力のみ、処理続行

## 関連ページ

- [[optional-dependencies]] — `CLAUDE_API_AVAILABLE` フラグ
- [[conversion-pipeline]] — OCRパスおよび標準パイプラインで利用
- [[image-extraction]] — 画像ブロック抽出時にClaude解析を呼び出し
