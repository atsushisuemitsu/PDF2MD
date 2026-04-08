---
title: DocumentAnalyzer
type: class
location: "pdf2md.py:187"
tags: [class, analysis, headings]
---

# DocumentAnalyzer

文書全体のフォントサイズを分析し、見出し階層（H1-H6）を自動推定するクラス。

## 位置

`pdf2md.py` line ~187

## 主要メソッド

### `analyze_document_structure(doc)`

PyMuPDF ドキュメント全ページを走査し、各spanのフォントサイズと太字属性を収集。

**処理フロー:**
1. 全ページの `get_text("dict")` からフォントサイズ頻度をカウント
2. `"bold"` / `"heavy"` を含むフォント名のサイズを記録
3. `_estimate_heading_sizes()` で見出し階層を決定

### `_estimate_heading_sizes(sorted_sizes, font_counts, bold_sizes)`

- 最頻フォントサイズを**本文サイズ**として特定
- 本文より大きいサイズを見出し候補とし、大きい順にH1-H6を割り当て（最大6レベル）
- 本文サイズでボールドのものはH3-H4として追加

### `get_heading_level(font_size, is_bold, text)`

テキストの見出しレベルを判定:
- 150文字超 → 見出しではない
- フォントサイズが見出しマップに一致 → そのレベル
- ボールドかつ80文字未満 → 最近傍の見出しサイズを参照、なければH4

## 設計上のポイント

- 文書全体を先に分析してから各ページの処理に入る（2パス方式）
- フォントサイズの5%許容範囲でマッチング
- 短いテキスト（200文字未満）のみを見出し候補として収集

## 関連ページ

- [[conversion-pipeline]] — 標準パイプラインで最初に呼ばれる
- [[AdvancedPDFConverter]] — `self.doc_analyzer` として保持
