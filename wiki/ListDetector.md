---
title: ListDetector
type: class
location: "pdf2md.py:523"
tags: [class, list, detection]
---

# ListDetector

テキストから箇条書き・番号付きリストを正規表現で検出し、階層構造を認識するクラス。

## 位置

`pdf2md.py` line ~523

## 検出パターン

### 箇条書きマーカー
`- * ● ○ ■ □ ・ ※ ★ ☆ → ⇒ ▶ ►`

### 番号付きパターン
- `1.` `2.` — ドット付き数字
- `(1)` `(2)` — 括弧付き数字
- `a)` `b)` — アルファベット
- `(a)` `(b)` — 括弧付きアルファベット
- `i.` `ii.` — ローマ数字

## 出力

`ListItem` データクラス:
- `text`: リスト項目テキスト
- `level`: インデントレベル
- `list_type`: "bullet" / "numbered"

## 関連ページ

- [[conversion-pipeline]] — テキストブロック処理時にリスト判定
- [[AdvancedPDFConverter]] — `self.list_detector` として保持
