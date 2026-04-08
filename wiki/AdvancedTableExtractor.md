---
title: AdvancedTableExtractor
type: class
location: "pdf2md.py:279"
tags: [class, table, extraction]
---

# AdvancedTableExtractor

2段階の表検出を行うクラス。罫線付き表とテキスト位置ベースの罫線なし表の両方を検出。

## 位置

`pdf2md.py` line ~279

## 検出アルゴリズム

### Stage 1: PyMuPDF `find_tables()`

PyMuPDFの組み込み表検出機能を使用。罫線で囲まれた表を高精度で検出。

### Stage 2: テキスト位置ベース検出

`find_tables()` で見つからなかった表を検出するフォールバック:
- テキストブロックのX/Y座標を分析
- 一定のグリッドパターンに配置されたテキストを表として認識
- データ一覧など罫線のない表に対応

## 出力

- `TableBlock` のリスト（セルデータ、位置、ヘッダー行検出）
- テーブル領域の座標リスト（後続の `_extract_text_blocks()` で除外するため）

## 設計上のポイント

- 表領域は座標で返され、テキスト抽出時に重複を排除するために使用される
- ヘッダー行はフォント太字や最初の行から自動判定

## 関連ページ

- [[conversion-pipeline]] — ページループの最初に呼ばれる
- [[AdvancedPDFConverter]] — `self.table_extractor` として保持
