---
title: LayoutAnalyzer
type: class
location: "pdf2md.py:636"
tags: [class, layout, columns, v4]
---

# LayoutAnalyzer

v4.0で追加された段組み（マルチカラム）検出クラス。ページのレイアウト構造を分析し、2段/3段組みを検出する。

## 位置

`pdf2md.py` line ~636

## 主要メソッド

### `analyze_page_layout(page, text_blocks) -> PageLayout`

1. ヘッダー領域（上部8%）とフッター領域（下部8%）を定義
2. ボディ領域のテキストブロックのみで `_detect_columns()` を実行
3. 各ブロックに `column_index` を割り当て

### `_detect_columns(blocks, page_width) -> (num_cols, boundaries)`

**カラム数検出アルゴリズム:**
1. 各テキストブロックの `x0`（左端X座標）を収集
2. ユニークな `x0` 値をソートし、隣接値間の最大ギャップを探す
3. ギャップ > コンテンツ幅の15% → 2段組み候補
4. 左右それぞれに20%以上のブロックが存在するか確認
5. 右側ブロック内でさらにギャップ > 12% → 3段組み

### `is_header_or_footer(block, layout) -> str`

ブロックのY座標から "header" / "footer" / "body" を判定。

## データクラス

### `PageLayout`
- `page_width`, `page_height`: ページサイズ
- `num_columns`: 検出されたカラム数 (1/2/3)
- `column_boundaries`: カラム境界のX座標リスト
- `header_y`, `footer_y`: ヘッダー/フッター境界Y座標

### `LayoutRegion`
- 座標 (`x0, y0, x1, y1`)
- `region_type`: "header" / "footer" / "body" / "sidebar"
- `column_index`: 所属カラム番号
- `blocks`: 領域内のブロックリスト

## 関連ページ

- [[conversion-pipeline]] — precise/autoモードでページごとに呼ばれる
- [[obsidian-layout]] — 検出結果を使ったHTML/CSS出力
- [[layout-modes]] — preciseモードで有効
