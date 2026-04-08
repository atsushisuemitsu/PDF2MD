---
title: Image Extraction
type: concept
tags: [images, raster, vector, extraction]
---

# Image Extraction

PDF内の画像を2段階で抽出するプロセス。[[AdvancedPDFConverter]] の `_extract_image_blocks()` が統合エントリーポイント。

## Stage 1: ラスター画像

**メソッド:** `_extract_raster_images()`

- `page.get_images()` で埋め込みビットマップ画像を列挙
- 50x50px未満のアイコンを除外
- PNG/JPEG形式で `{base_name}_images/page{N}_img{M}.png` に保存
- 保存後、OCR/Claude解析を適用（有効時）

## Stage 2: ベクター描画

**メソッド:** `_extract_drawing_blocks()`

1. `page.get_drawings()` でページ上のベクター描画要素を取得
2. `_cluster_drawing_rects()` で空間的に近い要素をクラスタリング
3. フィルタリング:
   - 単純な線（装飾線）→ 除外
   - コードブロック背景 → 除外
   - 表領域と重なる描画 → 除外
4. `page.get_pixmap(clip=rect)` でPNG画像としてレンダリング
5. **フルページクラスタ**（章タイトルページ等）→ ページ全体をレンダリング

## ベクター描画補完

**メソッド:** `_supplement_vector_drawings()`

pymupdf4llm変換成功後に実行される追加処理:

1. pymupdf4llmが抽出した画像と、ベクター描画から抽出した画像を比較
2. 重複画像を検出（座標のオーバーラップで判定）
3. 重複があれば高品質なベクター描画版に置換
4. 章タイトルページなどpymupdf4llmが見逃した画像を追加
5. 表紙画像をMarkdown先頭に配置

## 出力

`ImageBlock` データクラス:
- 画像ファイルパス
- 座標 (`x0, y0, x1, y1`)
- OCRテキスト（あれば）
- キャプション（紐付け後）
- Claude解析結果（あれば）

## 関連ページ

- [[conversion-pipeline]] — 画像抽出はページループ内で実行
- [[vector-drawing-supplement]] — pymupdf4llm変換後の補完詳細
- [[ClaudeDiagramAnalyzer]] — 画像のAI解析
- [[ocr-system]] — 画像内テキストのOCR
