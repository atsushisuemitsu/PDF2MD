---
title: Vector Drawing Supplement
type: concept
tags: [vector, images, pymupdf4llm, supplement]
---

# Vector Drawing Supplement

pymupdf4llm変換成功後に実行されるベクター描画の自動補完・重複排除処理。

## メソッド

`_supplement_vector_drawings()` — [[AdvancedPDFConverter]] のメソッド

## 処理フロー

1. **ベクター描画の抽出**: `_extract_drawing_blocks()` で全ページのベクター描画を検出
2. **フルページ描画の検出**: ページ面積の大部分を占める描画クラスタ（章タイトルページ等）を特定
3. **重複判定**: pymupdf4llmが出力したMarkdown内の画像参照と、ベクター描画画像の座標をオーバーラップで比較
4. **置換**: 重複があればpymupdf4llmの画像参照をベクター描画のフルページ版に差し替え（高品質化）
5. **追加**: pymupdf4llmが見逃した画像（テキストのない章タイトルページ等）をMarkdownに挿入
6. **表紙配置**: 表紙画像が検出された場合、Markdown先頭に移動

## 設計意図

pymupdf4llmはテキスト変換に優れるが、ベクター描画の画像化は行わない。この補完処理により:
- テキスト変換はpymupdf4llmの高品質な結果を活用
- ベクター描画は自前の画像化処理で補う
- 重複画像を排除して出力を整理

## 関連ページ

- [[conversion-pipeline]] — autoモードでpymupdf4llm成功後に呼ばれる
- [[image-extraction]] — ベクター描画抽出の詳細
- [[layout-modes]] — autoモードで有効、legacyモードでは実行されない
