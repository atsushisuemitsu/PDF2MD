---
title: AdvancedPDFConverter
type: class
location: "pdf2md.py:1023"
tags: [class, converter, core]
---

# AdvancedPDFConverter

PDF→Markdown変換の中核クラス。全変換パス、画像抽出、OCR処理、Markdown生成を統合。

## 位置

`pdf2md.py` line ~1023

## 初期化

```python
AdvancedPDFConverter(enable_ocr=True, ocr_languages=['ja', 'en'])
```

内部で各検出器を初期化:
- `self.doc_analyzer` = [[DocumentAnalyzer]]
- `self.table_extractor` = [[AdvancedTableExtractor]]
- `self.list_detector` = [[ListDetector]]
- `self.caption_detector` = [[CaptionDetector]]
- `self.layout_analyzer` = [[LayoutAnalyzer]]
- `self.claude_analyzer` = [[ClaudeDiagramAnalyzer]] （利用可能時のみ）
- `self.ocr_reader` = EasyOCR Reader（利用可能時のみ、GPU=False）

## 主要メソッド

### 変換エントリーポイント

| メソッド | 用途 |
|---------|------|
| `convert_file()` | メインエントリー、[[layout-modes]] に応じて分岐 |
| `_convert_with_pymupdf4llm()` | pymupdf4llm変換 |
| `_convert_with_page_ocr()` | OCRベース変換 |
| `_convert_as_page_images()` | ページ画像モード |

### 画像抽出

| メソッド | 用途 |
|---------|------|
| `_extract_image_blocks()` | ラスター + ベクターの統合抽出 |
| `_extract_raster_images()` | `page.get_images()` によるビットマップ抽出 |
| `_extract_drawing_blocks()` | ベクター描画のクラスタリング・レンダリング |
| `_supplement_vector_drawings()` | pymupdf4llm変換後の補完 |

### Markdown生成

| メソッド | 用途 |
|---------|------|
| `_generate_markdown()` | 標準Markdown出力 |
| `_generate_markdown_obsidian()` | [[obsidian-layout|Obsidian互換]] HTML/CSS出力 |
| `_format_multicolumn_page()` | 段組みページのHTMLレンダリング |
| `_format_single_column_page()` | 単カラムページ（画像フロート検出あり） |

### ユーティリティ

| メソッド | 用途 |
|---------|------|
| `_check_font_encoding_issue()` | [[font-encoding-detection|フォント問題検出]] |
| `_remove_text_in_drawing_images()` | 描画画像領域のテキスト重複排除 |
| `_associate_captions()` | キャプション紐付け |
| `_sort_blocks_with_columns()` | カラム対応ソート |
| `_render_page_image()` | ページ画像レンダリング |

## 関連ページ

- [[conversion-pipeline]] — `convert_file()` のフロー詳細
- [[image-extraction]] — 画像抽出の詳細
- [[PDF2MDGUI]] — GUIから `self.converter` として呼び出し
