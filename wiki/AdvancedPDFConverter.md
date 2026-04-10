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
- `self.ocr_reader` = EasyOCR Reader（フォールバック時のみ、GPU=False）
- `self.ndlocr_inferrer` = ndlocr_cli `OcrInferrer`（[[ocr-system|ndlocr_cli]] 利用可能時のみ、CPU、stages 2-3）

## 主要メソッド

### 変換エントリーポイント

| メソッド | 用途 |
|---------|------|
| `convert_file()` | メインエントリー、[[layout-modes]] に応じて分岐 |
| `_convert_with_pymupdf4llm()` | pymupdf4llm変換 |
| `_convert_with_page_ocr()` | OCRベース変換 |
| `_convert_as_page_images()` | ページ画像モード |
| `_convert_with_markitdown()` | MarkItDown変換（v4.2） |

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

### OCR（v4.3）

| メソッド | 用途 |
|---------|------|
| `_init_ndlocr()` | ndlocr_cli `OcrInferrer` を1回だけ初期化（モデル再利用） |
| `_ocr_page_with_ndlocr()` | ページ画像をndlocr_cliに渡し、テキストブロックと非テキスト領域（図版・表組等）を返す |
| `_crop_region_image()` | ndlocr_cli検出領域をページ画像から切り出し |
| `_classify_and_convert_region()` | 切り出した領域をClaude APIで分類し、PlantUML/WaveDrom/Markdown表に変換 |
| `_perform_ocr()` | 画像単位のOCR実行（ndlocr_cli → EasyOCR → pytesseract の順） |

### preserve-image-layout (v4.4)

| メソッド | 用途 |
|---------|------|
| `_extract_layout_regions_via_ndlocr()` | ndlocr_cli で図版/表組領域を検出し切り抜く。図版はPNG保存、表組はClaude APIでMarkdown表化 |
| `_postprocess_preserve_image_layout()` | 生成済みMDに図版/表組をY座標順で挿入する後処理 |
| `_insert_regions_by_y()` | 座標ありモードでの Y 順マージ挿入ヘルパ(text snippet ベース) |

`convert_file(preserve_image_layout=True)` で有効化。ndlocr_cli 必須。未導入時はオプション自体が無効化される。

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
