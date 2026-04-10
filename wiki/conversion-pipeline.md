---
title: Conversion Pipeline
type: concept
tags: [architecture, pipeline, core]
---

# Conversion Pipeline

[[AdvancedPDFConverter]] の `convert_file()` (line ~1415) がエントリーポイント。`layout_mode` パラメータに応じて4つのパスに分岐する。

## 全体フロー

```
convert_file(pdf_path, layout_mode, ...)
  │
  ├─ layout_mode == "page_image"
  │    → _convert_as_page_images()
  │    各ページをPNG画像としてレンダリング、<img>タグで出力
  │    → 終了
  │
  ├─ layout_mode == "markitdown" (v4.2)
  │    → _convert_with_markitdown()
  │    MarkItDown(Microsoft製)でテキスト/表変換
  │    + PyMuPDFで画像補完抽出 + Claude API解析
  │    → 終了
  │
  ├─ layout_mode == "legacy"
  │    → _convert_with_pymupdf4llm() のみ
  │    pymupdf4llm利用不可ならフォールスルー
  │
  ├─ _check_font_encoding_issue() ← 最初の3ページをサンプリング
  │    │
  │    ├─ [問題あり + OCR利用可] → _convert_with_page_ocr()
  │    │    300 DPIでページ画像化
  │    │    ├─ ndlocr_cli利用可 → レイアウト抽出 + 領域切り出し
  │    │    │   → 図版/表組/組織図をClaude APIで分類・構造化変換（v4.3）
  │    │    └─ それ以外 → EasyOCR/pytesseract → レイアウト再構築
  │    │
  │    └─ [問題なし、auto/legacy]
  │         → _convert_with_pymupdf4llm()
  │         成功時: + _supplement_vector_drawings() で補完
  │         失敗時: フォールスルー
  │
  └─ [precise、またはフォールバック] → 標準パイプライン
       │
       ├─ DocumentAnalyzer.analyze_document_structure()
       │    フォント分析 → 見出し階層推定
       │
       ├─ ページごとのループ:
       │    ├─ AdvancedTableExtractor.extract_tables()
       │    ├─ _extract_text_blocks() （表領域を除外）
       │    ├─ LayoutAnalyzer.analyze_page_layout() （precise/autoのみ）
       │    └─ _extract_image_blocks() （ラスター + ベクター描画）
       │
       ├─ _remove_text_in_drawing_images()
       │    フルページ描画画像と重複するテキストを除去
       │
       ├─ _associate_captions()
       │    図表キャプションと最近傍の図/表を紐付け
       │
       ├─ _sort_blocks_with_columns()
       │    ページ順 → カラムインデックス順 → Y順 → X順
       │
       └─ Markdown生成
            ├─ use_obsidian_layout → _generate_markdown_obsidian()
            └─ else → _generate_markdown()

  └─ [preserve_image_layout=True かつ ndlocr_cli 利用可能]
       → _extract_layout_regions_via_ndlocr()
       → _postprocess_preserve_image_layout()
       各モードのMD生成後、書き出し前に後処理で図版/表を元Y位置に挿入。
       座標ありモード(precise/標準パイプライン/page_ocr)では Y座標順マージ挿入、
       座標なしモード(pymupdf4llm/legacy/markitdown)では図版のみページ末尾集約、
       表は既存挙動のまま(重複回避)。
```

## 出力構造

変換結果はPDFと同じディレクトリ（または指定出力先）に生成される:

```
{base_name}.md          — Markdownファイル
{base_name}_images/     — 抽出画像ディレクトリ
  page{N}_img{M}.png    — 個別画像
```

## 関連ページ

- [[layout-modes]] — 4つのモードの詳細
- [[font-encoding-detection]] — フォント問題検出
- [[image-extraction]] — 画像抽出の2段階プロセス
- [[obsidian-layout]] — Obsidian互換出力
- [[vector-drawing-supplement]] — ベクター描画補完
