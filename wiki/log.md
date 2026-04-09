# PDF2MD Wiki Log

## [2026-04-10] update | ndlocr_cli関連メソッドをAdvancedPDFConverter/conversion-pipelineに追記

`_init_ndlocr` / `_ocr_page_with_ndlocr` / `_crop_region_image` / `_classify_and_convert_region` / `_perform_ocr`（ndlocr優先）を AdvancedPDFConverter.md のメソッド表に追加。conversion-pipeline.md の `_convert_with_page_ocr` 説明に ndlocr_cli レイアウト抽出+領域切り出しフローを追記。overview.md のバージョン履歴に v4.3 を追加。

更新ページ: AdvancedPDFConverter, conversion-pipeline, overview

## [2026-04-09] update | ndlocr_cli統合をWikiに反映

ndlocr_cli（国立国会図書館OCR）統合に伴うWiki更新。OCRエンジン優先順位変更、レイアウト抽出による領域検出・切り出し・Claude API構造化変換フローを追加。

更新ページ: ocr-system（ndlocr_cliフロー全面書き換え）, optional-dependencies（NDLOCR_AVAILABLEフラグ追加）

## [2026-04-09] update | v4.1/v4.2 変更をWikiに反映

v4.1（Claude API図表解析）およびv4.2（MarkItDown統合、WaveDromタイミングチャート対応、.env対応）の変更を既存Wikiページに反映。

更新ページ: ClaudeDiagramAnalyzer（WaveDrom/timing_chart追加）, layout-modes（markitdownモード追加）, optional-dependencies（MARKITDOWN_AVAILABLE, dotenv追加）, conversion-pipeline（markitdownパス追加）, dependencies（markitdown, python-dotenv追加）, cli-interface（markitdownオプション追加）, overview（v4.1/v4.2履歴追加）, AdvancedPDFConverter（_convert_with_markitdown追加）, PDF2MDGUI（MarkItDownラジオボタン・ステータス追加）

## [2026-04-08] ingest | pdf2md.py v4.0 初期Wiki構築

ソースコード `pdf2md.py`（3123行、v4.0）、`README.md`、`requirements.txt`、`PDF2MD.spec`、`build.bat` を読み込み、Wiki全体を初期構築。

作成ページ: overview, conversion-pipeline, layout-modes, optional-dependencies, DocumentAnalyzer, AdvancedTableExtractor, LayoutAnalyzer, ListDetector, CaptionDetector, ClaudeDiagramAnalyzer, AdvancedPDFConverter, PDF2MDGUI, image-extraction, font-encoding-detection, ocr-system, obsidian-layout, vector-drawing-supplement, build-system, cli-interface, dependencies, source-pdf2md-py, source-readme

合計: 22ページ + index + log
