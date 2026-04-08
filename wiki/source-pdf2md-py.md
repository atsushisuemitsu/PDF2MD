---
title: "Source: pdf2md.py v4.0"
type: source
tags: [source, code]
ingested: 2026-04-08
---

# Source: pdf2md.py v4.0

## 基本情報

- **ファイル**: `pdf2md.py`
- **行数**: 3123行
- **バージョン**: v4.0 (Layout-Aware Version)
- **言語**: Python 3.13
- **構成**: 単一ファイル、全クラス同一モジュール

## クラス一覧（出現順）

| クラス | 行 | 種別 | Wiki |
|--------|-----|------|------|
| `TextBlock` | ~100 | データクラス | — |
| `ImageBlock` | ~118 | データクラス | — |
| `TableBlock` | ~140 | データクラス | — |
| `ListItem` | ~155 | データクラス | — |
| `LayoutRegion` | ~164 | データクラス | [[LayoutAnalyzer]] |
| `PageLayout` | ~176 | データクラス | [[LayoutAnalyzer]] |
| `DocumentAnalyzer` | ~187 | 解析 | [[DocumentAnalyzer]] |
| `AdvancedTableExtractor` | ~279 | 解析 | [[AdvancedTableExtractor]] |
| `ListDetector` | ~523 | 解析 | [[ListDetector]] |
| `CaptionDetector` | ~596 | 解析 | [[CaptionDetector]] |
| `LayoutAnalyzer` | ~636 | 解析 | [[LayoutAnalyzer]] |
| `ClaudeDiagramAnalyzer` | ~773 | AI解析 | [[ClaudeDiagramAnalyzer]] |
| `AdvancedPDFConverter` | ~1023 | コア | [[AdvancedPDFConverter]] |
| `PDF2MDGUI` | ~2692 | GUI | [[PDF2MDGUI]] |

## v4.0 の主要追加機能

- [[layout-modes|レイアウトモード切替]] (auto/precise/page_image/legacy)
- [[LayoutAnalyzer|段組み検出]] (2/3段、X座標分布分析)
- [[obsidian-layout|Obsidian互換HTML/CSS出力]]
- ページ画像モード (`_convert_as_page_images`)
- 画像フロート配置検出
- ヘッダー/フッター分離
- フォントエンコーディング検出閾値の改善 (30%→25%)

## フィーチャーフラグ

line ~42-96 で定義。詳細は [[optional-dependencies]] を参照。

## エントリーポイント

`main()` 関数 (line ~2986):
- 引数なし → GUI起動
- `--context-menu` → 右クリックメニューモード
- ファイル/フォルダ指定 → CLI変換
