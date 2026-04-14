---
title: Layout Modes
type: concept
tags: [layout, modes, configuration]
---

# Layout Modes

`convert_file()` の `layout_mode`、GUI のラジオボタン、CLI の `--layout` で切り替える変換モード一覧。

## auto

標準モード。まずフォント問題を検出し、必要ならページ OCR にフォールバックする。問題がなければ `pymupdf4llm` を試し、失敗時は精密レイアウト経路に戻る。

## precise

自前の解析パイプラインを強制使用する。`LayoutAnalyzer` で段組みを検出し、`obsidian-layout` ベースの HTML/CSS でページ配置を再現する。

## page_image

各ページを PNG 化して `<img>` で出力する。内容保持を最優先し、テキスト再構成は行わない。

## legacy

`pymupdf4llm` のみを使う従来モード。ベクター描画補完や精密レイアウト再構築は行わない。

## markitdown

MarkItDown を起点にしつつ、PDF2MD 側の OCR / 表抽出 / 画像切り出しでページを再構築する。

- 文字: ページ全体 OCR から `TextBlock` を生成
- 表: `AdvancedTableExtractor` で Markdown 表に変換
- 図・フロー・画像: 既存の画像抽出で位置を維持
- 出力: `obsidian-layout` ベースの HTML/CSS

`OPENAI_API_KEY` が設定されている場合は MarkItDown OCR プラグインも自動で有効化する。モデルは `MARKITDOWN_LLM_MODEL`、未設定時は `gpt-4o`。

## 選び方

| 用途 | 推奨モード |
| --- | --- |
| 普通の PDF | `auto` |
| 段組みや回り込みを保ちたい | `precise` |
| 見た目そのままを優先 | `page_image` |
| 従来互換 | `legacy` |
| OCR + 表 + 画像配置をまとめて使う | `markitdown` |

## 関連

- [[conversion-pipeline]]
- [[LayoutAnalyzer]]
- [[obsidian-layout]]
- [[cli-interface]]
