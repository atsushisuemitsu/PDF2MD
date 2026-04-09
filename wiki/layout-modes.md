---
title: Layout Modes
type: concept
tags: [layout, modes, configuration]
---

# Layout Modes

v4.0で導入されたレイアウトモード切替。`convert_file()` の `layout_mode` パラメータ、GUIのラジオボタン、CLI の `--layout` オプションで選択。v4.2で markitdown モードを追加。

## モード一覧

### auto（デフォルト）

1. [[font-encoding-detection|フォント問題検出]] → 問題あればOCR変換
2. pymupdf4llm で変換を試行
3. 成功すれば [[vector-drawing-supplement|ベクター描画補完]] を実行
4. 失敗すれば精密レイアウト（precise相当）にフォールバック

最もインテリジェントなパス。ほとんどのPDFに対応。

### precise

標準パイプラインを強制使用。[[LayoutAnalyzer]] による段組み検出、[[obsidian-layout|Obsidian互換HTML/CSS]] 出力を行う。

pymupdf4llm をスキップし、全ての解析を自前で実行するため:
- 段組み(2/3段)をCSSで再現
- 画像とテキストのフロート配置
- ヘッダー/フッター分離

### page_image

各ページをPNG画像としてレンダリングし、`<img>` タグで出力。テキスト抽出は行わない。

DPIは `--dpi` オプション（デフォルト150）で制御。レイアウトが複雑すぎるPDFに有効。

### legacy

pymupdf4llm のみ使用。ベクター描画補完なし。v3.2以前の動作と同等。pymupdf4llm が利用不可の場合は標準パイプラインにフォールスルー。

### markitdown（v4.2）

Microsoft製 [MarkItDown](https://github.com/microsoft/markitdown) を使用した変換。`_convert_with_markitdown()` で処理。

- テキスト/表の抽出はMarkItDownが担当
- 画像抽出が有効な場合、PyMuPDFでラスター画像+ベクター描画を補完抽出
- Claude API有効時、抽出画像を解析し `## Diagrams` / `## Images` セクションとして末尾に追加
- `MARKITDOWN_AVAILABLE` が `False` の場合、GUIのラジオボタンは `disabled`

## 選択ガイド

| ユースケース | 推奨モード |
|-------------|-----------|
| 一般的なPDF | auto |
| 段組みPDFを正確に再現 | precise |
| レイアウトが崩れるPDF | page_image |
| pymupdf4llm変換のみで十分 | legacy |
| フォント化けするPDF | auto（自動OCR） |
| Microsoft MarkItDownを試す | markitdown |

## 関連ページ

- [[conversion-pipeline]] — 各モードがどこで分岐するか
- [[LayoutAnalyzer]] — precise/autoで使われる段組み検出
- [[obsidian-layout]] — precise/autoのHTML/CSS出力
- [[cli-interface]] — `--layout` オプション
- [[optional-dependencies]] — `MARKITDOWN_AVAILABLE` フラグ
