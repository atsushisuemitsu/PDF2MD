---
title: PDF2MDGUI
type: class
location: "pdf2md.py"
tags: [class, gui, tkinter]
---

# PDF2MDGUI

`tkinter` ベースの GUI アプリ。PDF の追加、出力先指定、変換オプション選択、進捗表示を担当する。

## 主な機能

- ファイル追加 / フォルダ追加 / クリア
- `ファイルを選択して変換` のワンボタン導線
- `Treeview` による対象ファイル一覧表示
- `threading.Thread` を使ったバックグラウンド変換
- `auto / precise / page_image / legacy / markitdown` のモード切替
- `MarkItDown OCR` 用モデル入力欄
- PyMuPDF / OCR / Claude / MarkItDown の利用可否表示

## MarkItDown OCR

`layout=markitdown` を選ぶと、`OPENAI_API_KEY` がある環境では OpenAI OCR を使った MarkItDown OCR プラグインを有効化する。

- モデル名: GUI 入力欄の値を使用
- 既定値: `gpt-4o`
- 反映先: `MARKITDOWN_LLM_MODEL`

## 進捗更新

重い処理はワーカースレッドで実行し、GUI 更新だけを `self.root.after()` でメインスレッドへ戻している。

## 関連

- [[AdvancedPDFConverter]]
- [[optional-dependencies]]
- [[layout-modes]]
