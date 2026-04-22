---
title: CLI Interface
type: infrastructure
tags: [cli, argparse, context-menu]
---

# CLI Interface

argparse ベースのCLIと、Windows右クリックメニュー統合。

## CLI オプション

```
pdf2md.py [inputs...] [options]

引数:
  inputs              PDF/Office ファイルまたはフォルダのパス（省略時はGUI起動）
                      対応形式: .pdf, .doc, .docx, .xls, .xlsx, .xlsm, .pptx

オプション:
  --layout MODE       レイアウトモード: auto|precise|page_image|legacy|markitdown (default: auto)
                      (Office ファイルに対しては無視され、常に markitdown にルーティング)
  --dpi N             画像レンダリングDPI (default: 150)
  -o, --output DIR    出力先フォルダ
  --ocr               OCRを有効にする (default: True)
  --no-ocr            OCRを無効にする
  --no-images         画像抽出を無効にする
  --no-claude         Claude API図表解析を無効にする
  --context-menu      右クリックメニューからの呼び出し用
  --silent            変換完了後のダイアログを抑制
```

## 動作モード

- **引数なし** → GUI起動 (`PDF2MDGUI().run()`)
- **ファイル/フォルダ指定** → CLI変換
- **`--context-menu`** → 右クリックメニューモード（ダイアログ表示付き）

## Office ファイル変換 (v4.5)

```bash
# Word 文書
python pdf2md.py document.docx
python pdf2md.py legacy.doc

# Excel
python pdf2md.py report.xlsx
python pdf2md.py macros.xlsm

# PowerPoint
python pdf2md.py slides.pptx

# フォルダ内の全サポートファイル (PDF + Office) を一括変換
python pdf2md.py ./mixed_folder/
```

- Office ファイル入力では `--layout` 指定は無視され、常に MarkItDown にルーティングされる
- `precise` など PDF 専用モードを指定した場合は stderr に警告が出る
- `--no-images` で埋め込み画像抽出を無効化可能
- docx/xlsx/xlsm/pptx は ZIP 構造から埋め込み画像を `{name}_images/` に抽出
- 旧バイナリ .doc/.xls は画像抽出非対応 (警告ログ)

## 右クリックメニュー統合

### インストール

`install_context_menu.bat` を管理者権限で実行。Windowsレジストリに PDF ファイルの右クリックメニュー「PDF → MD 変換」を登録。

### アンインストール

`uninstall_context_menu.bat` を管理者権限で実行。

### 動作

1. エクスプローラーでPDFを右クリック → 「PDF → MD 変換」
2. `pdf2md.py --context-menu <pdf_path>` が実行される
3. 変換完了/エラー時にダイアログを表示

## 関連ページ

- [[layout-modes]] — `--layout` オプションの詳細
- [[PDF2MDGUI]] — GUI モード
- [[AdvancedPDFConverter]] — CLI から直接呼び出し
