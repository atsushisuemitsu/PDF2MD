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
  inputs              PDFファイルまたはフォルダのパス（省略時はGUI起動）

オプション:
  --layout MODE       レイアウトモード: auto|precise|page_image|legacy|markitdown (default: auto)
  --dpi N             画像レンダリングDPI (default: 150)
  -o, --output DIR    出力先フォルダ
  --ocr               OCRを有効にする (default: True)
  --no-ocr            OCRを無効にする
  --no-images         画像抽出を無効にする
  --no-claude         Claude API図表解析を無効にする
  --preserve-image-layout  図版を切り抜いて元Y位置に保持、表をClaude APIでMD表化 (ndlocr_cli必須)
  --context-menu      右クリックメニューからの呼び出し用
  --silent            変換完了後のダイアログを抑制
```

## 使用例

```
  pdf2md.py document.pdf                             基本変換
  pdf2md.py --layout precise document.pdf           精密レイアウトモード
  pdf2md.py --preserve-image-layout doc.pdf         図版を元配置で保持、表をAIでMD化
```

## 動作モード

- **引数なし** → GUI起動 (`PDF2MDGUI().run()`)
- **ファイル/フォルダ指定** → CLI変換
- **`--context-menu`** → 右クリックメニューモード（ダイアログ表示付き）

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
