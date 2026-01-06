# PDF2MD - PDF to Markdown Converter

PDFファイルをMarkdown形式に変換するWindowsデスクトップアプリケーションです。

Microsoft [MarkItDown](https://github.com/microsoft/markitdown) を使用して、高精度なPDF→Markdown変換を実現します。

## 機能

- **GUI対応**: 直感的なグラフィカルインターフェース
- **ドラッグ&ドロップ**: PDFファイルをウィンドウにドロップして追加
- **バッチ処理**: 複数ファイルの一括変換
- **出力先選択**: 同じフォルダまたは指定フォルダに出力
- **CLIモード**: コマンドラインからも使用可能

## スクリーンショット

```
+------------------------------------------+
|      PDF to Markdown Converter           |
+------------------------------------------+
| [ファイル追加] [フォルダ追加] [クリア] [変換実行] |
+------------------------------------------+
| ファイルパス                    | 状態    |
|--------------------------------|---------|
| C:\docs\sample.pdf             | 待機中   |
| C:\docs\report.pdf             | 完了     |
+------------------------------------------+
| ○ 同じフォルダに出力                      |
| ○ 指定フォルダに出力: [          ] [参照] |
+------------------------------------------+
| [================    ] 60%               |
| 変換中: report.pdf                        |
+------------------------------------------+
```

## インストール

### 方法1: EXEファイルを使用（推奨）

1. [Releases](https://github.com/atsushisuemitsu/PDF2MD/releases) から最新の `PDF2MD.exe` をダウンロード
2. ダウンロードしたEXEを実行

### 方法2: Pythonから実行

```bash
# リポジトリをクローン
git clone https://github.com/atsushisuemitsu/PDF2MD.git
cd PDF2MD

# 依存関係をインストール
pip install -r requirements.txt

# 実行
python pdf2md.py
```

## 使い方

### GUIモード

1. `PDF2MD.exe` をダブルクリックして起動
2. 「ファイル追加」または「フォルダ追加」でPDFを選択
   - またはPDFファイルをウィンドウにドラッグ&ドロップ
3. 必要に応じて出力先を設定
4. 「変換実行」をクリック

### CLIモード

```bash
# 単一ファイル変換
python pdf2md.py document.pdf

# 複数ファイル変換
python pdf2md.py file1.pdf file2.pdf file3.pdf

# フォルダ内の全PDFを変換
python pdf2md.py ./pdf_folder/
```

## EXEのビルド方法

```bash
# 依存関係をインストール
pip install -r requirements.txt

# ビルド実行（Windows）
build.bat
# または
powershell -ExecutionPolicy Bypass -File build.ps1

# EXEは dist/PDF2MD.exe に生成されます
```

## 技術仕様

- **Python**: 3.9+
- **変換エンジン**: Microsoft MarkItDown
- **GUI**: tkinter
- **D&D対応**: tkinterdnd2（オプション）
- **EXE化**: PyInstaller

## 変換精度

Microsoft MarkItDownは以下の変換を高精度で行います：

- テキストの抽出と構造化
- 見出し階層の認識
- リスト構造の保持
- テーブルの変換
- 画像参照の保持

技術文書で90%以上の精度を達成（参考記事による）。

## 参考

- [PDFを高品質マークダウンに変換](https://note.com/suh_sunaneko/n/na6687b2e01c8)
- [Microsoft MarkItDown](https://github.com/microsoft/markitdown)

## ライセンス

MIT License

## 更新履歴

### v1.0.1 (2025-01-07)
- EXE化時のmagikaモデルファイル問題を修正
- PyInstaller specファイルを追加

### v1.0.0 (2025-01-07)
- 初回リリース
- GUI/CLIモード対応
- バッチ変換機能
- ドラッグ&ドロップ対応
