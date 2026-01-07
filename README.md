# PDF2MD - PDF to Markdown Converter

PDFファイルをMarkdown形式に変換するWindowsデスクトップアプリケーションです。

**PyMuPDF** と **EasyOCR** を使用して、テキストと画像を正確に抽出し、位置関係を保持したMarkdown変換を実現します。

## 主な機能

- **テキスト抽出**: PDFからテキストを位置情報付きで抽出
- **表抽出**: PDF内の表を検出してMarkdownテーブル形式に変換
- **画像抽出**: PDF内の図・画像を自動的に抽出して保存
- **OCR対応**: 図内のテキストをOCRで認識（日本語・英語対応）
- **位置保持**: テキストと画像と表の位置関係を正確にMarkdownに反映
- **GUI対応**: 直感的なグラフィカルインターフェース
- **ドラッグ&ドロップ**: PDFファイルをウィンドウにドロップして追加
- **バッチ処理**: 複数ファイルの一括変換
- **CLIモード**: コマンドラインからも使用可能

## スクリーンショット

```
+------------------------------------------+
|      PDF to Markdown Converter           |
+------------------------------------------+
| [ファイル追加] [フォルダ追加] [クリア] [変換実行] |
+------------------------------------------+
|   [x] 画像を抽出   [x] OCR（図内テキスト認識） |
+------------------------------------------+
| ファイルパス                    | 状態    |
|--------------------------------|---------|
| C:\docs\sample.pdf             | 待機中   |
| C:\docs\manual.pdf             | 完了     |
+------------------------------------------+
| ○ 同じフォルダに出力                      |
| ○ 指定フォルダに出力: [          ] [参照] |
+------------------------------------------+
| [================    ] 60%               |
| 変換中: manual.pdf                        |
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
3. オプションを設定:
   - **画像を抽出**: チェックするとPDF内の画像を別ファイルとして保存
   - **OCR**: チェックすると画像内のテキストをOCRで認識してMarkdownに含める
4. 必要に応じて出力先を設定
5. 「変換実行」をクリック

### CLIモード

```bash
# 単一ファイル変換（画像抽出あり）
python pdf2md.py document.pdf

# 複数ファイル変換
python pdf2md.py file1.pdf file2.pdf file3.pdf

# フォルダ内の全PDFを変換
python pdf2md.py ./pdf_folder/

# OCRを有効にして変換
python pdf2md.py --ocr document.pdf

# 画像抽出を無効にして変換
python pdf2md.py --no-images document.pdf
```

## 出力例

変換すると以下のファイルが生成されます：

```
input_folder/
├── manual.pdf          # 元のPDF

output_folder/
├── manual.md           # 変換されたMarkdown
└── manual_images/      # 抽出された画像
    ├── page1_img1.png
    ├── page1_img2.png
    └── page2_img1.png
```

### Markdownの出力形式

```markdown
# タイトル

テキストの内容がここに出力されます。

| 項目 | 説明 | 備考 |
| --- | --- | --- |
| 設定A | 値1 | - |
| 設定B | 値2 | オプション |

![図1](manual_images/page1_img1.png)

<details>
<summary>図内テキスト（OCR）</summary>

図から認識されたテキストがここに表示されます。

</details>

続きのテキスト内容...
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
- **PDF解析**: PyMuPDF (fitz)
- **OCR**: EasyOCR（日本語・英語対応）
- **画像処理**: Pillow
- **GUI**: tkinter
- **D&D対応**: tkinterdnd2（オプション）
- **EXE化**: PyInstaller

## 変換の仕組み

1. **PDF解析**: PyMuPDFでPDFを開き、各ページを解析
2. **表検出**: PyMuPDFの表検出機能で表を抽出し、Markdownテーブル形式に変換
3. **テキスト抽出**: 表領域を除外してテキストブロックを抽出
4. **位置情報取得**: 各要素のバウンディングボックス（x0, y0, x1, y1）を取得
5. **画像保存**: 画像をPNGファイルとして保存
6. **OCR処理**: EasyOCRで画像内のテキストを認識
7. **Markdown生成**: 位置情報に基づいて上から下、左から右の順序でMarkdownを構築

## 注意事項

- OCR機能を使用する場合、初回実行時にモデルファイル（約100MB）がダウンロードされます
- 大きな画像を含むPDFの場合、処理に時間がかかることがあります
- 複雑なレイアウトのPDFでは、位置関係が正確に再現されない場合があります

## 参考

- [PDFを高品質マークダウンに変換](https://note.com/suh_sunaneko/n/na6687b2e01c8)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

## ライセンス

MIT License

## 更新履歴

### v2.1.0 (2025-01-07)
- 表検出・抽出機能を追加（PyMuPDFのfind_tables使用）
- 表領域とテキスト領域の重複排除
- Markdownテーブル形式での出力

### v2.0.0 (2025-01-07)
- 変換エンジンをPyMuPDFに変更
- 画像抽出機能を追加
- OCR機能を追加（EasyOCR、日本語・英語対応）
- 位置情報を保持したMarkdown生成
- GUI にオプションチェックボックスを追加

### v1.0.1 (2025-01-07)
- EXE化時のmagikaモデルファイル問題を修正
- PyInstaller specファイルを追加

### v1.0.0 (2025-01-07)
- 初回リリース
- GUI/CLIモード対応
- バッチ変換機能
- ドラッグ&ドロップ対応
