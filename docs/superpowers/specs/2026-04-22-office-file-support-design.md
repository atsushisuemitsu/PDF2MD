# Office ファイル対応 設計書

**作成日**: 2026-04-22
**対象機能**: PDF2MD の変換対象に Microsoft Office ファイル (doc, docx, xls, xlsx, xlsm, pptx) を追加する
**バージョン**: v4.5 (予定)

## 1. 目的と背景

PDF2MD の入力対象を `.pdf` のみから Microsoft Office ファイルにも拡張する。対応形式は以下:

- `.doc` (Word 97-2003 バイナリ)
- `.docx` (Word 2007 以降 Open XML)
- `.xls` (Excel 97-2003 バイナリ)
- `.xlsx` (Excel 2007 以降 Open XML)
- `.xlsm` (マクロ有効 Excel 2007 以降 Open XML)
- `.pptx` (PowerPoint 2007 以降 Open XML)

既存の `markitdown` レイアウトモードは Microsoft 製 MarkItDown ライブラリを optional 依存として使用しており、上記の Office 形式はすべて MarkItDown がサポートする。これを拡張子ベースのディスパッチで自動起用する。

## 2. 基本方針

1. **拡張子ベースのディスパッチを入口に新設**: PDF は既存パイプライン、Office は新設する MarkItDown パイプラインに振り分ける
2. **Office ファイルは `--layout` 指定を無視して常に MarkItDown で変換**: PDF 用のレイアウトオプション (precise/legacy/page_image 等) は Office に適用できないため、指定されていた場合はログに「Office ファイルのため markitdown に自動切替」と警告を出す
3. **画像抽出は ZIP 構造から直接**: docx/xlsx/xlsm/pptx は内部的に ZIP 形式なので、`zipfile` (Python 標準ライブラリ) で `word/media/` `xl/media/` `ppt/media/` 配下を `{name}_images/` に展開する
4. **旧バイナリ形式 (.doc, .xls) は MarkItDown のテキストのみ**: 画像抽出は非対応、警告ログのみ出す
5. **MD 出力末尾に `## Embedded Images` セクションを追記**: 抽出した画像の参照リンクを列挙

**前提条件**:
- `markitdown` は optional 依存として既にインストール済みである前提 (既存の `markitdown` レイアウトモードと同じ)
- 旧形式 (.doc, .xls) のサポートには `markitdown[all]` でフル extras の導入が推奨される

**スコープ外**:
- Office → PDF 変換経由の処理 (LibreOffice / MS Office 依存となり EXE 配布と相性が悪い)
- Office ファイル向けの新規レイアウトモード (precise/page_image 等)
- 埋め込み画像の Claude API 解析 (PDF モード内の機構は流用しない)
- preserve-image-layout / ndlocr_cli 連携の Office 対応

## 3. アーキテクチャ

### 3.1 コンポーネント構成

既存 `pdf2md.py` への追加は最小限とし、新規クラスは導入せず既存 `AdvancedPDFConverter` にメソッドを追加する。

| 要素 | 種類 | 配置 | 責務 |
|------|------|------|------|
| `SUPPORTED_OFFICE_EXTS` | 定数 | module-level | `{'.doc', '.docx', '.xls', '.xlsx', '.xlsm', '.pptx'}` |
| `SUPPORTED_INPUT_EXTS` | 定数 | module-level | `{'.pdf'} \| SUPPORTED_OFFICE_EXTS` |
| `_is_office_file(path)` | 関数 | module-level | 拡張子 (lowercase) で Office 判定 |
| `_is_zip_based_office(path)` | 関数 | module-level | `.docx/.xlsx/.xlsm/.pptx` のいずれかを判定 (ZIP メディア抽出対象か) |
| `convert_file()` | 既存メソッド修正 | `AdvancedPDFConverter` | 入口でディスパッチ。Office なら `_convert_office_file()` を呼ぶ |
| `_convert_office_file()` | 新規メソッド | `AdvancedPDFConverter` | MarkItDown 実行 + ZIP からの画像抽出 + MD 末尾追記 + ファイル書き出し |
| `_extract_office_media_from_zip()` | 新規メソッド | `AdvancedPDFConverter` | ZIP 内メディアを `{name}_images/` にコピーし、抽出結果のリストを返す |
| `_append_embedded_images_section()` | 新規メソッド | `AdvancedPDFConverter` | MD 文字列末尾に `## Embedded Images` セクションを追加 |

### 3.2 CLI 修正

- ファイル判定 (現 line 1878, 3491, 3541 付近) の `.pdf` ハードコードを `SUPPORTED_INPUT_EXTS` 集合比較に置き換え
- フォルダ走査 (現 line 3307, 3329, 3513) の `glob('*.pdf')` を `SUPPORTED_INPUT_EXTS` 全拡張子に拡張
- `argparse` epilog に Office 変換の使用例を追加

### 3.3 GUI 修正 (`PDF2MDGUI`)

- D&D 判定 (現 line 3304, 3307) を拡張子セット比較に
- `askopenfilenames` の filetypes (現 line 3320) を `[("Supported files", "*.pdf *.doc *.docx *.xls *.xlsx *.xlsm *.pptx"), ("PDF files", "*.pdf"), ("Office files", "*.doc *.docx *.xls *.xlsx *.xlsm *.pptx"), ("All files", "*.*")]` に
- タイトル「PDFファイルを選択」は「ファイルを選択」に変更
- フォルダ走査 (現 line 3329) を同様に拡張

### 3.4 依存関係

`requirements.txt`:
- `markitdown` → `markitdown[all]` に変更 (doc/xls 対応を含むフル extras を確保)

## 4. データ構造

新規データクラスは追加しない。以下の軽量レコード型で十分:

```python
# 画像抽出結果 (list の要素、tuple で表現)
# (source_in_zip: str, dest_path: str, basename: str)
```

## 5. データフロー

```
入力ファイル (pdf_path; 後で input_path にリネーム)
  ↓
convert_file(input_path, ...)
  ↓
_is_office_file(input_path)
  │
  ├─ False → [既存 PDF パイプライン]
  │
  └─ True → _convert_office_file(input_path, output_path, images_dir, extract_images)
              ↓
              MARKITDOWN_AVAILABLE チェック → False → エラー return
              ↓
              MarkItDown().convert(input_path) → result.text_content
              ↓
              空文字チェック → 空 → エラー return
              ↓
              extract_images == True かつ _is_zip_based_office(input_path)
                ├─ True → _extract_office_media_from_zip(input_path, images_dir, base_name)
                │          → extracted: list[tuple[src, dest, basename]]
                │          → _append_embedded_images_section(md, extracted, base_name)
                └─ False (旧 doc/xls または extract_images=False)
                          → 警告ログ (旧形式の場合)
                          → md そのまま
              ↓
              write {name}.md (UTF-8)
              ↓
              return (True, 成功メッセージ)
```

## 6. エラーハンドリング

### 6.1 MarkItDown 未インストール

- `MARKITDOWN_AVAILABLE == False` のとき Office ファイル入力を受けたら、`_convert_office_file()` は即座に以下を返す:
  - `(False, "MarkItDown がインストールされていません。'pip install markitdown[all]' を実行してください。")`
- PDF 変換には影響させない (既存の degradation パターンと同じ)

### 6.2 旧バイナリ形式で MarkItDown が失敗

- .doc/.xls で MarkItDown 実行が例外を投げた場合 (例: `[xls]` extra 未インストール、環境依存) は、例外メッセージを含めたエラーを返す
- 「`pip install markitdown[all]` で全 extra を追加してください」というヒントを併記

### 6.3 ZIP 構造ではない Office ファイル

- `zipfile.is_zipfile(path) == False` の .docx/.xlsx/.xlsm/.pptx → 破損ファイルの可能性
- 警告ログ `[WARN] Not a valid ZIP file, skipping image extraction: {path}` を出してメディア抽出はスキップ、MarkItDown のテキスト出力は継続
- .doc/.xls は `_is_zip_based_office() == False` なので、そもそも ZIP 処理に入らない

### 6.4 ZIP 内にメディアなし

- `word/media/` `xl/media/` `ppt/media/` のいずれも存在しないまたは空 → 警告ログは出さず、`## Embedded Images` セクションも追加しない (空セクション抑止)

### 6.5 ファイル名衝突

- ZIP 内メディア (通常は `image1.png`, `image2.jpeg` 等) をコピーする際、拡張子は保持しつつ `{name}_images/office_img{N}{ext}` に連番リネーム
- `office_img` プレフィックスで PDF 抽出画像 (`page{N}_img{M}`) と名前空間を分離し、順序も保証

### 6.6 出力先重複

- 既存 `{name}_images/` ディレクトリが既にある場合の挙動は既存 PDF と同じ (`os.makedirs(exist_ok=True)` 的にスキップ、既存ファイルは上書きされない)

### 6.7 `--layout` と Office の相互作用

- Office ファイルは常に MarkItDown にルーティングする
- `--layout` が `markitdown` または `auto` (デフォルト) の場合は警告を出さない
- `--layout precise` / `legacy` / `page_image` を明示指定した状態で Office ファイルを渡されたとき:
  - 変換自体は実行し、stderr に `[WARN] Office file detected. --layout {value} is ignored; using markitdown.` を出力
- 判定方法: `argparse` のデフォルト値と比較、または `sys.argv` に `--layout` が含まれるかで「明示指定」を検出

### 6.8 エンコーディング

- 出力 MD は UTF-8 (既存と同じ)
- MarkItDown 側でエンコーディング問題が起きた場合はその例外をそのまま呼び出し元に伝播させ、`_convert_office_file()` が `(False, エラーメッセージ)` として返す

### 6.9 画像コピー失敗

- 特定の 1 枚のコピーに失敗したらログ警告の上、そのファイルだけスキップして処理を継続 (全画像抽出失敗にしない)
- ただし `{name}_images/` ディレクトリ作成自体が失敗した場合は例外を伝播させ変換を失敗扱いにする

## 7. 出力仕様

### 7.1 ファイル名・配置

既存と同一:
- 出力 Markdown: `{input_dir}/{name}.md` (または `-o` で指定されたパス)
- 画像ディレクトリ: `{output_dir}/{name}_images/`

### 7.2 Markdown 構造

```markdown
{MarkItDown が生成したコンテンツをそのまま}

## Embedded Images

![office_img1](./{name}_images/office_img1.png)
![office_img2](./{name}_images/office_img2.jpeg)
...
```

- `## Embedded Images` セクションは抽出画像がある場合のみ追加
- ファイル名はアルファベット順 (ZIP 内のソート順) で連番

### 7.3 ページ区切り

- PDF 出力にある `<!-- Page N -->` 区切りは MarkItDown 出力には存在しない
- MarkItDown の出力形式にそのまま従い、整形は加えない

## 8. テスト・検証方針

### 8.1 手動検証マトリクス

| 形式 | サンプル | テキスト | 表 | 画像 |
|------|---------|---------|-----|------|
| .docx | Word 2016+ の文書 | ✅ | ✅ | ZIP 抽出 |
| .doc | 同文書を Word で「97-2003形式」保存 | ✅ | ✅ | スキップ (警告) |
| .xlsx | 表と画像を含む Excel 2016+ | ✅ | ✅ | ZIP 抽出 |
| .xlsm | マクロ有 xlsx (マクロは MD に出ない) | ✅ | ✅ | ZIP 抽出 |
| .xls | 同資料を「97-2003形式」保存 | ✅ | ✅ | スキップ (警告) |
| .pptx | スライド + 図入り PowerPoint | ✅ | n/a | ZIP 抽出 |

### 8.2 自動チェック (手動起動の簡易スクリプト)

- `python pdf2md.py test_files/sample.docx` が exit code 0 で完了
- `sample.md` が生成される
- `sample_images/` 内に少なくとも 1 枚の画像があること (ZIP に画像が含まれていれば)
- 生成 MD に `## Embedded Images` セクションが含まれること (同上)
- 存在しない拡張子 (例: `.txt`) は従来通りエラー

### 8.3 リグレッション確認

- 既存 PDF 変換が影響を受けないこと: `sample.pdf` で precise/legacy/auto/markitdown/page_image の各モードが動くこと
- `--preserve-image-layout` も PDF でのみ動作、Office には影響しないこと

### 8.4 EXE ビルド検証

- `build.bat` で EXE 化後、docx/xlsx/pptx 各形式を D&D で GUI に投入し変換成功を確認
- EXE サイズが著しく増えないこと (markitdown[all] 追加分の範囲内)

### 8.5 エッジケース

- 空の docx (本文なし) → MarkItDown が空文字を返したら「変換結果が空です」エラー (既存 markitdown モードと同様)
- 大容量 xlsx (100 シート超) → MarkItDown に委譲、タイムアウトしないこと (MarkItDown 側のパフォーマンスに依存)

## 9. Wiki / ドキュメント更新

- `README.md`: サポート形式、インストール方法 (`markitdown[all]`) を記載
- `CLAUDE.md`: Project Overview、Build & Run Commands、Architecture 各節に反映
- `wiki/AdvancedPDFConverter.md`: 新メソッド (`_convert_office_file`, `_extract_office_media_from_zip`, `_append_embedded_images_section`) を説明
- `wiki/cli-interface.md`: Office ファイル変換の使用例を追加
- `wiki/PDF2MDGUI.md`: 対応ファイルダイアログ・D&D の拡張を追記
- `wiki/conversion-pipeline.md`: 入口ディスパッチ (PDF vs Office) を図に追加
- `wiki/overview.md`: バージョン "4.5"、history 追加
- `wiki/log.md`: `2026-04-22` 日付の機能追加エントリを追加

## 10. 非スコープ (YAGNI)

- Office → PDF 変換経由の処理 (LibreOffice などの外部ツール依存)
- Office ファイル向けの新規レイアウトモード (precise/page_image 等の追加)
- 埋め込み画像の Claude API 解析、PlantUML / WaveDrom 変換
- preserve-image-layout / ndlocr_cli 連携の Office 対応
- Excel のグラフオブジェクト、PowerPoint の動画など非画像メディアの抽出
- 暗号化された Office ファイルの対応 (MarkItDown が失敗するならそのまま失敗扱い)
