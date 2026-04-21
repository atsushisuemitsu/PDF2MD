# Office ファイル対応 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PDF2MD の変換対象を `.pdf` のみから Microsoft Office ファイル (doc/docx/xls/xlsx/xlsm/pptx) にも拡張する。

**Architecture:** 拡張子ベースのディスパッチを `AdvancedPDFConverter.convert_file()` の入口に追加し、Office ファイルは MarkItDown パイプラインに振り分ける。docx/xlsx/xlsm/pptx は内部 ZIP 構造から埋め込み画像を抽出し、`{name}_images/` に展開する。旧バイナリ形式 (.doc/.xls) はテキストのみ出力。

**Tech Stack:** Python 3.13, MarkItDown (`markitdown[all]`), zipfile (stdlib)

**Spec:** `docs/superpowers/specs/2026-04-22-office-file-support-design.md`

**Note (no tests framework):** このリポジトリにはテストフレームワーク・リンター・フォーマッターがない。各タスクの最後にスモークテスト (実ファイルでの動作確認) を行う。TDD は適用しない代わりに、都度動作確認を挟む。

---

## File Structure

変更対象ファイル:

- `pdf2md.py` (main, 1 file) — 単一ファイルアプリケーションの設計を踏襲
  - module-level 定数 + ヘルパー関数追加 (~line 140 付近)
  - `AdvancedPDFConverter` に新規メソッド 3 つ追加 (`_convert_with_markitdown` の後、~line 1843 付近)
  - `convert_file()` の入口修正 (line 1852-1912 付近)
  - `main()` / `_run_context_menu_mode()` のファイル判定修正 (line 3491, 3513, 3541)
  - argparse epilog / `inputs` help テキスト修正 (line 3434-3446)
  - `PDF2MDGUI` の D&D・ファイルダイアログ・フォルダ走査修正 (line 3304-3329)
- `requirements.txt` — `markitdown[all]` を明示追加
- `README.md` — サポート形式・インストール手順更新
- `CLAUDE.md` — Project Overview / Architecture 更新
- `wiki/overview.md` — バージョン "4.5"、history 追加
- `wiki/AdvancedPDFConverter.md` — 新メソッド説明
- `wiki/cli-interface.md` — Office 変換例
- `wiki/PDF2MDGUI.md` — ダイアログ / D&D 拡張
- `wiki/conversion-pipeline.md` — 入口ディスパッチ追加
- `wiki/log.md` — 2026-04-22 エントリ追加

---

## Task 1: 定数とヘルパー関数の追加

**Files:**
- Modify: `pdf2md.py` (module-level、DND_AVAILABLE の直後、line 141 付近)

- [ ] **Step 1: モジュールレベルに定数とヘルパー関数を追加**

`DND_AVAILABLE = False` のブロック直後 (line 140 の次) に以下を挿入:

```python

# Office ファイル対応 (v4.5)
SUPPORTED_OFFICE_EXTS = {'.doc', '.docx', '.xls', '.xlsx', '.xlsm', '.pptx'}
SUPPORTED_INPUT_EXTS = {'.pdf'} | SUPPORTED_OFFICE_EXTS
ZIP_BASED_OFFICE_EXTS = {'.docx', '.xlsx', '.xlsm', '.pptx'}


def _get_file_ext(path: str) -> str:
    """ファイル拡張子を lowercase で返す (ドット含む)"""
    return os.path.splitext(path)[1].lower()


def _is_office_file(path: str) -> bool:
    """入力パスが Office ファイルかを拡張子で判定"""
    return _get_file_ext(path) in SUPPORTED_OFFICE_EXTS


def _is_zip_based_office(path: str) -> bool:
    """ZIP 構造を持つ Office 形式 (docx/xlsx/xlsm/pptx) か"""
    return _get_file_ext(path) in ZIP_BASED_OFFICE_EXTS


def _is_supported_input(path: str) -> bool:
    """PDF または Office のサポート済み形式か"""
    return _get_file_ext(path) in SUPPORTED_INPUT_EXTS
```

- [ ] **Step 2: Python 構文チェック**

Run: `python -c "import pdf2md"`
Expected: エラーなし (インポート成功)

- [ ] **Step 3: ヘルパー関数の動作確認 (ad-hoc)**

Run:
```bash
python -c "import pdf2md; print(pdf2md._is_office_file('a.docx'), pdf2md._is_office_file('a.pdf'), pdf2md._is_zip_based_office('a.doc'), pdf2md._is_supported_input('a.txt'))"
```
Expected: `True False False False`

- [ ] **Step 4: Commit**

```bash
git add pdf2md.py
git commit -m "feat(v4.5): Office ファイル対応の定数とヘルパー関数を追加"
```

---

## Task 2: ZIP からメディアを抽出するメソッド

**Files:**
- Modify: `pdf2md.py` (`AdvancedPDFConverter` クラス、`_convert_with_markitdown` の後)

- [ ] **Step 1: import 文に zipfile と shutil を追加**

`pdf2md.py` の import セクション (line 25-39) を確認。`shutil` が未 import なら追加する。`zipfile` は Task 内でローカル import するため module-level には追加不要。

line 26 の `import sys` の直後に以下を追加 (`shutil` が既にある場合はスキップ):

```python
import shutil
```

Run: `python -c "import pdf2md"` でエラーがないこと。

- [ ] **Step 2: `_extract_office_media_from_zip` メソッドを追加**

`_convert_with_markitdown` メソッドの直後 (`_render_page_image` の前、~line 1842-1843 の間) に以下のメソッドを挿入:

```python
    def _extract_office_media_from_zip(self, input_path: str, images_dir: str, base_name: str) -> list:
        """
        docx/xlsx/xlsm/pptx の ZIP 構造から埋め込み画像を抽出し、
        {name}_images/ に office_img{N}{ext} として保存する。

        Args:
            input_path: 入力 Office ファイルパス
            images_dir: 出力画像ディレクトリ
            base_name: ファイル基本名 (拡張子なし)

        Returns:
            [(source_in_zip, dest_basename), ...] ソース名順
        """
        import zipfile

        MEDIA_PREFIXES = ('word/media/', 'xl/media/', 'ppt/media/')

        if not zipfile.is_zipfile(input_path):
            print(f"[WARN] Not a valid ZIP file, skipping image extraction: {input_path}")
            return []

        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)

        extracted = []
        with zipfile.ZipFile(input_path, 'r') as zf:
            media_members = sorted([
                name for name in zf.namelist()
                if name.startswith(MEDIA_PREFIXES) and not name.endswith('/')
            ])

            for idx, member in enumerate(media_members, start=1):
                ext = os.path.splitext(member)[1] or '.bin'
                dest_basename = f"office_img{idx}{ext}"
                dest_path = os.path.join(images_dir, dest_basename)
                try:
                    with zf.open(member) as src, open(dest_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append((member, dest_basename))
                except Exception as e:
                    print(f"[WARN] Failed to extract {member}: {e}")

        return extracted
```

- [ ] **Step 3: スモークテスト (手元の docx で確認)**

任意の画像入り docx で動作確認する。サンプル docx がない場合は PowerPoint / Excel のサンプルでも可。

Run:
```bash
python -c "
import os, pdf2md
conv = pdf2md.AdvancedPDFConverter(enable_ocr=False)
os.makedirs('test_out_images', exist_ok=True)
# サンプル docx のパスに置き換える
sample = 'sample.docx'
if os.path.exists(sample):
    result = conv._extract_office_media_from_zip(sample, 'test_out_images', 'sample')
    print(f'Extracted {len(result)} media files')
    for src, dst in result:
        print(f'  {src} -> {dst}')
else:
    print('skip (no sample docx)')
"
```
Expected: 画像枚数とファイル名が出力される、または "skip" と出る

- [ ] **Step 4: Commit**

```bash
git add pdf2md.py
git commit -m "feat(v4.5): Office ファイル ZIP 構造からの画像抽出メソッドを追加"
```

---

## Task 3: Embedded Images セクション生成メソッド

**Files:**
- Modify: `pdf2md.py` (`AdvancedPDFConverter` クラス、Task 2 のメソッドの直後)

- [ ] **Step 1: `_append_embedded_images_section` メソッドを追加**

`_extract_office_media_from_zip` の直後に挿入:

```python
    def _append_embedded_images_section(self, md_content: str, extracted: list, base_name: str) -> str:
        """
        Markdown 末尾に ## Embedded Images セクションを追加する。

        Args:
            md_content: MarkItDown が生成した Markdown 本体
            extracted: [(source_in_zip, dest_basename), ...]
            base_name: ファイル基本名 (拡張子なし、画像ディレクトリ名に使用)

        Returns:
            セクションを追加した新しい Markdown 文字列。extracted が空なら md_content そのまま。
        """
        if not extracted:
            return md_content

        section_lines = ["", "", "## Embedded Images", ""]
        for _, dest_basename in extracted:
            alt = os.path.splitext(dest_basename)[0]
            section_lines.append(f"![{alt}](./{base_name}_images/{dest_basename})")
        section_lines.append("")

        return md_content.rstrip() + "\n".join(section_lines)
```

- [ ] **Step 2: スモークテスト**

Run:
```bash
python -c "
import pdf2md
conv = pdf2md.AdvancedPDFConverter(enable_ocr=False)
md = 'Hello world.'
extracted = [('word/media/image1.png', 'office_img1.png'), ('word/media/image2.jpeg', 'office_img2.jpeg')]
result = conv._append_embedded_images_section(md, extracted, 'sample')
print(result)
print('---')
# empty case
print(repr(conv._append_embedded_images_section(md, [], 'sample')))
"
```
Expected:
```
Hello world.

## Embedded Images

![office_img1](./sample_images/office_img1.png)
![office_img2](./sample_images/office_img2.jpeg)

---
'Hello world.'
```

- [ ] **Step 3: Commit**

```bash
git add pdf2md.py
git commit -m "feat(v4.5): Embedded Images セクション生成メソッドを追加"
```

---

## Task 4: Office 変換メソッド本体

**Files:**
- Modify: `pdf2md.py` (`AdvancedPDFConverter` クラス、Task 3 のメソッドの直後)

- [ ] **Step 1: `_convert_office_file` メソッドを追加**

`_append_embedded_images_section` の直後に挿入:

```python
    def _convert_office_file(self, input_path: str, output_path: str,
                              images_dir: str, extract_images: bool) -> Tuple[bool, str]:
        """
        Office ファイル (doc/docx/xls/xlsx/xlsm/pptx) を MarkItDown で変換する。

        Args:
            input_path: 入力 Office ファイルパス
            output_path: 出力 Markdown パス
            images_dir: 画像出力ディレクトリ
            extract_images: 画像抽出を実行するか

        Returns:
            (成功フラグ, メッセージ)
        """
        if not MARKITDOWN_AVAILABLE:
            return False, "MarkItDown がインストールされていません。'pip install markitdown[all]' を実行してください。"

        print(f"[INFO] Converting Office file with MarkItDown: {input_path}")

        try:
            md_engine = _MarkItDown()
            result = md_engine.convert(input_path)
        except Exception as e:
            hint = ""
            if _get_file_ext(input_path) in {'.doc', '.xls'}:
                hint = " 旧バイナリ形式の処理に失敗しました。'pip install markitdown[all]' で全 extra を追加してください。"
            return False, f"MarkItDown conversion failed: {e}.{hint}"

        md_content = result.text_content if hasattr(result, 'text_content') else getattr(result, 'markdown', '')
        if not md_content or not md_content.strip():
            return False, "MarkItDown returned empty content"

        base_name = os.path.splitext(os.path.basename(input_path))[0]

        if extract_images and _is_zip_based_office(input_path):
            try:
                extracted = self._extract_office_media_from_zip(input_path, images_dir, base_name)
                md_content = self._append_embedded_images_section(md_content, extracted, base_name)
            except Exception as e:
                print(f"[WARN] Image extraction error (continuing with text only): {e}")
        elif extract_images and _is_office_file(input_path):
            print(f"[WARN] Image extraction not supported for legacy binary format: {input_path}")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
        except Exception as e:
            return False, f"Failed to write output: {e}"

        return True, f"Converted: {output_path}"
```

- [ ] **Step 2: 構文チェック**

Run: `python -c "import pdf2md"`
Expected: エラーなし

- [ ] **Step 3: Commit**

```bash
git add pdf2md.py
git commit -m "feat(v4.5): Office ファイル変換メソッド _convert_office_file を追加"
```

---

## Task 5: convert_file() 入口にディスパッチャを追加

**Files:**
- Modify: `pdf2md.py` line 1852-1912 付近 (`convert_file` メソッド)

- [ ] **Step 1: ファイル形式チェックとディスパッチを挿入**

line 1878-1879 の

```python
            if not pdf_path.lower().endswith('.pdf'):
                return False, f"PDFファイルではありません: {pdf_path}"
```

を次に置き換える:

```python
            if not _is_supported_input(pdf_path):
                return False, f"対応していないファイル形式です: {pdf_path} (サポート: .pdf, .doc, .docx, .xls, .xlsx, .xlsm, .pptx)"
```

- [ ] **Step 2: Office ディスパッチを追加**

line 1891 の `if extract_images and not os.path.exists(images_dir): os.makedirs(images_dir)` の直後、line 1893 の `# page_imageモード...` コメントの前に以下を挿入:

```python

            # Office ファイル (doc/docx/xls/xlsx/xlsm/pptx) は MarkItDown にルーティング
            if _is_office_file(pdf_path):
                if layout_mode not in ('markitdown', 'auto'):
                    print(
                        f"[WARN] Office file detected. --layout {layout_mode} is ignored; using markitdown.",
                        file=sys.stderr
                    )
                return self._convert_office_file(pdf_path, output_path, images_dir, extract_images)
```

- [ ] **Step 3: スモークテスト (PDF パスに影響がないこと)**

手元の PDF ファイルで既存変換が動くことを確認:

Run:
```bash
python pdf2md.py --no-images sample.pdf 2>&1 | head -5
```
Expected: 既存通りに変換開始メッセージが出ること (sample.pdf がなければ `ファイルが見つかりません` エラーで OK)

- [ ] **Step 4: スモークテスト (Office サンプルがあれば)**

Run:
```bash
python pdf2md.py sample.docx 2>&1 | head -10
```
Expected: 
- MarkItDown 未インストール時: `MarkItDown がインストールされていません。...`
- インストール済み時: `[INFO] Converting Office file with MarkItDown: sample.docx` → 成功

- [ ] **Step 5: Commit**

```bash
git add pdf2md.py
git commit -m "feat(v4.5): convert_file 入口に Office ディスパッチを追加"
```

---

## Task 6: CLI 対応 (argparse と main())

**Files:**
- Modify: `pdf2md.py` main() 関数と `_run_context_menu_mode()` (line 3429-3560 付近)

- [ ] **Step 1: argparse epilog と help メッセージを更新**

line 3432 の `description="PDF to Markdown Converter v4.0"` を `description="PDF/Office to Markdown Converter v4.5"` に変更。

line 3434-3445 の epilog を次に置き換える:

```python
        epilog="""
例:
  pdf2md.py                               GUIモードで起動
  pdf2md.py document.pdf                  PDFをMarkdownに変換
  pdf2md.py document.docx                 Word(.docx)をMarkdownに変換
  pdf2md.py report.xlsx                   Excel(.xlsx)をMarkdownに変換
  pdf2md.py slides.pptx                   PowerPoint(.pptx)をMarkdownに変換
  pdf2md.py --layout precise doc.pdf      精密レイアウトモードで変換
  pdf2md.py --layout page_image doc.pdf   ページ画像モードで変換
  pdf2md.py --layout markitdown doc.pdf   MarkItDownで変換 (PDF)
  pdf2md.py --ocr document.pdf            OCR有効で変換
  pdf2md.py --no-images document.pdf      画像抽出なしで変換
  pdf2md.py ./pdf_folder/                 フォルダ内の対応ファイルを一括変換
"""
```

line 3446 の `parser.add_argument("inputs", nargs="*", help="PDFファイルまたはフォルダのパス")` を次に変更:

```python
    parser.add_argument("inputs", nargs="*", help="PDF/Office ファイルまたはフォルダのパス")
```

- [ ] **Step 2: main() のファイル判定を拡張**

line 3491 の

```python
        if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
```

を次に置き換える:

```python
        if os.path.isfile(input_path) and _is_supported_input(input_path):
```

line 3511-3513 の

```python
        elif os.path.isdir(input_path):
            print(f"Converting folder: {input_path}")
            for pdf_path in Path(input_path).glob('*.pdf'):
```

を次に置き換える:

```python
        elif os.path.isdir(input_path):
            print(f"Converting folder: {input_path}")
            folder_files = []
            for ext in SUPPORTED_INPUT_EXTS:
                folder_files.extend(Path(input_path).glob(f'*{ext}'))
                folder_files.extend(Path(input_path).glob(f'*{ext.upper()}'))
            for pdf_path in sorted(set(folder_files)):
```

line 3532 の

```python
        else:
            print(f"  Skipping (not a PDF or folder): {input_path}")
```

を次に変更:

```python
        else:
            print(f"  Skipping (not a supported file or folder): {input_path}")
```

- [ ] **Step 3: `_run_context_menu_mode` のファイル判定を拡張**

line 3540-3547 の

```python
    pdf_path = args.inputs[0]
    if not os.path.isfile(pdf_path) or not pdf_path.lower().endswith('.pdf'):
        if not args.silent:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("エラー", f"PDFファイルではありません:\n{pdf_path}")
            root.destroy()
        return
```

を次に置き換える:

```python
    pdf_path = args.inputs[0]
    if not os.path.isfile(pdf_path) or not _is_supported_input(pdf_path):
        if not args.silent:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "エラー",
                f"対応していないファイル形式です:\n{pdf_path}\n\nサポート: PDF, DOC, DOCX, XLS, XLSX, XLSM, PPTX"
            )
            root.destroy()
        return
```

- [ ] **Step 4: スモークテスト (help 表示)**

Run: `python pdf2md.py --help`
Expected: 新しい epilog (Office 例を含む) が表示される

- [ ] **Step 5: スモークテスト (未対応拡張子)**

Run: `python pdf2md.py README.md`
Expected: `Skipping (not a supported file or folder): README.md`

- [ ] **Step 6: スモークテスト (Office サンプル、ある場合)**

手元の .docx があれば:
Run: `python pdf2md.py sample.docx`
Expected: 正常変換、`sample.md` と `sample_images/` 生成

- [ ] **Step 7: Commit**

```bash
git add pdf2md.py
git commit -m "feat(v4.5): CLI が Office ファイルを受け付けるよう拡張"
```

---

## Task 7: GUI 対応 (PDF2MDGUI)

**Files:**
- Modify: `pdf2md.py` `PDF2MDGUI` クラス (line 3299-3330 付近)

- [ ] **Step 1: D&D ハンドラを Office 対応に**

line 3299-3308 の `_on_drop` を次に置き換える:

```python
    def _on_drop(self, event):
        """D&Dイベントハンドラ"""
        files = self.root.tk.splitlist(event.data)
        for f in files:
            f = f.strip('{}')
            if os.path.isfile(f) and _is_supported_input(f):
                self._add_file_to_list(f)
            elif os.path.isdir(f):
                for ext in SUPPORTED_INPUT_EXTS:
                    for p in Path(f).glob(f'*{ext}'):
                        self._add_file_to_list(str(p))
                    for p in Path(f).glob(f'*{ext.upper()}'):
                        self._add_file_to_list(str(p))
```

- [ ] **Step 2: ファイル選択ダイアログを Office 対応に**

line 3316-3323 の `_add_files` を次に置き換える:

```python
    def _add_files(self):
        """ファイル選択ダイアログ"""
        files = filedialog.askopenfilenames(
            title="ファイルを選択",
            filetypes=[
                ("対応ファイル", "*.pdf *.doc *.docx *.xls *.xlsx *.xlsm *.pptx"),
                ("PDF files", "*.pdf"),
                ("Office files", "*.doc *.docx *.xls *.xlsx *.xlsm *.pptx"),
                ("All files", "*.*"),
            ]
        )
        for f in files:
            self._add_file_to_list(f)
```

- [ ] **Step 3: フォルダ選択ダイアログを Office 対応に**

line 3325-3330 の `_add_folder` を次に置き換える:

```python
    def _add_folder(self):
        """フォルダ選択ダイアログ"""
        folder = filedialog.askdirectory(title="フォルダを選択")
        if folder:
            seen = set()
            for ext in SUPPORTED_INPUT_EXTS:
                for p in Path(folder).glob(f'*{ext}'):
                    sp = str(p)
                    if sp not in seen:
                        seen.add(sp)
                        self._add_file_to_list(sp)
                for p in Path(folder).glob(f'*{ext.upper()}'):
                    sp = str(p)
                    if sp not in seen:
                        seen.add(sp)
                        self._add_file_to_list(sp)
```

- [ ] **Step 4: スモークテスト (GUI 起動)**

Run: `python pdf2md.py`
確認項目:
- GUI が起動すること
- 「ファイル追加」ボタン押下でダイアログが出て、filetypes が 「対応ファイル」「PDF files」「Office files」「All files」の順で表示されること
- docx ファイル (もしあれば) を D&D でリストに追加できること

エラーが出たら即座に修正する。

- [ ] **Step 5: Commit**

```bash
git add pdf2md.py
git commit -m "feat(v4.5): GUI が Office ファイルを受け付けるよう拡張"
```

---

## Task 8: requirements.txt の更新

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: `markitdown[all]` 行を追加**

`anthropic>=0.40.0` (line 18) の後、`tkinterdnd2>=0.3.0` の前に以下を挿入:

```
# MarkItDown (PDF + Office documents: doc/docx/xls/xlsx/xlsm/pptx)
# [all] extras includes legacy doc/xls support
markitdown[all]>=0.0.2
```

完成後の `requirements.txt` (要確認):

```
# PDF to Markdown Converter Dependencies
# Advanced version with image extraction, OCR, and layout preservation

# Core PDF parsing
PyMuPDF>=1.23.0

# Layout-aware conversion (recommended)
pymupdf4llm>=0.2.0
pymupdf-layout>=1.0.0

# Image processing
Pillow>=10.0.0

# OCR for figure text recognition (optional - requires 64-bit Python with PyTorch)
# easyocr>=1.7.0

# Optional: Claude API for diagram/flowchart analysis
anthropic>=0.40.0

# MarkItDown (PDF + Office documents: doc/docx/xls/xlsx/xlsm/pptx)
# [all] extras includes legacy doc/xls support
markitdown[all]>=0.0.2

# Optional: Drag & Drop support for Windows GUI
tkinterdnd2>=0.3.0

# For EXE building
pyinstaller>=6.0.0
```

- [ ] **Step 2: スモークテスト (インストール実行)**

Run: `pip install -r requirements.txt`
Expected: markitdown[all] が (未インストール or upgrade) インストールされる。既存インストール済みなら "already satisfied" が出る。エラーなく完了。

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps(v4.5): markitdown[all] を requirements に追加 (Office対応)"
```

---

## Task 9: Wiki / ドキュメント更新

**Files:**
- Modify: `README.md`, `CLAUDE.md`
- Modify: `wiki/overview.md`, `wiki/AdvancedPDFConverter.md`, `wiki/cli-interface.md`, `wiki/PDF2MDGUI.md`, `wiki/conversion-pipeline.md`, `wiki/log.md`

- [ ] **Step 1: `wiki/overview.md` を更新**

`wiki/overview.md` の内容を読み、以下を反映:

- `version: "4.3"` (または現行値) を `version: "4.5"` に更新
- サポート形式セクションに Office ファイル (doc/docx/xls/xlsx/xlsm/pptx) を追加
- 更新履歴セクションに次のエントリを追加:
  ```
  ### v4.5 (2026-04-22)
  - **Office ファイル対応**: doc/docx/xls/xlsx/xlsm/pptx を入力として受付
  - MarkItDown による変換、ZIP 構造からの埋め込み画像抽出
  - CLI・GUI 両方で Office ファイルを直接変換可能
  - 旧バイナリ形式 (.doc/.xls) はテキストのみ (画像抽出非対応)
  ```

- [ ] **Step 2: `wiki/AdvancedPDFConverter.md` を更新**

新規メソッドセクションを追加:

```markdown
### Office ファイル変換 (v4.5)

Office ファイル (doc/docx/xls/xlsx/xlsm/pptx) の変換は MarkItDown ライブラリに委譲される。
`convert_file()` の入口で拡張子が Office 形式なら `_convert_office_file()` にディスパッチされる。

#### メソッド

- `_convert_office_file(input_path, output_path, images_dir, extract_images)`
  - MarkItDown で Markdown 変換
  - docx/xlsx/xlsm/pptx は ZIP から画像を抽出して `{name}_images/` に保存
  - 旧バイナリ .doc/.xls は画像抽出をスキップ (警告ログ)

- `_extract_office_media_from_zip(input_path, images_dir, base_name)`
  - ZIP 構造の Office ファイルから `word/media/` `xl/media/` `ppt/media/` を抽出
  - `office_img{N}{ext}` として連番保存

- `_append_embedded_images_section(md_content, extracted, base_name)`
  - Markdown 末尾に `## Embedded Images` セクション追加
  - 抽出画像がない場合は追加しない
```

- [ ] **Step 3: `wiki/cli-interface.md` を更新**

Office 変換の使用例セクションを追加:

```markdown
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
- `precise` など PDF 専用のモードを指定した場合は stderr に警告が出る
- `--no-images` で埋め込み画像抽出を無効化可能
```

- [ ] **Step 4: `wiki/PDF2MDGUI.md` を更新**

対応形式セクションに以下を反映:
- ファイルダイアログの filetypes に Office 形式を追加
- D&D で docx/xlsx/pptx 等を受け付ける
- フォルダ選択で Office ファイルもリストアップされる

- [ ] **Step 5: `wiki/conversion-pipeline.md` を更新**

パイプライン図の入口に以下を追加:

```
入力ファイル
  │
  ├─ 拡張子判定 (_is_office_file)
  │
  ├─ Office 形式 → _convert_office_file() → MarkItDown + ZIP メディア抽出
  │
  └─ PDF → (既存パイプライン)
```

- [ ] **Step 6: `wiki/log.md` に日付エントリ追加**

ファイル先頭に以下を追加:

```markdown
## 2026-04-22 — Office ファイル対応 (v4.5)

- 入力形式を拡張: `.doc`, `.docx`, `.xls`, `.xlsx`, `.xlsm`, `.pptx`
- MarkItDown 統合により Office 変換を実装
- docx/xlsx/xlsm/pptx の ZIP 構造から埋め込み画像を `{name}_images/` に抽出
- 旧バイナリ形式 (.doc/.xls) はテキストのみ (画像抽出非対応)
- CLI / GUI 両方で Office ファイルを直接受付
- `--layout` 指定は Office ファイルに対して無視され、MarkItDown に自動ルーティング
```

- [ ] **Step 7: `README.md` を更新**

「対応ファイル形式」「インストール」セクションに以下を反映:
- サポート形式リストに Office ファイル追加
- `pip install markitdown[all]` を推奨インストール手順に追加

- [ ] **Step 8: `CLAUDE.md` を更新**

Project Overview セクションの説明を更新:
- 「PDF2MD is a Windows desktop application ... that converts PDF files to Markdown format.」
  → 「PDF2MD is a Windows desktop application ... that converts PDF and Microsoft Office files (doc/docx/xls/xlsx/xlsm/pptx) to Markdown format.」

Build & Run Commands に Office 変換例を追加:
```bash
python pdf2md.py document.docx
python pdf2md.py report.xlsx
python pdf2md.py slides.pptx
```

Architecture セクションの `convert_file()` 説明に Office ディスパッチを追記 (例: "Office files (doc/docx/xls/xlsx/xlsm/pptx) are dispatched to `_convert_office_file()` which uses MarkItDown for conversion and extracts embedded images from the ZIP structure.")

- [ ] **Step 9: Commit**

```bash
git add README.md CLAUDE.md wiki/
git commit -m "docs(v4.5): Office ファイル対応のドキュメント・Wiki 更新"
```

---

## Task 10: 動作検証と最終確認

**Files:** (変更なし、検証のみ)

- [ ] **Step 1: 構文チェック**

Run: `python -c "import pdf2md"`
Expected: エラーなし

- [ ] **Step 2: help 出力確認**

Run: `python pdf2md.py --help 2>&1 | grep -E "(docx|xlsx|pptx)"`
Expected: Office 形式の例が複数行出る

- [ ] **Step 3: 対応ファイル判定の網羅確認**

Run:
```bash
python -c "
import pdf2md
for p in ['a.pdf', 'a.docx', 'a.doc', 'a.xlsx', 'a.xls', 'a.xlsm', 'a.pptx', 'a.txt', 'a.PPTX', 'a.DOCX']:
    print(f'{p}: supported={pdf2md._is_supported_input(p)}, office={pdf2md._is_office_file(p)}, zip={pdf2md._is_zip_based_office(p)}')
"
```
Expected:
- .pdf/.docx/.doc/.xlsx/.xls/.xlsm/.pptx すべて `supported=True`
- Office 系 7 件が `office=True`、.pdf は False
- docx/xlsx/xlsm/pptx が `zip=True`、doc/xls は False
- .txt は全て False
- 大文字 `.PPTX` / `.DOCX` は全て `supported=True, office=True, zip=True`

- [ ] **Step 4: PDF 変換のリグレッション確認**

手元の PDF で既存変換が動作することを確認:

Run: `python pdf2md.py --no-images sample.pdf` (サンプルがなくても exit code は失敗でよい、未対応形式エラーでないこと)
Expected: 「ファイルが見つかりません」または実行成功 (「対応していないファイル形式」ではない)

- [ ] **Step 5: Office サンプル変換の実地確認**

以下の形式で手元にサンプルがあれば各形式を変換:

```bash
python pdf2md.py sample.docx
python pdf2md.py sample.xlsx
python pdf2md.py sample.pptx
# 旧形式 (ある場合)
python pdf2md.py sample.doc
python pdf2md.py sample.xls
```

確認項目:
- `sample.md` が生成される
- docx/xlsx/xlsm/pptx: `sample_images/` 内に `office_img{N}.{ext}` が生成される (画像入りなら)
- 生成 MD の末尾に `## Embedded Images` セクションがある (画像入りなら)
- .doc/.xls: 画像抽出スキップの警告ログが出る

- [ ] **Step 6: `--layout` 警告の動作確認**

Run: `python pdf2md.py --layout precise sample.docx 2>&1 | grep WARN`
Expected: `[WARN] Office file detected. --layout precise is ignored; using markitdown.`

- [ ] **Step 7: `--layout auto` および `markitdown` で警告なし**

Run: `python pdf2md.py --layout auto sample.docx 2>&1 | grep WARN`
Expected: 警告なし (または ndlocr 未利用の無関係な警告のみ)

Run: `python pdf2md.py --layout markitdown sample.docx 2>&1 | grep WARN`
Expected: 警告なし

- [ ] **Step 8: GUI 動作確認**

Run: `python pdf2md.py`
確認項目:
- 「ファイル追加」ダイアログが Office 形式を含む
- docx/xlsx/pptx を D&D でリストに追加可
- 変換実行 → `.md` と `_images/` が生成される

- [ ] **Step 9: EXE ビルド検証**

Run: `build.bat` (手動実行)
Expected: `dist\PDF2MD.exe` が生成される、ビルドエラーなし

EXE 実行後、docx ファイルで変換動作確認:
```
dist\PDF2MD.exe sample.docx
```
Expected: MD と画像が生成される

- [ ] **Step 10: 最終 Commit (検証で変更があれば)**

検証中に発見した問題を修正した場合のみ:

```bash
git add pdf2md.py
git commit -m "fix(v4.5): 動作検証で発見した問題を修正"
```

問題がなければ Commit なしでタスク完了。

---

## 付録: スコープ外確認

以下は本計画の対象外 (仕様書 §10 と対応):

- Office → PDF 変換経由 (LibreOffice 等の外部ツール依存)
- Office ファイル向けの新規レイアウトモード (precise/page_image 等)
- 埋め込み画像の Claude API 解析、PlantUML / WaveDrom 変換
- preserve-image-layout / ndlocr_cli 連携の Office 対応
- Excel のグラフオブジェクト、PowerPoint の動画など非画像メディアの抽出
- 暗号化 Office ファイルの特別対応 (MarkItDown 失敗→そのまま失敗扱い)
