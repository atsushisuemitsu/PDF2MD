# preserve-image-layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PDF から図版(画像・図・フローチャート)領域のみを切り抜いて PNG 保存し、表組は Claude API で Markdown 表に変換、元ページと同じ Y 位置で Markdown に挿入する新オプション `--preserve-image-layout` を追加する。

**Architecture:** 全 layout mode 共通の後処理フックとして実装。ndlocr_cli で図版/表組領域を検出 → 切り抜き → 図版は画像保存、表組は Claude API で Markdown 化 → 既存 MD テキストに Y 座標順でマージ。ndlocr_cli 未導入時はオプション自体を無効化し既存挙動にフォールバック。

**Tech Stack:** Python 3.13, PyMuPDF (fitz), ndlocr_cli, anthropic (Claude API Vision), tkinter, argparse

**Reference spec:** `docs/superpowers/specs/2026-04-10-preserve-image-layout-design.md`

**Note on tests:** このプロジェクトにはユニットテストフレームワークが無いため(`CLAUDE.md` 明記)、各タスクの検証は `python -c "import pdf2md"` による構文/インポートチェックを必須とし、最終タスクで実 PDF によるラウンドトリップ検証を行う。

---

## File Structure

すべての変更は単一ファイル `pdf2md.py` 内で行う(既存の単一ファイル構成を維持)。

| 変更対象 | 行数目安 | 内容 |
|---------|---------|------|
| `pdf2md.py` データクラス追加箇所 (line ~165 `ImageBlock` の下) | +20 | `PageFigureRegion`, `PageTableRegion` |
| `pdf2md.py` `AdvancedPDFConverter` クラス内 (line ~1381 `_convert_with_page_ocr` の上あたり) | +180 | `_extract_layout_regions_via_ndlocr`, `_postprocess_preserve_image_layout` |
| `pdf2md.py` `convert_file()` (line ~1852) | 修正 | `preserve_image_layout` 引数追加、各 mode 分岐で後処理呼び出し |
| `pdf2md.py` 標準パイプライン (line ~1944-1999) | 修正 | `page_text_positions` 構築、`AdvancedTableExtractor` スキップ分岐 |
| `pdf2md.py` `_convert_with_page_ocr` (line ~1381) | 修正 | `preserve_image_layout` ガード追加 |
| `pdf2md.py` CLI main (line ~3429) | +5 | `--preserve-image-layout` argparse 追加 |
| `pdf2md.py` `PDF2MDGUI` (line ~3225 option_frame) | +10 | チェックボックス追加、`_convert_thread` に伝搬 |
| `wiki/AdvancedPDFConverter.md` | +セクション | 新メソッド追記 |
| `wiki/cli-interface.md` | +行 | オプション追記 |
| `wiki/PDF2MDGUI.md` | +行 | チェックボックス追記 |
| `wiki/log.md` | +エントリ | 更新ログ |
| `wiki/overview.md` | +行 | v4.4 |

---

## Task 1: データクラス `PageFigureRegion` と `PageTableRegion` を追加

**Files:**
- Modify: `pdf2md.py` (existing dataclass ブロック付近、line ~165 の `ImageBlock` の直後)

- [ ] **Step 1: 追加箇所を確認**

Run: `Grep` で `@dataclass` と `class ImageBlock` の位置を特定

期待: `ImageBlock` と `TableBlock` が連続して定義されている。その後に追加する。

- [ ] **Step 2: データクラスを追加**

`class ImageBlock` の定義が終わった直後 (空行を挟んで) に以下を挿入:

```python
@dataclass
class PageFigureRegion:
    """preserve-image-layout: ndlocr_cli で検出した図版領域"""
    page_num: int              # 0-origin
    y: float                   # PDF 座標 (top)
    y_end: float
    x: float
    x_end: float
    image_path: str            # 保存済み画像の相対パス


@dataclass
class PageTableRegion:
    """preserve-image-layout: ndlocr_cli で検出した表組領域"""
    page_num: int
    y: float
    y_end: float
    x: float
    x_end: float
    markdown_table: Optional[str]       # Claude API 変換結果 (成功時)
    fallback_image_path: Optional[str]  # Claude API 不可時の画像パス
```

- [ ] **Step 3: 構文チェック**

Run:
```
python -c "import pdf2md; from pdf2md import PageFigureRegion, PageTableRegion; print('OK')"
```

Expected: `OK` と表示される。エラーなら import 順序や typo を修正。

- [ ] **Step 4: コミット**

```bash
git add pdf2md.py
git commit -m "feat(preserve-image-layout): PageFigureRegion/PageTableRegion データクラス追加"
```

---

## Task 2: `_extract_layout_regions_via_ndlocr` メソッドを追加

**Files:**
- Modify: `pdf2md.py` `AdvancedPDFConverter` クラス内 (`_convert_with_page_ocr` の直前、line ~1381 付近)

- [ ] **Step 1: 新規メソッドを追加**

`AdvancedPDFConverter` クラス内に以下のメソッドを追加:

```python
def _extract_layout_regions_via_ndlocr(
    self, doc, images_dir: str, base_name: str
) -> Tuple[Dict[int, List['PageFigureRegion']], Dict[int, List['PageTableRegion']]]:
    """ndlocr_cli でページから図版/表組領域を検出し、切り抜いて保存する。

    図版は PNG として保存し PageFigureRegion を返す。
    表組は Claude API で Markdown 表に変換し、失敗時は画像としてフォールバック保存する。

    Returns:
        (figure_regions, table_regions) それぞれ {page_num: [region, ...]} 形式
    """
    figure_regions: Dict[int, List[PageFigureRegion]] = {}
    table_regions: Dict[int, List[PageTableRegion]] = {}

    if not self.ndlocr_inferrer:
        print("[preserve-layout] ndlocr_cli not available, skipping")
        return figure_regions, table_regions

    total_pages = len(doc)

    for page_num in range(total_pages):
        print(f"[preserve-layout] Processing page {page_num + 1}/{total_pages}")
        page = doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height

        try:
            pix = page.get_pixmap(dpi=300)
            page_image_bytes = pix.tobytes("png")
        except Exception as e:
            print(f"[preserve-layout] Page {page_num + 1}: rendering failed: {e}")
            continue

        from PIL import Image as _PILImage
        try:
            _img = _PILImage.open(io.BytesIO(page_image_bytes))
            img_width = _img.width
            img_height = _img.height
        except Exception as e:
            print(f"[preserve-layout] Page {page_num + 1}: image decode failed: {e}")
            continue

        if img_width == 0 or img_height == 0:
            print(f"[preserve-layout] Page {page_num + 1}: zero image dimensions, skipping")
            continue

        scale_x = page_width / img_width
        scale_y = page_height / img_height

        try:
            _, region_blocks = self._ocr_page_with_ndlocr(page_image_bytes, page_num)
        except Exception as e:
            print(f"[preserve-layout] Page {page_num + 1}: ndlocr inference failed: {e}")
            continue

        page_figures: List[PageFigureRegion] = []
        page_tables: List[PageTableRegion] = []
        fig_index = 0
        tbl_index = 0

        for region in region_blocks:
            r_type = region.get('type', '')
            r_w = region.get('width', 0)
            r_h = region.get('height', 0)
            if r_w < 50 or r_h < 50:
                continue

            cropped_bytes = self._crop_region_image(
                page_image_bytes, region, img_width, img_height
            )
            if not cropped_bytes:
                print(f"[preserve-layout] Page {page_num + 1}: crop failed for {r_type}")
                continue

            pdf_y = region['y'] * scale_y
            pdf_y_end = (region['y'] + r_h) * scale_y
            pdf_x = region['x'] * scale_x
            pdf_x_end = (region['x'] + r_w) * scale_x

            if r_type == '図版':
                fig_index += 1
                image_filename = f"{base_name}_p{page_num + 1}_fig{fig_index}.png"
                image_path = os.path.join(images_dir, image_filename)
                try:
                    with open(image_path, 'wb') as f:
                        f.write(cropped_bytes)
                except Exception as e:
                    print(f"[preserve-layout] Page {page_num + 1}: figure save failed: {e}")
                    continue
                rel_path = f"{base_name}_images/{image_filename}"
                page_figures.append(PageFigureRegion(
                    page_num=page_num,
                    y=pdf_y, y_end=pdf_y_end,
                    x=pdf_x, x_end=pdf_x_end,
                    image_path=rel_path,
                ))

            elif r_type == '表組':
                tbl_index += 1
                markdown_table: Optional[str] = None
                fallback_image_path: Optional[str] = None

                claude_ok = (
                    self.claude_analyzer
                    and not getattr(self.claude_analyzer, 'disabled', False)
                )
                if claude_ok:
                    try:
                        result = self.claude_analyzer.analyze_single_image(cropped_bytes)
                        if result and result.get('type') == 'table' and result.get('content'):
                            markdown_table = result['content']
                    except Exception as e:
                        print(f"[preserve-layout] Claude API table conversion failed: {e}")

                if markdown_table is None:
                    image_filename = f"{base_name}_p{page_num + 1}_tbl{tbl_index}.png"
                    image_path = os.path.join(images_dir, image_filename)
                    try:
                        with open(image_path, 'wb') as f:
                            f.write(cropped_bytes)
                        fallback_image_path = f"{base_name}_images/{image_filename}"
                    except Exception as e:
                        print(f"[preserve-layout] Page {page_num + 1}: table fallback save failed: {e}")
                        continue

                page_tables.append(PageTableRegion(
                    page_num=page_num,
                    y=pdf_y, y_end=pdf_y_end,
                    x=pdf_x, x_end=pdf_x_end,
                    markdown_table=markdown_table,
                    fallback_image_path=fallback_image_path,
                ))

        if page_figures:
            figure_regions[page_num] = page_figures
        if page_tables:
            table_regions[page_num] = page_tables
        print(f"[preserve-layout] Page {page_num + 1}: extracted {len(page_figures)} figures, {len(page_tables)} tables")

    return figure_regions, table_regions
```

- [ ] **Step 2: 構文チェック**

Run:
```
python -c "import pdf2md; from pdf2md import AdvancedPDFConverter; print(hasattr(AdvancedPDFConverter, '_extract_layout_regions_via_ndlocr'))"
```

Expected: `True`

- [ ] **Step 3: コミット**

```bash
git add pdf2md.py
git commit -m "feat(preserve-image-layout): 図版/表組領域抽出メソッドを追加"
```

---

## Task 3: `_postprocess_preserve_image_layout` メソッドを追加

**Files:**
- Modify: `pdf2md.py` `AdvancedPDFConverter` クラス内 (Task 2 で追加した `_extract_layout_regions_via_ndlocr` の直後)

- [ ] **Step 1: 新規メソッドを追加**

```python
def _postprocess_preserve_image_layout(
    self,
    md_text: str,
    figure_regions: Dict[int, List['PageFigureRegion']],
    table_regions: Dict[int, List['PageTableRegion']],
    page_text_positions: Optional[Dict[int, List[Dict]]] = None,
) -> str:
    """生成済み Markdown に図版/表組を Y 座標順でマージする。

    page_text_positions が供給されている場合(座標ありモード)は Y 座標の
    マージソートで挿入する。供給されていない場合(座標なしモード)は各
    ページ末尾に図版セクションを追加し、表はスキップする(重複回避)。
    """
    if not figure_regions and not table_regions:
        return md_text

    # ページ区切りで分割。先頭に <!-- Page N --> が入る既存の区切り規則に従う。
    page_marker_re = re.compile(r'^<!-- Page (\d+) -->\s*$', re.MULTILINE)
    matches = list(page_marker_re.finditer(md_text))

    if not matches:
        # ページマーカーが無い場合は末尾に全ページ分まとめて追加
        tail_lines = []
        for page_num in sorted(set(list(figure_regions.keys()) + list(table_regions.keys()))):
            tail_lines.append(f"\n<!-- Page {page_num + 1} figures (preserve-image-layout) -->\n")
            for fig in figure_regions.get(page_num, []):
                tail_lines.append(f"\n![Figure p{page_num + 1}]({fig.image_path})\n")
        return md_text + "\n".join(tail_lines)

    # 各ページのスライス (start, end, page_num) を作成
    page_slices: List[Tuple[int, int, int]] = []
    for i, m in enumerate(matches):
        page_num = int(m.group(1)) - 1  # 0-origin
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        page_slices.append((start, end, page_num))

    # 後ろから処理(文字列置換のインデックスずれ防止)
    new_md = md_text
    for start, end, page_num in reversed(page_slices):
        figs = figure_regions.get(page_num, [])
        tbls = table_regions.get(page_num, [])
        if not figs and not tbls:
            continue

        page_slice = new_md[start:end]

        if page_text_positions and page_num in page_text_positions and page_text_positions[page_num]:
            # 座標ありモード: Y 順にマージ挿入
            page_slice = self._insert_regions_by_y(
                page_slice, figs, tbls, page_text_positions[page_num]
            )
        else:
            # 座標なしモード: ページ末尾に図版のみ集約(表は既存 MD に任せる)
            if figs:
                tail = "\n\n## Figures\n"
                for i, fig in enumerate(figs, 1):
                    tail += f"\n![Figure {i}]({fig.image_path})\n"
                page_slice = page_slice.rstrip() + tail + "\n"

        new_md = new_md[:start] + page_slice + new_md[end:]

    return new_md


def _insert_regions_by_y(
    self,
    page_slice: str,
    figures: List['PageFigureRegion'],
    tables: List['PageTableRegion'],
    text_positions: List[Dict],
) -> str:
    """座標ありモードの Y 順マージ挿入ヘルパ。

    text_positions は [{'y': float, 'line_offset': int}, ...] の形式。
    line_offset は page_slice 内の行オフセット(行番号)。
    各 region について、y < region.y を満たす最大 y のエントリの行の直後に
    region を挿入する。
    """
    # regions を Y 順にソート
    regions: List[Tuple[float, str, object]] = []
    for fig in figures:
        regions.append((fig.y, 'figure', fig))
    for tbl in tables:
        regions.append((tbl.y, 'table', tbl))
    regions.sort(key=lambda r: r[0])

    lines = page_slice.split('\n')
    # text_positions を y でソート
    sorted_positions = sorted(text_positions, key=lambda p: p.get('y', 0))

    # 各 region について挿入先の行インデックスを決定
    insertions: List[Tuple[int, str]] = []  # (line_index_after, markdown_block)
    for region_y, kind, obj in regions:
        target_line = 0
        for pos in sorted_positions:
            if pos.get('y', 0) < region_y:
                target_line = pos.get('line_offset', 0)
            else:
                break
        if kind == 'figure':
            block = f"\n![Figure]({obj.image_path})\n"
        else:  # table
            if obj.markdown_table:
                block = f"\n{obj.markdown_table}\n"
            else:
                block = f"\n![Table]({obj.fallback_image_path})\n"
        insertions.append((target_line, block))

    # 後ろから挿入してインデックスずれを防ぐ
    insertions.sort(key=lambda x: x[0], reverse=True)
    for line_idx, block in insertions:
        insert_at = min(line_idx + 1, len(lines))
        lines.insert(insert_at, block)

    return '\n'.join(lines)
```

- [ ] **Step 2: `re` import を確認**

Run: `Grep` で `^import re` をファイル内検索。

期待: 既に import されている。無ければファイル先頭の import セクションに `import re` を追加。

- [ ] **Step 3: 構文チェック**

Run:
```
python -c "import pdf2md; from pdf2md import AdvancedPDFConverter; print(hasattr(AdvancedPDFConverter, '_postprocess_preserve_image_layout'))"
```

Expected: `True`

- [ ] **Step 4: コミット**

```bash
git add pdf2md.py
git commit -m "feat(preserve-image-layout): MD後処理マージメソッドを追加"
```

---

## Task 4: `convert_file` に `preserve_image_layout` 引数を追加し、座標なしモード (pymupdf4llm/legacy/markitdown/page_image) に配線

**Files:**
- Modify: `pdf2md.py` `convert_file()` (line ~1852), `_convert_with_pymupdf4llm` (line ~1691), `_convert_with_markitdown` (line ~1724)

- [ ] **Step 1: `convert_file` のシグネチャを変更**

`convert_file()` メソッドの引数定義を次のように変更:

```python
def convert_file(self, pdf_path: str, output_path: str = None,
                 extract_images: bool = True,
                 layout_mode: str = "auto",
                 dpi: int = 150,
                 enable_claude: bool = True,
                 preserve_image_layout: bool = False) -> Tuple[bool, str]:
```

docstring にも `preserve_image_layout` の説明を追加:

```python
    preserve_image_layout: 図版を切り抜いて元Y位置に挿入し表をClaude APIで
                          Markdown表化する (ndlocr_cli 必須)
```

- [ ] **Step 2: 座標なしモードで後処理を適用するヘルパー関数を追加**

`convert_file` メソッドの**先頭(try ブロックの中)**に以下のクロージャを定義:

```python
            # preserve-image-layout 後処理ヘルパー
            def _apply_preserve_image_layout_to_file(md_output_path: str):
                """座標なしモード用: 既存の書き出し済み .md を読み、後処理して書き戻す"""
                if not preserve_image_layout:
                    return
                if not self.ndlocr_inferrer:
                    print("[preserve-layout] ndlocr_cli not available, skipping")
                    return
                try:
                    _doc = fitz.open(pdf_path)
                except Exception as e:
                    print(f"[preserve-layout] cannot reopen doc: {e}")
                    return
                try:
                    fig_regions, tbl_regions = self._extract_layout_regions_via_ndlocr(
                        _doc, images_dir, base_name
                    )
                finally:
                    _doc.close()
                try:
                    with open(md_output_path, 'r', encoding='utf-8') as f:
                        current_md = f.read()
                    new_md = self._postprocess_preserve_image_layout(
                        current_md, fig_regions, tbl_regions,
                        page_text_positions=None,
                    )
                    with open(md_output_path, 'w', encoding='utf-8') as f:
                        f.write(new_md)
                    print(f"[preserve-layout] post-processed {md_output_path}")
                except Exception as e:
                    print(f"[preserve-layout] post-process failed: {e}")
```

- [ ] **Step 3: page_image モード分岐に後処理フックを追加**

`convert_file` 内の `if layout_mode == "page_image":` ブロックを次のように変更:

```python
            # page_imageモード: 各ページをPNG画像として出力
            if layout_mode == "page_image":
                result = self._convert_as_page_images(
                    pdf_path, output_path, base_name, images_dir, dpi
                )
                # page_image は各ページ全体が画像なので preserve-image-layout は NOP
                return result
```

(変更なしコメントを追加するのみ)

- [ ] **Step 4: markitdown モード分岐に後処理フックを追加**

```python
            # markitdownモード: MarkItDown（Microsoft製）を使用
            if layout_mode == "markitdown":
                result = self._convert_with_markitdown(
                    pdf_path, output_path, images_dir, extract_images
                )
                if result[0]:
                    _apply_preserve_image_layout_to_file(output_path)
                return result
```

- [ ] **Step 5: legacy モード分岐に後処理フックを追加**

```python
            # legacyモード: pymupdf4llmのみ使用
            if layout_mode == "legacy":
                if PYMUPDF4LLM_AVAILABLE:
                    result = self._convert_with_pymupdf4llm(
                        pdf_path, output_path, images_dir, extract_images
                    )
                    if result[0]:
                        _apply_preserve_image_layout_to_file(output_path)
                    return result
                # pymupdf4llmが使えない場合はフォールスルー
```

- [ ] **Step 6: auto の pymupdf4llm 成功パスに後処理フックを追加**

`elif layout_mode in ("auto", "legacy"):` ブロック内、`if result[0]:  # 成功した場合` の `_supplement_vector_drawings` 呼び出しの直後に以下を追加:

```python
                    if result[0]:  # 成功した場合
                        # pymupdf4llm成功後もベクター描画画像を補完
                        if extract_images:
                            self._supplement_vector_drawings(
                                doc_for_drawings, output_path, images_dir, base_name
                            )
                        doc_for_drawings.close()
                        # preserve-image-layout 後処理
                        _apply_preserve_image_layout_to_file(output_path)
                        return result
```

- [ ] **Step 7: 構文チェック**

Run:
```
python -c "import pdf2md; c = pdf2md.AdvancedPDFConverter(enable_ocr=False); import inspect; print('preserve_image_layout' in inspect.signature(c.convert_file).parameters)"
```

Expected: `True`

- [ ] **Step 8: コミット**

```bash
git add pdf2md.py
git commit -m "feat(preserve-image-layout): convert_fileに引数追加、座標なしモードに配線"
```

---

## Task 5: 標準パイプライン(precise/auto fallback)に `page_text_positions` 構築と配線を追加

**Files:**
- Modify: `pdf2md.py` `convert_file()` 内の標準パイプライン部分 (line ~1944-2001)

- [ ] **Step 1: 表抽出スキップ分岐を追加**

標準パイプラインのページループ内、`table_blocks, table_regions = self.table_extractor.extract_tables(...)` の箇所を次のように変更:

```python
            for page_num in range(len(doc)):
                page = doc[page_num]

                # preserve-image-layout 有効時は ndlocr+Claude に表を任せるため既存抽出をスキップ
                if preserve_image_layout and self.ndlocr_inferrer:
                    table_blocks = []
                    table_regions = []
                else:
                    # 表ブロック抽出（テキストより先に抽出して重複を避ける）
                    table_blocks, table_regions = self.table_extractor.extract_tables(page, page_num)
                all_blocks.extend(table_blocks)
```

- [ ] **Step 2: `page_text_positions` を構築するコードを Markdown 生成直後に追加**

`markdown_content = self._generate_markdown_obsidian(...)` または `markdown_content = self._generate_markdown(...)` の後に、次のコードブロックを追加:

```python
            # Markdown生成
            if use_obsidian_layout and page_layouts:
                markdown_content = self._generate_markdown_obsidian(
                    all_blocks, base_name, page_layouts
                )
            else:
                markdown_content = self._generate_markdown(all_blocks, base_name)

            # preserve-image-layout 後処理 (座標ありモード)
            if preserve_image_layout and self.ndlocr_inferrer:
                # 再度 doc を開く(既に doc.close() 済み)
                _doc = fitz.open(pdf_path)
                try:
                    fig_regions, tbl_regions = self._extract_layout_regions_via_ndlocr(
                        _doc, images_dir, base_name
                    )
                finally:
                    _doc.close()

                # all_blocks から text_block タイプのみ抽出して page_text_positions 構築
                page_text_positions: Dict[int, List[Dict]] = {}
                for blk in all_blocks:
                    if not hasattr(blk, 'page_num') or not hasattr(blk, 'y0'):
                        continue
                    # 画像ブロック/表ブロックは除外、テキストブロックのみ
                    if type(blk).__name__ != 'TextBlock':
                        continue
                    page_text_positions.setdefault(blk.page_num, []).append({
                        'y': float(blk.y0),
                        'line_offset': 0,  # 行単位の正確な追跡が難しいため 0 固定
                    })
                # 各ページでページ先頭からの出現順に line_offset を振り直す
                for pnum, positions in page_text_positions.items():
                    positions.sort(key=lambda p: p['y'])
                    for i, p in enumerate(positions):
                        p['line_offset'] = i

                markdown_content = self._postprocess_preserve_image_layout(
                    markdown_content, fig_regions, tbl_regions,
                    page_text_positions=page_text_positions,
                )
```

注: `line_offset` は行単位の追跡が難しいため、各ページ内の y 順インデックスとして使用し、`_insert_regions_by_y` 側ではこれを「何番目のテキストブロックの後に挿入するか」として解釈する。このため Task 3 の `_insert_regions_by_y` の挿入位置は **MD の実行順テキスト行に対応するインデックス**ではなく**相対順序**になる。実 PDF で位置がずれる場合は Task 9 の検証ループで調整する。

- [ ] **Step 3: `doc.close()` の順序確認**

`doc.close()` は既存コードで Markdown 生成前にも呼ばれているため、preserve-image-layout 後処理で再度 `fitz.open(pdf_path)` している。これは既存挙動を壊さない。

- [ ] **Step 4: 構文チェック**

Run:
```
python -c "import pdf2md; print('OK')"
```

Expected: `OK` (エラーなし)

- [ ] **Step 5: コミット**

```bash
git add pdf2md.py
git commit -m "feat(preserve-image-layout): 標準パイプラインに配線"
```

---

## Task 6: `_convert_with_page_ocr` に `preserve_image_layout` ガードを追加

**Files:**
- Modify: `pdf2md.py` `_convert_with_page_ocr` (line ~1381) および `convert_file()` の呼び出し箇所

- [ ] **Step 1: `_convert_with_page_ocr` のシグネチャを拡張**

現在のシグネチャ:
```python
def _convert_with_page_ocr(self, doc, output_path: str, base_name: str,
                            images_dir: str, extract_images: bool,
                            enable_claude: bool = True) -> Tuple[bool, str]:
```

に `preserve_image_layout: bool = False` を追加:

```python
def _convert_with_page_ocr(self, doc, output_path: str, base_name: str,
                            images_dir: str, extract_images: bool,
                            enable_claude: bool = True,
                            preserve_image_layout: bool = False) -> Tuple[bool, str]:
```

- [ ] **Step 2: `_convert_with_page_ocr` の末尾(md保存直前)に後処理フックを追加**

メソッド末尾の `with open(output_path, 'w', encoding='utf-8') as f:` の直前に追加:

```python
        # preserve-image-layout 後処理
        # _convert_with_page_ocr 内では ndlocr_cli がそのページで既に実行済みのため
        # 二重実行を避けつつ統一 API を使うため、ここでは追加処理は行わず既存挙動を維持
        if preserve_image_layout and not self.ndlocr_inferrer:
            print("[preserve-layout] ndlocr_cli not available, OCR mode already skipped preserve-image-layout enhancements")
```

注: ndlocr_cli を使う `_convert_with_page_ocr` は既に図版/表組を領域ベースで処理している。preserve_image_layout が指定された場合の追加変更は不要で、既存挙動が要件を満たす(ただしファイル命名が `img{M}` のままなので、完全な一貫性のためには将来的に `fig{M}`/`tbl{M}` に統一する検討が必要 — 今回のスコープ外)。

- [ ] **Step 3: `convert_file` から `_convert_with_page_ocr` への引数伝搬を更新**

```python
            if has_font_issue:
                # フォント問題がある場合はOCRベースの変換
                if self.enable_ocr and self.ocr_reader:
                    print("[INFO] Font encoding issues detected, using OCR-based conversion...")
                    result = self._convert_with_page_ocr(
                        doc, output_path, base_name, images_dir, extract_images,
                        enable_claude, preserve_image_layout=preserve_image_layout
                    )
                    doc.close()
                    return result
```

- [ ] **Step 4: 構文チェック**

Run:
```
python -c "import pdf2md; import inspect; c = pdf2md.AdvancedPDFConverter(enable_ocr=False); print('preserve_image_layout' in inspect.signature(c._convert_with_page_ocr).parameters)"
```

Expected: `True`

- [ ] **Step 5: コミット**

```bash
git add pdf2md.py
git commit -m "feat(preserve-image-layout): _convert_with_page_ocr に引数追加"
```

---

## Task 7: CLI `--preserve-image-layout` オプションを追加

**Files:**
- Modify: `pdf2md.py` `main()` 関数内の argparse ブロック (line ~3431)、および CLI 実行部 (line ~3500)

- [ ] **Step 1: argparse に引数追加**

`parser.add_argument("--no-claude", ...)` の直後に追加:

```python
    parser.add_argument("--preserve-image-layout", action="store_true",
                       help="画像・図形を切り抜いて元配置で保持、表はClaude APIでMD表化 (ndlocr_cli必須)")
```

- [ ] **Step 2: ヘルプ epilog に例を追加**

`epilog` 内の例リストに追加:

```python
  pdf2md.py --preserve-image-layout doc.pdf   図版を元配置で保持、表をAIでMD化
```

- [ ] **Step 3: CLI 実行部で convert_file 呼び出しに引数を渡す**

```python
            success, result = converter.convert_file(
                input_path, out_path,
                extract_images=extract_images,
                layout_mode=args.layout,
                dpi=args.dpi,
                enable_claude=enable_claude,
                preserve_image_layout=args.preserve_image_layout,
            )
```

同じ変更を `elif os.path.isdir(input_path):` ブロック内の convert_file 呼び出しにも適用:

```python
                success, result = converter.convert_file(
                    str(pdf_path), out_path,
                    extract_images=extract_images,
                    layout_mode=args.layout,
                    dpi=args.dpi,
                    enable_claude=enable_claude,
                    preserve_image_layout=args.preserve_image_layout,
                )
```

- [ ] **Step 4: ヘルプ出力確認**

Run:
```
python pdf2md.py --help 2>&1 | grep -i preserve
```

Expected: `--preserve-image-layout` を含む行が表示される

- [ ] **Step 5: コミット**

```bash
git add pdf2md.py
git commit -m "feat(preserve-image-layout): CLI --preserve-image-layout オプション追加"
```

---

## Task 8: GUI チェックボックスを追加

**Files:**
- Modify: `pdf2md.py` `PDF2MDGUI` クラス (option_frame 付近 line ~3225、`_convert_thread` line ~3372)

- [ ] **Step 1: option_frame にチェックボックスを追加**

`claude_cb` の直後(line ~3246)に以下を追加:

```python
        self.preserve_image_layout_var = tk.BooleanVar(value=False)
        preserve_cb = ttk.Checkbutton(option_frame, text="元配置で画像を保持",
                                     variable=self.preserve_image_layout_var)
        preserve_cb.pack(side="left", padx=5)
        if not NDLOCR_AVAILABLE:
            preserve_cb.configure(state="disabled")
```

- [ ] **Step 2: `_convert_thread` で値を取得し convert_file に渡す**

`enable_claude = self.enable_claude_var.get()` の直後に:

```python
        preserve_image_layout = self.preserve_image_layout_var.get()
```

そして convert_file 呼び出しを次のように修正:

```python
            # 変換実行
            success, message = self.converter.convert_file(
                filepath, out_path, extract_images=extract_images,
                layout_mode=layout_mode, enable_claude=enable_claude,
                preserve_image_layout=preserve_image_layout,
            )
```

- [ ] **Step 3: GUI 起動テスト**

Run: GUI を短時間起動して動作確認(インタラクティブなので目視):
```
python pdf2md.py
```

期待: アプリが起動し、変換オプション欄に「元配置で画像を保持」チェックボックスが表示される。`NDLOCR_AVAILABLE=True` なら有効、False なら disabled になる。すぐ閉じてよい。

- [ ] **Step 4: コミット**

```bash
git add pdf2md.py
git commit -m "feat(preserve-image-layout): GUI チェックボックスを追加"
```

---

## Task 9: 実 PDF によるラウンドトリップ検証と反復修正

**Files:**
- ユーザー指定の検証用 PDF(実装直前に提示される)
- 必要に応じて `pdf2md.py` の挿入ロジック、`_crop_region_image` パディング、`ClaudeDiagramAnalyzer` プロンプト

- [ ] **Step 1: ユーザーに検証用 PDF のパスを依頼**

実装者は次のようにユーザーに尋ねる:

> 実装が完了したので、検証用 PDF のパスを教えてください。`python pdf2md.py --layout precise --preserve-image-layout <path>` で変換を実行し、生成された MD を元 PDF と比較します。

- [ ] **Step 2: 前提条件の確認**

Run:
```
python -c "import pdf2md; print('NDLOCR_AVAILABLE:', pdf2md.NDLOCR_AVAILABLE); print('CLAUDE_API_AVAILABLE:', pdf2md.CLAUDE_API_AVAILABLE)"
```

Expected: 両方とも `True`。False の場合はそれぞれセットアップを確認。

- [ ] **Step 3: 変換実行**

Run:
```
python pdf2md.py --layout precise --preserve-image-layout <ユーザー指定PDF>
```

Expected: 標準出力に `[preserve-layout] Processing page ...` が各ページ分表示され、最後にエラーなく出力 MD パスが表示される。

- [ ] **Step 4: 生成 MD と元 PDF の比較(目視)**

- 生成された `{base}.md` を Obsidian または VS Code Markdown プレビュアーで開く
- 元 PDF を PDF ビューアーで並べて表示
- 仕様書の比較チェックリストに沿って確認:
  - [ ] 各ページの見出し階層が一致
  - [ ] 本文の段落順序が元 PDF の読み順と一致
  - [ ] 図版が元 PDF と同じ Y 位置(テキストの段落間)に表示される
  - [ ] 図版の切り抜き範囲に周辺テキストが混入していない
  - [ ] 表が Markdown 表として生成され、行数・列数が元 PDF と一致
  - [ ] 表の位置が元 PDF と同じ Y 位置に表示される
  - [ ] 抽出画像ファイルに想定通りのコンテンツが含まれている
  - [ ] ページ区切り `---` が正しく入っている
  - [ ] `{base}_images/` に `p{N}_fig{M}.png` / `p{N}_tbl{M}.png` が共存

- [ ] **Step 5: 不一致がある場合の修正ループ**

以下の対応表に従って修正し、修正後に Step 3-4 を再実行:

| 不一致内容 | 修正箇所 | 修正方法 |
|-----------|---------|---------|
| 図版位置が元 PDF よりずれる | `_insert_regions_by_y` | 挿入位置を「y < region.y の最大」から「最も近いエントリの前後」に変更 |
| 図版の切り抜きに周辺テキスト混入 | `_crop_region_image` | パディング ±5 を 0 または -5(内側) に変更 |
| 表内容が不正確 | `ClaudeDiagramAnalyzer` のプロンプト | 「表の行数と列数を厳密に保持してください」を明記 |
| 表位置ズレ | 図版と同様 `_insert_regions_by_y` | 同じ方法 |
| 本文と図版のテキスト重複 | `_postprocess_preserve_image_layout` | 図版 Y 範囲に属するテキスト行を MD から除外するフィルタを追加 |
| ページ単位でのズレ | `_extract_layout_regions_via_ndlocr` の座標変換 | `scale_x`/`scale_y` の計算式を再確認 |

修正ループを 3 周しても一致しない項目が残る場合は、ユーザーに報告し挿入ロジックを絶対座標 HTML 方式へ切り替える設計見直しを提案する。

- [ ] **Step 6: 副次的サニティチェック**

以下をすべて実行し、既存挙動が壊れていないことを確認:

```
python pdf2md.py --layout precise <PDF>            # オプション無しで既存挙動維持
python pdf2md.py --layout auto <PDF>               # auto で既存挙動
python pdf2md.py --layout legacy --preserve-image-layout <PDF>  # 座標なしモードで図版末尾集約
```

Expected: 3 つとも成功。オプション無し実行で生成される MD は preserve-image-layout フラグなしの既存バージョンと(画像セクション部分を除き)同一。

- [ ] **Step 7: 検証完了のコミット(修正があった場合)**

```bash
git add pdf2md.py
git commit -m "fix(preserve-image-layout): 検証フィードバックに基づく調整"
```

---

## Task 10: Wiki 更新

**Files:**
- Modify: `wiki/AdvancedPDFConverter.md`, `wiki/cli-interface.md`, `wiki/PDF2MDGUI.md`, `wiki/conversion-pipeline.md`, `wiki/layout-modes.md`, `wiki/log.md`, `wiki/overview.md`

プロジェクトルール: ソース変更時に wiki/ を必ず更新する(`MEMORY.md` / `feedback_wiki_maintenance.md`)。

- [ ] **Step 1: `wiki/AdvancedPDFConverter.md` 更新**

「OCR(v4.3)」セクションの下に「preserve-image-layout(v4.4)」セクションを追加:

```markdown
### preserve-image-layout (v4.4)

| メソッド | 用途 |
|---------|------|
| `_extract_layout_regions_via_ndlocr()` | ndlocr_cli で図版/表組領域を検出し切り抜く。図版はPNG保存、表組はClaude APIでMarkdown表化 |
| `_postprocess_preserve_image_layout()` | 生成済みMDに図版/表組をY座標順で挿入する後処理 |
| `_insert_regions_by_y()` | 座標ありモードでの Y 順マージ挿入ヘルパ |

`convert_file(preserve_image_layout=True)` で有効化。ndlocr_cli 必須。
```

- [ ] **Step 2: `wiki/cli-interface.md` 更新**

オプション表に追加:

```markdown
  --preserve-image-layout  図版を切り抜いて元Y位置に保持、表をClaude APIでMD表化 (ndlocr_cli必須)
```

- [ ] **Step 3: `wiki/PDF2MDGUI.md` 更新**

変換オプション一覧のコード図とステータス説明に以下を追記:

```
| [x] 画像を抽出  [x] OCR  [ ] Claude AI図表解析  [ ] 元配置で画像を保持 |
```

- [ ] **Step 4: `wiki/conversion-pipeline.md` 更新**

フロー図の末尾近く(書き出し直前)に以下を追記:

```
  └─ [preserve_image_layout=True かつ ndlocr_cli 利用可能]
       → _extract_layout_regions_via_ndlocr()
       → _postprocess_preserve_image_layout()
       各モードのMD生成後、書き出し前に後処理で図版/表を元Y位置に挿入
```

- [ ] **Step 5: `wiki/layout-modes.md` 更新**

「モード一覧」の後に注記を追加:

```markdown
## 全モード共通オプション: preserve-image-layout (v4.4)

`--preserve-image-layout` は layout mode とは独立したオプションで、
どのモードでも使用できる。ndlocr_cli で図版/表組領域を検出し、
図版は PNG として切り抜き、表組は Claude API で Markdown 表化、
元ページの Y 位置に挿入する。座標を持たないモード
(pymupdf4llm/legacy/markitdown)では図版のみページ末尾に集約。
ndlocr_cli 未導入時はオプション無効化。
```

- [ ] **Step 6: `wiki/log.md` 更新**

先頭に新しいエントリを追加:

```markdown
## [2026-04-10] update | preserve-image-layout 機能を追加

ndlocr_cli で検出した図版領域を切り抜いて PNG 保存し、表組は Claude API で
Markdown 表化する新機能 `--preserve-image-layout` を追加。全layout mode共通
オプションで、座標ありモード(precise/標準パイプライン/page_ocr)では Y座標順に
マージ挿入、座標なしモード(pymupdf4llm/legacy/markitdown)では図版のみページ
末尾に集約。

更新ページ: AdvancedPDFConverter(新メソッド追加), cli-interface(オプション追加),
PDF2MDGUI(チェックボックス追加), conversion-pipeline(後処理フック追記),
layout-modes(共通オプション節追加), overview(v4.4履歴追加)
```

- [ ] **Step 7: `wiki/overview.md` 更新**

frontmatter を `version: "4.4"` に更新。バージョン履歴に追加:

```markdown
| v4.4 | preserve-image-layout オプション: 図版切り抜き+Y位置保持、表のClaude API Markdown化 |
```

- [ ] **Step 8: Wiki 変更のコミット**

```bash
git add wiki/
git commit -m "docs(wiki): preserve-image-layout 機能をWikiに反映"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] `--preserve-image-layout` CLI オプション → Task 7
- [x] GUI チェックボックス + ndlocr 未導入時 disabled → Task 8
- [x] 全 layout mode 共通オプション → Task 4, 5, 6(各 mode に配線)
- [x] `PageFigureRegion`/`PageTableRegion` データクラス → Task 1
- [x] `_extract_layout_regions_via_ndlocr` → Task 2
- [x] `_postprocess_preserve_image_layout` → Task 3
- [x] 座標ありモードで `AdvancedTableExtractor` スキップ → Task 5 Step 1
- [x] 座標なしモードで図版のみ末尾集約 → Task 3 `_postprocess` 内、Task 4 配線
- [x] 画像命名規則 `p{N}_fig{M}.png` / `p{N}_tbl{M}.png` → Task 2
- [x] エラー処理(ndlocr 未導入/Claude 失敗/保存失敗) → Task 2, 3 各所
- [x] `[preserve-layout]` ログプリフィックス → Task 2, 3
- [x] 実 PDF ラウンドトリップ検証 → Task 9
- [x] 副次的サニティチェック(オプション無しで既存挙動) → Task 9 Step 6
- [x] Wiki 更新 → Task 10

**Placeholder scan:** 全タスクにコード本体あり、TBD/TODO なし。

**Type consistency:**
- `PageFigureRegion`/`PageTableRegion` の属性名(page_num, y, y_end, x, x_end, image_path / markdown_table / fallback_image_path)は Task 1-5 で一貫
- `_extract_layout_regions_via_ndlocr` の戻り値型 `Tuple[Dict[int, List[...]], Dict[int, List[...]]]` は Task 4, 5 の呼び出し箇所と一致
- `_postprocess_preserve_image_layout` の引数 `page_text_positions: Optional[Dict[int, List[Dict]]]` は Task 3 定義と Task 5 呼び出しで一致

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-10-preserve-image-layout.md`.**
