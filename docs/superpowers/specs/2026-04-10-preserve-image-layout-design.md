# 設計書: 元ページ配置を保持する画像抽出機能 (preserve-image-layout)

- **作成日**: 2026-04-10
- **対象ファイル**: `pdf2md.py`
- **ステータス**: ドラフト(ユーザーレビュー待ち)

## 目的

PDF ページから「画像・絵・図形・フローチャート」に該当する領域のみを切り抜いて画像ファイルとして保存し、生成される Markdown ファイルが元の PDF ページと**同じ配置構成**で表示されるようにする。文章はテキスト、表は AI による Markdown 表として出力する。

## スコープ

- 対象: 図版(画像・図・フローチャート・イラスト)、および表組
- 非対象: 数式、化学式、組織図などは別途処理せず本文扱い
- 前提: `ndlocr_cli` が利用可能であること。未導入環境ではオプション自体を無効化し、警告ログを出した上で既存挙動を維持する。

## 全体アーキテクチャ

### 新規 CLI オプション
```
--preserve-image-layout
```
`argparse` に追加。`convert_file(preserve_image_layout: bool = False)` として各 layout mode に伝搬。

### 新規 GUI チェックボックス
ラベル「元配置で画像を保持」を変換オプション群に追加。`NDLOCR_AVAILABLE=False` の場合は `state='disabled'`。

### 配置方針
- 全 layout mode (`auto` / `precise` / `page_image` / `legacy` / `markitdown`) 共通のオプションとして実装
- 処理は各 mode の MD 本体生成**後**、ファイル書き出し**前**の共通後処理フックとして実行
- 座標を持つモード(precise / 標準パイプライン / `_convert_with_page_ocr`)では Y 座標順に図版と表を挿入
- 座標を持たないモード(pymupdf4llm / legacy / markitdown)では図版のみページ末尾に集約し、表は既存挙動を維持(重複回避のため AI 変換しない)
- `page_image` モードはページ全体が画像なので NOP

### 新規データクラス
```python
@dataclass
class PageFigureRegion:
    page_num: int           # 0-origin
    y: float                # PDF 座標 (top)
    y_end: float
    x: float
    x_end: float
    image_path: str         # {base}_images/{base}_p{N}_fig{M}.png

@dataclass
class PageTableRegion:
    page_num: int
    y: float
    y_end: float
    x: float
    x_end: float
    markdown_table: Optional[str]       # Claude API 変換結果
    fallback_image_path: Optional[str]  # Claude 不可時の画像パス
```

## コンポーネント詳細

### A. `_extract_layout_regions_via_ndlocr(doc, images_dir, base_name) -> Tuple[Dict, Dict]`
`AdvancedPDFConverter` の新規メソッド。戻り値は `(figure_regions, table_regions)`、どちらも `Dict[int, List[...]]`(ページ番号キー)。

処理フロー:
1. 各ページを 300 DPI でレンダリング (`pix.tobytes("png")`)
2. 既存の `_ocr_page_with_ndlocr()` を再利用して `(text_blocks, region_blocks)` を取得
3. `region_blocks` を `TYPE` でフィルタ:
   - `TYPE == '図版'` かつ width≥50, height≥50 → 図版処理へ
   - `TYPE == '表組'` かつ width≥50, height≥50 → 表組処理へ
   - それ以外はスキップ
4. 図版処理: `_crop_region_image()` → `{base}_images/{base}_p{N}_fig{M}.png` に保存、`PageFigureRegion` を作成
5. 表組処理:
   - `_crop_region_image()` で切り抜き(バイト列)
   - Claude API 利用可能時: `self.claude_analyzer.analyze_single_image(bytes)`
     - `result.type == 'table'` かつ `result.content` が非空 → `PageTableRegion.markdown_table = result.content`
     - それ以外 → fallback 画像として `{base}_p{N}_tbl{M}.png` に保存し `fallback_image_path` を設定
   - Claude API 不可時: 常に fallback 画像モード
6. ピクセル座標 → PDF 座標変換 (`scale_x = page_width / img_width` など) を適用して `PageFigureRegion` / `PageTableRegion` の座標を PDF 座標系で保持

### B. `_postprocess_preserve_image_layout(md_text, figure_regions, table_regions, page_text_positions, base_name, images_dir) -> str`
`AdvancedPDFConverter` の新規メソッド。

引数:
- `md_text`: 生成済みの Markdown 全文(ページ区切り `---` 含む)
- `figure_regions`, `table_regions`: 上記 A の出力
- `page_text_positions`: `Optional[Dict[int, List[Dict]]]`。座標ありモードのみ供給。各エントリは `{'y': float, 'md_line_index': int, 'text_snippet': str}`

処理:
1. `md_text` をページ単位に分割(既存のページ区切り規則 `<!-- Page N -->` / `---` を使用)
2. 各ページで `figure_regions[page] + table_regions[page]` を Y 座標でソート
3. 座標ありモード(`page_text_positions` が供給されている):
   - 各領域 `region` について、`page_text_positions[page]` のうち `y < region.y` を満たす最大 `y` のエントリの `md_line_index` の直後に挿入
   - 挿入内容:
     - Figure → `\n![Figure {M}]({image_path})\n`
     - Table with markdown_table → `\n{markdown_table}\n`
     - Table without markdown_table → `\n![Table {M}]({fallback_image_path})\n`
   - 挿入後のインデックスずれを考慮し、オフセットを加算しながら処理
4. 座標なしモード(`page_text_positions is None`):
   - 各ページ末尾に `\n## Figures\n` セクションを追加し、`figure_regions[page]` を順に `![](...)` で並べる
   - 表組は**スキップ**(既存 MD 本文に含まれているため重複回避)
5. ページを再結合して `md_text` として返す

### C. 各 layout mode からのフック

| モード | page_text_positions 生成 | 表抽出スキップ | 後処理適用 |
|--------|--------------------------|----------------|-----------|
| precise / 標準パイプライン | `text_blocks` から構築 | `AdvancedTableExtractor` をスキップ | Yes |
| `_convert_with_page_ocr` | 内部 `text_blocks` から構築 | 既存 ndlocr 経路でガード | Yes(強化) |
| pymupdf4llm (auto/legacy) | `None` | - | Yes(図版のみ末尾集約) |
| markitdown | `None` | - | Yes(図版のみ末尾集約) |
| page_image | - | - | No(NOP) |

### D. オプション伝搬経路
```
CLI argparse (--preserve-image-layout)
  → main() → convert_file(..., preserve_image_layout=True)
  → AdvancedPDFConverter.convert_file(..., preserve_image_layout)
  → 各 layout mode 分岐で True の場合:
       既存の MD 生成ロジックを呼び出した後、
       _extract_layout_regions_via_ndlocr() → _postprocess_preserve_image_layout()
       → ファイル書き出し

GUI: self.preserve_image_var = tk.BooleanVar()
  → 変換実行時に preserve_image_layout=self.preserve_image_var.get()
  → NDLOCR_AVAILABLE=False の時は cb.configure(state='disabled')
```

### E. 画像ファイル命名規則
- 図版: `{base}_p{page_num+1}_fig{fig_index}.png`
- 表組の fallback 画像: `{base}_p{page_num+1}_tbl{tbl_index}.png`

`fig_index` と `tbl_index` はそれぞれページ内で 1 から独立にカウント。既存の `{base}_p{N}_img{M}.*`(埋め込みラスター画像/ベクター描画)とは接頭辞が異なるため衝突しない。

## データフロー

```
convert_file(pdf, layout_mode, preserve_image_layout=True)
  │
  ├─ doc = fitz.open(pdf)
  ├─ [layout_mode 分岐で MD 本体生成]
  │    ├─ precise/標準パイプライン
  │    │    if preserve_image_layout and ndlocr_inferrer:
  │    │        AdvancedTableExtractor をスキップ
  │    │    _generate_markdown*() で md_text 生成
  │    │    page_text_positions を構築
  │    │
  │    ├─ _convert_with_page_ocr
  │    │    preserve_image_layout=True ガード追加
  │    │    page_text_positions を内部 text_blocks から構築
  │    │
  │    └─ pymupdf4llm / legacy / markitdown
  │         既存どおり md_text 生成、page_text_positions = None
  │
  ├─ if preserve_image_layout and self.ndlocr_inferrer:
  │     figure_regions, table_regions =
  │         _extract_layout_regions_via_ndlocr(doc, images_dir, base_name)
  │     md_text = _postprocess_preserve_image_layout(
  │         md_text, figure_regions, table_regions,
  │         page_text_positions, base_name, images_dir)
  │
  └─ Path(output).write_text(md_text, encoding='utf-8')
```

## エラー処理

| ケース | 挙動 |
|--------|------|
| `preserve_image_layout=True` かつ `ndlocr_inferrer is None` | warning ログ `[preserve-layout] ndlocr_cli not available, skipping`、後処理スキップ |
| ndlocr 推論中に例外 | ページ単位で try/except、空リストで継続、error ログ |
| `_crop_region_image` 失敗 | 当該領域スキップ、warning ログ |
| Claude API 表変換中に `AuthenticationError` | `claude_analyzer.disabled = True`(既存挙動)、以降全表は fallback 画像モード |
| Claude API 表変換中にその他例外 / timeout | 当該表のみ fallback 画像モード、warning ログ |
| Claude API 応答が `type != 'table'` または `content` 空 | 当該表のみ fallback 画像モード |
| 画像保存失敗 | 例外を main に伝播、変換全体を失敗扱い(既存 `_extract_raster_images` と同じ方針) |
| `page_text_positions` が空のページ | 座標ありモードでも座標なしモード同様にページ末尾集約にフォールバック |
| ピクセル→PDF 座標変換時のゼロ除算 (`img_width==0`) | 該当ページスキップ、error ログ |

### ログプリフィックス
`[preserve-layout]` で統一。主要ログ:
- `[preserve-layout] ndlocr_cli not available, skipping`
- `[preserve-layout] Processing page {N}/{total}`
- `[preserve-layout] Page {N}: extracted {M} figures, {K} tables`
- `[preserve-layout] Page {N}: table crop failed, using fallback image`
- `[preserve-layout] Claude API table conversion failed: {err}`

## テスト / 検証

本プロジェクトにはユニットテストフレームワークが無いため、**実 PDF によるラウンドトリップ検証**を主軸とする。

### 前提条件確認
- ndlocr_cli が `D:\Github\ndlocr_cli` に存在し、起動ログで `NDLOCR_AVAILABLE=True` を確認
- `ANTHROPIC_API_KEY` が `.env` から読み込まれ `CLAUDE_API_AVAILABLE=True` を確認

### 検証手順
1. ユーザー指定の検証用 PDF(パスは実装直前に提示)に対して以下を実行:
   ```
   python pdf2md.py --layout precise --preserve-image-layout <test.pdf>
   ```
2. 生成された `{base}.md` を Markdown プレビュアー(Obsidian または VS Code)で表示
3. 元 PDF を PDF ビューアーで並べて表示し、目視比較

### 比較チェックリスト
- [ ] 各ページの見出し階層が一致
- [ ] 本文の段落順序が元 PDF の読み順と一致
- [ ] 図版が元 PDF と同じ Y 位置(テキストの段落間)に表示される
- [ ] 図版の切り抜き範囲に周辺テキストが混入していない
- [ ] 表が Markdown 表として生成され、行数・列数が元 PDF と一致
- [ ] 表の位置が元 PDF と同じ Y 位置に表示される
- [ ] 抽出画像ファイルに想定通りのコンテンツが含まれている(目視)
- [ ] ページ区切り `---` が正しく入っている
- [ ] `{base}_images/` 配下に `p{N}_fig{M}.png` と既存の `p{N}_img{M}.*` が共存(衝突なし)

### 不一致時の修正ループ
- **図版位置のズレ** → `_postprocess_preserve_image_layout` の Y 近傍挿入ロジックを「最大 Y」から「Y ± tol 範囲」などに調整
- **図版の切り抜き範囲ズレ** → `_crop_region_image` のパディング量(現在 ±5px)を調整
- **表の内容ミス** → `ClaudeDiagramAnalyzer` のプロンプトに「表の行列構造を厳密に保持」を追加
- **表位置のズレ** → 図版と同じ挿入ロジックなので同じ方法で調整
- **図版と本文の重複** → 既存 `exclusion_regions` 方式を適用、該当 Y 範囲のテキストを MD から除去
- **ページ単位でのズレ** → `scale_x`/`scale_y` の計算を再確認

### 副次的サニティチェック
- `--preserve-image-layout` 無しで同じ PDF を変換 → 既存挙動が維持されていること(後方互換)
- ndlocr_cli を一時無効化して実行 → warning ログ + 既存挙動にフォールバック
- Claude API キーを外して実行 → 全表が fallback 画像モードになる

### 検証終了条件
上記チェックリストの全項目がパスし、ユーザーが「元の PDF と同じ見た目になった」と承認した時点で完了。修正ループを 3 周しても一致しない項目が残る場合は、挿入ロジックを座標マージ方式から絶対座標 HTML 方式へ切り替える設計見直しを検討する。

### 非対象(YAGNI)
- 自動回帰テストスクリプト
- ピクセル差分比較(PDF→画像→MD プレビュー→画像比較)
- CI 統合

## Wiki 更新対象

実装完了後、以下の Wiki ページを更新する(プロジェクトルール: ソース変更時に wiki/ を必ず更新):
- `wiki/AdvancedPDFConverter.md` — `_extract_layout_regions_via_ndlocr` / `_postprocess_preserve_image_layout` を新セクションで追加
- `wiki/cli-interface.md` — `--preserve-image-layout` オプションを追加
- `wiki/PDF2MDGUI.md` — 「元配置で画像を保持」チェックボックスを追加
- `wiki/conversion-pipeline.md` — 後処理フックの位置を図示
- `wiki/layout-modes.md` — 本機能は全モード共通オプションである旨を追記
- `wiki/log.md` — 更新ログエントリ追加
- `wiki/overview.md` — バージョン履歴に v4.4 として追加

## 非対象

- 数式・化学式・組織図の構造化変換(Claude API 経由にせよ画像にせよ、本機能では扱わない)
- 画像のテキスト回り込み (CSS float) — Markdown の制約上実現しない
- PDF との完全ピクセル一致(座標ありモードでのみ近似再現、座標なしモードではページ末尾集約)
