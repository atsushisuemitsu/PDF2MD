#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF to Markdown Converter (Advanced Version v3.0)
PDFファイルをMarkdown形式に変換するGUIツール

機能:
- 高精度テキスト抽出（位置情報付き）
- 高度な表構造認識（罫線なし表も対応）
- 見出し階層の自動認識
- リスト構造の階層保持
- 画像抽出と保存
- OCRによる図内テキスト認識
- 図表キャプション対応
- フローチャート・図形のテキスト抽出

使用ライブラリ: PyMuPDF, Pillow, easyocr/pytesseract
"""

import os
import sys
import re
import io
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import defaultdict

# PyMuPDF
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. Run: pip install PyMuPDF")

# Pillow
try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

# OCR - easyocrを優先（pytesseractよりインストールが簡単）
OCR_ENGINE = None
try:
    import easyocr
    OCR_ENGINE = "easyocr"
except ImportError:
    try:
        import pytesseract
        OCR_ENGINE = "pytesseract"
    except ImportError:
        pass

# ドラッグ&ドロップ対応（Windows）
try:
    import tkinterdnd2 as tkdnd
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False


@dataclass
class TextBlock:
    """テキストブロック（位置情報付き）"""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_num: int
    font_size: float = 12.0
    is_bold: bool = False
    block_type: str = "text"  # text, heading1-6, list, caption
    indent_level: int = 0
    font_name: str = ""


@dataclass
class ImageBlock:
    """画像ブロック（位置情報付き）"""
    image_data: bytes
    x0: float
    y0: float
    x1: float
    y1: float
    page_num: int
    image_index: int
    ocr_text: str = ""
    image_path: str = ""
    caption: str = ""
    figure_number: str = ""


@dataclass
class TableBlock:
    """表ブロック（位置情報付き）"""
    cells: List[List[str]]  # 2D array of cell contents
    x0: float
    y0: float
    x1: float
    y1: float
    page_num: int
    header_rows: int = 1
    caption: str = ""
    table_number: str = ""
    col_widths: List[int] = field(default_factory=list)


@dataclass
class ListItem:
    """リストアイテム"""
    text: str
    level: int = 0
    list_type: str = "bullet"  # bullet, numbered
    number: str = ""


class DocumentAnalyzer:
    """文書構造解析クラス"""

    def __init__(self):
        self.font_sizes = []
        self.heading_sizes = []
        self.page_width = 0
        self.page_height = 0

    def analyze_document_structure(self, doc) -> Dict:
        """文書全体の構造を分析"""
        font_size_counts = defaultdict(int)
        bold_sizes = set()

        for page_num in range(len(doc)):
            page = doc[page_num]
            if page_num == 0:
                self.page_width = page.rect.width
                self.page_height = page.rect.height

            text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            size = round(span.get("size", 12), 1)
                            text = span.get("text", "").strip()
                            if text and len(text) < 200:  # 短いテキスト＝見出し候補
                                font_size_counts[size] += 1
                                font_name = span.get("font", "").lower()
                                if "bold" in font_name or "heavy" in font_name:
                                    bold_sizes.add(size)

        # フォントサイズを頻度でソート
        sorted_sizes = sorted(font_size_counts.keys(), reverse=True)

        # 見出しサイズの推定（上位のサイズを見出しとして扱う）
        self.heading_sizes = self._estimate_heading_sizes(sorted_sizes, font_size_counts, bold_sizes)

        return {
            "font_sizes": sorted_sizes,
            "heading_sizes": self.heading_sizes,
            "bold_sizes": bold_sizes
        }

    def _estimate_heading_sizes(self, sorted_sizes: List[float],
                                 font_counts: Dict, bold_sizes: set) -> Dict[float, int]:
        """見出しサイズと階層レベルを推定"""
        heading_map = {}

        # 本文サイズを推定（最も多いサイズ）
        if font_counts:
            body_size = max(font_counts.keys(), key=lambda x: font_counts[x])
        else:
            body_size = 12.0

        # 本文より大きいサイズを見出し候補とする
        heading_candidates = [s for s in sorted_sizes if s > body_size]

        # 階層を割り当て
        level = 1
        for size in heading_candidates[:6]:  # 最大H6まで
            heading_map[size] = level
            level += 1

        # 本文サイズでボールドのものはH3-H4として扱う可能性
        if body_size in bold_sizes and body_size not in heading_map:
            heading_map[body_size] = min(level, 4)

        return heading_map

    def get_heading_level(self, font_size: float, is_bold: bool, text: str) -> Optional[int]:
        """テキストの見出しレベルを取得"""
        # 長すぎるテキストは見出しではない
        if len(text) > 150:
            return None

        # フォントサイズベースの判定
        if font_size in self.heading_sizes:
            return self.heading_sizes[font_size]

        # ボールドで短いテキストは見出しの可能性
        if is_bold and len(text) < 80:
            # 最も近い見出しサイズを探す
            for size, level in sorted(self.heading_sizes.items(), reverse=True):
                if font_size >= size * 0.95:  # 5%の許容範囲
                    return level
            return 4  # デフォルトでH4

        return None


class AdvancedTableExtractor:
    """高度な表抽出クラス"""

    def __init__(self):
        self.min_cols = 2
        self.min_rows = 2

    def extract_tables(self, page, page_num: int) -> Tuple[List[TableBlock], List[Tuple]]:
        """ページから表を抽出"""
        blocks = []
        table_regions = []

        try:
            # PyMuPDFの表検出機能を使用
            tables = page.find_tables()

            for table in tables:
                try:
                    cells = table.extract()
                    if not cells or len(cells) < self.min_rows:
                        continue

                    # 列数チェック
                    max_cols = max(len(row) for row in cells)
                    if max_cols < self.min_cols:
                        continue

                    bbox = table.bbox
                    x0, y0, x1, y1 = bbox
                    table_regions.append((x0, y0, x1, y1))

                    # セル内容のクリーンアップ
                    cleaned_cells = self._clean_cells(cells)

                    # ヘッダー行数を推定
                    header_rows = self._estimate_header_rows(cleaned_cells)

                    # 列幅を計算
                    col_widths = self._calculate_col_widths(cleaned_cells)

                    if cleaned_cells:
                        blocks.append(TableBlock(
                            cells=cleaned_cells,
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                            page_num=page_num,
                            header_rows=header_rows,
                            col_widths=col_widths
                        ))

                except Exception as e:
                    print(f"Warning: Failed to extract table: {e}")
                    continue

        except Exception as e:
            print(f"Warning: Table detection failed on page {page_num + 1}: {e}")

        # 表が見つからない場合、テキスト構造から表を検出
        if not blocks:
            text_tables = self._detect_text_tables(page, page_num)
            for tbl in text_tables:
                if not self._is_in_region((tbl.x0, tbl.y0, tbl.x1, tbl.y1), table_regions):
                    blocks.append(tbl)
                    table_regions.append((tbl.x0, tbl.y0, tbl.x1, tbl.y1))

        return blocks, table_regions

    def _clean_cells(self, cells: List[List]) -> List[List[str]]:
        """セル内容のクリーンアップ"""
        cleaned = []
        for row in cells:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # 改行をスペースに置換、空白の正規化
                    cell_text = str(cell).replace('\n', ' ').strip()
                    cell_text = re.sub(r'\s+', ' ', cell_text)
                    cleaned_row.append(cell_text)
            cleaned.append(cleaned_row)
        return cleaned

    def _estimate_header_rows(self, cells: List[List[str]]) -> int:
        """ヘッダー行数を推定"""
        if not cells or len(cells) < 2:
            return 1

        first_row = cells[0]
        second_row = cells[1] if len(cells) > 1 else []

        # 最初の行が短いテキスト（ラベル的）で、2行目が数値や長いテキストなら1行ヘッダー
        first_row_avg_len = sum(len(c) for c in first_row) / max(len(first_row), 1)
        second_row_avg_len = sum(len(c) for c in second_row) / max(len(second_row), 1)

        # 数値の割合
        def numeric_ratio(row):
            if not row:
                return 0
            numeric_count = sum(1 for c in row if re.match(r'^[\d,.\-+%]+$', c.strip()))
            return numeric_count / len(row)

        first_numeric = numeric_ratio(first_row)
        second_numeric = numeric_ratio(second_row)

        # 最初の行が非数値で、2行目が数値的ならヘッダー1行
        if first_numeric < 0.3 and second_numeric > 0.5:
            return 1

        # 両方とも非数値で、最初の行が短いならヘッダー1行
        if first_numeric < 0.3 and first_row_avg_len < second_row_avg_len:
            return 1

        return 1  # デフォルト

    def _calculate_col_widths(self, cells: List[List[str]]) -> List[int]:
        """各列の最大幅を計算"""
        if not cells:
            return []

        max_cols = max(len(row) for row in cells)
        col_widths = [0] * max_cols

        for row in cells:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        return col_widths

    def _detect_text_tables(self, page, page_num: int) -> List[TableBlock]:
        """テキスト構造から表を検出（罫線なし表）"""
        tables = []

        # テキストブロックを取得して位置でグルーピング
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        # 行ごとにテキストをグループ化
        lines_by_y = defaultdict(list)
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        y_key = round(bbox[1] / 5) * 5  # 5ptで丸める
                        lines_by_y[y_key].append({
                            "text": span.get("text", ""),
                            "x0": bbox[0],
                            "x1": bbox[2],
                            "y0": bbox[1],
                            "y1": bbox[3]
                        })

        # 同一Y座標に複数のテキストがある行を検出
        potential_table_rows = []
        for y, items in sorted(lines_by_y.items()):
            if len(items) >= 2:
                items.sort(key=lambda x: x["x0"])
                potential_table_rows.append((y, items))

        # 連続する行で列位置が揃っているものを表として検出
        if len(potential_table_rows) >= 2:
            table_data = self._group_aligned_rows(potential_table_rows)
            for tbl_rows, bbox in table_data:
                if len(tbl_rows) >= 2:
                    tables.append(TableBlock(
                        cells=tbl_rows,
                        x0=bbox[0],
                        y0=bbox[1],
                        x1=bbox[2],
                        y1=bbox[3],
                        page_num=page_num,
                        header_rows=1
                    ))

        return tables

    def _group_aligned_rows(self, rows: List) -> List[Tuple[List[List[str]], Tuple]]:
        """列位置が揃った行をグルーピング"""
        if not rows:
            return []

        tables = []
        current_table = []
        current_bbox = [float('inf'), float('inf'), 0, 0]
        prev_x_positions = None

        for y, items in rows:
            x_positions = [round(item["x0"] / 10) * 10 for item in items]

            # 前の行と列位置が概ね一致するか
            if prev_x_positions is not None:
                if len(x_positions) == len(prev_x_positions):
                    # 列数が同じで位置も近い場合は同じ表
                    row_data = [item["text"] for item in items]
                    current_table.append(row_data)
                    current_bbox[0] = min(current_bbox[0], min(item["x0"] for item in items))
                    current_bbox[1] = min(current_bbox[1], min(item["y0"] for item in items))
                    current_bbox[2] = max(current_bbox[2], max(item["x1"] for item in items))
                    current_bbox[3] = max(current_bbox[3], max(item["y1"] for item in items))
                else:
                    # 列数が変わったら新しい表
                    if len(current_table) >= 2:
                        tables.append((current_table, tuple(current_bbox)))
                    current_table = [[item["text"] for item in items]]
                    current_bbox = [
                        min(item["x0"] for item in items),
                        min(item["y0"] for item in items),
                        max(item["x1"] for item in items),
                        max(item["y1"] for item in items)
                    ]
            else:
                row_data = [item["text"] for item in items]
                current_table.append(row_data)
                current_bbox = [
                    min(item["x0"] for item in items),
                    min(item["y0"] for item in items),
                    max(item["x1"] for item in items),
                    max(item["y1"] for item in items)
                ]

            prev_x_positions = x_positions

        # 最後の表を追加
        if len(current_table) >= 2:
            tables.append((current_table, tuple(current_bbox)))

        return tables

    def _is_in_region(self, bbox: Tuple, regions: List[Tuple]) -> bool:
        """bboxが既存領域と重複するか"""
        x0, y0, x1, y1 = bbox
        for rx0, ry0, rx1, ry1 in regions:
            # 重複面積を計算
            overlap_x = max(0, min(x1, rx1) - max(x0, rx0))
            overlap_y = max(0, min(y1, ry1) - max(y0, ry0))
            overlap_area = overlap_x * overlap_y
            box_area = (x1 - x0) * (y1 - y0)
            if box_area > 0 and overlap_area / box_area > 0.3:
                return True
        return False


class ListDetector:
    """リスト構造検出クラス"""

    # リストマーカーパターン
    BULLET_PATTERNS = [
        r'^[\-\•\●\○\■\□\・\※\★\☆]\s*',
        r'^[→⇒▶►]\s*',
    ]

    NUMBERED_PATTERNS = [
        r'^(\d+)[\.\)）]\s*',
        r'^\((\d+)\)\s*',
        r'^([a-zA-Z])[\.\)）]\s*',
        r'^\(([a-zA-Z])\)\s*',
        r'^([ivxIVX]+)[\.\)）]\s*',
    ]

    def detect_list_items(self, text: str, x0: float, page_width: float) -> List[ListItem]:
        """テキストからリストアイテムを検出"""
        items = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # インデントレベルを推定
            indent_level = self._estimate_indent_level(x0, page_width)

            # 箇条書きチェック
            for pattern in self.BULLET_PATTERNS:
                match = re.match(pattern, line)
                if match:
                    content = line[match.end():].strip()
                    items.append(ListItem(
                        text=content,
                        level=indent_level,
                        list_type="bullet"
                    ))
                    break
            else:
                # 番号付きリストチェック
                for pattern in self.NUMBERED_PATTERNS:
                    match = re.match(pattern, line)
                    if match:
                        content = line[match.end():].strip()
                        items.append(ListItem(
                            text=content,
                            level=indent_level,
                            list_type="numbered",
                            number=match.group(1)
                        ))
                        break

        return items

    def _estimate_indent_level(self, x0: float, page_width: float) -> int:
        """X座標からインデントレベルを推定"""
        # ページ左端からの相対位置
        left_margin = page_width * 0.1  # 10%を左マージンと仮定
        indent_unit = 20  # 20ptを1インデントレベルと仮定

        if x0 < left_margin + indent_unit:
            return 0
        elif x0 < left_margin + indent_unit * 2:
            return 1
        elif x0 < left_margin + indent_unit * 3:
            return 2
        else:
            return 3


class CaptionDetector:
    """図表キャプション検出クラス"""

    FIGURE_PATTERNS = [
        r'^(図|Fig\.?|Figure)\s*(\d+[\-\.\d]*)',
        r'^(グラフ|Chart)\s*(\d+[\-\.\d]*)',
        r'^(写真|Photo)\s*(\d+[\-\.\d]*)',
    ]

    TABLE_PATTERNS = [
        r'^(表|Tab\.?|Table)\s*(\d+[\-\.\d]*)',
    ]

    def detect_caption(self, text: str, block_type: str = "any") -> Tuple[str, str, str]:
        """
        キャプションを検出

        Returns:
            (caption_type, number, caption_text)
        """
        text = text.strip()

        patterns = []
        if block_type in ["figure", "any"]:
            patterns.extend([(p, "figure") for p in self.FIGURE_PATTERNS])
        if block_type in ["table", "any"]:
            patterns.extend([(p, "table") for p in self.TABLE_PATTERNS])

        for pattern, cap_type in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                number = match.group(2)
                caption = text[match.end():].strip()
                # コロンやピリオドを除去
                caption = re.sub(r'^[\s\:\.：。\-]+', '', caption)
                return cap_type, number, caption

        return "", "", ""


class AdvancedPDFConverter:
    """高度なPDF to Markdown変換クラス"""

    def __init__(self, enable_ocr: bool = True, ocr_languages: List[str] = None):
        """
        Args:
            enable_ocr: OCRを有効にするか
            ocr_languages: OCR対象言語 ['ja', 'en']
        """
        self.enable_ocr = enable_ocr and OCR_ENGINE is not None
        self.ocr_languages = ocr_languages or ['ja', 'en']
        self.ocr_reader = None

        if self.enable_ocr and OCR_ENGINE == "easyocr":
            try:
                self.ocr_reader = easyocr.Reader(self.ocr_languages, gpu=False)
            except Exception as e:
                print(f"Warning: EasyOCR initialization failed: {e}")
                self.enable_ocr = False

        # 各種検出器
        self.doc_analyzer = DocumentAnalyzer()
        self.table_extractor = AdvancedTableExtractor()
        self.list_detector = ListDetector()
        self.caption_detector = CaptionDetector()

    def convert_file(self, pdf_path: str, output_path: str = None,
                     extract_images: bool = True) -> Tuple[bool, str]:
        """
        PDFファイルをMarkdownに変換

        Args:
            pdf_path: 入力PDFファイルパス
            output_path: 出力Markdownファイルパス
            extract_images: 画像を抽出するか

        Returns:
            (成功フラグ, メッセージまたはエラー内容)
        """
        if not PYMUPDF_AVAILABLE:
            return False, "PyMuPDF is not installed"

        try:
            if not os.path.exists(pdf_path):
                return False, f"ファイルが見つかりません: {pdf_path}"

            if not pdf_path.lower().endswith('.pdf'):
                return False, f"PDFファイルではありません: {pdf_path}"

            # 出力パス決定
            if output_path is None:
                output_path = os.path.splitext(pdf_path)[0] + '.md'

            # 画像出力フォルダ
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.dirname(output_path) or '.'
            images_dir = os.path.join(output_dir, f"{base_name}_images")

            if extract_images and not os.path.exists(images_dir):
                os.makedirs(images_dir)

            # PDF解析
            doc = fitz.open(pdf_path)

            # 文書構造分析（見出しサイズの推定など）
            self.doc_analyzer.analyze_document_structure(doc)

            all_blocks = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # 表ブロック抽出（テキストより先に抽出して重複を避ける）
                table_blocks, table_regions = self.table_extractor.extract_tables(page, page_num)
                all_blocks.extend(table_blocks)

                # テキストブロック抽出（表領域を除外）
                text_blocks = self._extract_text_blocks(page, page_num, table_regions)
                all_blocks.extend(text_blocks)

                # 画像ブロック抽出
                if extract_images:
                    image_blocks = self._extract_image_blocks(
                        page, page_num, images_dir, base_name
                    )
                    all_blocks.extend(image_blocks)

            doc.close()

            # キャプションと図表の紐付け
            all_blocks = self._associate_captions(all_blocks)

            # ブロックをソート（ページ順 → Y座標順 → X座標順）
            all_blocks.sort(key=lambda b: (b.page_num, b.y0, b.x0))

            # Markdown生成
            markdown_content = self._generate_markdown(all_blocks, base_name)

            # 保存
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            return True, output_path

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)

    def _is_in_table_region(self, bbox: Tuple, table_regions: List[Tuple]) -> bool:
        """テキストブロックが表領域内にあるかチェック"""
        x0, y0, x1, y1 = bbox
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2

        for tx0, ty0, tx1, ty1 in table_regions:
            # 中心点が表領域内にあるか
            if tx0 <= center_x <= tx1 and ty0 <= center_y <= ty1:
                return True
            # 重複面積の計算
            overlap_x = max(0, min(x1, tx1) - max(x0, tx0))
            overlap_y = max(0, min(y1, ty1) - max(y0, ty0))
            overlap_area = overlap_x * overlap_y
            block_area = (x1 - x0) * (y1 - y0)
            if block_area > 0 and overlap_area / block_area > 0.5:
                return True

        return False

    def _extract_text_blocks(self, page, page_num: int,
                             table_regions: List[Tuple] = None) -> List[TextBlock]:
        """ページからテキストブロックを抽出"""
        blocks = []
        table_regions = table_regions or []

        # テキストを辞書形式で取得（位置情報付き）
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # テキストブロック
                block_text = ""
                max_font_size = 0
                is_bold = False
                font_name = ""

                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                        font_size = span.get("size", 12)
                        if font_size > max_font_size:
                            max_font_size = font_size
                        fn = span.get("font", "").lower()
                        if "bold" in fn or "heavy" in fn:
                            is_bold = True
                        if not font_name:
                            font_name = fn
                    block_text += line_text + "\n"

                block_text = block_text.strip()
                if block_text:
                    bbox = block.get("bbox", [0, 0, 0, 0])

                    # 表領域内のテキストはスキップ
                    if table_regions and self._is_in_table_region(bbox, table_regions):
                        continue

                    # ブロックタイプの推定
                    block_type = self._detect_block_type(block_text, max_font_size, is_bold)

                    # インデントレベル推定
                    indent_level = self._estimate_indent_level(bbox[0], page.rect.width)

                    blocks.append(TextBlock(
                        text=block_text,
                        x0=bbox[0],
                        y0=bbox[1],
                        x1=bbox[2],
                        y1=bbox[3],
                        page_num=page_num,
                        font_size=max_font_size,
                        is_bold=is_bold,
                        block_type=block_type,
                        indent_level=indent_level,
                        font_name=font_name
                    ))

        return blocks

    def _detect_block_type(self, text: str, font_size: float, is_bold: bool) -> str:
        """ブロックタイプを推定"""
        lines = text.split('\n')
        first_line = lines[0].strip() if lines else ""

        # キャプション検出
        cap_type, _, _ = self.caption_detector.detect_caption(first_line)
        if cap_type:
            return "caption"

        # 見出し判定（文書分析に基づく）
        heading_level = self.doc_analyzer.get_heading_level(font_size, is_bold, first_line)
        if heading_level:
            return f"heading{heading_level}"

        # リスト判定
        list_patterns = [
            r'^[\-\•\●\○\■\□\・\※\★\☆→⇒▶►]\s',
            r'^\d+[\.\)）]\s',
            r'^\([a-zA-Z\d]+\)\s',
            r'^[a-zA-Z][\.\)）]\s'
        ]
        for pattern in list_patterns:
            if re.match(pattern, first_line):
                return "list"

        return "text"

    def _estimate_indent_level(self, x0: float, page_width: float) -> int:
        """X座標からインデントレベルを推定"""
        left_margin = page_width * 0.1
        indent_unit = 20

        relative_x = x0 - left_margin
        if relative_x < indent_unit:
            return 0
        else:
            return min(int(relative_x / indent_unit), 4)

    def _extract_image_blocks(self, page, page_num: int,
                              images_dir: str, base_name: str) -> List[ImageBlock]:
        """ページから画像ブロックを抽出"""
        blocks = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]

                # 画像データ取得
                base_image = page.parent.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image.get("ext", "png")

                # 小さすぎる画像はスキップ（アイコンなど）
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                if width < 50 or height < 50:
                    continue

                # 画像の位置を取得
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rect = img_rects[0]
                    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                else:
                    x0, y0, x1, y1 = 100, page_num * 100, 500, page_num * 100 + 200

                # 画像保存
                img_filename = f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
                img_path = os.path.join(images_dir, img_filename)

                with open(img_path, 'wb') as f:
                    f.write(image_bytes)

                # OCR実行
                ocr_text = ""
                if self.enable_ocr and self.ocr_reader:
                    ocr_text = self._perform_ocr(image_bytes)

                blocks.append(ImageBlock(
                    image_data=image_bytes,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    page_num=page_num,
                    image_index=img_index,
                    ocr_text=ocr_text,
                    image_path=f"{base_name}_images/{img_filename}"
                ))

            except Exception as e:
                print(f"Warning: Failed to extract image {img_index}: {e}")
                continue

        return blocks

    def _perform_ocr(self, image_bytes: bytes) -> str:
        """画像にOCRを実行"""
        try:
            if OCR_ENGINE == "easyocr" and self.ocr_reader:
                image = Image.open(io.BytesIO(image_bytes))
                results = self.ocr_reader.readtext(image)
                texts = [result[1] for result in results]
                return "\n".join(texts)

            elif OCR_ENGINE == "pytesseract":
                image = Image.open(io.BytesIO(image_bytes))
                text = pytesseract.image_to_string(image, lang='jpn+eng')
                return text.strip()

        except Exception as e:
            print(f"OCR error: {e}")

        return ""

    def _associate_captions(self, blocks: List) -> List:
        """キャプションと図表を紐付け"""
        # キャプションブロックを特定
        caption_blocks = []
        other_blocks = []

        for block in blocks:
            if isinstance(block, TextBlock) and block.block_type == "caption":
                caption_blocks.append(block)
            else:
                other_blocks.append(block)

        # 各キャプションを最も近い図または表に紐付け
        for cap in caption_blocks:
            cap_type, number, caption_text = self.caption_detector.detect_caption(cap.text)

            if cap_type == "figure":
                # 最も近い画像を探す
                closest_img = None
                min_dist = float('inf')
                for block in other_blocks:
                    if isinstance(block, ImageBlock) and block.page_num == cap.page_num:
                        dist = abs(block.y1 - cap.y0) if block.y1 < cap.y0 else abs(cap.y1 - block.y0)
                        if dist < min_dist:
                            min_dist = dist
                            closest_img = block
                if closest_img and min_dist < 100:  # 100pt以内
                    closest_img.figure_number = number
                    closest_img.caption = caption_text

            elif cap_type == "table":
                # 最も近い表を探す
                closest_tbl = None
                min_dist = float('inf')
                for block in other_blocks:
                    if isinstance(block, TableBlock) and block.page_num == cap.page_num:
                        dist = abs(block.y0 - cap.y1) if cap.y1 < block.y0 else abs(cap.y0 - block.y1)
                        if dist < min_dist:
                            min_dist = dist
                            closest_tbl = block
                if closest_tbl and min_dist < 100:
                    closest_tbl.table_number = number
                    closest_tbl.caption = caption_text

        # キャプションブロックは除外（図表に紐付けたので）
        return [b for b in blocks if not (isinstance(b, TextBlock) and b.block_type == "caption")]

    def _generate_markdown(self, blocks: List, base_name: str) -> str:
        """ブロックからMarkdownを生成"""
        md_lines = []
        current_page = -1
        prev_block_type = None

        for block in blocks:
            # ページ区切り
            if block.page_num != current_page:
                if current_page >= 0:
                    md_lines.append("\n---\n")
                current_page = block.page_num
                md_lines.append(f"\n<!-- Page {current_page + 1} -->\n")

            if isinstance(block, TextBlock):
                md_lines.append(self._format_text_block(block, prev_block_type))
                prev_block_type = block.block_type

            elif isinstance(block, ImageBlock):
                md_lines.append(self._format_image_block(block))
                prev_block_type = "image"

            elif isinstance(block, TableBlock):
                md_lines.append(self._format_table_block(block))
                prev_block_type = "table"

        return "\n".join(md_lines)

    def _format_text_block(self, block: TextBlock, prev_type: str = None) -> str:
        """テキストブロックをMarkdown形式に変換"""
        text = block.text.strip()

        # 見出し
        if block.block_type.startswith("heading"):
            level = int(block.block_type[-1]) if block.block_type[-1].isdigit() else 2
            level = min(level, 6)  # H6が最大
            return f"\n{'#' * level} {text}\n"

        # リスト
        elif block.block_type == "list":
            return self._format_list_block(block)

        # 通常のテキスト
        else:
            # 空行の追加（前のブロックが見出しでない場合）
            prefix = "\n" if prev_type and not prev_type.startswith("heading") else ""
            return f"{prefix}{text}\n"

    def _format_list_block(self, block: TextBlock) -> str:
        """リストブロックをMarkdown形式に変換"""
        lines = block.text.split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            indent = "  " * block.indent_level

            # 箇条書きマーカーの正規化
            for pattern in ListDetector.BULLET_PATTERNS:
                if re.match(pattern, line):
                    line = re.sub(pattern, '', line)
                    formatted_lines.append(f"{indent}- {line}")
                    break
            else:
                # 番号付きリストの正規化
                for pattern in ListDetector.NUMBERED_PATTERNS:
                    match = re.match(pattern, line)
                    if match:
                        content = line[match.end():]
                        num = match.group(1)
                        formatted_lines.append(f"{indent}{num}. {content}")
                        break
                else:
                    # マーカーなしの場合
                    formatted_lines.append(f"{indent}- {line}")

        return "\n".join(formatted_lines) + "\n"

    def _format_image_block(self, block: ImageBlock) -> str:
        """画像ブロックをMarkdown形式に変換"""
        # 図番号とキャプション
        if block.figure_number:
            alt_text = f"図{block.figure_number}"
            if block.caption:
                alt_text += f": {block.caption}"
        else:
            alt_text = f"図{block.page_num + 1}-{block.image_index + 1}"

        md = f"\n![{alt_text}]({block.image_path})\n"

        # キャプション（図の下に追加）
        if block.caption and block.figure_number:
            md += f"\n*図{block.figure_number}: {block.caption}*\n"

        # OCRテキストがあれば追加
        if block.ocr_text:
            ocr_lines = block.ocr_text.strip().split('\n')
            if ocr_lines:
                md += "\n<details>\n<summary>図内テキスト（OCR）</summary>\n\n"
                md += "```\n"
                md += "\n".join(ocr_lines)
                md += "\n```\n\n</details>\n"

        return md

    def _format_table_block(self, block: TableBlock) -> str:
        """表ブロックをMarkdown形式に変換"""
        if not block.cells or len(block.cells) == 0:
            return ""

        md_lines = []

        # 表番号とキャプション（表の上に追加）
        if block.table_number and block.caption:
            md_lines.append(f"\n**表{block.table_number}: {block.caption}**\n")
        elif block.table_number:
            md_lines.append(f"\n**表{block.table_number}**\n")

        md_lines.append("")

        # 列数を取得（最大列数）
        max_cols = max(len(row) for row in block.cells)

        # 列幅を計算（見やすさのため）
        col_widths = [3] * max_cols  # 最小幅3
        for row in block.cells:
            for i, cell in enumerate(row):
                if i < max_cols:
                    col_widths[i] = max(col_widths[i], len(cell))

        for row_idx, row in enumerate(block.cells):
            # 行の各セルを整形（列数を揃える）
            cells = list(row) + [""] * (max_cols - len(row))

            # セル内の|をエスケープ
            escaped_cells = [cell.replace("|", "\\|") for cell in cells]

            # パディングを追加して見やすく
            padded_cells = [
                cell.ljust(col_widths[i]) if i < len(col_widths) else cell
                for i, cell in enumerate(escaped_cells)
            ]

            md_lines.append("| " + " | ".join(padded_cells) + " |")

            # ヘッダー行の後にセパレータを追加
            if row_idx == block.header_rows - 1:
                separator_parts = ["-" * max(col_widths[i], 3) for i in range(max_cols)]
                md_lines.append("| " + " | ".join(separator_parts) + " |")

        md_lines.append("")
        return "\n".join(md_lines)


class PDF2MDGUI:
    """GUIアプリケーションクラス"""

    def __init__(self):
        # ウィンドウ作成
        if DND_AVAILABLE:
            self.root = tkdnd.Tk()
        else:
            self.root = tk.Tk()

        self.root.title("PDF to Markdown Converter v3.0 (Advanced)")
        self.root.geometry("750x550")
        self.root.minsize(650, 450)

        # 変換エンジン
        self.converter = AdvancedPDFConverter(enable_ocr=True)

        # ファイルリスト
        self.file_list = []

        # UI構築
        self._create_widgets()

        # ドラッグ&ドロップ設定
        if DND_AVAILABLE:
            self._setup_dnd()

    def _create_widgets(self):
        """UIウィジェット作成"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # タイトル
        title_label = ttk.Label(main_frame, text="PDF to Markdown Converter v3.0",
                               font=("", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))

        # 機能ステータス表示
        status_text = f"PyMuPDF: {'✓' if PYMUPDF_AVAILABLE else '✗'} | "
        status_text += f"OCR: {'✓ ' + OCR_ENGINE if OCR_ENGINE else '✗'}"
        status_label = ttk.Label(main_frame, text=status_text, foreground="gray")
        status_label.grid(row=0, column=0, sticky="e")

        # ボタンフレーム
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=1, column=0, sticky="ew", pady=5)

        ttk.Button(btn_frame, text="ファイル追加",
                  command=self._add_files).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="フォルダ追加",
                  command=self._add_folder).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="クリア",
                  command=self._clear_list).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="変換実行",
                  command=self._convert).pack(side="right", padx=2)

        # ファイルリスト
        list_frame = ttk.LabelFrame(main_frame, text="変換対象ファイル", padding="5")
        list_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        # Treeview
        columns = ("path", "status")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings",
                                 selectmode="extended")
        self.tree.heading("path", text="ファイルパス")
        self.tree.heading("status", text="状態")
        self.tree.column("path", width=550)
        self.tree.column("status", width=120)

        # スクロールバー
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical",
                                  command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # D&Dヒント
        if DND_AVAILABLE:
            hint_text = "PDFファイルをドラッグ&ドロップで追加できます"
        else:
            hint_text = "※ tkinterdnd2をインストールするとD&Dが使えます"

        hint_label = ttk.Label(list_frame, text=hint_text, foreground="gray")
        hint_label.grid(row=1, column=0, columnspan=2, pady=(5, 0))

        # オプションフレーム
        option_frame = ttk.LabelFrame(main_frame, text="変換オプション", padding="5")
        option_frame.grid(row=3, column=0, sticky="ew", pady=5)

        self.extract_images_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(option_frame, text="画像を抽出",
                       variable=self.extract_images_var).pack(side="left", padx=5)

        self.enable_ocr_var = tk.BooleanVar(value=OCR_ENGINE is not None)
        ocr_cb = ttk.Checkbutton(option_frame, text="OCR（図内テキスト認識）",
                                variable=self.enable_ocr_var)
        ocr_cb.pack(side="left", padx=5)
        if OCR_ENGINE is None:
            ocr_cb.configure(state="disabled")

        # 出力設定フレーム
        output_frame = ttk.LabelFrame(main_frame, text="出力設定", padding="5")
        output_frame.grid(row=4, column=0, sticky="ew", pady=5)
        output_frame.columnconfigure(1, weight=1)

        self.output_var = tk.StringVar(value="same")
        ttk.Radiobutton(output_frame, text="同じフォルダに出力",
                       variable=self.output_var, value="same").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(output_frame, text="指定フォルダに出力:",
                       variable=self.output_var, value="custom").grid(row=1, column=0, sticky="w")

        self.output_path = tk.StringVar()
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path, state="readonly")
        output_entry.grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(output_frame, text="参照",
                  command=self._select_output_folder).grid(row=1, column=2)

        # 進捗バー
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var,
                                            maximum=100)
        self.progress_bar.grid(row=5, column=0, sticky="ew", pady=5)

        # ステータスラベル
        self.status_var = tk.StringVar(value="準備完了")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=6, column=0, sticky="w")

    def _setup_dnd(self):
        """ドラッグ&ドロップ設定"""
        self.tree.drop_target_register(tkdnd.DND_FILES)
        self.tree.dnd_bind('<<Drop>>', self._on_drop)

    def _on_drop(self, event):
        """D&Dイベントハンドラ"""
        files = self.root.tk.splitlist(event.data)
        for f in files:
            f = f.strip('{}')
            if os.path.isfile(f) and f.lower().endswith('.pdf'):
                self._add_file_to_list(f)
            elif os.path.isdir(f):
                for pdf in Path(f).glob('*.pdf'):
                    self._add_file_to_list(str(pdf))

    def _add_file_to_list(self, filepath: str):
        """ファイルをリストに追加"""
        if filepath not in self.file_list:
            self.file_list.append(filepath)
            self.tree.insert("", "end", values=(filepath, "待機中"))

    def _add_files(self):
        """ファイル選択ダイアログ"""
        files = filedialog.askopenfilenames(
            title="PDFファイルを選択",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        for f in files:
            self._add_file_to_list(f)

    def _add_folder(self):
        """フォルダ選択ダイアログ"""
        folder = filedialog.askdirectory(title="フォルダを選択")
        if folder:
            for pdf in Path(folder).glob('*.pdf'):
                self._add_file_to_list(str(pdf))

    def _clear_list(self):
        """リストクリア"""
        self.file_list.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.progress_var.set(0)
        self.status_var.set("準備完了")

    def _select_output_folder(self):
        """出力フォルダ選択"""
        folder = filedialog.askdirectory(title="出力フォルダを選択")
        if folder:
            self.output_path.set(folder)
            self.output_var.set("custom")

    def _convert(self):
        """変換実行"""
        if not self.file_list:
            messagebox.showwarning("警告", "変換するファイルがありません")
            return

        if not PYMUPDF_AVAILABLE:
            messagebox.showerror("エラー",
                "PyMuPDFがインストールされていません。\n"
                "pip install PyMuPDF を実行してください。")
            return

        # 出力フォルダ確認
        output_folder = None
        if self.output_var.get() == "custom":
            output_folder = self.output_path.get()
            if not output_folder:
                messagebox.showwarning("警告", "出力フォルダを指定してください")
                return

        # バックグラウンドで変換
        thread = threading.Thread(target=self._convert_thread, args=(output_folder,))
        thread.daemon = True
        thread.start()

    def _convert_thread(self, output_folder):
        """変換処理（バックグラウンド）"""
        total = len(self.file_list)
        success_count = 0

        # 変換オプション更新
        self.converter.enable_ocr = self.enable_ocr_var.get() and OCR_ENGINE is not None
        extract_images = self.extract_images_var.get()

        for i, filepath in enumerate(self.file_list):
            # UI更新
            self.root.after(0, lambda p=(i/total)*100: self.progress_var.set(p))
            self.root.after(0, lambda f=filepath:
                          self.status_var.set(f"変換中: {os.path.basename(f)}"))

            # Treeview更新
            item = self.tree.get_children()[i]
            self.root.after(0, lambda it=item:
                          self.tree.set(it, "status", "変換中..."))

            # 出力パス決定
            if output_folder:
                out_path = os.path.join(output_folder,
                                       os.path.splitext(os.path.basename(filepath))[0] + '.md')
            else:
                out_path = None

            # 変換実行
            success, message = self.converter.convert_file(
                filepath, out_path, extract_images=extract_images
            )

            if success:
                success_count += 1
                status = "完了"
            else:
                status = f"エラー: {message[:20]}..."

            self.root.after(0, lambda it=item, s=status:
                          self.tree.set(it, "status", s))

        # 完了
        self.root.after(0, lambda: self.progress_var.set(100))
        self.root.after(0, lambda:
                       self.status_var.set(f"完了: {success_count}/{total} ファイル変換成功"))
        self.root.after(0, lambda:
                       messagebox.showinfo("完了",
                                          f"変換が完了しました\n成功: {success_count}/{total}"))

    def run(self):
        """アプリケーション起動"""
        self.root.mainloop()


def main():
    """メイン関数"""
    # コマンドライン引数がある場合はCLIモード
    if len(sys.argv) > 1:
        converter = AdvancedPDFConverter(enable_ocr=True)

        for arg in sys.argv[1:]:
            if os.path.isfile(arg) and arg.lower().endswith('.pdf'):
                print(f"Converting: {arg}")
                success, result = converter.convert_file(arg)
                if success:
                    print(f"  -> {result}")
                else:
                    print(f"  Error: {result}")
            elif os.path.isdir(arg):
                print(f"Converting folder: {arg}")
                for pdf_path in Path(arg).glob('*.pdf'):
                    print(f"  Converting: {pdf_path.name}")
                    success, result = converter.convert_file(str(pdf_path))
                    status = "OK" if success else f"Error: {result}"
                    print(f"    {status}")
    else:
        # GUIモード
        app = PDF2MDGUI()
        app.run()


if __name__ == "__main__":
    main()
