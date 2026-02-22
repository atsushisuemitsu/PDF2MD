#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF to Markdown Converter (Layout-Aware Version v4.0)
PDFファイルをMarkdown形式に変換するGUI/CLIツール

機能:
- 高精度テキスト抽出（位置情報付き）
- 高度な表構造認識（罫線なし表も対応）
- 見出し階層の自動認識
- リスト構造の階層保持
- 画像抽出と保存（サイズ情報保持）
- OCRによる図内テキスト認識（コントラスト強化・信頼度フィルタ付き）
- 図表キャプション対応
- 段組み（2/3段）自動検出・再現
- 画像フロート配置（テキストとの横並び）
- Obsidian互換レイアウト（CSS/HTML埋め込み）
- ページ画像モード（各ページをPNG出力）
- argparse CLIインターフェース
- Windows右クリックメニュー統合

使用ライブラリ: PyMuPDF, Pillow, easyocr/pytesseract
"""

import os
import sys
import re
import io
import base64
import json
import argparse
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

# PyMuPDF4LLM (レイアウト保持変換)
PYMUPDF4LLM_AVAILABLE = False
try:
    # レイアウト機能を有効化（pymupdf4llmより先にインポート）
    try:
        import pymupdf.layout
        LAYOUT_AVAILABLE = True
    except ImportError:
        LAYOUT_AVAILABLE = False

    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    LAYOUT_AVAILABLE = False

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

# Claude API（図表・フローチャートAI解析）
CLAUDE_API_AVAILABLE = False
try:
    import anthropic
    CLAUDE_API_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY"))
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
    color: int = 0  # テキスト色（RGB整数値）
    column_index: int = 0  # 段組みでのカラムインデックス


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
    width_px: int = 0  # 画像の実ピクセル幅
    height_px: int = 0  # 画像の実ピクセル高さ
    display_width: float = 0.0  # PDF上の表示幅（pt）
    display_height: float = 0.0  # PDF上の表示高さ（pt）
    claude_analysis: str = ""   # Claude解析結果（MD表/PlantUML）
    analysis_type: str = ""     # table/flowchart/sequence_diagram/diagram等


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


@dataclass
class LayoutRegion:
    """レイアウト領域"""
    x0: float
    y0: float
    x1: float
    y1: float
    region_type: str = "body"  # header, footer, body, sidebar
    column_index: int = 0
    blocks: List = field(default_factory=list)


@dataclass
class PageLayout:
    """ページレイアウト情報"""
    page_width: float = 0.0
    page_height: float = 0.0
    num_columns: int = 1
    column_boundaries: List[float] = field(default_factory=list)  # カラム境界のX座標
    header_y: float = 0.0  # ヘッダー領域の下端Y座標
    footer_y: float = 0.0  # フッター領域の上端Y座標
    regions: List[LayoutRegion] = field(default_factory=list)


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


class LayoutAnalyzer:
    """段組み・レイアウト検出クラス"""

    def __init__(self):
        self.header_ratio = 0.08  # ページ上部8%をヘッダー領域
        self.footer_ratio = 0.08  # ページ下部8%をフッター領域

    def analyze_page_layout(self, page, text_blocks: List[TextBlock]) -> PageLayout:
        """ページのレイアウト構造を分析"""
        page_width = page.rect.width
        page_height = page.rect.height

        layout = PageLayout(
            page_width=page_width,
            page_height=page_height,
            header_y=page_height * self.header_ratio,
            footer_y=page_height * (1 - self.footer_ratio),
        )

        # ボディ領域のテキストブロックのみ使用
        body_blocks = [
            b for b in text_blocks
            if layout.header_y < b.y0 < layout.footer_y
        ]

        if not body_blocks:
            layout.num_columns = 1
            layout.column_boundaries = [0, page_width]
            return layout

        # カラム数を検出
        num_cols, boundaries = self._detect_columns(body_blocks, page_width)
        layout.num_columns = num_cols
        layout.column_boundaries = boundaries

        # ブロックにカラムインデックスを割り当て
        self._assign_column_indices(text_blocks, layout)

        return layout

    def _detect_columns(self, blocks: List[TextBlock],
                        page_width: float) -> Tuple[int, List[float]]:
        """テキストブロックのX座標分布からカラム数を検出"""
        if not blocks:
            return 1, [0, page_width]

        # 各ブロックの中心X座標を収集
        x_centers = [(b.x0 + b.x1) / 2 for b in blocks]
        x_starts = [b.x0 for b in blocks]

        # ヒストグラムでX座標分布を分析
        margin = page_width * 0.1
        content_width = page_width - 2 * margin
        bin_width = content_width / 20  # 20分割

        bins = defaultdict(int)
        for x in x_starts:
            if margin < x < page_width - margin:
                bin_idx = int((x - margin) / bin_width)
                bins[bin_idx] += 1

        if not bins:
            return 1, [0, page_width]

        # ギャップ（テキストが無い領域）を検出してカラム境界を推定
        sorted_bins = sorted(bins.keys())
        if not sorted_bins:
            return 1, [0, page_width]

        # X座標の開始位置で2つのクラスタに分かれるか確認
        # ブロックのx0をソートして大きなギャップを探す
        unique_x0 = sorted(set(round(b.x0, 0) for b in blocks))
        if len(unique_x0) < 2:
            return 1, [0, page_width]

        # 隣接するx0間の最大ギャップを探す
        max_gap = 0
        gap_pos = 0
        for i in range(len(unique_x0) - 1):
            gap = unique_x0[i + 1] - unique_x0[i]
            if gap > max_gap:
                max_gap = gap
                gap_pos = (unique_x0[i] + unique_x0[i + 1]) / 2

        # ギャップがコンテンツ幅の15%以上あれば2段組みと判定
        if max_gap > content_width * 0.15:
            # ギャップの左右にそれぞれ十分なブロックがあるか確認
            left_count = sum(1 for b in blocks if b.x0 < gap_pos)
            right_count = sum(1 for b in blocks if b.x0 >= gap_pos)
            total = len(blocks)

            if left_count >= total * 0.2 and right_count >= total * 0.2:
                # さらに3段組みの可能性をチェック
                left_blocks = [b for b in blocks if b.x0 < gap_pos]
                right_blocks = [b for b in blocks if b.x0 >= gap_pos]

                # 右側ブロック内でさらにギャップがあるか
                right_x0 = sorted(set(round(b.x0, 0) for b in right_blocks))
                max_gap2 = 0
                gap_pos2 = 0
                for i in range(len(right_x0) - 1):
                    gap2 = right_x0[i + 1] - right_x0[i]
                    if gap2 > max_gap2:
                        max_gap2 = gap2
                        gap_pos2 = (right_x0[i] + right_x0[i + 1]) / 2

                if max_gap2 > content_width * 0.12:
                    r_left = sum(1 for b in right_blocks if b.x0 < gap_pos2)
                    r_right = sum(1 for b in right_blocks if b.x0 >= gap_pos2)
                    if r_left >= len(right_blocks) * 0.2 and r_right >= len(right_blocks) * 0.2:
                        return 3, [0, gap_pos, gap_pos2, page_width]

                return 2, [0, gap_pos, page_width]

        return 1, [0, page_width]

    def _assign_column_indices(self, blocks: List[TextBlock],
                               layout: PageLayout) -> None:
        """ブロックにカラムインデックスを割り当て"""
        for block in blocks:
            center_x = (block.x0 + block.x1) / 2
            col_idx = 0
            for i in range(len(layout.column_boundaries) - 1):
                if layout.column_boundaries[i] <= center_x < layout.column_boundaries[i + 1]:
                    col_idx = i
                    break
            block.column_index = col_idx

    def is_header_or_footer(self, block, layout: PageLayout) -> str:
        """ブロックがヘッダーまたはフッターかを判定"""
        if block.y0 < layout.header_y:
            return "header"
        if block.y1 > layout.footer_y:
            return "footer"
        return "body"


class ClaudeDiagramAnalyzer:
    """Claude APIを使用した図表・フローチャート解析クラス"""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.model = "claude-sonnet-4-20250514"
        self.disabled = False  # 認証エラー時に自動無効化

        # APIキーの有効性を検証
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=16,
                messages=[{"role": "user", "content": "test"}],
            )
            print("[Claude API] API key verified successfully")
        except anthropic.AuthenticationError as e:
            print(f"[Claude API] ERROR: APIキーが無効です: {e}")
            print("[Claude API] ANTHROPIC_API_KEY環境変数に正しいキーを設定してください")
            print("[Claude API] https://console.anthropic.com/ でキーを取得できます")
            self.disabled = True
        except Exception:
            # 認証以外のエラー（レート制限等）はキー自体は有効
            pass

    def analyze_page(self, page_image_bytes: bytes, page_height: float, page_width: float = 0) -> list:
        """ページ画像全体から図表・フローチャートを検出・構造化"""
        if self.disabled:
            return []
        try:
            image_b64 = base64.standard_b64encode(page_image_bytes).decode("utf-8")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": self._build_page_prompt(),
                        },
                    ],
                }],
            )
            return self._parse_page_response(response.content[0].text, page_height)
        except anthropic.AuthenticationError:
            print("[Claude API] 認証エラー: 以降のClaude解析を無効化します")
            self.disabled = True
            return []
        except Exception as e:
            print(f"[Claude API] Page analysis error: {e}")
            return []

    def analyze_single_image(self, image_bytes: bytes) -> Optional[dict]:
        """個別画像を解析して図表・フローチャートを構造化"""
        if self.disabled:
            return None
        try:
            # 画像フォーマット判定
            media_type = "image/png"
            if image_bytes[:2] == b'\xff\xd8':
                media_type = "image/jpeg"

            image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": self._build_image_prompt(),
                        },
                    ],
                }],
            )
            return self._parse_image_response(response.content[0].text)
        except anthropic.AuthenticationError:
            print("[Claude API] 認証エラー: 以降のClaude解析を無効化します")
            self.disabled = True
            return None
        except Exception as e:
            print(f"[Claude API] Image analysis error: {e}")
            return None

    def _build_page_prompt(self) -> str:
        return """このPDFページの画像を解析し、以下の視覚的要素を全て検出してください:
- 表（罫線あり・なし両方）
- フローチャート
- シーケンス図
- 状態遷移図
- 組織図・ツリー図
- その他の構造図（ブロック図、ネットワーク図等）

【重要】各要素の位置を正確に返してください。
- y_start_ratio: 要素の上端位置（ページ上端=0.0、下端=1.0）
- y_end_ratio: 要素の下端位置
この範囲内のOCRテキストは除去されるため、正確な範囲指定が必須です。

変換ルール:
- 表 → Markdown表形式（|ヘッダー|...|で記述、セル内テキストを正確に読み取る）
- フローチャート → PlantUML形式（@startuml/@enduml）
- シーケンス図 → PlantUML形式
- 状態遷移図 → PlantUML形式
- 組織図 → PlantUML形式
- その他の構造図 → PlantUML形式またはテキスト構造説明

JSON配列で返してください（図表が無い場合は空配列[]）:
[{
  "type": "table",
  "y_start_ratio": 0.30,
  "y_end_ratio": 0.55,
  "content": "| ヘッダー1 | ヘッダー2 |\\n|---|---|\\n| セル1 | セル2 |",
  "caption": "表1 サンプル表"
}]

typeの値: table, flowchart, sequence_diagram, state_diagram, org_chart, diagram

注意:
- y_start_ratio/y_end_ratioは図表の外枠全体を包含する範囲にすること
- キャプション（「図1」「表2」等）も範囲に含める
- 日本語テキストはそのまま保持
- 表のセル内テキストは正確に読み取ること
- 通常のテキスト段落や見出しは検出しない（図表・図形のみ）
- JSONのみを出力し、他の説明文は含めないでください"""

    def _build_image_prompt(self) -> str:
        return """この画像を解析してください。
画像が以下のいずれかに該当する場合、構造化された形式に変換してください:
- 表 → Markdown表形式
- フローチャート → PlantUML形式
- シーケンス図 → PlantUML形式
- 状態遷移図 → PlantUML形式
- 組織図 → PlantUML形式
- その他の図形 → テキストによる構造説明

該当する場合、以下のJSON形式で返してください:
{"type": "flowchart", "content": "@startuml\\n...\\n@enduml"}

typeの値: table, flowchart, sequence_diagram, state_diagram, org_chart, diagram

写真・イラスト・装飾画像など構造化できない画像の場合はnullを返してください。
JSONまたはnullのみを出力し、他の説明文は含めないでください"""

    def _parse_page_response(self, text: str, page_height: float) -> list:
        """ページ解析レスポンスをパース"""
        text = text.strip()
        # コードブロックで囲まれている場合を処理
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # JSON配列部分を抽出
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(data, list):
            return []

        results = []
        for item in data:
            if not isinstance(item, dict):
                continue
            elem_type = item.get("type", "diagram")
            content = item.get("content", "")
            caption = item.get("caption", "")
            if not content:
                continue

            # y_start_ratio / y_end_ratio（新形式）を優先
            y_start_ratio = item.get("y_start_ratio")
            y_end_ratio = item.get("y_end_ratio")

            if y_start_ratio is not None and y_end_ratio is not None:
                y_start = y_start_ratio * page_height
                y_end = y_end_ratio * page_height
            else:
                # 旧形式 y_position_ratio からのフォールバック
                y_ratio = item.get("y_position_ratio", 0.5)
                y_start = y_ratio * page_height
                # 範囲不明のため高さの10%を仮の範囲とする
                y_end = y_start + page_height * 0.10

            results.append({
                "type": elem_type,
                "y_start": y_start,
                "y_end": y_end,
                "content": content,
                "caption": caption,
            })
        return results

    def _parse_image_response(self, text: str) -> Optional[dict]:
        """個別画像解析レスポンスをパース"""
        text = text.strip()
        if text.lower() == "null" or not text:
            return None

        # コードブロックで囲まれている場合を処理
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        if not isinstance(data, dict) or "content" not in data:
            return None

        return {
            "type": data.get("type", "diagram"),
            "content": data["content"],
        }


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
                import traceback
                traceback.print_exc()
                self.enable_ocr = False

        # 各種検出器
        self.doc_analyzer = DocumentAnalyzer()
        self.table_extractor = AdvancedTableExtractor()
        self.list_detector = ListDetector()
        self.caption_detector = CaptionDetector()
        self.layout_analyzer = LayoutAnalyzer()

        # Claude API図表解析
        self.claude_analyzer = None
        if CLAUDE_API_AVAILABLE:
            try:
                self.claude_analyzer = ClaudeDiagramAnalyzer()
            except Exception as e:
                print(f"Warning: Claude API initialization failed: {e}")

    def _check_font_encoding_issue(self, doc) -> bool:
        """PDFのフォントエンコーディング問題をチェック"""
        # 最初の数ページのテキストをサンプリング
        sample_text = ""
        for i in range(min(3, len(doc))):
            sample_text += doc[i].get_text()

        if len(sample_text) < 50:
            return False

        # 文字化けパターン検出（連続する置換文字、制御文字）
        replacement_chars = sum(1 for c in sample_text if c in '\ufffd\x00\x01\x02\x03')
        if len(sample_text) > 0 and replacement_chars / len(sample_text) > 0.1:
            print(f"[INFO] Using page-level OCR due to garbled text detected...")
            return True

        # CID/Identity-Hフォントの文字化け: 同じUnicode Private Use Areaの文字が多い
        pua_count = sum(1 for c in sample_text if 0xE000 <= ord(c) <= 0xF8FF)
        if len(sample_text) > 0 and pua_count / len(sample_text) > 0.15:
            print(f"[INFO] Using page-level OCR due to CID font encoding issues...")
            return True

        # 単一文字の繰り返しが多い場合はエンコーディング問題
        char_counts = {}
        for char in sample_text:
            if char.isalpha():
                char_counts[char] = char_counts.get(char, 0) + 1

        if char_counts:
            most_common_char = max(char_counts.keys(), key=lambda x: char_counts[x])
            ratio = char_counts[most_common_char] / len(sample_text)
            # 1文字が25%以上を占める場合は問題あり（閾値を30%→25%に改善）
            if ratio > 0.25:
                print(f"[INFO] Using page-level OCR due to font encoding issues...")
                return True
        return False

    def _convert_with_page_ocr(self, doc, output_path: str, base_name: str,
                                images_dir: str, extract_images: bool,
                                enable_claude: bool = True) -> Tuple[bool, str]:
        """ページ全体をOCRで変換（フォント問題があるPDF用）- レイアウト保持版"""
        import numpy as np

        md_lines = []
        total_pages = len(doc)
        image_count = 0

        for page_num in range(total_pages):
            print(f"[OCR] Processing page {page_num + 1}/{total_pages}...")
            page = doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height

            # ページ区切り
            if page_num > 0:
                md_lines.append("\n---\n")
            md_lines.append(f"\n<!-- Page {page_num + 1} -->\n")

            # 1. 画像を先に抽出（位置情報付き）
            image_blocks = []
            if extract_images:
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        # 画像の位置を取得
                        img_rects = page.get_image_rects(xref)
                        if img_rects:
                            img_rect = img_rects[0]
                            # 画像データを抽出
                            base_image = doc.extract_image(xref)
                            if base_image:
                                image_bytes = base_image["image"]
                                image_ext = base_image.get("ext", "png")

                                # 小さすぎる画像はスキップ（アイコン等）
                                if img_rect.width < 50 or img_rect.height < 50:
                                    continue

                                # 画像を保存
                                image_count += 1
                                image_filename = f"{base_name}_p{page_num + 1}_img{image_count}.{image_ext}"
                                image_path = os.path.join(images_dir, image_filename)

                                with open(image_path, 'wb') as f:
                                    f.write(image_bytes)

                                # 画像ブロックを記録（Y座標でソート用）
                                rel_path = f"{base_name}_images/{image_filename}"
                                disp_w = int(img_rect.width * 1.333)
                                disp_h = int(img_rect.height * 1.333)

                                # Claude APIで埋め込み画像を解析
                                img_md = f'\n<img src="{rel_path}" alt="Image {image_count}" width="{disp_w}" height="{disp_h}">\n'
                                claude_result = None
                                px_w = base_image.get("width", 0)
                                px_h = base_image.get("height", 0)
                                if enable_claude and self.claude_analyzer and px_w > 100 and px_h > 100:
                                    claude_result = self.claude_analyzer.analyze_single_image(image_bytes)
                                    if claude_result:
                                        c_type = claude_result.get("type", "")
                                        c_content = claude_result.get("content", "")
                                        print(f"[Claude API] Embedded image p{page_num+1}_img{image_count}: {c_type}")
                                        if c_type == "table":
                                            img_md += f"\n{c_content}\n"
                                        elif c_type in ("flowchart", "sequence_diagram", "state_diagram", "org_chart"):
                                            img_md += f"\n```plantuml\n{c_content}\n```\n"
                                        elif c_content:
                                            img_md += f"\n{c_content}\n"

                                image_blocks.append({
                                    'y': img_rect.y0,
                                    'x': img_rect.x0,
                                    'y_end': img_rect.y1,
                                    'width': img_rect.width,
                                    'height': img_rect.height,
                                    'markdown': img_md,
                                    'has_claude': claude_result is not None,
                                })
                    except Exception as e:
                        pass

            # 2. ページを画像としてレンダリングしてOCR
            pix = page.get_pixmap(dpi=300)  # 高解像度でOCR精度向上
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            # スケール係数（OCR座標→PDF座標変換用）
            scale_x = page_width / img.width
            scale_y = page_height / img.height

            # 大きい場合はリサイズ
            max_size = 3500
            resize_ratio = 1.0
            if max(img.size) > max_size:
                resize_ratio = max_size / max(img.size)
                new_size = (int(img.width * resize_ratio), int(img.height * resize_ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # RGBに変換
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # OCR実行（位置情報付き）
            img_np = np.array(img)
            results = self.ocr_reader.readtext(img_np)

            # 3. テキストブロックを構築（位置情報付き）
            text_blocks = []
            for result in results:
                bbox, text, confidence = result
                if not text.strip():
                    continue
                # 信頼度30%未満のテキストはフィルタ
                if confidence < 0.3:
                    continue

                # バウンディングボックスからPDF座標を計算
                # bbox = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                ocr_x0 = min(p[0] for p in bbox) / resize_ratio
                ocr_y0 = min(p[1] for p in bbox) / resize_ratio
                ocr_x1 = max(p[0] for p in bbox) / resize_ratio
                ocr_y1 = max(p[1] for p in bbox) / resize_ratio

                # PDF座標に変換
                pdf_x0 = ocr_x0 * scale_x
                pdf_y0 = ocr_y0 * scale_y
                pdf_x1 = ocr_x1 * scale_x
                pdf_y1 = ocr_y1 * scale_y

                text_blocks.append({
                    'y': pdf_y0,
                    'x': pdf_x0,
                    'x1': pdf_x1,
                    'y1': pdf_y1,
                    'text': text.strip(),
                    'height': pdf_y1 - pdf_y0
                })

            # 3.5. Claude APIで図表・フローチャートを解析
            claude_elements = []
            if enable_claude and self.claude_analyzer:
                print(f"[Claude API] Analyzing page {page_num + 1} for diagrams...")
                claude_elements = self.claude_analyzer.analyze_page(
                    pix.tobytes("png"), page_height, page_width
                )
                if claude_elements:
                    print(f"[Claude API] Found {len(claude_elements)} diagram(s) on page {page_num + 1}")

            # 3.6. 図表領域内のOCRテキストを除去（重複防止）
            # Claude検出領域 + Claude解析済み埋め込み画像の領域を収集
            exclusion_regions = []
            for ce in claude_elements:
                exclusion_regions.append((ce['y_start'], ce['y_end']))
            for ib in image_blocks:
                if ib.get('has_claude'):
                    exclusion_regions.append((ib['y'], ib.get('y_end', ib['y'] + ib['height'])))

            if exclusion_regions:
                filtered_text_blocks = []
                for tb in text_blocks:
                    tb_center_y = (tb['y'] + tb['y1']) / 2
                    in_excluded = False
                    for y_start, y_end in exclusion_regions:
                        if y_start <= tb_center_y <= y_end:
                            in_excluded = True
                            break
                    if not in_excluded:
                        filtered_text_blocks.append(tb)
                removed = len(text_blocks) - len(filtered_text_blocks)
                if removed > 0:
                    print(f"[Claude API] Removed {removed} OCR text blocks within diagram/image regions")
                text_blocks = filtered_text_blocks

            # 4. テキストと画像とClaude要素を位置でソートして結合
            all_elements = []
            for tb in text_blocks:
                all_elements.append(('text', tb['y'], tb['x'], tb))
            for ib in image_blocks:
                all_elements.append(('image', ib['y'], ib['x'], ib))
            for ce in claude_elements:
                # Claude要素はy_startの位置に挿入
                all_elements.append(('claude', ce['y_start'], 0, ce))

            # Y座標でソート、同じY座標ならX座標でソート
            all_elements.sort(key=lambda e: (e[1], e[2]))

            # テキストの中央値サイズを計算（見出し判定の基準）
            text_heights = [tb['height'] for tb in text_blocks if tb['height'] > 5]
            if text_heights:
                text_heights.sort()
                median_height = text_heights[len(text_heights) // 2]
            else:
                median_height = 15

            # 5. レイアウトを考慮したMarkdown生成
            line_buffer = []
            line_y = None
            line_threshold = median_height * 0.8  # 同じ行とみなすY座標の閾値

            for elem_type, y, x, elem in all_elements:
                if elem_type == 'image':
                    # 行バッファをフラッシュ
                    if line_buffer:
                        md_lines.append(' '.join(line_buffer))
                        line_buffer = []
                        line_y = None
                    md_lines.append(elem['markdown'])
                elif elem_type == 'claude':
                    # Claude解析要素（図表・フローチャート等）
                    if line_buffer:
                        md_lines.append(' '.join(line_buffer))
                        line_buffer = []
                        line_y = None
                    if elem.get('caption'):
                        md_lines.append(f"\n**{elem['caption']}**\n")
                    content = elem['content']
                    if elem['type'] == 'table':
                        md_lines.append(f"\n{content}\n")
                    elif elem['type'] in ('flowchart', 'sequence_diagram', 'state_diagram', 'org_chart'):
                        md_lines.append(f"\n```plantuml\n{content}\n```\n")
                    else:
                        md_lines.append(f"\n{content}\n")
                else:
                    # テキスト要素
                    text = elem['text']

                    # 段落判定（大きなY間隔）
                    if line_y is not None and abs(y - line_y) > line_threshold:
                        # 新しい行/段落
                        if line_buffer:
                            md_lines.append(' '.join(line_buffer))
                            line_buffer = []

                        # 大きな間隔は段落区切り
                        if abs(y - line_y) > line_threshold * 2:
                            md_lines.append("")

                    # 見出し判定（テキストサイズが中央値より1.5倍以上大きい場合）
                    is_heading = (
                        elem['height'] > median_height * 1.5 and
                        len(text) < 80 and
                        len(text) > 1 and
                        not text.startswith('・') and
                        not text.startswith('-')
                    )

                    if is_heading:
                        if line_buffer:
                            md_lines.append(' '.join(line_buffer))
                            line_buffer = []
                        if elem['height'] > median_height * 2:
                            md_lines.append(f"\n## {text}\n")
                        else:
                            md_lines.append(f"\n### {text}\n")
                        line_y = y
                        continue

                    line_buffer.append(text)
                    line_y = y

            # 残りのバッファをフラッシュ
            if line_buffer:
                md_lines.append(' '.join(line_buffer))

            md_lines.append("")

        # 保存
        markdown_content = "\n".join(md_lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        return True, output_path

    def _convert_with_pymupdf4llm(self, pdf_path: str, output_path: str,
                                   images_dir: str, extract_images: bool) -> Tuple[bool, str]:
        """PyMuPDF4LLMを使用したレイアウト保持変換"""
        if not PYMUPDF4LLM_AVAILABLE:
            return False, "pymupdf4llm is not available"

        try:
            print("[INFO] Converting with PyMuPDF4LLM (layout-aware)...")

            # 画像抽出オプション
            kwargs = {
                'write_images': extract_images,
                'image_path': images_dir,
                'image_format': 'png',
                'dpi': 150,
                'page_chunks': False,
            }

            # レイアウト機能が有効な場合はOCRも有効化
            if LAYOUT_AVAILABLE:
                kwargs['use_ocr'] = True
                kwargs['ocr_dpi'] = 300

            md_content = pymupdf4llm.to_markdown(pdf_path, **kwargs)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

            return True, output_path

        except Exception as e:
            return False, f"PyMuPDF4LLM conversion failed: {e}"

    def _render_page_image(self, page, page_num: int, images_dir: str,
                           base_name: str, dpi: int = 150) -> str:
        """ページ全体を高解像度PNGとしてレンダリング"""
        pix = page.get_pixmap(dpi=dpi)
        img_filename = f"{base_name}_page{page_num + 1}.png"
        img_path = os.path.join(images_dir, img_filename)
        pix.save(img_path)
        return f"{base_name}_images/{img_filename}"

    def convert_file(self, pdf_path: str, output_path: str = None,
                     extract_images: bool = True,
                     layout_mode: str = "auto",
                     dpi: int = 150,
                     enable_claude: bool = True) -> Tuple[bool, str]:
        """
        PDFファイルをMarkdownに変換

        Args:
            pdf_path: 入力PDFファイルパス
            output_path: 出力Markdownファイルパス
            extract_images: 画像を抽出するか
            layout_mode: レイアウトモード (auto/precise/page_image/legacy)
            dpi: 画像レンダリングDPI
            enable_claude: Claude APIによる図表解析を有効にするか

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

            # page_imageモード: 各ページをPNG画像として出力
            if layout_mode == "page_image":
                return self._convert_as_page_images(
                    pdf_path, output_path, base_name, images_dir, dpi
                )

            # legacyモード: pymupdf4llmのみ使用
            if layout_mode == "legacy":
                if PYMUPDF4LLM_AVAILABLE:
                    return self._convert_with_pymupdf4llm(
                        pdf_path, output_path, images_dir, extract_images
                    )
                # pymupdf4llmが使えない場合はフォールスルー

            # PDF解析
            doc = fitz.open(pdf_path)

            # フォントエンコーディング問題をチェック
            has_font_issue = self._check_font_encoding_issue(doc)

            if has_font_issue:
                # フォント問題がある場合はOCRベースの変換
                if self.enable_ocr and self.ocr_reader:
                    print("[INFO] Font encoding issues detected, using OCR-based conversion...")
                    result = self._convert_with_page_ocr(doc, output_path, base_name, images_dir, extract_images, enable_claude)
                    doc.close()
                    return result
                else:
                    print("[WARNING] Font encoding issues detected but OCR is not available")
            elif layout_mode in ("auto", "legacy"):
                # フォント問題がない場合かつauto/legacyならpymupdf4llmを試す
                if PYMUPDF4LLM_AVAILABLE:
                    doc_for_drawings = doc  # ベクター描画抽出用に保持
                    result = self._convert_with_pymupdf4llm(pdf_path, output_path, images_dir, extract_images)
                    if result[0]:  # 成功した場合
                        # pymupdf4llm成功後もベクター描画画像を補完
                        if extract_images:
                            self._supplement_vector_drawings(
                                doc_for_drawings, output_path, images_dir, base_name
                            )
                        doc_for_drawings.close()
                        return result
                    print(f"[WARNING] PyMuPDF4LLM failed, falling back to standard conversion: {result[1]}")
                    # doc はまだ開いている

            # preciseモード or autoモードのフォールバック: 精密レイアウト変換
            use_obsidian_layout = layout_mode in ("precise", "auto")

            # 文書構造分析（見出しサイズの推定など）
            self.doc_analyzer.analyze_document_structure(doc)

            all_blocks = []
            page_layouts = {}

            for page_num in range(len(doc)):
                page = doc[page_num]

                # 表ブロック抽出（テキストより先に抽出して重複を避ける）
                table_blocks, table_regions = self.table_extractor.extract_tables(page, page_num)
                all_blocks.extend(table_blocks)

                # テキストブロック抽出（表領域を除外）
                text_blocks = self._extract_text_blocks(page, page_num, table_regions)
                all_blocks.extend(text_blocks)

                # レイアウト分析（段組み検出）
                if use_obsidian_layout:
                    page_layout = self.layout_analyzer.analyze_page_layout(page, text_blocks)
                    page_layouts[page_num] = page_layout

                # 画像ブロック抽出（ラスター画像 + ベクター描画領域）
                if extract_images:
                    image_blocks = self._extract_image_blocks(
                        page, page_num, images_dir, base_name,
                        table_regions, enable_claude=enable_claude
                    )
                    all_blocks.extend(image_blocks)

            doc.close()

            # 描画画像領域と重複するテキストブロックを除外
            # （ベクター描画を画像化した際、内部テキストの重複を防ぐ）
            all_blocks = self._remove_text_in_drawing_images(all_blocks)

            # キャプションと図表の紐付け
            all_blocks = self._associate_captions(all_blocks)

            # ブロックをソート（ページ順、カラム検出対応）
            all_blocks = self._sort_blocks_with_columns(all_blocks)

            # Markdown生成
            if use_obsidian_layout and page_layouts:
                markdown_content = self._generate_markdown_obsidian(
                    all_blocks, base_name, page_layouts
                )
            else:
                markdown_content = self._generate_markdown(all_blocks, base_name)

            # 保存
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            return True, output_path

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)

    def _convert_as_page_images(self, pdf_path: str, output_path: str,
                                 base_name: str, images_dir: str,
                                 dpi: int = 150) -> Tuple[bool, str]:
        """各ページをPNG画像として出力するモード"""
        try:
            doc = fitz.open(pdf_path)
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)

            md_lines = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                print(f"[PAGE_IMAGE] Rendering page {page_num + 1}/{len(doc)}...")

                rel_path = self._render_page_image(
                    page, page_num, images_dir, base_name, dpi
                )
                w = int(page.rect.width * 1.333)
                h = int(page.rect.height * 1.333)

                if page_num > 0:
                    md_lines.append("\n---\n")
                md_lines.append(f"\n<!-- Page {page_num + 1} -->\n")
                md_lines.append(f'<img src="{rel_path}" alt="Page {page_num + 1}" width="{w}" height="{h}">\n')

            doc.close()

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(md_lines))

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
                              images_dir: str, base_name: str,
                              table_regions: List[Tuple] = None,
                              enable_claude: bool = True) -> List[ImageBlock]:
        """ページから画像ブロックを抽出（ラスター画像 + ベクター描画領域）"""
        blocks = []
        table_regions = table_regions or []
        img_counter = [0]  # mutable counter for shared indexing

        # 1. ラスター画像の抽出（従来方式）
        raster_blocks = self._extract_raster_images(
            page, page_num, images_dir, base_name, img_counter,
            enable_claude=enable_claude
        )
        blocks.extend(raster_blocks)

        # 2. ベクター描画領域の抽出（新規）
        drawing_blocks = self._extract_drawing_blocks(
            page, page_num, images_dir, base_name, table_regions, img_counter
        )
        blocks.extend(drawing_blocks)

        return blocks

    def _extract_raster_images(self, page, page_num: int,
                               images_dir: str, base_name: str,
                               img_counter: list,
                               enable_claude: bool = True) -> List[ImageBlock]:
        """ラスター画像（埋め込みビットマップ）を抽出"""
        blocks = []
        image_list = page.get_images(full=True)

        for img_info in image_list:
            try:
                xref = img_info[0]

                # 画像データ取得
                base_image = page.parent.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image.get("ext", "png")

                # 画像の実ピクセルサイズ
                px_width = base_image.get("width", 0)
                px_height = base_image.get("height", 0)

                # 小さすぎる画像はスキップ（アイコンなど）
                if px_width < 50 or px_height < 50:
                    continue

                # 画像の位置を取得（PDF上の表示サイズ）
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rect = img_rects[0]
                    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                else:
                    x0, y0, x1, y1 = 100, page_num * 100, 500, page_num * 100 + 200

                img_counter[0] += 1
                img_idx = img_counter[0]

                display_w = x1 - x0
                display_h = y1 - y0

                # 画像保存
                img_filename = f"page{page_num + 1}_img{img_idx}.{image_ext}"
                img_path = os.path.join(images_dir, img_filename)

                with open(img_path, 'wb') as f:
                    f.write(image_bytes)

                # OCR実行
                ocr_text = ""
                if self.enable_ocr and self.ocr_reader:
                    ocr_text = self._perform_ocr(image_bytes)

                # Claude APIで個別画像を解析（図表・フローの場合）
                claude_analysis = ""
                analysis_type = ""
                if enable_claude and self.claude_analyzer and px_width > 100 and px_height > 100:
                    result = self.claude_analyzer.analyze_single_image(image_bytes)
                    if result:
                        claude_analysis = result.get("content", "")
                        analysis_type = result.get("type", "")
                        print(f"[Claude API] Image p{page_num+1}_img{img_index+1}: detected {analysis_type}")

                blocks.append(ImageBlock(
                    image_data=image_bytes,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    page_num=page_num,
                    image_index=img_idx,
                    ocr_text=ocr_text,
                    image_path=f"{base_name}_images/{img_filename}",
                    width_px=px_width,
                    height_px=px_height,
                    display_width=display_w,
                    display_height=display_h,
                    claude_analysis=claude_analysis,
                    analysis_type=analysis_type,
                ))

            except Exception as e:
                print(f"Warning: Failed to extract raster image: {e}")
                continue

        return blocks

    def _extract_drawing_blocks(self, page, page_num: int,
                                images_dir: str, base_name: str,
                                table_regions: List[Tuple],
                                img_counter: list) -> List[ImageBlock]:
        """ベクター描画領域を検出し、画像としてレンダリング・抽出"""
        blocks = []

        drawings = page.get_drawings()
        if not drawings:
            return blocks

        page_width = page.rect.width
        page_height = page.rect.height
        page_area = page_width * page_height

        # 描画要素の矩形を収集
        draw_rects = []
        draw_infos = []
        for d in drawings:
            r = d.get("rect")
            if r and r.width > 0 and r.height > 0:
                draw_rects.append(r)
                draw_infos.append(d)
            elif r and (r.width > 0 or r.height > 0):
                # 線（幅または高さが0）も収集
                draw_rects.append(r)
                draw_infos.append(d)

        if not draw_rects:
            return blocks

        # Change Location-2026/02/16 - Increase clustering proximity for vector drawings
        # Original Code
        # clusters = self._cluster_drawing_rects(draw_rects, proximity=5)
        # Updated Code
        # proximity=5 was too small: connector pin diagrams etc. fragmented into
        # many tiny clusters that got filtered out. proximity=10 merges them correctly.
        clusters = self._cluster_drawing_rects(draw_rects, proximity=10)
        # Change Location-2026/02/16 - Increase clustering proximity for vector drawings

        for cluster_rect, member_indices in clusters:
            cw = cluster_rect.x1 - cluster_rect.x0
            ch = cluster_rect.y1 - cluster_rect.y0
            c_area = cw * ch

            # --- フィルタリング ---

            # 小さすぎるクラスタをスキップ（装飾要素やアイコン）
            if cw < 80 or ch < 30:
                continue

            # 単一の線（セパレータ等）をスキップ
            if len(member_indices) == 1:
                d = draw_infos[member_indices[0]]
                items = d.get("items", [])
                if len(items) == 1 and items[0][0] == "l":
                    continue

            # 単一の塗りつぶし矩形（コードブロック背景等）をスキップ
            # テキスト内容がメインの場合、画像化は不要
            if len(member_indices) == 1:
                d = draw_infos[member_indices[0]]
                items = d.get("items", [])
                fill = d.get("fill")
                # 塗りつぶし矩形（角丸含む）でアイテム数が少ない = 背景ボックス
                if fill is not None and len(items) <= 8:
                    # 矩形/角丸矩形アイテムのみかチェック
                    item_types = set(it[0] for it in items)
                    if item_types.issubset({"re", "c", "l", "qu"}):
                        continue

            # 表領域との重複チェック
            if self._overlaps_regions(cluster_rect, table_regions, threshold=0.5):
                continue

            # フルページ背景かチェック（章タイトルページ等）
            is_full_page = (c_area / page_area > 0.85)

            if is_full_page:
                # フルページ描画: ページ全体をレンダリング
                render_rect = page.rect
            else:
                # 部分描画: クラスタ領域に少しマージンを追加
                margin = 2
                render_rect = fitz.Rect(
                    max(0, cluster_rect.x0 - margin),
                    max(0, cluster_rect.y0 - margin),
                    min(page_width, cluster_rect.x1 + margin),
                    min(page_height, cluster_rect.y1 + margin)
                )

            try:
                # ベクター描画領域をPNG画像としてレンダリング
                # 解像度: 2倍（高品質）、フルページは1.5倍
                scale = 1.5 if is_full_page else 2.0
                mat = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat, clip=render_rect)
                image_bytes = pix.tobytes("png")

                img_counter[0] += 1
                img_idx = img_counter[0]

                img_filename = f"page{page_num + 1}_img{img_idx}.png"
                img_path = os.path.join(images_dir, img_filename)

                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                # OCR実行
                ocr_text = ""
                if self.enable_ocr and self.ocr_reader:
                    ocr_text = self._perform_ocr(image_bytes)

                blocks.append(ImageBlock(
                    image_data=image_bytes,
                    x0=render_rect.x0,
                    y0=render_rect.y0,
                    x1=render_rect.x1,
                    y1=render_rect.y1,
                    page_num=page_num,
                    image_index=img_idx,
                    ocr_text=ocr_text,
                    image_path=f"{base_name}_images/{img_filename}"
                ))

            except Exception as e:
                print(f"Warning: Failed to render drawing region on page {page_num + 1}: {e}")
                continue

        return blocks

    def _cluster_drawing_rects(self, rects: list, proximity: float = 5) -> list:
        """描画矩形を空間的にクラスタリング

        Returns:
            List of (cluster_rect, member_indices)
        """
        n = len(rects)
        used = [False] * n
        clusters = []

        for i in range(n):
            if used[i]:
                continue

            cluster_rect = fitz.Rect(rects[i])
            members = [i]
            used[i] = True

            changed = True
            while changed:
                changed = False
                for j in range(n):
                    if used[j]:
                        continue
                    # クラスタ矩形を少し拡張して近接チェック
                    expanded = fitz.Rect(cluster_rect)
                    expanded.x0 -= proximity
                    expanded.y0 -= proximity
                    expanded.x1 += proximity
                    expanded.y1 += proximity
                    if expanded.intersects(rects[j]):
                        cluster_rect |= rects[j]  # union
                        members.append(j)
                        used[j] = True
                        changed = True

            clusters.append((cluster_rect, members))

        return clusters

    def _overlaps_regions(self, rect, regions: List[Tuple],
                          threshold: float = 0.5) -> bool:
        """矩形が既存領域と一定以上重複するかチェック"""
        rx0, ry0, rx1, ry1 = rect.x0, rect.y0, rect.x1, rect.y1
        rect_area = (rx1 - rx0) * (ry1 - ry0)
        if rect_area <= 0:
            return False

        for region in regions:
            tx0, ty0, tx1, ty1 = region
            overlap_x = max(0, min(rx1, tx1) - max(rx0, tx0))
            overlap_y = max(0, min(ry1, ty1) - max(ry0, ty0))
            overlap_area = overlap_x * overlap_y
            if overlap_area / rect_area > threshold:
                return True

        return False

    def _perform_ocr(self, image_bytes: bytes) -> str:
        """画像にOCRを実行"""
        try:
            if OCR_ENGINE == "easyocr" and self.ocr_reader:
                import numpy as np
                image = Image.open(io.BytesIO(image_bytes))

                # 大きな画像はリサイズしてOCRを高速化（最大2000px）
                max_size = 2000
                if image.width > max_size or image.height > max_size:
                    ratio = min(max_size / image.width, max_size / image.height)
                    new_size = (int(image.width * ratio), int(image.height * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)

                # RGBに変換（RGBA等の場合）
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # コントラスト強化前処理
                try:
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(1.5)
                except ImportError:
                    pass

                # EasyOCRはnumpy arrayを必要とする
                image_np = np.array(image)
                results = self.ocr_reader.readtext(image_np)
                # 信頼度30%未満のテキストをフィルタ
                texts = [result[1] for result in results if result[2] >= 0.3]
                return "\n".join(texts)

            elif OCR_ENGINE == "pytesseract":
                image = Image.open(io.BytesIO(image_bytes))
                text = pytesseract.image_to_string(image, lang='jpn+eng')
                return text.strip()

        except Exception as e:
            print(f"OCR error: {e}")

        return ""

    def _remove_text_in_drawing_images(self, blocks: List) -> List:
        """フルページ描画画像に含まれるテキストブロックを除外

        章タイトルページ等のフルページ画像に含まれるテキストは、
        画像自体にレンダリングされているため重複を防ぐ。
        部分的な描画領域（コードブロック等）のテキストは保持する。
        """
        # フルページ画像ブロック（ページの85%以上を占める画像）のみ対象
        full_page_images = []
        for block in blocks:
            if isinstance(block, ImageBlock):
                img_w = block.x1 - block.x0
                img_h = block.y1 - block.y0
                # フルページ判定: 画像が大きい場合のみ
                if img_w > 600 and img_h > 400:
                    full_page_images.append(block)

        if not full_page_images:
            return blocks

        filtered = []
        for block in blocks:
            if isinstance(block, TextBlock):
                contained = False
                for img in full_page_images:
                    if (block.page_num == img.page_num and
                        img.x0 <= block.x0 and block.x1 <= img.x1 and
                        img.y0 <= block.y0 and block.y1 <= img.y1):
                        contained = True
                        break
                if not contained:
                    filtered.append(block)
            else:
                filtered.append(block)

        removed = len(blocks) - len(filtered)
        if removed > 0:
            print(f"Info: Removed {removed} text blocks overlapping with full-page images")

        return filtered

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

    def _supplement_vector_drawings(self, doc, output_path: str,
                                     images_dir: str, base_name: str):
        """pymupdf4llm変換後にベクター描画画像を補完

        pymupdf4llmがテキスト変換に成功した後、ベクター描画で構成される
        ビジュアル要素（章タイトルページ等）を追加抽出する。
        pymupdf4llmが既に同一ページの画像を抽出済みの場合はそちらを
        ベクター描画画像で置換する（フルページ描画の方が高品質なため）。
        """
        existing_images = set(os.listdir(images_dir)) if os.path.exists(images_dir) else set()
        img_counter = [len(existing_images)]
        added_images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            table_regions = []
            try:
                tables = list(page.find_tables())
                table_regions = [t.bbox for t in tables]
            except Exception:
                pass

            drawing_blocks = self._extract_drawing_blocks(
                page, page_num, images_dir, base_name, table_regions, img_counter
            )

            for img_block in drawing_blocks:
                added_images.append((page_num, img_block))

        if not added_images:
            return

        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                md_content = f.read()

            # pymupdf4llmが生成した画像参照を検出（パターン: base_name.pdf-{page}-{idx}.png）
            import re
            pymupdf4llm_img_pattern = re.compile(
                r'!\[([^\]]*)\]\(([^)]*' + re.escape(base_name) + r'[^)]*?-(\d+)-\d+\.png)\)'
            )

            # Change Location-2026/02/16 - Handle multiple vector drawings per page
            # Original Code
            # (single image per page: only one replacement or insertion per page)
            # Updated Code

            # ページ番号→pymupdf4llm画像参照のマッピングを作成
            pymupdf4llm_pages = {}
            for m in pymupdf4llm_img_pattern.finditer(md_content):
                p4l_page = int(m.group(3))
                pymupdf4llm_pages[p4l_page] = m.group(0)  # full match

            replaced_count = 0
            inserted_count = 0
            files_to_delete = []

            # ページごとにベクター描画をグループ化
            from collections import defaultdict as _defaultdict
            page_drawings = _defaultdict(list)
            for page_num, img_block in added_images:
                page_drawings[page_num].append(img_block)

            # ページごとに処理（逆順で挿入位置がずれないように）
            for page_num in sorted(page_drawings.keys(), reverse=True):
                img_blocks = page_drawings[page_num]
                # Y座標順にソート（ページ内の正しい位置順）
                img_blocks.sort(key=lambda b: b.y0)

                # 全画像のMarkdownテキストを生成
                all_img_md = "\n".join(
                    f"![図{page_num + 1}]({b.image_path})" for b in img_blocks
                )

                if page_num in pymupdf4llm_pages:
                    # pymupdf4llm画像を全ベクター描画で置換
                    old_ref = pymupdf4llm_pages[page_num]
                    md_content = md_content.replace(old_ref, all_img_md)
                    replaced_count += 1
                    inserted_count += len(img_blocks) - 1
                    # 置換されたpymupdf4llm画像ファイルを削除対象に
                    old_match = re.search(r'\(([^)]+)\)', old_ref)
                    if old_match:
                        old_path = old_match.group(1)
                        if old_path.startswith('./'):
                            old_path = old_path[2:]
                        abs_old_path = os.path.join(os.path.dirname(output_path), old_path)
                        files_to_delete.append(abs_old_path)
                elif page_num == 0:
                    # 表紙（ページ1）はMarkdownの先頭に挿入
                    md_content = all_img_md + "\n\n" + md_content
                    inserted_count += len(img_blocks)
                else:
                    # 対応するページ番号テキストの前に挿入を試みる
                    img_md_block = "\n" + all_img_md + "\n"
                    page_marker = f"\n{page_num + 1}\n"
                    idx = md_content.find(page_marker)
                    if idx >= 0:
                        md_content = md_content[:idx] + img_md_block + md_content[idx:]
                    else:
                        # 見つからない場合は末尾に追加
                        md_content += f"\n<!-- Page {page_num + 1} -->\n{img_md_block}"
                    inserted_count += len(img_blocks)
            # Change Location-2026/02/16 - Handle multiple vector drawings per page

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

            # 置換済みの旧画像ファイルを削除
            for fpath in files_to_delete:
                try:
                    if os.path.exists(fpath):
                        os.remove(fpath)
                except Exception:
                    pass

            total = replaced_count + inserted_count
            print(f"Info: Supplemented {total} vector drawing images "
                  f"(replaced {replaced_count}, inserted {inserted_count})")

        except Exception as e:
            print(f"Warning: Failed to supplement vector drawings: {e}")

    def _sort_blocks_with_columns(self, blocks: List) -> List:
        """ブロックをページ順にソート（マルチカラムレイアウト対応）

        2カラムレイアウトを検出した場合、左カラム→右カラムの順で出力し、
        テキストの読み順を正しく保持する。
        """
        if not blocks:
            return blocks

        # ページごとにブロックをグルーピング
        pages = defaultdict(list)
        for b in blocks:
            pages[b.page_num].append(b)

        sorted_blocks = []
        for page_num in sorted(pages.keys()):
            page_blocks = pages[page_num]

            # フルページ画像のみのページはそのまま
            if len(page_blocks) == 1 and isinstance(page_blocks[0], ImageBlock):
                sorted_blocks.extend(page_blocks)
                continue

            # テキスト/表ブロックのX座標分布を分析してカラム検出
            x_positions = []
            for b in page_blocks:
                if isinstance(b, (TextBlock, TableBlock)):
                    x_center = (b.x0 + b.x1) / 2
                    x_positions.append(x_center)

            if len(x_positions) >= 4:
                # X座標の中央値でカラム境界を推定
                x_positions.sort()
                x_median = x_positions[len(x_positions) // 2]

                # 左右に分散しているか確認
                left_count = sum(1 for x in x_positions if x < x_median - 50)
                right_count = sum(1 for x in x_positions if x > x_median + 50)

                if left_count >= 2 and right_count >= 2:
                    # 2カラムレイアウト検出: カラム境界 = x_median
                    col_boundary = x_median

                    left_blocks = []
                    right_blocks = []
                    full_width_blocks = []

                    for b in page_blocks:
                        b_center = (b.x0 + b.x1) / 2
                        b_width = b.x1 - b.x0

                        # フルページ画像やページ幅の70%以上を占める要素
                        if isinstance(b, ImageBlock) and b_width > col_boundary * 1.4:
                            full_width_blocks.append(b)
                        elif b_center < col_boundary:
                            left_blocks.append(b)
                        else:
                            right_blocks.append(b)

                    # フルページ要素 → 左カラム(Y順) → 右カラム(Y順)
                    full_width_blocks.sort(key=lambda b: (b.y0, b.x0))
                    left_blocks.sort(key=lambda b: (b.y0, b.x0))
                    right_blocks.sort(key=lambda b: (b.y0, b.x0))
                    sorted_blocks.extend(full_width_blocks)
                    sorted_blocks.extend(left_blocks)
                    sorted_blocks.extend(right_blocks)
                    continue

            # 単カラム: Y座標順 → X座標順
            page_blocks.sort(key=lambda b: (b.y0, b.x0))
            sorted_blocks.extend(page_blocks)

        return sorted_blocks

    def _generate_css_header(self) -> str:
        """Obsidian互換のCSSスタイル定義を生成"""
        return """<style>
.pdf-page { margin-bottom: 2em; page-break-after: always; }
.pdf-columns { display: flex; gap: 1.5em; align-items: flex-start; }
.pdf-column { flex: 1; min-width: 0; }
.pdf-float-left { float: left; margin: 0 1em 0.5em 0; }
.pdf-float-right { float: right; margin: 0 0 0.5em 1em; }
.pdf-clearfix::after { content: ""; display: table; clear: both; }
.pdf-header, .pdf-footer { color: #888; font-size: 0.85em; text-align: center; margin: 0.5em 0; }
</style>
"""

    def _generate_markdown_obsidian(self, blocks: List, base_name: str,
                                     page_layouts: Dict[int, PageLayout]) -> str:
        """Obsidian向けレイアウト保持Markdown生成"""
        md_lines = [self._generate_css_header()]
        current_page = -1

        # ページごとにブロックをグループ化
        pages = defaultdict(list)
        for block in blocks:
            pages[block.page_num].append(block)

        for page_num in sorted(pages.keys()):
            page_blocks = pages[page_num]
            layout = page_layouts.get(page_num)

            if current_page >= 0:
                md_lines.append("\n---\n")
            current_page = page_num
            md_lines.append(f"\n<!-- Page {page_num + 1} -->\n")
            md_lines.append(f'<div class="pdf-page">\n')

            if layout and layout.num_columns > 1:
                md_lines.append(self._format_multicolumn_page(
                    page_blocks, layout, base_name
                ))
            else:
                md_lines.append(self._format_single_column_page(
                    page_blocks, layout, base_name
                ))

            md_lines.append("</div>\n")

        return "\n".join(md_lines)

    def _format_multicolumn_page(self, blocks: List, layout: PageLayout,
                                  base_name: str) -> str:
        """段組みページをHTML divで再現"""
        # ヘッダー/フッターを分離
        header_blocks = []
        footer_blocks = []
        body_blocks = []

        for block in blocks:
            region = self.layout_analyzer.is_header_or_footer(block, layout)
            if region == "header":
                header_blocks.append(block)
            elif region == "footer":
                footer_blocks.append(block)
            else:
                body_blocks.append(block)

        lines = []

        # ヘッダー
        if header_blocks:
            lines.append('<div class="pdf-header">')
            for b in header_blocks:
                lines.append(self._render_block_html(b))
            lines.append("</div>")

        # カラムにブロックを分配
        columns = defaultdict(list)
        for block in body_blocks:
            col_idx = getattr(block, 'column_index', 0)
            columns[col_idx].append(block)

        lines.append(f'<div class="pdf-columns">')
        for col_idx in range(layout.num_columns):
            lines.append(f'<div class="pdf-column">')
            col_blocks = columns.get(col_idx, [])
            prev_type = None
            for block in col_blocks:
                lines.append(self._render_block_html(block, prev_type))
                if isinstance(block, TextBlock):
                    prev_type = block.block_type
                elif isinstance(block, ImageBlock):
                    prev_type = "image"
                elif isinstance(block, TableBlock):
                    prev_type = "table"
            lines.append("</div>")
        lines.append("</div>")

        # フッター
        if footer_blocks:
            lines.append('<div class="pdf-footer">')
            for b in footer_blocks:
                lines.append(self._render_block_html(b))
            lines.append("</div>")

        return "\n".join(lines)

    def _format_single_column_page(self, blocks: List, layout: Optional[PageLayout],
                                    base_name: str) -> str:
        """単一カラムページ：画像とテキストの横並び（フロート）を検出"""
        lines = []
        prev_type = None
        i = 0

        while i < len(blocks):
            block = blocks[i]

            # 画像とテキストの横並び検出
            if isinstance(block, ImageBlock) and i + 1 < len(blocks):
                next_block = blocks[i + 1]
                if isinstance(next_block, TextBlock):
                    # Y座標が近く、X座標が離れている → 横並び
                    y_overlap = (min(block.y1, next_block.y1) - max(block.y0, next_block.y0))
                    block_height = max(block.y1 - block.y0, 1)
                    if y_overlap > block_height * 0.3:
                        # フロート配置
                        if block.x0 < next_block.x0:
                            # 画像が左、テキストが右
                            lines.append(f'<div class="pdf-clearfix">')
                            lines.append(self._render_block_html(block, float_dir="left"))
                            lines.append(self._render_block_html(next_block))
                            lines.append("</div>")
                        else:
                            # テキストが左、画像が右
                            lines.append(f'<div class="pdf-clearfix">')
                            lines.append(self._render_block_html(block, float_dir="right"))
                            lines.append(self._render_block_html(next_block))
                            lines.append("</div>")
                        prev_type = "float"
                        i += 2
                        continue

            lines.append(self._render_block_html(block, prev_type))
            if isinstance(block, TextBlock):
                prev_type = block.block_type
            elif isinstance(block, ImageBlock):
                prev_type = "image"
            elif isinstance(block, TableBlock):
                prev_type = "table"
            i += 1

        return "\n".join(lines)

    def _render_block_html(self, block, prev_type=None, float_dir=None) -> str:
        """ブロックをHTML/Markdown文字列にレンダリング"""
        if isinstance(block, TextBlock):
            return self._format_text_block(block, prev_type)
        elif isinstance(block, ImageBlock):
            img_html = self._format_image_block(block, use_html=True)
            if float_dir == "left":
                return f'<div class="pdf-float-left">{img_html}</div>'
            elif float_dir == "right":
                return f'<div class="pdf-float-right">{img_html}</div>'
            return img_html
        elif isinstance(block, TableBlock):
            return self._format_table_block(block)
        return ""

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

    def _format_image_block(self, block: ImageBlock, use_html: bool = False) -> str:
        """画像ブロックをMarkdown形式に変換"""
        # 図番号とキャプション
        if block.figure_number:
            alt_text = f"図{block.figure_number}"
            if block.caption:
                alt_text += f": {block.caption}"
        else:
            alt_text = f"図{block.page_num + 1}-{block.image_index + 1}"

        # サイズ指定付きHTML imgタグ or Markdown
        if use_html and block.display_width > 0:
            # PDF上の表示サイズをpx換算（72dpi基準: 1pt ≈ 1.333px）
            w = int(block.display_width * 1.333)
            h = int(block.display_height * 1.333)
            md = f'\n<img src="{block.image_path}" alt="{alt_text}" width="{w}" height="{h}">\n'
        elif block.display_width > 0:
            # Obsidian互換: width指定付きHTML
            w = int(block.display_width * 1.333)
            h = int(block.display_height * 1.333)
            md = f'\n<img src="{block.image_path}" alt="{alt_text}" width="{w}" height="{h}">\n'
        else:
            md = f"\n![{alt_text}]({block.image_path})\n"

        # キャプション（図の下に追加）
        if block.caption and block.figure_number:
            md += f"\n*図{block.figure_number}: {block.caption}*\n"

        # Claude解析結果がある場合、構造化出力を追加
        if block.claude_analysis:
            if block.analysis_type == "table":
                md += f"\n{block.claude_analysis}\n"
            elif block.analysis_type in ("flowchart", "sequence_diagram", "state_diagram", "org_chart"):
                md += f"\n```plantuml\n{block.claude_analysis}\n```\n"
            else:
                md += f"\n{block.claude_analysis}\n"
        # OCRテキストがあれば追加（Claude解析がない場合のみ）
        elif block.ocr_text:
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

        self.root.title("PDF to Markdown Converter v4.0 (Layout-Aware)")
        self.root.geometry("750x620")
        self.root.minsize(650, 520)

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
        title_label = ttk.Label(main_frame, text="PDF to Markdown Converter v4.0",
                               font=("", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))

        # 機能ステータス表示
        status_text = f"PyMuPDF: {'✓' if PYMUPDF_AVAILABLE else '✗'} | "
        status_text += f"OCR: {'✓ ' + OCR_ENGINE if OCR_ENGINE else '✗'} | "
        status_text += f"Claude: {'✓' if CLAUDE_API_AVAILABLE else '✗'}"
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

        self.enable_claude_var = tk.BooleanVar(value=CLAUDE_API_AVAILABLE)
        claude_cb = ttk.Checkbutton(option_frame, text="Claude AI図表解析",
                                   variable=self.enable_claude_var)
        claude_cb.pack(side="left", padx=5)
        if not CLAUDE_API_AVAILABLE:
            claude_cb.configure(state="disabled")

        # レイアウトモード選択フレーム
        layout_frame = ttk.LabelFrame(main_frame, text="レイアウトモード", padding="5")
        layout_frame.grid(row=4, column=0, sticky="ew", pady=5)

        self.layout_mode_var = tk.StringVar(value="auto")
        ttk.Radiobutton(layout_frame, text="自動",
                       variable=self.layout_mode_var, value="auto").pack(side="left", padx=5)
        ttk.Radiobutton(layout_frame, text="精密（段組み・フロート対応）",
                       variable=self.layout_mode_var, value="precise").pack(side="left", padx=5)
        ttk.Radiobutton(layout_frame, text="ページ画像",
                       variable=self.layout_mode_var, value="page_image").pack(side="left", padx=5)
        ttk.Radiobutton(layout_frame, text="従来",
                       variable=self.layout_mode_var, value="legacy").pack(side="left", padx=5)

        # 出力設定フレーム
        output_frame = ttk.LabelFrame(main_frame, text="出力設定", padding="5")
        output_frame.grid(row=5, column=0, sticky="ew", pady=5)
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
        self.progress_bar.grid(row=6, column=0, sticky="ew", pady=5)

        # ステータスラベル
        self.status_var = tk.StringVar(value="準備完了")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=7, column=0, sticky="w")

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
        layout_mode = self.layout_mode_var.get()
        enable_claude = self.enable_claude_var.get()

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
                filepath, out_path, extract_images=extract_images,
                layout_mode=layout_mode, enable_claude=enable_claude
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
    parser = argparse.ArgumentParser(
        description="PDF to Markdown Converter v4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  pdf2md.py                          GUIモードで起動
  pdf2md.py document.pdf             PDFをMarkdownに変換
  pdf2md.py --layout precise doc.pdf 精密レイアウトモードで変換
  pdf2md.py --layout page_image doc.pdf ページ画像モードで変換
  pdf2md.py --ocr document.pdf       OCR有効で変換
  pdf2md.py --no-images document.pdf 画像抽出なしで変換
  pdf2md.py ./pdf_folder/            フォルダ内のPDFを一括変換
"""
    )
    parser.add_argument("inputs", nargs="*", help="PDFファイルまたはフォルダのパス")
    parser.add_argument("--layout", choices=["auto", "precise", "page_image", "legacy"],
                       default="auto", help="レイアウトモード (default: auto)")
    parser.add_argument("--dpi", type=int, default=150,
                       help="画像レンダリングDPI (default: 150)")
    parser.add_argument("-o", "--output", help="出力先フォルダ")
    parser.add_argument("--ocr", action="store_true", default=True,
                       help="OCRを有効にする (default: True)")
    parser.add_argument("--no-ocr", action="store_true",
                       help="OCRを無効にする")
    parser.add_argument("--no-images", action="store_true",
                       help="画像抽出を無効にする")
    parser.add_argument("--no-claude", action="store_true",
                       help="Claude APIによる図表解析を無効にする")
    parser.add_argument("--context-menu", action="store_true",
                       help="右クリックメニューからの呼び出し用")
    parser.add_argument("--silent", action="store_true",
                       help="変換完了後のダイアログを抑制")

    # 引数がない場合はGUIモード
    if len(sys.argv) == 1:
        app = PDF2MDGUI()
        app.run()
        return

    args = parser.parse_args()

    # 右クリックメニューからの呼び出し
    if args.context_menu:
        _run_context_menu_mode(args)
        return

    # 入力がない場合はGUI起動
    if not args.inputs:
        app = PDF2MDGUI()
        app.run()
        return

    # CLIモード
    enable_ocr = not args.no_ocr
    extract_images = not args.no_images
    enable_claude = not args.no_claude
    converter = AdvancedPDFConverter(enable_ocr=enable_ocr)

    for input_path in args.inputs:
        if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
            print(f"Converting: {input_path}")
            out_path = None
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                out_path = os.path.join(
                    args.output,
                    os.path.splitext(os.path.basename(input_path))[0] + '.md'
                )
            success, result = converter.convert_file(
                input_path, out_path,
                extract_images=extract_images,
                layout_mode=args.layout,
                dpi=args.dpi,
                enable_claude=enable_claude,
            )
            if success:
                print(f"  -> {result}")
            else:
                print(f"  Error: {result}")
        elif os.path.isdir(input_path):
            print(f"Converting folder: {input_path}")
            for pdf_path in Path(input_path).glob('*.pdf'):
                print(f"  Converting: {pdf_path.name}")
                out_path = None
                if args.output:
                    os.makedirs(args.output, exist_ok=True)
                    out_path = os.path.join(
                        args.output,
                        os.path.splitext(pdf_path.name)[0] + '.md'
                    )
                success, result = converter.convert_file(
                    str(pdf_path), out_path,
                    extract_images=extract_images,
                    layout_mode=args.layout,
                    dpi=args.dpi,
                    enable_claude=enable_claude,
                )
                status = "OK" if success else f"Error: {result}"
                print(f"    {status}")
        else:
            print(f"  Skipping (not a PDF or folder): {input_path}")


def _run_context_menu_mode(args):
    """右クリックメニューからの変換実行"""
    if not args.inputs:
        return

    pdf_path = args.inputs[0]
    if not os.path.isfile(pdf_path) or not pdf_path.lower().endswith('.pdf'):
        if not args.silent:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("エラー", f"PDFファイルではありません:\n{pdf_path}")
            root.destroy()
        return

    converter = AdvancedPDFConverter(enable_ocr=True)
    success, result = converter.convert_file(
        pdf_path, layout_mode=args.layout, dpi=args.dpi
    )

    if not args.silent:
        root = tk.Tk()
        root.withdraw()
        if success:
            messagebox.showinfo("PDF2MD 変換完了",
                              f"変換が完了しました\n\n入力: {os.path.basename(pdf_path)}\n出力: {result}")
        else:
            messagebox.showerror("PDF2MD 変換エラー",
                               f"変換に失敗しました\n\n{result}")
        root.destroy()


if __name__ == "__main__":
    main()
