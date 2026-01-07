#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF to Markdown Converter (Advanced Version)
PDFファイルをMarkdown形式に変換するGUIツール

機能:
- テキスト抽出（位置情報付き）
- 画像抽出と保存
- OCRによる図内テキスト認識
- 位置関係を保持したMarkdown生成

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
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

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
    block_type: str = "text"  # text, heading, list, table


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
            all_blocks = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # テキストブロック抽出
                text_blocks = self._extract_text_blocks(page, page_num)
                all_blocks.extend(text_blocks)

                # 画像ブロック抽出
                if extract_images:
                    image_blocks = self._extract_image_blocks(
                        page, page_num, images_dir, base_name
                    )
                    all_blocks.extend(image_blocks)

            doc.close()

            # ブロックをソート（ページ順 → Y座標順 → X座標順）
            all_blocks.sort(key=lambda b: (b.page_num, b.y0, b.x0))

            # Markdown生成
            markdown_content = self._generate_markdown(all_blocks, base_name)

            # 保存
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            return True, output_path

        except Exception as e:
            return False, str(e)

    def _extract_text_blocks(self, page, page_num: int) -> List[TextBlock]:
        """ページからテキストブロックを抽出"""
        blocks = []

        # テキストを辞書形式で取得（位置情報付き）
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # テキストブロック
                block_text = ""
                max_font_size = 0
                is_bold = False

                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                        font_size = span.get("size", 12)
                        if font_size > max_font_size:
                            max_font_size = font_size
                        font_name = span.get("font", "").lower()
                        if "bold" in font_name or "heavy" in font_name:
                            is_bold = True
                    block_text += line_text + "\n"

                block_text = block_text.strip()
                if block_text:
                    bbox = block.get("bbox", [0, 0, 0, 0])

                    # ブロックタイプの推定
                    block_type = self._detect_block_type(block_text, max_font_size, is_bold)

                    blocks.append(TextBlock(
                        text=block_text,
                        x0=bbox[0],
                        y0=bbox[1],
                        x1=bbox[2],
                        y1=bbox[3],
                        page_num=page_num,
                        font_size=max_font_size,
                        is_bold=is_bold,
                        block_type=block_type
                    ))

        return blocks

    def _detect_block_type(self, text: str, font_size: float, is_bold: bool) -> str:
        """ブロックタイプを推定"""
        lines = text.split('\n')
        first_line = lines[0].strip() if lines else ""

        # 見出し判定
        if font_size >= 16 or (is_bold and len(first_line) < 100):
            if font_size >= 20:
                return "heading1"
            elif font_size >= 16:
                return "heading2"
            elif is_bold:
                return "heading3"

        # リスト判定
        list_patterns = [r'^[\-\•\●\○\■\□\・]\s', r'^\d+[\.\)]\s', r'^[a-zA-Z][\.\)]\s']
        for pattern in list_patterns:
            if re.match(pattern, first_line):
                return "list"

        return "text"

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

                # 画像の位置を取得
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rect = img_rects[0]
                    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                else:
                    # 位置が取得できない場合はページ中央に仮配置
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
                # PIL Imageに変換
                image = Image.open(io.BytesIO(image_bytes))
                # OCR実行
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

    def _generate_markdown(self, blocks: List, base_name: str) -> str:
        """ブロックからMarkdownを生成"""
        md_lines = []
        current_page = -1

        for block in blocks:
            # ページ区切り
            if block.page_num != current_page:
                if current_page >= 0:
                    md_lines.append("\n---\n")
                current_page = block.page_num
                md_lines.append(f"\n<!-- Page {current_page + 1} -->\n")

            if isinstance(block, TextBlock):
                md_lines.append(self._format_text_block(block))

            elif isinstance(block, ImageBlock):
                md_lines.append(self._format_image_block(block))

        return "\n".join(md_lines)

    def _format_text_block(self, block: TextBlock) -> str:
        """テキストブロックをMarkdown形式に変換"""
        text = block.text.strip()

        if block.block_type == "heading1":
            return f"\n# {text}\n"
        elif block.block_type == "heading2":
            return f"\n## {text}\n"
        elif block.block_type == "heading3":
            return f"\n### {text}\n"
        elif block.block_type == "list":
            # リストアイテムの整形
            lines = text.split('\n')
            formatted_lines = []
            for line in lines:
                line = line.strip()
                # 番号付きリストまたは箇条書きリストの正規化
                if re.match(r'^[\-\•\●\○\■\□\・]\s*', line):
                    line = re.sub(r'^[\-\•\●\○\■\□\・]\s*', '- ', line)
                elif re.match(r'^\d+[\.\)]\s*', line):
                    line = re.sub(r'^(\d+)[\.\)]\s*', r'\1. ', line)
                formatted_lines.append(line)
            return "\n".join(formatted_lines) + "\n"
        else:
            return f"\n{text}\n"

    def _format_image_block(self, block: ImageBlock) -> str:
        """画像ブロックをMarkdown形式に変換"""
        md = f"\n![図{block.page_num + 1}-{block.image_index + 1}]({block.image_path})\n"

        # OCRテキストがあれば追加
        if block.ocr_text:
            ocr_lines = block.ocr_text.strip().split('\n')
            if ocr_lines:
                md += "\n<details>\n<summary>図内テキスト（OCR）</summary>\n\n"
                md += "```\n"
                md += "\n".join(ocr_lines)
                md += "\n```\n\n</details>\n"

        return md


class PDF2MDGUI:
    """GUIアプリケーションクラス"""

    def __init__(self):
        # ウィンドウ作成
        if DND_AVAILABLE:
            self.root = tkdnd.Tk()
        else:
            self.root = tk.Tk()

        self.root.title("PDF to Markdown Converter (Advanced)")
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
        title_label = ttk.Label(main_frame, text="PDF to Markdown Converter (Advanced)",
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
