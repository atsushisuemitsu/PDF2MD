#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF to Markdown Converter
PDFファイルをMarkdown形式に変換するGUIツール

使用ライブラリ: Microsoft MarkItDown
参考: https://note.com/suh_sunaneko/n/na6687b2e01c8
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from pathlib import Path
from datetime import datetime

try:
    from markitdown import MarkItDown
except ImportError:
    print("Error: markitdown is not installed. Run: pip install markitdown")
    sys.exit(1)

# ドラッグ&ドロップ対応（Windows）
try:
    import tkinterdnd2 as tkdnd
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False


class PDF2MDConverter:
    """PDF to Markdown変換クラス"""

    def __init__(self):
        # enable_plugins=Falseでmagikaなどの追加機能を無効化
        # これによりEXE化時のモデルファイル問題を回避
        try:
            self.md = MarkItDown(enable_plugins=False)
        except TypeError:
            # 古いバージョンのmarkitdownの場合
            self.md = MarkItDown()

    def convert_file(self, pdf_path: str, output_path: str = None) -> tuple[bool, str]:
        """
        単一PDFファイルをMarkdownに変換

        Args:
            pdf_path: 入力PDFファイルパス
            output_path: 出力Markdownファイルパス（省略時は同じディレクトリに.md拡張子で出力）

        Returns:
            (成功フラグ, メッセージまたはエラー内容)
        """
        try:
            if not os.path.exists(pdf_path):
                return False, f"ファイルが見つかりません: {pdf_path}"

            if not pdf_path.lower().endswith('.pdf'):
                return False, f"PDFファイルではありません: {pdf_path}"

            # 変換実行
            result = self.md.convert(pdf_path)

            # 出力パス決定
            if output_path is None:
                output_path = os.path.splitext(pdf_path)[0] + '.md'

            # Markdown保存
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.text_content)

            return True, output_path

        except Exception as e:
            return False, str(e)

    def convert_folder(self, folder_path: str, output_folder: str = None,
                       callback=None) -> list[tuple[str, bool, str]]:
        """
        フォルダ内のPDFファイルを一括変換

        Args:
            folder_path: 入力フォルダパス
            output_folder: 出力フォルダパス（省略時は同じフォルダに出力）
            callback: 進捗コールバック関数 (current, total, filename)

        Returns:
            [(ファイル名, 成功フラグ, メッセージ), ...]
        """
        results = []
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        total = len(pdf_files)

        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i, filename in enumerate(pdf_files):
            if callback:
                callback(i + 1, total, filename)

            pdf_path = os.path.join(folder_path, filename)

            if output_folder:
                output_path = os.path.join(output_folder,
                                          os.path.splitext(filename)[0] + '.md')
            else:
                output_path = None

            success, message = self.convert_file(pdf_path, output_path)
            results.append((filename, success, message))

        return results


class PDF2MDGUI:
    """GUIアプリケーションクラス"""

    def __init__(self):
        # ウィンドウ作成
        if DND_AVAILABLE:
            self.root = tkdnd.Tk()
        else:
            self.root = tk.Tk()

        self.root.title("PDF to Markdown Converter")
        self.root.geometry("700x500")
        self.root.minsize(600, 400)

        # 変換エンジン
        self.converter = PDF2MDConverter()

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
        title_label = ttk.Label(main_frame, text="PDF to Markdown Converter",
                               font=("", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))

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
        self.tree.column("path", width=500)
        self.tree.column("status", width=100)

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

        # 出力設定フレーム
        output_frame = ttk.LabelFrame(main_frame, text="出力設定", padding="5")
        output_frame.grid(row=3, column=0, sticky="ew", pady=5)
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
        self.progress_bar.grid(row=4, column=0, sticky="ew", pady=5)

        # ステータスラベル
        self.status_var = tk.StringVar(value="準備完了")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=5, column=0, sticky="w")

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
            success, message = self.converter.convert_file(filepath, out_path)

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
        converter = PDF2MDConverter()

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
                results = converter.convert_folder(arg)
                for filename, success, message in results:
                    status = "OK" if success else f"Error: {message}"
                    print(f"  {filename}: {status}")
    else:
        # GUIモード
        app = PDF2MDGUI()
        app.run()


if __name__ == "__main__":
    main()
