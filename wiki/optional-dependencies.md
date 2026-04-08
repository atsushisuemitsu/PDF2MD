---
title: Optional Dependencies Pattern
type: concept
tags: [architecture, imports, feature-flags]
---

# Optional Dependencies Pattern

PDF2MDは多数のオプショナル依存を持ち、利用可能な機能に応じて動作を切り替える。`pdf2md.py` 冒頭（line ~42-96）で try/except import パターンを使用。

## フィーチャーフラグ一覧

| フラグ | ライブラリ | 用途 | 不在時の動作 |
|--------|-----------|------|-------------|
| `PYMUPDF_AVAILABLE` | PyMuPDF (fitz) | PDF解析全般 | アプリ起動不可 |
| `PYMUPDF4LLM_AVAILABLE` | pymupdf4llm | レイアウト認識変換 | 標準パイプラインのみ |
| `LAYOUT_AVAILABLE` | pymupdf.layout | ML レイアウト検出 | pymupdf4llm内OCR無効 |
| `OCR_ENGINE` | easyocr / pytesseract | 図内テキスト認識 | OCR機能無効 |
| `CLAUDE_API_AVAILABLE` | anthropic + API key | 図表AI解析 | AI解析スキップ |
| `DND_AVAILABLE` | tkinterdnd2 | D&D対応 | ボタン操作のみ |
| `PILLOW_AVAILABLE` | Pillow | 画像処理 | 画像関連機能制限 |

## OCRエンジンの優先順位

```python
OCR_ENGINE = None
try:
    import easyocr        # 優先
    OCR_ENGINE = "easyocr"
except ImportError:
    try:
        import pytesseract  # フォールバック
        OCR_ENGINE = "pytesseract"
    except ImportError:
        pass  # OCR無効
```

## Claude API の条件

`anthropic` パッケージのインポート成功 **かつ** `ANTHROPIC_API_KEY` 環境変数が設定されている場合のみ有効。さらに [[ClaudeDiagramAnalyzer]] は初期化時にテスト呼び出しでAPIキーの有効性を検証し、無効なら `self.disabled = True` で自動無効化する。

## GUI での表示

[[PDF2MDGUI]] はステータスバーに各機能の有効/無効を表示:
```
PyMuPDF: ✓ | OCR: ✓ easyocr | Claude: ✗
```

OCRやClaude APIが無効な場合、対応するチェックボックスは `disabled` 状態になる。

## 設計意図

- 最小構成（PyMuPDFのみ）でも基本変換が動作する
- 追加パッケージをインストールするほど機能が拡張される
- インポート失敗時にクラッシュせず、警告メッセージのみ出力

## 関連ページ

- [[dependencies]] — 各パッケージの詳細
- [[ClaudeDiagramAnalyzer]] — Claude API統合
- [[ocr-system]] — OCRエンジン詳細
