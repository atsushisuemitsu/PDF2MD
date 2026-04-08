---
title: Font Encoding Detection
type: concept
tags: [font, encoding, ocr, fallback]
---

# Font Encoding Detection

PDFのフォントエンコーディング問題を自動検出し、OCRベース変換にフォールバックする仕組み。

## 検出メソッド

`AdvancedPDFConverter._check_font_encoding_issue(doc)` (line ~1060)

### 検出アルゴリズム

最初の3ページのテキストをサンプリングし、以下の条件をチェック:

1. **文字化けパターン**: 置換文字 (`\ufffd`) や制御文字がテキストの10%超
2. **CID/Identity-Hフォント**: Unicode Private Use Area (U+E000-U+F8FF) の文字が15%超
3. **単一文字支配**: 1つのアルファベット文字がテキスト全体の25%超（v4.0で閾値を30%→25%に改善）

いずれかに該当 → `True` を返し、OCR変換にルーティング。

## OCR変換パス

`_convert_with_page_ocr()`:

1. 各ページを200 DPIで画像にレンダリング
2. 画像からラスター/ベクター画像を抽出
3. EasyOCRでバウンディングボックス付きテキスト認識
4. Y/X座標に基づきレイアウトを再構築
5. 相対テキストサイズから見出しを推定
6. テキストと画像を正しい読み順で結合

## 対象となるPDF

- CIDフォントを使用した古い日本語PDF
- フォント埋め込みのないPDF
- 独自エンコーディングのPDF

## 関連ページ

- [[conversion-pipeline]] — フォント問題検出は分岐の第一段階
- [[ocr-system]] — OCRエンジンの詳細
- [[AdvancedPDFConverter]] — 検出メソッドの所在
