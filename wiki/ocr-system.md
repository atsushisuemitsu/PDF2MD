---
title: OCR System
type: concept
tags: [ocr, easyocr, pytesseract]
---

# OCR System

PDF内の画像からテキストを認識するOCRシステム。

## エンジン優先順位

1. **EasyOCR** (優先) — 日本語・英語対応、バウンディングボックス座標付き
2. **pytesseract** (フォールバック) — Tesseract OCRのPythonラッパー

`OCR_ENGINE` グローバル変数にエンジン名が格納される。両方インストールされていなければ `None`。

## EasyOCR の特性

- 初回実行時にモデルファイル（約100MB）をダウンロード
- GPU=False で初期化（CPUモード）
- 対象言語: `['ja', 'en']`
- `requirements.txt` ではコメントアウト（64-bit Python + PyTorch が必要なため手動インストール）

## OCR Reader の初期化

[[AdvancedPDFConverter]] の `__init__()` で一度だけ初期化し、全変換でリーダーを再利用:

```python
if self.enable_ocr and OCR_ENGINE == "easyocr":
    self.ocr_reader = easyocr.Reader(self.ocr_languages, gpu=False)
```

## 利用箇所

1. **画像内テキスト認識**: `_extract_image_blocks()` で抽出した画像にOCRを適用
2. **ページレベルOCR**: `_convert_with_page_ocr()` でページ全体をOCR（[[font-encoding-detection|フォント問題時]]）

## 関連ページ

- [[optional-dependencies]] — `OCR_ENGINE` フラグ
- [[font-encoding-detection]] — OCRフォールバックのトリガー
- [[image-extraction]] — 画像抽出後のOCR適用
- [[dependencies]] — EasyOCRパッケージ詳細
