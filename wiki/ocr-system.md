---
title: OCR System
type: concept
tags: [ocr, ndlocr, easyocr, pytesseract]
---

# OCR System

PDF内の画像からテキストを認識するOCRシステム。v4.3でndlocr_cli（国立国会図書館OCR）を統合。

## エンジン優先順位

1. **ndlocr_cli** (最優先) — 国立国会図書館製日本語OCR。レイアウト抽出+文字認識。領域検出（図版・表組・組織図等）付き
2. **EasyOCR** (フォールバック) — 日本語・英語対応、バウンディングボックス座標付き
3. **pytesseract** (最終フォールバック) — Tesseract OCRのPythonラッパー

`OCR_ENGINE` グローバル変数に `"ndlocr"`, `"easyocr"`, `"pytesseract"` のいずれかが格納される。`NDLOCR_AVAILABLE` フラグも別途存在。

## ndlocr_cli 統合 (v4.3)

### 検出

起動時に以下のパスからndlocr_cliを探索:
- `../ndlocr_cli`（pdf2md.pyからの相対パス）
- `D:\Github\ndlocr_cli`（固定パス）

`main.py` と `config.yml` の存在、および `cv2` (opencv-python) のインポート可否で判定。

### 初期化

[[AdvancedPDFConverter]] の `__init__()` → `_init_ndlocr()` で:
1. ndlocr_cliのサブモジュールパスをsys.pathに追加
2. `OcrInferrer` をインポートしインスタンス化（**モデルを1回だけロード**）
3. `layout_extraction.device = 'cpu'` でCPU動作
4. stages 2-3（レイアウト抽出+文字認識）のみ実行

### ページOCR処理フロー

```
ページ画像 (300 DPI PNG)
  → _ocr_page_with_ndlocr()
    → ndlocr_cli._infer() (stage2: レイアウト抽出, stage3: 文字認識)
    → XML解析
      → TEXTBLOCK/LINE → テキストブロック（位置+テキスト+TYPE）
      → BLOCK → 非テキスト領域（図版/表組/組織図等）
  → 非テキスト領域を切り出し (_crop_region_image)
  → 分類・変換 (_classify_and_convert_region)
    → 表組 → Claude API → Markdown表
    → 組織図 → Claude API → PlantUML
    → 図版 → Claude APIで判定:
        写真/イラスト → 画像ファイルとして保存
        フローチャート/ダイアグラム → PlantUML / WaveDrom
  → 全要素を位置でソートしてMarkdown生成
```

### ndlocr_cli BLOCK TYPE一覧

| TYPE | 説明 | 変換先 |
|------|------|--------|
| 図版 | 図・写真・イラスト | Claude API判定 → 画像 or PlantUML |
| 表組 | 表 | Claude API → Markdown表 |
| 組織図 | 組織図・ツリー | Claude API → PlantUML |
| 数式 | 数式 | Claude API → 構造化 |
| 化学式 | 化学式 | Claude API → 構造化 |
| 広告 | 広告領域 | 画像として保存 |
| 柱 | ヘッダー | テキスト抽出 |
| ノンブル | ページ番号 | テキスト抽出 |

## EasyOCR (フォールバック)

- 初回実行時にモデルファイル（約100MB）をダウンロード
- GPU=False で初期化（CPUモード）
- 対象言語: `['ja', 'en']`
- `requirements.txt` ではコメントアウト（64-bit Python + PyTorch が必要なため手動インストール）

## 利用箇所

1. **画像内テキスト認識**: `_perform_ocr()` で抽出した画像にOCRを適用（ndlocr_cli優先）
2. **ページレベルOCR**: `_convert_with_page_ocr()` でページ全体をOCR（[[font-encoding-detection|フォント問題時]]）

## 関連ページ

- [[optional-dependencies]] — `OCR_ENGINE`, `NDLOCR_AVAILABLE` フラグ
- [[font-encoding-detection]] — OCRフォールバックのトリガー
- [[image-extraction]] — 画像抽出後のOCR適用
- [[ClaudeDiagramAnalyzer]] — 図表領域のAI構造化変換
- [[dependencies]] — OCRパッケージ詳細
