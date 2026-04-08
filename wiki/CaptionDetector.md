---
title: CaptionDetector
type: class
location: "pdf2md.py:596"
tags: [class, caption, detection]
---

# CaptionDetector

図表キャプションを自動検出し、最近傍の図/表ブロックと紐付けるクラス。

## 位置

`pdf2md.py` line ~596

## 検出パターン

### 図キャプション
- 図1, Fig. 1, Figure 1
- グラフ1, Chart 1
- 写真1, Photo 1

### 表キャプション
- 表1, Tab. 1, Table 1

## 紐付けロジック

`_associate_captions()` で100pt以内の距離にある図/表ブロックとマッチング。Y座標の近さで最近傍を選択。

## 関連ページ

- [[conversion-pipeline]] — ブロック抽出後に紐付け処理
- [[AdvancedPDFConverter]] — `self.caption_detector` として保持
