---
title: Obsidian Layout
type: concept
tags: [obsidian, layout, css, html, v4]
---

# Obsidian Layout

v4.0で追加されたObsidian互換のマルチカラムHTML/CSS出力。precise/autoモード（フォールバック時）で使用。

## 生成メソッド

`_generate_markdown_obsidian()` (line ~2363)

## HTML構造

```html
<style>/* CSSヘッダー */</style>

<!-- Page 1 -->
<div class="pdf-page">
  <div class="pdf-header">...</div>
  <div class="pdf-columns">
    <div class="pdf-column">/* 左カラム */</div>
    <div class="pdf-column">/* 右カラム */</div>
  </div>
  <div class="pdf-footer">...</div>
</div>

---

<!-- Page 2 -->
<div class="pdf-page">
  /* 単カラムページ */
</div>
```

## ページレンダリング

### マルチカラムページ (`_format_multicolumn_page`)

1. [[LayoutAnalyzer]] の結果でヘッダー/フッター/ボディを分離
2. ボディブロックを `column_index` でカラムに分配
3. `<div class="pdf-columns">` 内に各カラムの `<div class="pdf-column">` を配置

### 単カラムページ (`_format_single_column_page`)

画像とテキストの横並び（フロート配置）を検出。画像の幅がページの50%未満の場合、フロートレイアウトを適用。

## CSS

`_generate_css_header()` で出力。Obsidianのプレビュー/リーディングモードで段組みが再現されるようにデザイン。

## 関連ページ

- [[LayoutAnalyzer]] — カラム検出結果の供給元
- [[layout-modes]] — precise/autoモードで有効
- [[conversion-pipeline]] — `use_obsidian_layout` 分岐
