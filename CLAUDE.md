# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF2MD is a Windows desktop application (single-file: `pdf2md.py`, ~2600 lines) that converts PDF files to Markdown format. It uses PyMuPDF and PyMuPDF4LLM for PDF parsing with layout-aware conversion, EasyOCR for optical character recognition, and optionally Claude API for diagram/flowchart analysis (converting visual elements to Markdown tables or PlantUML). The application supports both GUI (tkinter) and CLI (argparse) modes with layout-aware conversion.

Primary language: Python 3.13 on Windows. UI labels and comments are in Japanese.

## Build Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application (GUI mode)
python pdf2md.py

# Run in CLI mode
python pdf2md.py document.pdf
python pdf2md.py --layout precise document.pdf
python pdf2md.py --layout page_image document.pdf
python pdf2md.py --layout legacy document.pdf
python pdf2md.py --no-ocr document.pdf
python pdf2md.py --no-images document.pdf
python pdf2md.py --no-claude document.pdf
python pdf2md.py -o ./output/ document.pdf
python pdf2md.py ./pdf_folder/

# Build EXE (Windows) - output: dist/PDF2MD.exe
build.bat
# or
powershell -ExecutionPolicy Bypass -File build.ps1
```

There are no tests in this project. No linter or formatter is configured.

## Architecture

Single-file application (`pdf2md.py`) with all classes in one module. No package structure.

### Conversion Pipeline (the critical path to understand)

`convert_file()` (line ~870) is the main entry point. It branches into **two distinct conversion paths** based on font encoding detection:

```
PDF opened with PyMuPDF
  │
  ├─ _check_font_encoding_issue() samples first 3 pages
  │    └─ If single char > 30% of text → font encoding problem detected
  │
  ├─ [Font issues] → _convert_with_page_ocr()
  │    Renders pages as images (300 DPI) → EasyOCR with bbox → reconstruct layout
  │    + Claude API: detect diagrams/flowcharts → Markdown tables / PlantUML
  │
  └─ [Normal] → _convert_with_pymupdf4llm() (preferred)
       │    Uses pymupdf4llm for layout-aware conversion
       │    Falls back to standard conversion on failure
       │
       └─ [Fallback] Standard conversion pipeline:
            DocumentAnalyzer → per-page extraction → Markdown generation
```

### Standard Conversion Pipeline (fallback path)

```
DocumentAnalyzer.analyze_document_structure()  → font size → heading hierarchy
  → Per page:
      → AdvancedTableExtractor.extract_tables()  (PyMuPDF find_tables + position-based)
      → _extract_text_blocks()  (excludes table regions)
      → _extract_image_blocks() + OCR + Claude API diagram analysis
  → CaptionDetector._associate_captions()  (within 100pt distance)
  → Sort by page → Y → X
  → _generate_markdown()
```

### Key Classes

| Class | Line | Role |
|-------|------|------|
| `DocumentAnalyzer` | ~139 | Analyzes font sizes across all pages to map heading levels (H1-H6) |
| `AdvancedTableExtractor` | ~231 | Two-stage: PyMuPDF `find_tables()` + text-position grid detection for borderless tables |
| `ListDetector` | ~475 | Regex-based bullet/numbered list detection with hierarchy |
| `CaptionDetector` | ~548 | Matches patterns like 図1, Fig. 1, 表1, Table 1 |
| `ClaudeDiagramAnalyzer` | ~774 | Claude API vision for diagram/flowchart detection → Markdown tables / PlantUML |
| `AdvancedPDFConverter` | ~990 | Main converter with both conversion paths, image extraction, OCR, Claude API |
| `PDF2MDGUI` | ~2130 | tkinter GUI with Treeview file list, threading, drag-and-drop |

### Data Classes

- `TextBlock`: Text with position, font info, block type (heading1-6, list, caption, text)
- `ImageBlock`: Image with position, OCR text, caption association, Claude analysis results
- `TableBlock`: Table cells with position, header row detection
- `ListItem`: List item with level and type (bullet/numbered)

### Optional Dependencies Pattern

The codebase uses try/except imports with availability flags (`PYMUPDF_AVAILABLE`, `PYMUPDF4LLM_AVAILABLE`, `LAYOUT_AVAILABLE`, `OCR_ENGINE`, `CLAUDE_API_AVAILABLE`, `DND_AVAILABLE`). These control feature degradation at runtime. When modifying import handling, maintain this pattern.

### Output Structure

Conversion produces `{name}.md` alongside the input PDF, plus `{name}_images/` directory containing extracted images as `page{N}_img{M}.png/jpeg`.

**ListDetector**: Detects list items using regex patterns for:
- Bullet markers: `- * ● ○ ■ □ ・ ※ ★ ☆ → ⇒ ▶ ►`
- Numbered patterns: `1.`, `(1)`, `a)`, `(a)`, `i.`, etc.

**CaptionDetector**: Identifies figure/table captions matching patterns like:
- 図1, Fig. 1, Figure 1, グラフ1, Chart 1, 写真1, Photo 1
- 表1, Tab. 1, Table 1

**AdvancedPDFConverter**: Main conversion pipeline:
1. Font encoding issue detection (auto-detect problematic PDFs)
2. PyMuPDF4LLM conversion (for normal PDFs with layout preservation)
   - After success: `_supplement_vector_drawings()` detects and inserts chapter title page images
   - Duplicate detection: replaces pymupdf4llm images with higher-quality full-page vector drawings
   - Cover page insertion: places cover at markdown beginning
3. Page-level OCR conversion (for PDFs with font encoding issues)
4. Document structure analysis (heading sizes)
5. Per-page extraction: tables → text (excluding table regions) → images (raster + vector drawings)
6. Remove text blocks overlapping full-page drawing images
7. Caption-to-figure/table association (within 100pt distance)
8. Sort blocks with column detection (`_sort_blocks_with_columns()` -- detects two-column layout via X-coordinate distribution)
9. Generate Markdown with page separators

### Image Extraction (two-stage)

- **Raster images** (`_extract_raster_images`) -- traditional `page.get_images()` for embedded bitmaps, skip < 50x50px
- **Vector drawings** (`_extract_drawing_blocks`) -- `page.get_drawings()` → spatial clustering (`_cluster_drawing_rects`) → filtering (skip single lines, code block backgrounds, table overlaps) → render via `page.get_pixmap(clip=rect)` as PNG. Full-page drawing clusters (chapter pages) render the whole page; partial clusters render with margin.

### OCR Conversion Features (v3.1+)

**Font Encoding Detection** (`_check_font_encoding_issue`):
- Samples first 3 pages of text
- Detects when single character dominates (>30% of text)
- Automatically switches to page-level OCR when issues detected

**Page-Level OCR** (`_convert_with_page_ocr`):
- Renders each page as image (200 DPI)
- Extracts images with position information
- Uses EasyOCR with bounding box coordinates
- Reconstructs layout based on Y/X positions
- Detects headings based on relative text size
- Combines text and images in correct reading order

**PyMuPDF4LLM Integration** (`_convert_with_pymupdf4llm`):
- Uses pymupdf4llm for layout-aware conversion
- Supports automatic OCR (with pymupdf-layout)
- Extracts images with proper markdown references

**PDF2MDGUI**: tkinter GUI with:
- File list via Treeview
- Drag & drop support (requires tkinterdnd2)
- Background conversion threading
- Progress tracking

### Conversion Flow

```
PDF → Font encoding check
    → (if issue) Page-level OCR conversion
    → (if pymupdf4llm available) Layout-aware conversion
        → _supplement_vector_drawings() (detect/insert chapter title images, replace duplicates)
    → (fallback) Standard pipeline:
        → DocumentAnalyzer (font analysis)
        → Per page:
            → AdvancedTableExtractor.extract_tables()
            → _extract_text_blocks() (excluding table regions)
            → _extract_image_blocks() (raster + vector drawings)
        → _remove_text_in_drawing_images() (full-page images only)
        → _associate_captions() (link captions to nearest figure/table)
        → _sort_blocks_with_columns() (two-column detection and ordering)
        → _generate_markdown()
```

## Key Dependencies

- **PyMuPDF (fitz)**: PDF parsing, text extraction, table detection, image extraction, vector drawing rendering
- **pymupdf4llm**: Layout-aware PDF to Markdown conversion
- **pymupdf-layout**: Enhanced layout detection with ML (optional, enables OCR)
- **EasyOCR**: OCR for Japanese and English text in images (pytesseract as fallback)
- **anthropic**: Claude API SDK for diagram/flowchart AI analysis (optional, requires ANTHROPIC_API_KEY env var)
- **Pillow**: Image processing
- **numpy**: Array processing for OCR
- **tkinter/tkinterdnd2**: GUI and drag-drop support
- **PyInstaller**: EXE packaging (uses PDF2MD.spec for config)

Note: EasyOCR is commented out in `requirements.txt` because it requires 64-bit Python with PyTorch. Install manually: `pip install easyocr`

The application is designed for Japanese documents with English support. OCR defaults to `['ja', 'en']`. The README and GUI are in Japanese.

## Important Conventions

- OCR reader is initialized once in `AdvancedPDFConverter.__init__()` and reused across conversions
- GUI runs conversions in a background thread to keep UI responsive
- Images smaller than 50px are excluded as icons
- Page separators (`---`) are inserted between pages in Markdown output
- The EXE build uses `console=True` in the spec file to show conversion progress
