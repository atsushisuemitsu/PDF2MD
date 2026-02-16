# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF2MD is a Windows desktop application (v3.1) that converts PDF files to Markdown format. It uses PyMuPDF for PDF parsing and EasyOCR for optical character recognition. The application supports both GUI (tkinter) and CLI modes. The entire application is a single Python file: `pdf2md.py`.

Primary language: Python 3.13 on Windows. UI labels and comments are in Japanese.

## Build Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application (GUI mode)
python pdf2md.py

# Run in CLI mode
python pdf2md.py document.pdf
python pdf2md.py --ocr document.pdf
python pdf2md.py ./pdf_folder/

# Build EXE (Windows)
build.bat
# or
powershell -ExecutionPolicy Bypass -File build.ps1

# The EXE is generated at dist/PDF2MD.exe
```

## Architecture

The codebase is a single-file application (`pdf2md.py`) with the following key components:

### Data Classes
- `TextBlock`: Text with position, font info, and block type (heading1-6, list, caption, text)
- `ImageBlock`: Image with position, OCR text, and caption association
- `TableBlock`: Table cells with position and header row detection
- `ListItem`: List item with level and type (bullet/numbered)

### Core Classes

**DocumentAnalyzer**: Analyzes entire document to determine heading hierarchy by:
- Collecting font sizes across all pages
- Identifying body text size (most frequent)
- Mapping larger sizes to heading levels (H1-H6)

**AdvancedTableExtractor**: Two-stage table detection:
1. PyMuPDF's `find_tables()` for bordered tables
2. Text position analysis for borderless tables (grids aligned by X coordinates)

**ListDetector**: Detects list items using regex patterns for:
- Bullet markers: `- • ● ○ ■ □ ・ ※ ★ ☆ → ⇒ ▶ ►`
- Numbered patterns: `1.`, `(1)`, `a)`, `(a)`, `i.`, etc.

**CaptionDetector**: Identifies figure/table captions matching patterns like:
- 図1, Fig. 1, Figure 1, グラフ1, Chart 1, 写真1, Photo 1
- 表1, Tab. 1, Table 1

**AdvancedPDFConverter**: Main conversion pipeline:
1. Font encoding issue detection (auto-detect problematic PDFs)
2. PyMuPDF4LLM conversion (for normal PDFs with layout preservation)
3. Page-level OCR conversion (for PDFs with font encoding issues)
4. Document structure analysis (heading sizes)
5. Per-page extraction: tables → text (excluding table regions) → images (raster + vector drawings)
6. Remove text blocks overlapping full-page drawing images
7. Caption-to-figure/table association (within 100pt distance)
8. Sort blocks by page → Y → X coordinates
9. Generate Markdown with page separators

### Image Extraction (two-stage)

- **Raster images** (`_extract_raster_images`) — traditional `page.get_images()` for embedded bitmaps, skip < 50x50px
- **Vector drawings** (`_extract_drawing_blocks`) — `page.get_drawings()` → spatial clustering (`_cluster_drawing_rects`) → filtering (skip single lines, code block backgrounds, table overlaps) → render via `page.get_pixmap(clip=rect)` as PNG. Full-page drawing clusters (chapter pages) render the whole page; partial clusters render with margin.

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
    → (fallback) Standard pipeline:
        → DocumentAnalyzer (font analysis)
        → Per page:
            → AdvancedTableExtractor.extract_tables()
            → _extract_text_blocks() (excluding table regions)
            → _extract_image_blocks() (raster + vector drawings)
        → _remove_text_in_drawing_images() (full-page images only)
        → _associate_captions() (link captions to nearest figure/table)
        → Sort by position
        → _generate_markdown()
```

## Key Dependencies

- **PyMuPDF (fitz)**: PDF parsing, text extraction, table detection, image extraction, vector drawing rendering
- **pymupdf4llm**: Layout-aware PDF to Markdown conversion
- **pymupdf-layout**: Enhanced layout detection with ML (optional, enables OCR)
- **EasyOCR**: OCR for Japanese and English text in images (pytesseract as fallback)
- **Pillow**: Image processing
- **numpy**: Array processing for OCR
- **tkinter/tkinterdnd2**: GUI and drag-drop support
- **PyInstaller**: EXE packaging (uses PDF2MD.spec for config)

## Language Support

The application is designed for Japanese documents with English support. OCR defaults to `['ja', 'en']`. The README and GUI are in Japanese.

## Known Gaps

- CLI flags `--ocr` and `--no-images` are documented in README but not implemented in `main()` — CLI always processes with OCR enabled and images extracted.
