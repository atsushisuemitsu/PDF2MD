# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF2MD is a Windows desktop application (single-file: `pdf2md.py`, ~3700 lines, v4.5) that converts PDF and Microsoft Office files (doc/docx/xls/xlsx/xlsm/pptx) to Markdown format. It uses PyMuPDF and PyMuPDF4LLM for PDF parsing, EasyOCR for OCR, MarkItDown for Office and PDF fallback conversion, and optionally Claude API for diagram/flowchart analysis. Supports both GUI (tkinter) and CLI (argparse) modes.

Primary language: Python 3.13 on Windows. UI labels and comments are in Japanese.

## Build & Run Commands

```bash
pip install -r requirements.txt

# GUI mode
python pdf2md.py

# CLI mode - PDF
python pdf2md.py document.pdf
python pdf2md.py --layout precise document.pdf
python pdf2md.py --layout page_image document.pdf
python pdf2md.py --layout legacy document.pdf
python pdf2md.py --no-ocr document.pdf
python pdf2md.py --no-images document.pdf
python pdf2md.py --no-claude document.pdf
python pdf2md.py --dpi 300 document.pdf
python pdf2md.py -o ./output/ document.pdf
python pdf2md.py ./pdf_folder/

# CLI mode - Office (v4.5)
python pdf2md.py document.docx
python pdf2md.py legacy.doc
python pdf2md.py report.xlsx
python pdf2md.py macros.xlsm
python pdf2md.py slides.pptx

# Build EXE (Windows) → dist/PDF2MD.exe
build.bat
```

No tests, no linter, no formatter configured.

## Architecture

Single-file application (`pdf2md.py`) with all classes in one module.

### Layout Modes (v4.0)

`convert_file()` accepts `layout_mode` parameter:
- **auto** (default): pymupdf4llm → fallback to precise layout pipeline
- **precise**: Forces the standard pipeline with Obsidian-compatible multi-column layout (CSS/HTML)
- **page_image**: Renders each page as PNG, outputs `<img>` tags
- **legacy**: pymupdf4llm only, no vector drawing supplement

### Conversion Pipeline

```
convert_file() (line ~1997)
  │
  ├─ _is_supported_input() — extension check (.pdf + Office formats)
  │
  ├─ _is_office_file() (v4.5) → _convert_office_file()
  │    MarkItDown conversion + ZIP media extraction (docx/xlsx/xlsm/pptx)
  │    Legacy .doc/.xls: text only (image extraction skipped with warning)
  │    → End
  │
  ├─ page_image mode → _convert_as_page_images() (render each page as PNG)
  │
  ├─ legacy mode → _convert_with_pymupdf4llm() only
  │
  ├─ _check_font_encoding_issue() samples first 3 pages
  │    └─ Single char > 30% of text → font encoding problem
  │
  ├─ [Font issues] → _convert_with_page_ocr()
  │    Renders pages as images (200 DPI) → EasyOCR → reconstruct layout
  │
  ├─ [auto/legacy, no font issues] → _convert_with_pymupdf4llm()
  │    + _supplement_vector_drawings() on success (chapter title pages, duplicate replacement)
  │
  └─ [precise, or pymupdf4llm fallback] → Standard pipeline:
       DocumentAnalyzer → LayoutAnalyzer (column detection)
       → Per page: tables → text → images (raster + vector)
       → _remove_text_in_drawing_images()
       → _associate_captions()
       → _sort_blocks_with_columns()
       → _generate_markdown_obsidian() (multi-column CSS) or _generate_markdown()
```

### Key Classes

| Class | Line | Role |
|-------|------|------|
| `DocumentAnalyzer` | ~187 | Font size analysis → heading hierarchy (H1-H6) |
| `AdvancedTableExtractor` | ~279 | PyMuPDF `find_tables()` + text-position grid for borderless tables |
| `ListDetector` | ~523 | Regex-based bullet/numbered list detection with hierarchy |
| `CaptionDetector` | ~596 | Matches 図1, Fig. 1, 表1, Table 1, etc. |
| `LayoutAnalyzer` | ~636 | Column detection (2/3-column) via X-coordinate distribution, header/footer detection |
| `ClaudeDiagramAnalyzer` | ~773 | Claude API vision for diagram → Markdown tables / PlantUML |
| `AdvancedPDFConverter` | ~1023 | Main converter with all conversion paths, image extraction, OCR |
| `PDF2MDGUI` | ~2692 | tkinter GUI with Treeview, threading, drag-and-drop, layout mode radio buttons |

### Data Classes

- `TextBlock`: Text with position, font info, block type, `column_index`
- `ImageBlock`: Image with position, OCR text, caption, Claude analysis
- `TableBlock`: Table cells with position, header row detection
- `ListItem`: List item with level and type
- `LayoutRegion` / `PageLayout` (v4.0): Column boundaries, header/footer regions

### Image Extraction (two-stage)

- **Raster images** (`_extract_raster_images`): `page.get_images()` for embedded bitmaps, skip < 50x50px
- **Vector drawings** (`_extract_drawing_blocks`): `page.get_drawings()` → spatial clustering → filtering (skip lines, code backgrounds, table overlaps) → render via `page.get_pixmap(clip=rect)`. Full-page clusters render the whole page.

### Optional Dependencies Pattern

Try/except imports with availability flags: `PYMUPDF_AVAILABLE`, `PYMUPDF4LLM_AVAILABLE`, `LAYOUT_AVAILABLE`, `OCR_ENGINE`, `CLAUDE_API_AVAILABLE`, `DND_AVAILABLE`. These control feature degradation at runtime. Maintain this pattern when modifying imports.

### Output Structure

`{name}.md` alongside input PDF + `{name}_images/` with `page{N}_img{M}.png/jpeg`.

## Key Dependencies

- **PyMuPDF (fitz)**: PDF parsing, text/table/image extraction, vector drawing rendering
- **pymupdf4llm**: Layout-aware PDF→Markdown (multi-column support)
- **pymupdf-layout**: Enhanced ML layout detection (optional, enables OCR in pymupdf4llm)
- **EasyOCR**: OCR for Japanese+English (`['ja', 'en']`). Requires 64-bit Python+PyTorch. Commented out in requirements.txt — install manually.
- **anthropic**: Claude API for diagram analysis (optional, needs `ANTHROPIC_API_KEY` env var)
- **tkinterdnd2**: GUI drag-and-drop (optional)

## Important Conventions

- OCR reader initialized once in `AdvancedPDFConverter.__init__()` and reused
- GUI runs conversions in a background thread
- Page separators (`---`) between pages in output
- EXE build uses `console=True` in spec file for progress output
- Windows Explorer right-click integration via `install_context_menu.bat` (requires admin)

  ---

  

  Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

  **Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

  ## 1. Think Before Coding

  **Don't assume. Don't hide confusion. Surface tradeoffs.**

  Before implementing:
  - State your assumptions explicitly. If uncertain, ask.
  - If multiple interpretations exist, present them - don't pick silently.
  - If a simpler approach exists, say so. Push back when warranted.
  - If something is unclear, stop. Name what's confusing. Ask.

  ## 2. Simplicity First

  **Minimum code that solves the problem. Nothing speculative.**

  - No features beyond what was asked.
  - No abstractions for single-use code.
  - No "flexibility" or "configurability" that wasn't requested.
  - No error handling for impossible scenarios.
  - If you write 200 lines and it could be 50, rewrite it.

  Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

  ## 3. Surgical Changes

  **Touch only what you must. Clean up only your own mess.**

  When editing existing code:
  - Don't "improve" adjacent code, comments, or formatting.
  - Don't refactor things that aren't broken.
  - Match existing style, even if you'd do it differently.
  - If you notice unrelated dead code, mention it - don't delete it.

  When your changes create orphans:
  - Remove imports/variables/functions that YOUR changes made unused.
  - Don't remove pre-existing dead code unless asked.

  The test: Every changed line should trace directly to the user's request.

  ## 4. Goal-Driven Execution

  **Define success criteria. Loop until verified.**

  Transform tasks into verifiable goals:
  - "Add validation" → "Write tests for invalid inputs, then make them pass"
  - "Fix the bug" → "Write a test that reproduces it, then make it pass"
  - "Refactor X" → "Ensure tests pass before and after"

  For multi-step tasks, state a brief plan:
  ```
  1. [Step] → verify: [check]
  2. [Step] → verify: [check]
  3. [Step] → verify: [check]
  ```

  Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

  ---

  **These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
