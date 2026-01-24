# Document Understanding Module - Consolidated Structure

## Files

### 1. **document_understanding_engine.py** (Self-Contained Engine)
- **Purpose**: Single, comprehensive module containing all document understanding functionality
- **Contents**:
  - Visualization utilities (plot_bbox, draw_detections)
  - PDF processing utilities (pdf_to_image, pdf_page_count)
  - Spatial association utilities (calculate_distance, crop_and_combine)
  - Model loading logic
  - Inference logic
  - DocumentUnderstandingEngine class with 15+ methods
  
### 2. **doc_layout_extraction.py** (Examples & Reference)
- **Purpose**: Original DocLayout-YOLO demonstration script
- **Contents**: Comprehensive examples of document layout extraction from PDFs
- **Use**: Reference for raw DocLayout-YOLO API usage

## Quick Start

```python
from PIL import Image
from document_understanding_engine import create_engine

# Create and load engine
engine = create_engine()

# Process a single image
image = Image.open("document_page.jpg")
detections = engine.detect_layout(image)

# Process entire PDF
results = engine.analyze_document("document.pdf")

# Extract figure-caption pairs
figure_pairs = engine.extract_figure_caption_pairs("document.pdf")

# Extract table-caption pairs
table_pairs = engine.extract_table_caption_pairs("document.pdf")

# Extract all pairs at once
all_pairs = engine.extract_all_pairs("document.pdf")
```

## Engine Methods

### Detection Tasks
- `detect_layout()` - Detect layout elements in an image
- `detect_layout_with_page()` - Detect with page number tracking
- `visualize_detections()` - Visualize detected elements

### PDF Processing Tasks
- `process_pdf_page()` - Process a single PDF page
- `process_pdf()` - Process entire PDF document

### Association Tasks
- `associate_figures_with_captions()` - Associate figures with captions using spatial proximity
- `associate_tables_with_captions()` - Associate tables with captions (handles above/below)

### Extraction Tasks
- `extract_figure_caption_pairs()` - Extract all figure-caption pairs from PDF
- `extract_table_caption_pairs()` - Extract all table-caption pairs from PDF
- `extract_all_pairs()` - Extract both figures and tables in one pass

### Utility Methods
- `analyze_document()` - Comprehensive document analysis with statistics
- `display_extracted_pairs()` - Display extracted pairs visually

## Visualization Functions

Available as module-level functions:
- `plot_bbox(image, data)` - Plot bounding boxes on image
- `draw_detections(detections_df)` - Draw all detections with colored boxes

## PDF Utilities

Available as module-level functions:
- `pdf_to_image(pdf_path, page_number, dpi)` - Convert PDF page to image
- `pdf_page_count(pdf_path)` - Get total page count

## Spatial Association Utilities

Available as module-level functions:
- `calculate_distance(main_bbox, caption_bbox)` - Calculate spatial proximity
- `crop_and_combine(main_bbox, caption_bbox, page_image, pair_type, page_number)` - Crop and combine regions

## Architecture

```
document_understanding_engine.py
│
├── Visualization Utilities
│   ├── COLORMAP
│   ├── plot_bbox()
│   └── draw_detections()
│
├── PDF Utilities
│   ├── pdf_to_image()
│   └── pdf_page_count()
│
├── Spatial Association Utilities
│   ├── calculate_distance()
│   └── crop_and_combine()
│
└── DocumentUnderstandingEngine
    ├── __init__()
    ├── load()
    ├── _ensure_loaded()
    ├── Detection Methods (3)
    ├── PDF Processing Methods (2)
    ├── Association Methods (2)
    ├── Extraction Methods (3)
    └── Utility Methods (2)
```

## Supported Document Elements

The engine can detect and extract:
- **Figures** - Images, diagrams, charts
- **Figure Captions** - Text descriptions for figures
- **Tables** - Tabular data structures
- **Table Captions** - Text descriptions for tables
- **Text Blocks** - Paragraphs and text regions
- **Headings** - Section titles
- **Lists** - Bulleted and numbered lists
- **Equations** - Mathematical formulas
- **Footnotes** - Reference notes

## Spatial Association Logic

### Figure-Caption Association
- **Primary Strategy**: Finds captions **below** figures
- **Proximity Metrics**:
  1. Vertical distance (caption top to figure bottom)
  2. Horizontal overlap (prefers aligned captions)
- **Constraints**: Same page only, one-to-one mapping

### Table-Caption Association
- **Primary Strategy**: Finds captions **above or below** tables
- **Proximity Metrics**:
  1. Combined vertical + horizontal distance
  2. Minimizes total proximity score
- **Constraints**: Same page only, one-to-one mapping

## Benefits of Consolidation

1. **Single Import** - One file to import, no dependencies
2. **Self-Contained** - All functionality in one place
3. **Easy Deployment** - Copy one file to use anywhere
4. **No Module Confusion** - Clear, simple structure
5. **Maintainable** - All related code together
6. **PDF-Ready** - Built-in multi-page PDF processing
7. **Smart Association** - Automatic figure/table-caption pairing

## Example Workflow

```python
from document_understanding_engine import create_engine

# Initialize engine
engine = create_engine()

# Analyze entire document
analysis = engine.analyze_document("research_paper.pdf")

print(f"Pages: {analysis['page_count']}")
print(f"Total detections: {analysis['total_detections']}")
print(f"Class distribution: {analysis['class_counts']}")
print(f"Figure-caption pairs: {analysis['figure_caption_pairs']}")
print(f"Table-caption pairs: {analysis['table_caption_pairs']}")

# Access extracted pairs
figures = analysis['extracted_pairs']['figures']
tables = analysis['extracted_pairs']['tables']

# Display first 3 figure pairs
engine.display_extracted_pairs(figures, max_display=3)

# Display all table pairs
engine.display_extracted_pairs(tables, max_display=len(tables))
```

## Dependencies

- doclayout-yolo
- huggingface-hub
- PyMuPDF (fitz)
- opencv-python (cv2)
- pandas
- PIL/Pillow
- matplotlib
- numpy

## Installation

```bash
pip install doclayout-yolo huggingface-hub PyMuPDF opencv-python pandas Pillow matplotlib numpy
```

## Model Information

- **Default Model**: `juliozhao/DocLayout-YOLO-DocStructBench`
- **Model File**: `doclayout_yolo_docstructbench_imgsz1024.pt`
- **Input Size**: 1024x1024 (optimized)
- **Framework**: YOLOv10 architecture
- **Source**: HuggingFace Hub (auto-downloaded on first use)

## Performance Considerations

- **DPI Setting**: Default 200 DPI balances quality and speed
- **Batch Processing**: Processes entire PDFs page-by-page
- **Memory Usage**: Stores all page images in memory during processing
- **Speed**: ~1-3 seconds per page (depending on complexity)

## Total LOC

- **document_understanding_engine.py**: ~700 lines
- **doc_layout_extraction.py**: ~798 lines (examples)

**Total**: ~1,500 lines of clean, well-documented code

## Future Enhancements

- [ ] OCR integration for caption text extraction
- [ ] Multi-column layout support
- [ ] Cross-page figure/table handling
- [ ] Export to structured formats (JSON, XML)
- [ ] Batch processing optimization
- [ ] GPU acceleration support
- [ ] Custom model fine-tuning utilities
