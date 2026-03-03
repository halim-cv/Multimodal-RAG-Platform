# Multimodal-RAG-Platform

A modular **Retrieval-Augmented Generation (RAG)** platform that ingests and understands content across four modalities: **text, image, audio,** and a **frontend** chat interface.

---

## Repository Structure

```
Multimodal-RAG-Platform/
├── image-encoder/          ← Visual content pipeline (this doc)
│   ├── document_understanding/  ← PDF layout detection + figure/table extraction
│   ├── scene_understanding/     ← Image captioning, OCR, VQA, segmentation
│   ├── image_pipeline.py        ← Two-phase orchestrator
│   ├── Folder-simulation/       ← Document extraction demo
│   ├── Image-simulation/        ← Scene understanding demo
│   ├── Input/                   ← Drop PDFs / images here
│   ├── output/                  ← Pipeline results (JSON + PNGs)
│   ├── requirements.txt
│   └── tests/
│       └── test_spatial_association.py
├── Text-encoding/          ← PDF text extraction + embedding
├── audio-encoder/          ← Audio transcription + understanding
└── frontend/               ← Chat interface (HTML/CSS/JS)
```

---

## Image Encoder – Overview

The image encoder converts unstructured PDFs and images into structured, searchable metadata through a **two-phase pipeline**:

```
Input (PDFs + Images)
       │
       ▼
┌─────────────────────────────────┐
│  Phase 1: Document Understanding │  DocLayout-YOLO (YOLOv10)
│  • Detect page layout elements   │
│  • Associate figures ↔ captions  │
│  • Crop & save figure-caption PNGs│
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Phase 2: Scene Understanding    │  Florence-2 (microsoft/Florence-2-base)
│  • Generate detailed captions    │
│  • Extract text via OCR          │
│  • Save metadata JSON            │
└─────────────────────────────────┘
                 │
                 ▼
      output/all_metadata.json
```

### Models Used

| Phase | Model | Source | License |
|---|---|---|---|
| Document Understanding | `juliozhao/DocLayout-YOLO-DocStructBench` (YOLOv10) | HuggingFace Hub | Apache 2.0 |
| Scene Understanding | `microsoft/Florence-2-base` | HuggingFace Hub | MIT |

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA 12.x (optional but strongly recommended — CPU fallback is ~10× slower)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Multimodal-RAG-Platform.git
cd Multimodal-RAG-Platform

# 2. Create conda environment
conda create -n rag-platform python=3.10
conda activate rag-platform

# 3. Install PyTorch with CUDA (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install image-encoder dependencies
pip install -r image-encoder/requirements.txt
```

> **CPU-only machines:** The pipeline will automatically detect the absence of CUDA and configure appropriate thread limits. No extra steps needed.

---

## Usage

### Full Pipeline (PDFs + Images → Metadata)

```bash
cd image-encoder

# Place your PDFs and/or images in the Input/ directory, then:
python image_pipeline.py
```

Results will appear in:
```
image-encoder/output/
├── extracted_figures/       ← Cropped figure-caption PNG pairs
├── metadata/                ← Per-image JSON metadata files
└── all_metadata.json        ← Combined metadata for all images
```

### Document Extraction Simulation (figure extraction only)

```bash
cd image-encoder
python Folder-simulation/code/run_simulation.py
```

Extracted figure PNGs are saved to `Folder-simulation/assets/`.

### Scene Understanding Simulation (caption + OCR on one image)

```bash
cd image-encoder
python Image-simulation/code/run_simulation.py
```

Results saved to `Image-simulation/assets/`:
- `annotated_image.png` — original image with OCR bounding boxes drawn
- `extracted_text.txt` — detailed caption + per-region OCR text

### Python API

```python
from PIL import Image

# ── Document Understanding ──────────────────────────────────────────
from document_understanding.document_understanding_engine import create_engine

engine = create_engine()  # downloads model on first run (~150 MB)

# Detect layout elements in a single image
from PIL import Image
img = Image.open("page.png")
detections = engine.detect_layout(img)  # → pandas DataFrame

# Extract all figure + table-caption pairs from a PDF
pairs = engine.extract_all_pairs("document.pdf")
# → {'figures': [...], 'tables': [...], 'total_pairs': N}

# Full document analysis with statistics
analysis = engine.analyze_document("document.pdf")
print(f"Pages: {analysis['page_count']}")
print(f"Figures found: {analysis['figure_caption_pairs']}")


# ── Scene Understanding ─────────────────────────────────────────────
from scene_understanding.scene_understanding_engine import create_engine as create_scene_engine

scene = create_scene_engine()  # downloads Florence-2-base (~460 MB)

img = Image.open("figure.png")

# Captions at 3 levels of detail
print(scene.caption(img))              # short
print(scene.detailed_caption(img))    # paragraph
print(scene.more_detailed_caption(img))  # full description

# OCR
text = scene.ocr(img)                 # plain text
regions = scene.ocr_with_region(img)  # {quad_boxes, labels}

# Object detection
objects = scene.object_detection(img)  # {bboxes, labels}

# Visual Question Answering
answer = scene.visual_question_answering(img, "What does the legend show?")

# Run multiple tasks at once
results = scene.analyze_image(img, tasks=['caption', 'ocr', 'object_detection'])
```

---

## Output Schema

Each processed image produces a metadata record:

```json
{
  "path":             "/absolute/path/to/image.png",
  "filename":         "HighlightedV1_page4_fig1.png",
  "file_source":      "/absolute/path/to/source.pdf",
  "source_type":      "document",
  "detailed_caption": "The image is a diagram that shows the multi-modal projector...",
  "extracted_text":   "Mixer Layer x N ... Figure 2: Multi-modal projector...",
  "page_number":      4,
  "pair_type":        "figure-caption"
}
```

| Field | Type | Description |
|---|---|---|
| `source_type` | `"document"` \| `"original_image"` | Whether extracted from PDF or input directly |
| `pair_type` | `"figure-caption"` \| `"table-caption"` | Only present for PDF-extracted items |
| `detailed_caption` | string | Florence-2 `MORE_DETAILED_CAPTION` output |
| `extracted_text` | string | Florence-2 `OCR` output |

---

## Hardware Requirements

| Scenario | Minimum | Recommended |
|---|---|---|
| Full pipeline (both phases) | 8 GB RAM, CPU | 16 GB RAM + GPU with 6 GB VRAM |
| Document Understanding only | 4 GB RAM, CPU | GPU with 2 GB VRAM |
| Scene Understanding only | 8 GB RAM, CPU | GPU with 4 GB VRAM |

**Approximate latency (16-page PDF + 1 image):**

| Hardware | Phase 1 (DocLayout-YOLO) | Phase 2 (Florence-2) | Total |
|---|---|---|---|
| CPU (8-core) | ~25–40 s | ~3–6 min | ~4–7 min |
| GPU (T4, 16 GB VRAM) | ~5–8 s | ~20–40 s | ~30–50 s |

> The pipeline auto-offloads each model from GPU memory before loading the next, so both models never need to be in VRAM simultaneously.

---

## Running Tests

```bash
cd image-encoder

# Run all unit tests (no GPU or model download required)
pytest tests/ -v

# Run the CPU constraint test
python test_cpu_constraints.py
```

The unit tests in `tests/test_spatial_association.py` cover:
- `calculate_distance()` — 5 spatial geometry scenarios
- `crop_and_combine()` — dtype, shape, width/height, metadata fields
- `associate_figures_with_captions()` — greedy matching, exclusivity, multi-figure cases

---

## Document Understanding – Spatial Association Logic

Figures are matched to captions using a **greedy one-to-one nearest-neighbour algorithm**:

1. For each detected figure, collect all captions on the **same page**
2. Rank candidates by:
   - **Primary key:** vertical distance (caption top → figure bottom)
   - **Tiebreaker:** horizontal proximity (prefers captions with more x-overlap)
3. Assign the best candidate and **remove it** from the pool (one-to-one constraint)

**Figure captions** → must be **below** the figure (standard academic convention)  
**Table captions** → can be **above or below** the table

---

## Scene Understanding – Supported Tasks

| Category | Method | Florence-2 Prompt |
|---|---|---|
| Captioning | `caption()` | `<CAPTION>` |
| Captioning | `detailed_caption()` | `<DETAILED_CAPTION>` |
| Captioning | `more_detailed_caption()` | `<MORE_DETAILED_CAPTION>` |
| Detection | `object_detection()` | `<OD>` |
| Detection | `dense_region_caption()` | `<DENSE_REGION_CAPTION>` |
| Detection | `phrase_grounding(phrase)` | `<CAPTION_TO_PHRASE_GROUNDING>` |
| Detection | `open_vocabulary_detection(query)` | `<OPEN_VOCABULARY_DETECTION>` |
| Segmentation | `referring_expression_segmentation(expr)` | `<REFERRING_EXPRESSION_SEGMENTATION>` |
| OCR | `ocr()` | `<OCR>` |
| OCR | `ocr_with_region()` | `<OCR_WITH_REGION>` |
| VQA | `visual_question_answering(question)` | cascaded |

---

## Limitations & Known Issues

- **Multi-column PDFs:** Table-caption association may mis-pair elements across adjacent columns (column boundaries not modelled)
- **Cross-page elements:** Figures or tables that span two pages are not supported
- **OCR in caption text:** `extract_with_ocr()` in `extractors.py` is a placeholder — caption text is not yet extracted as a separate field (the full image OCR from Phase 2 covers it indirectly)
- **PyMuPDF licence:** This library is AGPL-3.0 — if you distribute this project, your code must also be AGPL-3.0 unless you purchase a commercial licence

---

## License

> **⚠️ To be decided.** PyMuPDF is AGPL-3.0, which propagates to derivative works if distributed publicly. Other dependencies (Florence-2: MIT, DocLayout-YOLO: Apache 2.0) are permissive. Choose between AGPL-3.0 for open distribution or purchase a commercial PyMuPDF licence.