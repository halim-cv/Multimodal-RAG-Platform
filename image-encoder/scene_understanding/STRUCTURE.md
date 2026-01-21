# Scene Understanding Module - Consolidated Structure

## Files

### 1. **scene_understanding_engine.py** (Self-Contained Engine)
- **Purpose**: Single, comprehensive module containing all scene understanding functionality
- **Contents**:
  - Visualization utilities (plot_bbox, draw_polygons, draw_ocr_bboxes, convert_to_od_format)
  - Model loading logic
  - Inference logic
  - SceneUnderstandingEngine class with 20+ methods
  
### 2. **base_functionalities.py** (Examples & Reference)
- **Purpose**: Original Florence-2 demonstration script
- **Contents**: Comprehensive examples of all Florence-2 capabilities
- **Use**: Reference for raw Florence-2 API usage

### 3. **test_engine.py** (Test Suite)
- **Purpose**: Comprehensive test script for the Scene Understanding Engine
- **Contents**: 8 test scenarios covering all major functionalities
- **Use**: Validation and demonstration of engine capabilities

## Quick Start

```python
from PIL import Image
from scene_understanding_engine import create_engine

# Load image and create engine
image = Image.open("image.jpg")
engine = create_engine()

# Use any of 20+ methods
caption = engine.caption(image)
objects = engine.object_detection(image)
text = engine.ocr(image)
```

## Engine Methods

### Captioning
- `caption()` - Basic caption
- `detailed_caption()` - Detailed caption
- `more_detailed_caption()` - More detailed caption
- `all_captions()` - All levels at once

### Object Detection
- `object_detection()` - Standard object detection
- `dense_region_caption()` - Dense region descriptions
- `region_proposal()` - Region proposals
- `phrase_grounding()` - Locate objects by phrase
- `open_vocabulary_detection()` - Open vocabulary detection

### Segmentation
- `referring_expression_segmentation()` - Segment by expression
- `region_to_segmentation()` - Region to mask

### Region Understanding
- `region_to_category()` - Categorize region
- `region_to_description()` - Describe region

### OCR
- `ocr()` - Extract text
- `ocr_with_region()` - Text with bounding boxes

### Utilities
- `caption_and_ground()` - Cascaded caption + grounding
- `analyze_image()` - Batch processing
- `process_image_path()` - Process from file path

## Visualization Functions

Available as module-level functions:
- `plot_bbox(image, data)` - Plot bounding boxes
- `draw_polygons(image, prediction, fill_mask)` - Draw segmentation masks
- `draw_ocr_bboxes(image, prediction, scale)` - Draw OCR boxes
- `convert_to_od_format(data)` - Convert to standard OD format

## Architecture

```
scene_understanding_engine.py
│
├── Visualization Utilities
│   ├── COLORMAP
│   ├── plot_bbox()
│   ├── draw_polygons()
│   ├── draw_ocr_bboxes()
│   └── convert_to_od_format()
│
└── SceneUnderstandingEngine
    ├── __init__()
    ├── load()
    ├── _run_task()
    ├── Captioning Methods (4)
    ├── Detection Methods (5)
    ├── Segmentation Methods (2)
    ├── Region Methods (2)
    ├── OCR Methods (2)
    └── Utility Methods (3)
```

## Benefits of Consolidation

1. **Single Import** - One file to import, no dependencies
2. **Self-Contained** - All functionality in one place
3. **Easy Deployment** - Copy one file to use anywhere
4. **No Module Confusion** - Clear, simple structure
5. **Maintainable** - All related code together

## Running Tests

```powershell
python test_engine.py
```

This will run 8 comprehensive tests covering all major engine features.

## Dependencies

- transformers
- torch
- PIL/Pillow
- matplotlib
- numpy

## Total LOC

- **scene_understanding_engine.py**: ~500 lines
- **base_functionalities.py**: ~374 lines (examples)
- **test_engine.py**: ~105 lines (tests)

**Total**: ~980 lines of clean, well-documented code
