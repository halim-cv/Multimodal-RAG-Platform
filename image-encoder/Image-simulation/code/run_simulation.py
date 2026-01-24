"""
Simulation Script for Scene Understanding Engine
Performs OCR text extraction, OCR bounding box annotation, and more detailed caption
"""

import sys
from pathlib import Path
from PIL import Image

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scene_understanding'))

from scene_understanding.scene_understanding_engine import SceneUnderstandingEngine, draw_ocr_bboxes

# Paths
#DATA_RAW_DIR = project_root / 'data' / 'raw'
TEST_IMAGE_PATH = Path(r'C:\Users\Sam-tech\Desktop\Github\Multimodal-RAG-Platform\image-encoder\Folder-simulation\assets\figure_page10_3.png')
ASSETS_DIR = current_dir.parent / 'assets'

# Output paths
OCR_TEXT_OUTPUT = ASSETS_DIR / 'extracted_text.txt'
ANNOTATED_IMAGE_OUTPUT = ASSETS_DIR / 'annotated_image.png'

# Model configuration
MODEL_ID = 'microsoft/Florence-2-base'
DEVICE = 'cuda'


def main():
    print("Scene Understanding Engine - Simulation")
    print("-" * 50)
    
    # Validate and load image
    if not TEST_IMAGE_PATH.exists():
        print(f"Error: Test image not found at {TEST_IMAGE_PATH}")
        return
    
    try:
        test_image = Image.open(TEST_IMAGE_PATH)
        test_image.verify()
        test_image = Image.open(TEST_IMAGE_PATH)
    except Exception as e:
        print(f"Error: Invalid image file - {e}")
        return
    
    print(f"Loading image: {TEST_IMAGE_PATH}")

    
    # Initialize engine
    print(f"Loading model: {MODEL_ID}")
    engine = SceneUnderstandingEngine(model_id=MODEL_ID, device=DEVICE)
    engine.load()
    
    # Perform OCR with regions
    print("Performing OCR text extraction...")
    ocr_with_regions = engine.ocr_with_region(test_image)
    
    # Generate more detailed caption
    print("Generating more detailed caption...")
    detailed_caption = engine.more_detailed_caption(test_image)
    
    # Create assets directory
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save extracted text
    print(f"Saving extracted text to {OCR_TEXT_OUTPUT}")
    with open(OCR_TEXT_OUTPUT, 'w', encoding='utf-8') as f:
        f.write("OCR Text Extraction Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Image: {TEST_IMAGE_PATH.name}\n\n")
        f.write("More Detailed Caption:\n")
        f.write(detailed_caption + "\n\n")
        f.write("Extracted Text Regions:\n")
        f.write("-" * 50 + "\n")
        for i, label in enumerate(ocr_with_regions.get('labels', []), 1):
            f.write(f"[Region {i}] {label}\n")
    
    # Save annotated image with OCR bounding boxes
    print(f"Saving annotated image to {ANNOTATED_IMAGE_OUTPUT}")
    annotated_image = test_image.copy()
    if ocr_with_regions.get('quad_boxes'):
        annotated_image = draw_ocr_bboxes(annotated_image, ocr_with_regions)
    annotated_image.save(ANNOTATED_IMAGE_OUTPUT)
    
    # Offload model from GPU to free memory
    print("Offloading model from GPU...")
    if hasattr(engine, 'model') and engine.model is not None:
        del engine.model
    if hasattr(engine, 'processor') and engine.processor is not None:
        del engine.processor
    del engine
    
    # Clear CUDA cache
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("-" * 50)
    print("Simulation complete")
    print(f"Results saved to: {ASSETS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
