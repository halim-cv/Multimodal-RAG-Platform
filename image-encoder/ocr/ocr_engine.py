"""
OCR functionality (EasyOCR only)
"""
from pathlib import Path
from typing import Dict, List, Any
import logging
from PIL import Image, ImageDraw

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False
    logging.warning("easyocr not available")

import config

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.4


def draw_ocr_boxes(image_path: Path, boxes: List[Dict[str, Any]], output_path: Path) -> Path:
    """
    Draw bounding boxes on image
    
    Args:
        image_path: Path to source image
        boxes: List of box dicts (must have 'bbox' with x,y,width,height)
        output_path: Path to save annotated image
        
    Returns:
        output_path
    """
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        for item in boxes:
            bbox = item['bbox']
            # Draw rectangle
            rect = [
                bbox['x'], 
                bbox['y'], 
                bbox['x'] + bbox['width'], 
                bbox['y'] + bbox['height']
            ]
            draw.rectangle(rect, outline="red", width=2)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        logger.info(f"Saved annotated image: {output_path.name}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to draw boxes: {e}")
        return None


def run_ocr(image_path: Path) -> Dict[str, Any]:
    """
    Run OCR on an image (EasyOCR only)
    
    Args:
        image_path: Path to image
    
    Returns:
        Dictionary with 'text' and 'boxes' (list of {text, bbox})
    """
    if not config.OCR_ENABLED:
        logger.debug("OCR disabled in config")
        return {"text": None, "boxes": []}
    
    logger.info(f"Running OCR on {image_path.name} with EasyOCR (GPU)")
    return run_easyocr_ocr(image_path)


def run_easyocr_ocr(image_path: Path) -> Dict[str, Any]:
    """
    Run EasyOCR
    
    Returns:
        Dictionary with 'text' and 'boxes'
    """
    if not HAS_EASYOCR:
        logger.error("EasyOCR not available")
        return {"text": None, "boxes": []}
    
    try:
        # Initialize reader (cached globally in production)
        reader = easyocr.Reader(['en'], gpu=True)
        
        # Read text
        results = reader.readtext(str(image_path))
        
        # Parse results
        text_parts = []
        boxes = []
        
        for (bbox_coords, text, confidence) in results:
            # Check confidence
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            text_parts.append(text)
            
            # Convert bbox format and types (numpy to native)
            x_coords = [pt[0] for pt in bbox_coords]
            y_coords = [pt[1] for pt in bbox_coords]
            
            x_min = int(min(x_coords))
            y_min = int(min(y_coords))
            x_max = int(max(x_coords))
            y_max = int(max(y_coords))
            
            bbox = {
                'x': x_min,
                'y': y_min,
                'width': x_max - x_min,
                'height': y_max - y_min
            }
            
            boxes.append({
                'text': text,
                'bbox': bbox,
                'confidence': float(confidence)
            })
        
        full_text = ' '.join(text_parts)
        logger.info(f"EasyOCR extracted {len(full_text)} chars, {len(boxes)} boxes")
        
        return {
            "text": full_text if full_text else None,
            "boxes": boxes
        }
        
    except Exception as e:
        logger.error(f"EasyOCR failed for {image_path.name}: {e}")
        return {"text": None, "boxes": []}
