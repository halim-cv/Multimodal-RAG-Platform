"""
OCR functionality (optional) using Tesseract or EasyOCR
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    logging.warning("pytesseract not available")

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False
    logging.warning("easyocr not available")

import config

logger = logging.getLogger(__name__)


def run_ocr(image_path: Path, engine: str = None) -> Dict[str, Any]:
    """
    Run OCR on an image
    
    Args:
        image_path: Path to image
        engine: OCR engine to use ('tesseract' or 'easyocr'), defaults to config
    
    Returns:
        Dictionary with 'text' and 'boxes' (list of {text, bbox})
    """
    if not config.OCR_ENABLED:
        logger.debug("OCR disabled in config")
        return {"text": None, "boxes": []}
    
    if engine is None:
        engine = config.OCR_ENGINE
    
    logger.info(f"Running OCR on {image_path.name} with {engine}")
    
    if engine == "tesseract":
        return run_tesseract_ocr(image_path)
    elif engine == "easyocr":
        return run_easyocr_ocr(image_path)
    else:
        logger.error(f"Unknown OCR engine: {engine}")
        return {"text": None, "boxes": []}


def run_tesseract_ocr(image_path: Path) -> Dict[str, Any]:
    """
    Run Tesseract OCR
    
    Returns:
        Dictionary with 'text' and 'boxes'
    """
    if not HAS_TESSERACT:
        logger.error("Tesseract not available")
        return {"text": None, "boxes": []}
    
    try:
        img = Image.open(image_path)
        
        # Extract text
        text = pytesseract.image_to_string(img)
        
        # Extract bounding boxes
        boxes = []
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                if int(data['conf'][i]) > 0:  # confidence threshold
                    bbox = {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    }
                    boxes.append({
                        'text': data['text'][i],
                        'bbox': bbox,
                        'confidence': data['conf'][i]
                    })
        except Exception as e:
            logger.warning(f"Failed to extract bounding boxes: {e}")
        
        logger.info(f"Tesseract OCR extracted {len(text)} chars, {len(boxes)} boxes")
        
        return {
            "text": text.strip() if text else None,
            "boxes": boxes
        }
        
    except Exception as e:
        logger.error(f"Tesseract OCR failed for {image_path.name}: {e}")
        return {"text": None, "boxes": []}


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
        reader = easyocr.Reader(['en'], gpu=False)
        
        # Read text
        results = reader.readtext(str(image_path))
        
        # Parse results
        text_parts = []
        boxes = []
        
        for (bbox_coords, text, confidence) in results:
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
