"""
Figure Extraction Module using Document Understanding Engine
Provides high-level interface for extracting figures from PDFs
"""

from typing import List, Union, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
import gc

from .document_understanding_engine import DocumentUnderstandingEngine


@dataclass
class ExtractedFigureRecord:
    """Record for an extracted figure with its metadata."""
    temp_path: Path
    caption: Optional[str]
    page_number: int
    bbox: List[float]
    image: np.ndarray


# Global engine instance for reuse
_global_engine: Optional[DocumentUnderstandingEngine] = None


def get_engine() -> DocumentUnderstandingEngine:
    """
    Get or create the global document understanding engine.
    
    Returns:
        Loaded DocumentUnderstandingEngine instance
    """
    global _global_engine
    
    if _global_engine is None:
        print("Initializing Document Understanding Engine...")
        _global_engine = DocumentUnderstandingEngine()
        _global_engine.load()
        print("Engine loaded successfully!")
    
    return _global_engine


def cleanup_models():
    """
    Cleanup and offload models from GPU memory.
    Call this after processing to free GPU resources.
    """
    global _global_engine
    
    if _global_engine is not None:
        print("Cleaning up Document Understanding Engine...")
        
        # Delete model reference
        if hasattr(_global_engine, 'model') and _global_engine.model is not None:
            del _global_engine.model
            _global_engine.model = None
            _global_engine._is_loaded = False
        
        # Clear the global engine
        _global_engine = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared")
        
        print("Cleanup complete")


def extract_figures_from_pdf(pdf_path: Union[str, Path], 
                             output_dir: Union[str, Path],
                             dpi: int = 200) -> List[ExtractedFigureRecord]:
    """
    Extract all figures with captions from a PDF document.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted figures
        dpi: Resolution for PDF conversion
        
    Returns:
        List of ExtractedFigureRecord objects
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get engine
    engine = get_engine()
    
    print(f"Processing PDF: {pdf_path.name}")
    
    # Extract figure-caption pairs
    figure_pairs = engine.extract_figure_caption_pairs(pdf_path, dpi=dpi)
    
    # Convert to ExtractedFigureRecord objects
    extracted_figures = []
    
    for idx, pair in enumerate(figure_pairs):
        page_number = pair['page_number']
        combined_image = pair['combined_image']
        figure_bbox = pair['main_bbox']
        
        # Generate filename
        figure_filename = f"figure_page{page_number + 1}_{idx + 1}.png"
        figure_path = output_dir / figure_filename
        
        # Save the combined figure-caption image
        Image.fromarray(combined_image).save(figure_path)
        
        # Create record (caption extraction would require OCR, placeholder for now)
        record = ExtractedFigureRecord(
            temp_path=figure_path,
            caption=None,  # OCR integration needed for caption text
            page_number=page_number,
            bbox=figure_bbox,
            image=combined_image
        )
        
        extracted_figures.append(record)
        
        print(f"  Extracted: {figure_filename} from page {page_number + 1}")
    
    return extracted_figures


def extract_from_file(pdf_path: Union[str, Path], 
                     output_dir: Union[str, Path],
                     dpi: int = 200) -> List[ExtractedFigureRecord]:
    """
    Main entry point for extracting figures from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted figures
        dpi: Resolution for PDF conversion
        
    Returns:
        List of ExtractedFigureRecord objects
    """
    return extract_figures_from_pdf(pdf_path, output_dir, dpi)


def extract_with_ocr(pdf_path: Union[str, Path], 
                     output_dir: Union[str, Path],
                     dpi: int = 200,
                     ocr_engine: Optional[any] = None) -> List[ExtractedFigureRecord]:
    """
    Extract figures with OCR-based caption text extraction.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted figures
        dpi: Resolution for PDF conversion
        ocr_engine: Optional OCR engine for caption text extraction
        
    Returns:
        List of ExtractedFigureRecord objects with caption text
        
    Note:
        This is a placeholder for future OCR integration.
        Currently returns same results as extract_from_file.
    """
    # TODO: Integrate OCR for caption text extraction
    # For now, use the standard extraction
    return extract_from_file(pdf_path, output_dir, dpi)
