"""
Figure Extraction Simulation
Extracts figures from PDF using the intelligent figure detection system
"""

import sys
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'extraction'))

from extraction.extractors import extract_from_file, cleanup_models
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Paths
INPUT_DIR = project_root / 'Input'
INPUT_PDF = INPUT_DIR / 'HighlightedV1.pdf'
ASSETS_DIR = current_dir.parent / 'assets'


def main():
    print("Figure Extraction Simulation")
    print("-" * 50)
    
    # Validate input PDF
    if not INPUT_PDF.exists():
        print(f"Error: PDF not found at {INPUT_PDF}")
        return
    
    print(f"Input PDF: {INPUT_PDF.name}")
    print(f"Output directory: {ASSETS_DIR}")
    
    # Create assets directory
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract figures
    print("\nExtracting figures from PDF...")
    figures = extract_from_file(INPUT_PDF, ASSETS_DIR)
    
    print("-" * 50)
    print(f"Extraction complete: {len(figures)} figures extracted")
    
    # Display results
    if figures:
        print("\nExtracted figures:")
        for fig in figures:
            print(f"  - {fig.temp_path.name}")
            if fig.caption:
                print(f"    Caption: {fig.caption[:80]}...")
            print(f"    Page: {fig.page_number}, BBox: {fig.bbox}")
    
    print(f"\nResults saved to: {ASSETS_DIR}")
    
    # Offload GPU resources
    print("\nOffloading models from GPU...")
    cleanup_models()
    print("GPU cleanup complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
