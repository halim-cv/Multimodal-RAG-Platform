import json
import logging
from pathlib import Path
import config
from ingestion.extractors import extract_from_pdf
from scene_understanding.scene_understanding_engine import create_engine
from preprocessing.utils import get_image_dimensions, compute_sha256
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    # Define paths
    pdf_path = config.RAW_DIR / "HighlightedV1.pdf"
    output_json_path = config.DATA_DIR / "metadata.jsonl"

    logger.info(f"Looking for PDF at: {pdf_path}")
    if not pdf_path.exists():
        logger.error("PDF file not found!")
        return

    # 1. Extract Images
    # Using ingestion utility
    logger.info("Starting image extraction...")
    extracted_records = extract_from_pdf(pdf_path)
    
    if not extracted_records:
        logger.warning("No images extracted.")
        return

    logger.info(f"Extracted {len(extracted_records)} images.")

    # 2. Initialize Scene Understanding Engine
    # Using scene_understanding utility
    logger.info("Initializing Scene Understanding Engine...")
    engine = create_engine()

    # 3. Process Images (Captioning)
    results = []
    
    for record in extracted_records:
        image_path = record.temp_path
        logger.info(f"Processing image: {image_path.name}")
        
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Generate Caption
            caption = engine.detailed_caption(image)
            
            # Use preprocessing utilities
            width, height = get_image_dimensions(image_path)
            file_hash = compute_sha256(image_path)

            # Create structured metadata entry
            entry = {
                "file_name": image_path.name,
                "file_path": str(image_path),
                "source_file": record.source,
                "page_number": record.source_index + 1 if record.source_index is not None else None,
                "width": width,
                "height": height,
                "sha256": file_hash,
                "caption": caption,
                "analysis_type": "detailed_caption"
            }
            results.append(entry)
            
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")

    # 4. Save to JSONL (Structured/Beautified)
    logger.info(f"Saving metadata to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        # Saving as a pretty-printed JSON array
        json.dump(results, f, indent=4)
        
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
