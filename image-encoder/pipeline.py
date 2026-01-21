"""
Main pipeline orchestrator for image ingestion and preprocessing
"""
import uuid
from pathlib import Path
from typing import List, Dict, Any
import logging

import config
from ingestion import extract_from_file, download_from_url
from preprocessing import (
    compute_sha256,
    normalize_image,
    make_thumbnail,
    get_image_dimensions
)
from ocr import run_ocr, draw_ocr_boxes
from metadata_store import get_metadata_store

logger = logging.getLogger(__name__)


class ImagePipeline:
    """Main pipeline for processing images"""
    
    def __init__(self):
        self.metadata_store = get_metadata_store()
        self.stats = {
            "processed": 0,
            "skipped_duplicates": 0,
            "errors": 0
        }
    
    def process_file(self, file_path: Path) -> List[str]:
        """
        Process a single file (extract and process all images)
        
        Args:
            file_path: Path to input file
        
        Returns:
            List of record IDs created
        """
        logger.info(f"Processing file: {file_path}")
        
        # Step 1: Extract images
        extracted_records = extract_from_file(file_path)
        
        if not extracted_records:
            logger.warning(f"No images extracted from {file_path}")
            return []
        
        logger.info(f"Extracted {len(extracted_records)} images from {file_path.name}")
        
        # Step 2: Process each extracted image
        record_ids = []
        for ext_record in extracted_records:
            record_id = self._process_single_image(ext_record)
            if record_id:
                record_ids.append(record_id)
        
        logger.info(f"Processing complete: {len(record_ids)} images processed")
        return record_ids
    
    def process_url(self, url: str) -> List[str]:
        """
        Download and process image from URL
        
        Args:
            url: Image URL
        
        Returns:
            List of record IDs
        """
        logger.info(f"Processing URL: {url}")
        
        # Download image
        downloaded_path = download_from_url(url)
        if not downloaded_path:
            logger.error(f"Failed to download from {url}")
            return []
        
        # Process as file
        return self.process_file(downloaded_path)
    
    def _process_single_image(self, ext_record) -> str:
        """
        Process a single extracted image through the full pipeline
        
        Args:
            ext_record: ExtractedImageRecord
        
        Returns:
            Record ID if successful, None otherwise
        """
        temp_path = ext_record.temp_path
        logger.info(f"Processing image: {temp_path.name}")
        
        try:
            # Step 1: Compute hash for deduplication
            sha256 = compute_sha256(temp_path)
            
            # Step 2: Check for duplicates
            if self.metadata_store.exists_by_sha(sha256):
                logger.info(f"Skipping duplicate (SHA256: {sha256[:16]}...)")
                self.stats["skipped_duplicates"] += 1
                
                # Cleanup logic: Only unlink if it's a temp file (not the original source)
                # This prevents deleting the user's input file if duplicate detected.
                should_unlink = True
                if hasattr(ext_record, 'source_type') and ext_record.source_type == 'image':
                    try:
                        # If temp_path refers to the same file as source, DON'T delete
                        if temp_path.resolve() == Path(ext_record.source).resolve():
                            should_unlink = False
                    except Exception:
                        pass
                
                if should_unlink:
                    temp_path.unlink()  # Clean up temp file
                
                return None
            
            # Step 3: Get dimensions
            width, height = get_image_dimensions(temp_path)
            
            # Step 4: Generate unique ID and filenames
            record_id = str(uuid.uuid4())
            base_name = f"{record_id}"
            
            # Paths for processed files
            # Use the existing temp_path as the raw_path (don't rename to UUID)
            raw_path = temp_path
            
            # Normalized and thumbnail still get UUID names for consistency
            normalized_path = config.PROCESSED_DIR / f"{base_name}_norm.png"
            thumbnail_path = config.PROCESSED_DIR / f"{base_name}_thumb.png"
            annotated_path = config.PROCESSED_DIR / f"{base_name}_annotated.png"
            
            # Step 5: File is already in raw location (handled by extractors), so no move needed
            # Unless it's not in RAW_DIR for some reason
            if raw_path.parent != config.RAW_DIR:
                dest_path = config.RAW_DIR / raw_path.name
                if raw_path != dest_path:
                    raw_path.rename(dest_path)
                    raw_path = dest_path
                    logger.debug(f"Moved to raw dir: {raw_path.name}")
            
            # Step 6: Normalize image
            try:
                norm_width, norm_height = normalize_image(raw_path, normalized_path)
            except Exception as e:
                logger.error(f"Normalization failed: {e}")
                raise
            
            # Step 7: Generate thumbnail
            try:
                thumb_width, thumb_height = make_thumbnail(raw_path, thumbnail_path)
            except Exception as e:
                logger.error(f"Thumbnail generation failed: {e}")
                raise
            
            # Step 8: Run OCR (if enabled)
            ocr_result = {"text": None, "boxes": []}
            if config.OCR_ENABLED:
                try:
                    ocr_result = run_ocr(normalized_path)
                    
                    # Draw boxes if any found
                    if ocr_result.get("boxes"):
                        draw_ocr_boxes(normalized_path, ocr_result["boxes"], annotated_path)
                except Exception as e:
                    logger.warning(f"OCR failed: {e}")
            
            # Step 9: Build metadata record
            paths_dict = {
                "raw": str(raw_path),
                "normalized": str(normalized_path),
                "thumbnail": str(thumbnail_path)
            }
            if annotated_path.exists():
                paths_dict["annotated"] = str(annotated_path)

            record = {
                "id": record_id,
                "sha256": sha256,
                "file_name": raw_path.name,
                "source": ext_record.source,
                "source_type": ext_record.source_type,
                "source_page": ext_record.source_index,
                "width": norm_width,
                "height": norm_height,
                "paths": paths_dict,
                "ocr_text": ocr_result.get("text"),
                "ocr_boxes": ocr_result.get("boxes", []),
                "tags": [],
                "status": "ready"
            }
            
            # Step 10: Save metadata
            self.metadata_store.save_metadata(record)
            
            self.stats["processed"] += 1
            logger.info(f"Successfully processed image {record_id}")
            
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to process {temp_path.name}: {e}", exc_info=True)
            self.stats["errors"] += 1
            
            # Try to save error record
            try:
                error_record = {
                    "id": str(uuid.uuid4()),
                    "sha256": compute_sha256(temp_path) if temp_path.exists() else "unknown",
                    "file_name": temp_path.name,
                    "source": ext_record.source,
                    "source_type": ext_record.source_type,
                    "status": "error",
                    "notes": str(e)
                }
                self.metadata_store.save_metadata(error_record)
            except Exception as meta_error:
                logger.error(f"Failed to save error record: {meta_error}")
            
            return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get pipeline statistics"""
        return self.stats.copy()


def process_directory(directory: Path, recursive: bool = False) -> Dict[str, Any]:
    """
    Process all files in a directory
    
    Args:
        directory: Directory path
        recursive: Whether to process subdirectories
    
    Returns:
        Processing statistics
    """
    pipeline = ImagePipeline()
    
    pattern = "**/*" if recursive else "*"
    files = list(directory.glob(pattern))
    
    logger.info(f"Processing {len(files)} files from {directory}")
    
    all_record_ids = []
    for file_path in files:
        if file_path.is_file():
            try:
                record_ids = pipeline.process_file(file_path)
                all_record_ids.extend(record_ids)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
    
    stats = pipeline.get_stats()
    stats["total_records"] = len(all_record_ids)
    
    logger.info(f"Directory processing complete: {stats}")
    
    return stats
