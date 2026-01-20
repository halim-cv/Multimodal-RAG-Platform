#!/usr/bin/env python3
"""
Command-line interface for image ingestion pipeline
"""
import argparse
import logging
import sys
from pathlib import Path

import config
from pipeline import ImagePipeline, process_directory

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.DATA_DIR / "pipeline.log")
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Image Ingestion & Preprocessing Pipeline"
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to input file or directory, or URL"
    )
    
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Process directory recursively"
    )
    
    parser.add_argument(
        "--url",
        action="store_true",
        help="Treat input as URL"
    )
    
    parser.add_argument(
        "--max-side",
        type=int,
        default=config.MAX_SIDE,
        help=f"Maximum side for normalized images (default: {config.MAX_SIDE})"
    )
    
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=config.THUMBNAIL_SIZE,
        help=f"Thumbnail size (default: {config.THUMBNAIL_SIZE})"
    )
    
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR"
    )
    
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR"
    )
    
    args = parser.parse_args()
    
    # Override config with CLI args
    if args.max_side:
        config.MAX_SIDE = args.max_side
    if args.thumbnail_size:
        config.THUMBNAIL_SIZE = args.thumbnail_size
    if args.ocr:
        config.OCR_ENABLED = True
    if args.no_ocr:
        config.OCR_ENABLED = False
    
    logger.info("=" * 60)
    logger.info("Image Ingestion & Preprocessing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Max Side: {config.MAX_SIDE}")
    logger.info(f"Thumbnail Size: {config.THUMBNAIL_SIZE}")
    logger.info(f"OCR Enabled: {config.OCR_ENABLED}")
    logger.info("=" * 60)
    
    pipeline = ImagePipeline()
    
    try:
        # Process URL
        if args.url:
            record_ids = pipeline.process_url(args.input)
            logger.info(f"Processed {len(record_ids)} images from URL")
        
        # Process directory
        elif Path(args.input).is_dir():
            stats = process_directory(Path(args.input), recursive=args.recursive)
            logger.info(f"Directory processing stats: {stats}")
        
        # Process single file
        elif Path(args.input).is_file():
            record_ids = pipeline.process_file(Path(args.input))
            logger.info(f"Processed {len(record_ids)} images from file")
        
        else:
            logger.error(f"Input not found: {args.input}")
            sys.exit(1)
        
        # Print final stats
        stats = pipeline.get_stats()
        logger.info("=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Processed: {stats['processed']}")
        logger.info(f"Skipped (duplicates): {stats['skipped_duplicates']}")
        logger.info(f"Errors: {stats['errors']}")
        logger.info("=" * 60)
        
        if stats['errors'] > 0:
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
