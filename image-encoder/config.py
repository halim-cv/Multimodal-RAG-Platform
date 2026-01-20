"""
Configuration for Image Ingestion & Preprocessing Pipeline
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXAMPLES_DIR = BASE_DIR / "examples"

# Image processing config
MAX_SIDE = int(os.getenv("MAX_SIDE", "1024"))
THUMBNAIL_SIZE = int(os.getenv("THUMBNAIL_SIZE", "256"))
PAD_TO_SQUARE = os.getenv("PAD_TO_SQUARE", "false").lower() == "true"

# OCR config
OCR_ENABLED = os.getenv("OCR_ENABLED", "false").lower() == "true"
OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract")  # tesseract or easyocr

# File validation
IMAGE_MIME_WHITELIST = [
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/tiff",
    "image/bmp"
]
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# Storage config
STORAGE = os.getenv("STORAGE", "local")  # local or s3
METADATA_BACKEND = os.getenv("METADATA_BACKEND", "jsonl")  # jsonl or sqlite

# Metadata file
METADATA_FILE = DATA_DIR / "metadata.jsonl"
METADATA_DB = DATA_DIR / "metadata.db"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
