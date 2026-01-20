"""
Integration test for full pipeline
"""
import pytest
from pathlib import Path
import tempfile
import shutil
import json
from PIL import Image

from pipeline import ImagePipeline, process_directory
from metadata_store import get_metadata_store
import config


@pytest.fixture
def test_env(tmp_path):
    """Set up test environment with temporary directories"""
    # Create test directories
    test_data = tmp_path / "data"
    test_raw = test_data / "raw"
    test_processed = test_data / "processed"
    test_examples = tmp_path / "examples"
    
    test_raw.mkdir(parents=True)
    test_processed.mkdir(parents=True)
    test_examples.mkdir(parents=True)
    
    # Override config paths
    original_data_dir = config.DATA_DIR
    original_raw_dir = config.RAW_DIR
    original_processed_dir = config.PROCESSED_DIR
    original_metadata_file = config.METADATA_FILE
    
    config.DATA_DIR = test_data
    config.RAW_DIR = test_raw
    config.PROCESSED_DIR = test_processed
    config.METADATA_FILE = test_data / "metadata.jsonl"
    
    # Create sample image
    sample_img = test_examples / "sample.png"
    img = Image.new("RGB", (1200, 800), color="blue")
    img.save(sample_img)
    
    yield {
        "data_dir": test_data,
        "raw_dir": test_raw,
        "processed_dir": test_processed,
        "examples_dir": test_examples,
        "sample_image": sample_img
    }
    
    # Restore config
    config.DATA_DIR = original_data_dir
    config.RAW_DIR = original_raw_dir
    config.PROCESSED_DIR = original_processed_dir
    config.METADATA_FILE = original_metadata_file


def test_pipeline_single_image(test_env):
    """Test processing a single image through full pipeline"""
    pipeline = ImagePipeline()
    sample_image = test_env["sample_image"]
    
    # Process the image
    record_ids = pipeline.process_file(sample_image)
    
    # Assertions
    assert len(record_ids) == 1
    record_id = record_ids[0]
    
    # Check files were created
    raw_files = list(test_env["raw_dir"].glob("*.png"))
    processed_files = list(test_env["processed_dir"].glob("*.png"))
    
    assert len(raw_files) == 1  # Original
    assert len(processed_files) == 2  # Normalized + thumbnail
    
    # Check metadata
    metadata_store = get_metadata_store()
    records = metadata_store.get_all()
    assert len(records) == 1
    
    record = records[0]
    assert record["id"] == record_id
    assert record["status"] == "ready"
    assert record["width"] == config.MAX_SIDE  # Should be resized
    assert "sha256" in record
    
    # Check stats
    stats = pipeline.get_stats()
    assert stats["processed"] == 1
    assert stats["skipped_duplicates"] == 0
    assert stats["errors"] == 0


def test_pipeline_duplicate_detection(test_env):
    """Test that duplicate images are skipped"""
    pipeline = ImagePipeline()
    sample_image = test_env["sample_image"]
    
    # Process same image twice
    record_ids_1 = pipeline.process_file(sample_image)
    record_ids_2 = pipeline.process_file(sample_image)
    
    # First should succeed
    assert len(record_ids_1) == 1
    
    # Second should be skipped
    assert len(record_ids_2) == 0
    
    # Check stats
    stats = pipeline.get_stats()
    assert stats["processed"] == 1
    assert stats["skipped_duplicates"] == 1


def test_process_directory(test_env):
    """Test processing a directory of images"""
    examples_dir = test_env["examples_dir"]
    
    # Create multiple test images
    for i in range(3):
        img_path = examples_dir / f"image_{i}.png"
        img = Image.new("RGB", (600, 400), color=(i * 50, 100, 200))
        img.save(img_path)
    
    # Process directory
    stats = process_directory(examples_dir)
    
    # Check stats
    assert stats["processed"] == 4  # 3 new + 1 sample
    assert stats["total_records"] == 4


def test_pipeline_error_handling(test_env):
    """Test that pipeline handles errors gracefully"""
    pipeline = ImagePipeline()
    
    # Try to process non-existent file
    fake_path = test_env["examples_dir"] / "nonexistent.png"
    record_ids = pipeline.process_file(fake_path)
    
    assert len(record_ids) == 0


def test_metadata_persistence(test_env):
    """Test that metadata is correctly persisted"""
    pipeline = ImagePipeline()
    sample_image = test_env["sample_image"]
    
    # Process image
    record_ids = pipeline.process_file(sample_image)
    record_id = record_ids[0]
    
    # Read metadata file directly
    metadata_file = config.METADATA_FILE
    assert metadata_file.exists()
    
    # Parse JSONL
    with open(metadata_file, "r") as f:
        lines = f.readlines()
    
    assert len(lines) == 1
    record = json.loads(lines[0])
    
    # Verify structure
    assert record["id"] == record_id
    assert "sha256" in record
    assert "extracted_at" in record
    assert "paths" in record
    assert record["paths"]["raw"]
    assert record["paths"]["normalized"]
    assert record["paths"]["thumbnail"]
    assert record["source_type"] == "image"


def test_normalized_image_quality(test_env):
    """Test that normalized image meets requirements"""
    pipeline = ImagePipeline()
    sample_image = test_env["sample_image"]
    
    # Process
    record_ids = pipeline.process_file(sample_image)
    
    # Find normalized image
    norm_files = list(test_env["processed_dir"].glob("*_norm.png"))
    assert len(norm_files) == 1
    
    norm_img = Image.open(norm_files[0])
    
    # Check it's RGB
    assert norm_img.mode == "RGB"
    
    # Check dimensions
    width, height = norm_img.size
    assert max(width, height) == config.MAX_SIDE
    
    # Check aspect ratio preserved (approximately)
    original_aspect = 1200 / 800
    new_aspect = width / height
    assert abs(original_aspect - new_aspect) < 0.01


def test_thumbnail_quality(test_env):
    """Test that thumbnail meets requirements"""
    pipeline = ImagePipeline()
    sample_image = test_env["sample_image"]
    
    # Process
    pipeline.process_file(sample_image)
    
    # Find thumbnail
    thumb_files = list(test_env["processed_dir"].glob("*_thumb.png"))
    assert len(thumb_files) == 1
    
    thumb_img = Image.open(thumb_files[0])
    
    # Check dimensions
    width, height = thumb_img.size
    assert max(width, height) <= config.THUMBNAIL_SIZE
    assert thumb_img.mode == "RGB"
