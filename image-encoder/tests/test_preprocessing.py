"""
Unit tests for preprocessing utilities
"""
import pytest
from pathlib import Path
from PIL import Image
import tempfile
import shutil

from preprocessing import (
    compute_sha256,
    normalize_image,
    make_thumbnail,
    validate_image,
    get_image_dimensions
)
import config


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image"""
    img_path = temp_dir / "test_image.png"
    img = Image.new("RGB", (800, 600), color="red")
    img.save(img_path)
    return img_path


def test_compute_sha256(sample_image):
    """Test SHA256 hash computation is deterministic"""
    hash1 = compute_sha256(sample_image)
    hash2 = compute_sha256(sample_image)
    
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex is 64 chars
    assert isinstance(hash1, str)


def test_normalize_image_rgb_conversion(temp_dir):
    """Test that non-RGB images are converted to RGB"""
    # Create RGBA image
    rgba_path = temp_dir / "rgba.png"
    img = Image.new("RGBA", (400, 300), color=(255, 0, 0, 128))
    img.save(rgba_path)
    
    # Normalize
    output_path = temp_dir / "normalized.png"
    width, height = normalize_image(rgba_path, output_path)
    
    # Check output
    assert output_path.exists()
    result_img = Image.open(output_path)
    assert result_img.mode == "RGB"
    assert width == 400
    assert height == 300


def test_normalize_image_resize(temp_dir, sample_image):
    """Test image resizing respects MAX_SIDE"""
    max_side = 512
    output_path = temp_dir / "normalized.png"
    
    width, height = normalize_image(sample_image, output_path, max_side=max_side)
    
    assert output_path.exists()
    assert max(width, height) == max_side
    
    # Check aspect ratio preserved
    original_aspect = 800 / 600
    new_aspect = width / height
    assert abs(original_aspect - new_aspect) < 0.01


def test_normalize_image_no_resize_if_smaller(temp_dir):
    """Test that images smaller than MAX_SIDE are not upscaled"""
    small_img_path = temp_dir / "small.png"
    img = Image.new("RGB", (200, 150), color="blue")
    img.save(small_img_path)
    
    output_path = temp_dir / "normalized.png"
    width, height = normalize_image(small_img_path, output_path, max_side=1024)
    
    assert width == 200
    assert height == 150


def test_make_thumbnail(temp_dir, sample_image):
    """Test thumbnail generation"""
    thumb_path = temp_dir / "thumbnail.png"
    thumb_size = 256
    
    width, height = make_thumbnail(sample_image, thumb_path, thumb_size=thumb_size)
    
    assert thumb_path.exists()
    assert max(width, height) <= thumb_size
    
    # Check it's actually smaller
    assert width < 800 or height < 600


def test_validate_image_valid(sample_image):
    """Test validation accepts valid images"""
    assert validate_image(sample_image) is True


def test_validate_image_invalid(temp_dir):
    """Test validation rejects invalid files"""
    invalid_path = temp_dir / "invalid.png"
    invalid_path.write_text("This is not an image")
    
    assert validate_image(invalid_path) is False


def test_validate_image_too_large(temp_dir):
    """Test validation rejects oversized files"""
    # Create a large image (simulate by creating large file)
    large_path = temp_dir / "large.png"
    
    # Create image larger than MAX_FILE_SIZE_MB
    original_max = config.MAX_FILE_SIZE_MB
    config.MAX_FILE_SIZE_MB = 0.001  # 1KB limit for test
    
    img = Image.new("RGB", (1000, 1000), color="green")
    img.save(large_path)
    
    result = validate_image(large_path)
    
    # Restore config
    config.MAX_FILE_SIZE_MB = original_max
    
    assert result is False


def test_get_image_dimensions(sample_image):
    """Test dimension extraction"""
    width, height = get_image_dimensions(sample_image)
    
    assert width == 800
    assert height == 600


def test_normalize_with_padding(temp_dir):
    """Test padding to square"""
    non_square_path = temp_dir / "non_square.png"
    img = Image.new("RGB", (400, 200), color="yellow")
    img.save(non_square_path)
    
    output_path = temp_dir / "padded.png"
    width, height = normalize_image(
        non_square_path,
        output_path,
        max_side=400,
        pad_to_square=True
    )
    
    assert width == height  # Should be square now
    assert width == 400
