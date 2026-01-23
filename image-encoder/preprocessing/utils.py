"""
Preprocessing utilities: normalization, hashing
"""
import hashlib
from pathlib import Path
from typing import Tuple
from PIL import Image
import logging

import config

logger = logging.getLogger(__name__)

def normalize_image(
    in_path: Path,
    out_path: Path,
    max_side: int = None,
    pad_to_square: bool = False
) -> Tuple[int, int]:
    """
    Normalize image: convert to RGB, resize to max_side, optionally pad
    
    Args:
        in_path: Input image path
        out_path: Output path for normalized image
        max_side: Maximum dimension (defaults to config.MAX_SIDE)
        pad_to_square: Whether to pad to square (defaults to config.PAD_TO_SQUARE)
    
    Returns:
        (width, height) of normalized image
    """
    if max_side is None:
        max_side = config.MAX_SIDE
    if pad_to_square is None:
        pad_to_square = config.PAD_TO_SQUARE
    
    logger.info(f"Normalizing {in_path.name} -> {out_path.name}")
    
    # Load image
    img = Image.open(in_path)
    
    # Convert to RGB if needed
    if img.mode != "RGB":
        logger.debug(f"Converting {img.mode} to RGB")
        img = img.convert("RGB")
    
    # Resize if larger than max_side
    width, height = img.size
    if max(width, height) > max_side:
        scale_factor = max_side / max(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.debug(f"Resized from {width}x{height} to {new_width}x{new_height}")
        width, height = new_width, new_height
    
    # Optional: pad to square
    if pad_to_square:
        max_dim = max(width, height)
        if width != height:
            padded = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
            paste_x = (max_dim - width) // 2
            paste_y = (max_dim - height) // 2
            padded.paste(img, (paste_x, paste_y))
            img = padded
            logger.debug(f"Padded to square {max_dim}x{max_dim}")
            width, height = max_dim, max_dim
    
    # Save normalized image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    logger.info(f"Saved normalized image: {out_path.name} ({width}x{height})")
    
    return width, height


def validate_image(file_path: Path) -> bool:
    """
    Validate image file
    
    Args:
        file_path: Path to image file
    
    Returns:
        True if valid, False otherwise
    """
    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > config.MAX_FILE_SIZE_MB:
        logger.warning(f"{file_path.name} exceeds max size: {file_size_mb:.2f}MB")
        return False
    
    # Try to open with PIL
    try:
        img = Image.open(file_path)
        img.verify()
        logger.debug(f"Validated image: {file_path.name}")
        return True
    except Exception as e:
        logger.warning(f"Invalid image {file_path.name}: {e}")
        return False


def get_image_dimensions(file_path: Path) -> Tuple[int, int]:
    """Get image width and height"""
    with Image.open(file_path) as img:
        return img.size
