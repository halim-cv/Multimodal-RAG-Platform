# Image Ingestion & Preprocessing Pipeline

A production-ready pipeline for extracting, validating, normalizing, and storing images from various sources (PDFs, PPTX, ZIP, raw images, URLs) with optional OCR support.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd image-encoder

# Install dependencies
pip install -r requirements.txt

# Optional: Install Tesseract for OCR
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
```

### Basic Usage

```bash
# Process a single image
python run_pipeline.py examples/sample.jpg

# Process a PDF
python run_pipeline.py examples/document.pdf

# Process a directory
python run_pipeline.py examples/ --recursive

# Download and process from URL
python run_pipeline.py https://example.com/image.jpg --url

# Enable OCR
python run_pipeline.py examples/sample.jpg --ocr

# Custom settings
python run_pipeline.py examples/sample.jpg --max-side 2048 --thumbnail-size 512
```

## 📁 Project Structure

```
image-encoder/
├── data/
│   ├── raw/              # Original extracted images
│   ├── processed/        # Normalized images + thumbnails
│   └── metadata.jsonl    # Image metadata records
├── ingestion/            # Image extraction modules
│   ├── __init__.py
│   └── extractors.py     # PDF, PPTX, ZIP, URL extractors
├── preprocessing/        # Image processing utilities
│   ├── __init__.py
│   └── utils.py          # Normalize, thumbnail, hash
├── ocr/                  # OCR functionality
│   ├── __init__.py
│   └── ocr_engine.py     # Tesseract/EasyOCR wrapper
├── tests/                # Unit & integration tests
│   ├── test_preprocessing.py
│   └── test_pipeline.py
├── examples/             # Sample files for testing
├── config.py             # Configuration settings
├── metadata_store.py     # Metadata persistence (JSONL/SQLite)
├── pipeline.py           # Main pipeline orchestrator
├── run_pipeline.py       # CLI interface
├── run-tests.py          # Test runner
└── requirements.txt      # Python dependencies
```

## ⚙️ Configuration

Configure via environment variables or edit `config.py`:

```bash
# Image processing
export MAX_SIDE=1024              # Max dimension for normalized images
export THUMBNAIL_SIZE=256         # Thumbnail max dimension
export PAD_TO_SQUARE=false        # Pad images to square

# OCR
export OCR_ENABLED=false          # Enable/disable OCR
export OCR_ENGINE=tesseract       # tesseract or easyocr

# Storage
export STORAGE=local              # local or s3
export METADATA_BACKEND=jsonl     # jsonl or sqlite

# Validation
export MAX_FILE_SIZE_MB=50        # Max file size to process

# Logging
export LOG_LEVEL=INFO             # DEBUG, INFO, WARNING, ERROR
```

## 🔧 Features

### Supported Input Formats
- **Images**: JPG, PNG, WebP, TIFF, BMP, GIF
- **Documents**: PDF (extract embedded images)
- **Presentations**: PPTX (extract slide images)
- **Archives**: ZIP (containing images)
- **URLs**: Download images from web

### Processing Pipeline
1. **Extraction**: Extract images from various sources
2. **Validation**: Check file size and image integrity
3. **Deduplication**: Skip duplicates using SHA256 hashing
4. **Normalization**: Convert to RGB, resize to MAX_SIDE
5. **Thumbnail Generation**: Create thumbnails
6. **OCR** (optional): Extract text and bounding boxes
7. **Metadata Storage**: Save metadata in JSONL or SQLite

### Metadata Schema

Each processed image gets a metadata record:

```json
{
  "id": "uuid4",
  "sha256": "hex...",
  "file_name": "sample-0001.png",
  "source": "examples/sample.pdf",
  "source_type": "pdf",
  "source_page": 2,
  "extracted_at": "2026-01-20T18:15:00Z",
  "width": 1024,
  "height": 768,
  "paths": {
    "raw": "data/raw/uuid.png",
    "normalized": "data/processed/uuid_norm.png",
    "thumbnail": "data/processed/uuid_thumb.png"
  },
  "ocr_text": "Extracted text...",
  "ocr_boxes": [{"text": "word", "bbox": {...}, "confidence": 0.95}],
  "tags": [],
  "status": "ready"
}
```

## 🧪 Testing

```bash
# Run all tests
python run-tests.py

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest --cov=. tests/
```

### Test Coverage
- ✅ SHA256 hashing (deterministic)
- ✅ Image normalization (RGB conversion, resizing)
- ✅ Thumbnail generation
- ✅ Image validation
- ✅ Full pipeline integration
- ✅ Duplicate detection
- ✅ Directory processing
- ✅ Error handling

## 📊 Performance & Scaling

### Current Implementation (PoC)
- **Throughput**: ~10-50 images/second (depending on size)
- **Storage**: Local filesystem
- **Concurrency**: Single-threaded

### Scaling Recommendations
For production with thousands of images:

1. **Parallel Processing**
   ```python
   from multiprocessing import Pool
   # Use worker pool for concurrent image processing
   ```

2. **Distributed Storage**
   - Use S3/GCS for image blobs
   - Use PostgreSQL for metadata (instead of JSONL)

3. **Job Queue**
   - Add RabbitMQ/Redis for asynchronous processing
   - Separate ingestion and processing workers

4. **Batch Processing**
   - Process images in batches to reduce memory usage
   - Use streaming for large files

## 🔒 Security & Privacy

- ✅ File size limits enforced (default 50MB)
- ✅ MIME type validation
- ✅ Sanitized filenames
- ⚠️ **TODO**: Add encryption for sensitive images
- ⚠️ **TODO**: Access control for metadata

## 📈 Monitoring & Observability

The pipeline logs:
- Extraction counts per source
- Processing times
- Error rates
- Duplicate detection hits

Logs are written to:
- `stdout` (real-time)
- `data/pipeline.log` (persistent)

Key metrics tracked:
```python
stats = {
    "processed": <count>,
    "skipped_duplicates": <count>,
    "errors": <count>
}
```

## 🐛 Troubleshooting

### Common Issues

**1. OCR not working**
```bash
# Ensure Tesseract is installed and in PATH
tesseract --version

# Windows: Add to PATH
# C:\Program Files\Tesseract-OCR
```

**2. PDF extraction fails**
```bash
# Install PyMuPDF
pip install PyMuPDF
```

**3. Out of memory**
```bash
# Reduce MAX_SIDE or process fewer files at once
export MAX_SIDE=512
```

**4. Metadata file growing too large**
```bash
# Switch to SQLite backend
export METADATA_BACKEND=sqlite
```

## 🚀 Next Steps (Task 2)

After completing this pipeline, the next phase involves:

1. **Image Encoding**
   - Integrate CLIP for visual embeddings
   - Add Gemini/Vision API for rich embeddings
   - Store embeddings in vector database (FAISS/Qdrant)

2. **Retrieval System**
   - Implement semantic search
   - Add similarity queries
   - Build ranking/scoring

3. **API Layer**
   - REST API for ingestion
   - Search endpoints
   - Batch processing endpoints

## 📝 License

MIT License (or specify your license)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python run-tests.py`
5. Submit a pull request

## 📧 Support

For issues or questions, please open an issue on GitHub.

---

**Version**: 1.0.0  
**Last Updated**: 2026-01-20  
**Status**: Production-ready PoC
