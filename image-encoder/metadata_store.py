"""
Metadata storage and retrieval for processed images
Supports JSONL and SQLite backends
"""
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import config

logger = logging.getLogger(__name__)


class MetadataStore:
    """Abstract base for metadata storage"""
    
    def save_metadata(self, record: Dict[str, Any]) -> str:
        """Save a metadata record, return the id"""
        raise NotImplementedError

    def update_status(self, record_id: str, status: str, notes: str = "") -> None:
        """Update status of a record"""
        raise NotImplementedError
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all records"""
        raise NotImplementedError


class JSONLStore(MetadataStore):
    """JSONL-based metadata store"""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            self.filepath.touch()
    
    def save_metadata(self, record: Dict[str, Any]) -> str:
        """Append record to JSONL file"""
        if "id" not in record:
            record["id"] = str(uuid.uuid4())
        if "extracted_at" not in record:
            record["extracted_at"] = datetime.utcnow().isoformat() + "Z"
        
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        
        logger.info(f"Saved metadata for {record['id']}")
        return record["id"]
    
    def update_status(self, record_id: str, status: str, notes: str = "") -> None:
        """Update status - rewrite entire file (inefficient but simple for PoC)"""
        if not self.filepath.exists():
            return
        
        records = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if record.get("id") == record_id:
                        record["status"] = status
                        if notes:
                            record["notes"] = notes
                    records.append(record)
        
        with open(self.filepath, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        
        logger.info(f"Updated status for {record_id} to {status}")
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Load all records"""
        if not self.filepath.exists():
            return []
        
        records = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records


class SQLiteStore(MetadataStore):
    """SQLite-based metadata store"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                file_name TEXT,
                source TEXT,
                source_type TEXT,
                source_page INTEGER,
                extracted_at TEXT,
                width INTEGER,
                height INTEGER,
                path_raw TEXT,
                path_normalized TEXT,
                ocr_text TEXT,
                ocr_boxes TEXT,
                tags TEXT,
                status TEXT,
                notes TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def save_metadata(self, record: Dict[str, Any]) -> str:
        """Insert record into SQLite"""
        if "id" not in record:
            record["id"] = str(uuid.uuid4())
        if "extracted_at" not in record:
            record["extracted_at"] = datetime.utcnow().isoformat() + "Z"
        
        paths = record.get("paths", {})
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO images (
                    id, file_name, source, source_type, source_page,
                    extracted_at, width, height, path_raw, path_normalized,
                    ocr_text, ocr_boxes, tags, status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record["id"],
                record.get("file_name"),
                record.get("source"),
                record.get("source_type"),
                record.get("source_page"),
                record["extracted_at"],
                record.get("width"),
                record.get("height"),
                paths.get("raw"),
                paths.get("normalized"),
                record.get("ocr_text"),
                json.dumps(record.get("ocr_boxes", [])),
                json.dumps(record.get("tags", [])),
                record.get("status", "pending"),
                record.get("notes", "")
            ))
            conn.commit()
            logger.info(f"Saved metadata for {record['id']}")
        finally:
            conn.close()
        
        return record["id"]
    

    
    def update_status(self, record_id: str, status: str, notes: str = "") -> None:
        """Update status"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE images SET status = ?, notes = ? WHERE id = ?",
            (status, notes, record_id)
        )
        conn.commit()
        conn.close()
        logger.info(f"Updated status for {record_id} to {status}")
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all records"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM images")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]


def get_metadata_store() -> MetadataStore:
    """Factory: return configured metadata store"""
    if config.METADATA_BACKEND == "sqlite":
        return SQLiteStore(config.METADATA_DB)
    else:
        return JSONLStore(config.METADATA_FILE)
