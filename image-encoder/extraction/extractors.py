from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import io
import logging

from PIL import Image, ImageDraw
import numpy as np
import torch
import fitz
import layoutparser as lp
import easyocr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    raise RuntimeError("GPU is required for OCR (torch.cuda.is_available() is False)")

_EASYOCR_READER = easyocr.Reader(["en"], gpu=True)
_LAYOUT_MODEL = lp.AutoLayoutModel(
    "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

@dataclass
class ExtractedFigureRecord:
    source: str
    page_number: int
    figure_number: Optional[int]
    bbox: Tuple[float, float, float, float]
    temp_path: Path
    caption: Optional[str] = None
    figure_type: str = "figure"
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def extract_from_file(src_path: Path, output_dir: Path, *, min_figure_size: int = 10000, dpi: int = 300, layout_score_threshold: float = 0.6) -> List[ExtractedFigureRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        doc = fitz.open(src_path)
    except Exception as e:
        logger.error("open pdf failed: %s", e)
        return []
    extracted: List[ExtractedFigureRecord] = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        try:
            recs = _extract_figures_from_page(page, page_idx, src_path, output_dir, min_figure_size, dpi, layout_score_threshold)
            extracted.extend(recs)
        except Exception as e:
            logger.exception("page extraction error: %s", e)
    doc.close()
    return extracted

def extract_figures_from_pdf(pdf_path: Path, output_dir: Path, *, min_figure_size: int = 10000, dpi: int = 300, layout_score_threshold: float = 0.6) -> List[ExtractedFigureRecord]:
    """Alias for extract_from_file for backward compatibility"""
    return extract_from_file(pdf_path, output_dir, min_figure_size=min_figure_size, dpi=dpi, layout_score_threshold=layout_score_threshold)

def _extract_figures_from_page(page: 'fitz.Page', page_num: int, pdf_path: Path, output_dir: Path, min_figure_size: int, dpi: int, layout_score_threshold: float) -> List[ExtractedFigureRecord]:
    layout_regions = _detect_layout_regions(page, score_threshold=layout_score_threshold)
    image_blocks = _get_image_blocks(page)
    vector_blocks = _get_vector_drawing_regions(page)
    if layout_regions:
        visual_candidates = [tuple(r["bbox"]) for r in layout_regions if r.get("type") in ("Figure", "figure")]
        caption_regions = [tuple(r["bbox"]) for r in layout_regions if r.get("type") in ("Caption", "caption")]
    else:
        visual_candidates = _merge_nearby_images(image_blocks + vector_blocks, threshold=20)
        caption_regions = []
    text_blocks = _get_text_blocks(page)
    text_regions = [tuple(b["bbox"]) for b in text_blocks]
    figure_candidates = _filter_text_heavy_regions(visual_candidates, text_regions, safe_regions=image_blocks, overlap_threshold=0.40)
    figures: List[ExtractedFigureRecord] = []
    for idx, cand in enumerate(figure_candidates, start=1):
        x0, y0, x1, y1 = cand
        area = max((x1 - x0) * (y1 - y0), 0.0)
        if area < min_figure_size:
            continue
        caption = None
        if caption_regions:
            cap_bbox = _find_closest_caption_region(caption_regions, cand)
            if cap_bbox:
                cap_img = _render_bbox_region(page, cap_bbox, dpi=150)
                if cap_img:
                    caption = _ocr_image(cap_img)
        else:
            caption = _find_caption_near_bbox(text_blocks, cand, search_distance=140)
        pil_img = _render_visual_content_only(page, cand, text_regions, dpi=dpi)
        if pil_img is None:
            continue
        if pil_img.width < 50 or pil_img.height < 50:
            continue
        filename = f"{pdf_path.stem}_page{page_num+1}_fig{idx}.png"
        temp_path = output_dir / filename
        pil_img.save(temp_path, format="PNG", dpi=(dpi, dpi))
        record = ExtractedFigureRecord(source=str(pdf_path), page_number=page_num + 1, figure_number=idx, bbox=cand, temp_path=temp_path, caption=caption, figure_type="figure")
        figures.append(record)
    return figures

def _detect_layout_regions(page: 'fitz.Page', score_threshold: float = 0.6) -> List[Dict[str, Any]]:
    render_dpi = 150
    zoom = render_dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    lp_img = lp.Image(img)
    layout = _LAYOUT_MODEL.detect(lp_img)
    regions: List[Dict[str, Any]] = []
    inv_zoom = 72.0 / render_dpi
    for el in layout:
        score = getattr(el, "score", 1.0)
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = el.block.x_1, el.block.y_1, el.block.x_2, el.block.y_2
        pdf_bbox = (x1 * inv_zoom, y1 * inv_zoom, x2 * inv_zoom, y2 * inv_zoom)
        regions.append({"type": el.type, "bbox": pdf_bbox, "score": score})
    return regions

def _get_image_blocks(page: 'fitz.Page') -> List[Tuple[float, float, float, float]]:
    rects: List[Tuple[float, float, float, float]] = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        for rect in page.get_image_rects(xref):
            rects.append((rect.x0, rect.y0, rect.x1, rect.y1))
    return rects

def _get_vector_drawing_regions(page: 'fitz.Page') -> List[Tuple[float, float, float, float]]:
    vector_regions: List[Tuple[float, float, float, float]] = []
    drawings = page.get_drawings()
    groups: List[List[Tuple[float, float, float, float]]] = []
    for d in drawings:
        rect = d.get("rect")
        if not rect:
            continue
        bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
        if (rect.width < 5 and rect.height < 5):
            continue
        added = False
        for g in groups:
            for ex in g:
                if _bboxes_are_close(bbox, ex, threshold=50):
                    g.append(bbox)
                    added = True
                    break
            if added:
                break
        if not added:
            groups.append([bbox])
    for g in groups:
        if len(g) >= 2:
            merged = _merge_bboxes(g)
            w = merged[2] - merged[0]
            h = merged[3] - merged[1]
            if w > 100 and h > 100:
                vector_regions.append(merged)
    return vector_regions

def _get_text_blocks(page: 'fitz.Page') -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    raw = page.get_text("dict")
    for b in raw.get("blocks", []):
        if b.get("type") != 0:
            continue
        bbox = b.get("bbox", (0, 0, 0, 0))
        text_chunks: List[str] = []
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                text_chunks.append(span.get("text", ""))
        text = " ".join(text_chunks).strip()
        blocks.append({"bbox": tuple(bbox), "text": text})
    return blocks

def _filter_text_heavy_regions(candidates: List[Tuple[float, float, float, float]], text_regions: List[Tuple[float, float, float, float]], safe_regions: Optional[List[Tuple[float, float, float, float]]] = None, overlap_threshold: float = 0.4) -> List[Tuple[float, float, float, float]]:
    filtered: List[Tuple[float, float, float, float]] = []
    for cand in candidates:
        cx0, cy0, cx1, cy1 = cand
        cand_area = max((cx1 - cx0) * (cy1 - cy0), 1.0)
        protected = False
        if safe_regions:
            for s in safe_regions:
                ix0 = max(cx0, s[0]); iy0 = max(cy0, s[1]); ix1 = min(cx1, s[2]); iy1 = min(cy1, s[3])
                if ix1 > ix0 and iy1 > iy0:
                    inter_area = (ix1 - ix0) * (iy1 - iy0)
                    if inter_area / cand_area > 0.1:
                        filtered.append(cand)
                        protected = True
                        break
            if protected:
                continue
        total_text_overlap = 0.0
        for t in text_regions:
            ix0 = max(cx0, t[0]); iy0 = max(cy0, t[1]); ix1 = min(cx1, t[2]); iy1 = min(cy1, t[3])
            if ix1 > ix0 and iy1 > iy0:
                total_text_overlap += (ix1 - ix0) * (iy1 - iy0)
        frac = total_text_overlap / cand_area
        if frac < overlap_threshold:
            filtered.append(cand)
    return filtered

def _render_bbox_region(page: 'fitz.Page', bbox: Tuple[float, float, float, float], dpi: int = 300) -> Optional[Image.Image]:
    rect = fitz.Rect(bbox)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
    return img

def _render_visual_content_only(page: 'fitz.Page', bbox: Tuple[float, float, float, float], text_regions: List[Tuple[float, float, float, float]], dpi: int = 300, mask_color: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> Optional[Image.Image]:
    img = _render_bbox_region(page, bbox, dpi=dpi)
    if img is None:
        return None
    if not text_regions:
        return img.convert("RGB")
    zoom = dpi / 72.0
    x0, y0, x1, y1 = bbox
    width_px, height_px = img.size
    draw = ImageDraw.Draw(img)
    for t in text_regions:
        tx0, ty0, tx1, ty1 = t
        ix0 = max(x0, tx0); iy0 = max(y0, ty0); ix1 = min(x1, tx1); iy1 = min(y1, ty1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        px0 = int(round((ix0 - x0) * zoom)); py0 = int(round((iy0 - y0) * zoom))
        px1 = int(round((ix1 - x0) * zoom)); py1 = int(round((iy1 - y0) * zoom))
        px0 = max(0, min(width_px, px0)); px1 = max(0, min(width_px, px1))
        py0 = max(0, min(height_px, py0)); py1 = max(0, min(height_px, py1))
        if (px1 - px0) < 3 or (py1 - py0) < 3:
            continue
        draw.rectangle([(px0, py0), (px1, py1)], fill=mask_color)
    return img.convert("RGB")

def _merge_nearby_images(bboxes: List[Tuple[float, float, float, float]], threshold: float = 20) -> List[Tuple[float, float, float, float]]:
    if not bboxes:
        return []
    boxes = sorted(bboxes, key=lambda b: (b[1], b[0]))
    merged: List[Tuple[float, float, float, float]] = []
    cur = list(boxes[0])
    for b in boxes[1:]:
        x0, y0, x1, y1 = b
        cx0, cy0, cx1, cy1 = cur
        vertical_gap = y0 - cy1
        horizontal_overlap = not (x1 < cx0 - threshold or x0 > cx1 + threshold)
        if vertical_gap < threshold and horizontal_overlap:
            cur[0] = min(cx0, x0); cur[1] = min(cy0, y0); cur[2] = max(cx1, x1); cur[3] = max(cy1, y1)
        else:
            merged.append(tuple(cur))
            cur = list(b)
    merged.append(tuple(cur))
    return merged

def _merge_bboxes(bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    x0 = min(b[0] for b in bboxes); y0 = min(b[1] for b in bboxes); x1 = max(b[2] for b in bboxes); y1 = max(b[3] for b in bboxes)
    return (x0, y0, x1, y1)

def _bboxes_are_close(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float], threshold: float) -> bool:
    horiz_overlap = not (b1[2] < b2[0] or b2[2] < b1[0])
    vert_overlap = not (b1[3] < b2[1] or b2[3] < b1[1])
    if horiz_overlap and vert_overlap:
        return True
    horiz_gap = min(abs(b1[0] - b2[2]), abs(b2[0] - b1[2]))
    vert_gap = min(abs(b1[1] - b2[3]), abs(b2[1] - b1[3]))
    return (horiz_gap < threshold or vert_gap < threshold)

def _find_closest_caption_region(caption_regions: List[Tuple[float, float, float, float]], bbox: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
    bx0, by0, bx1, by1 = bbox
    best = None
    best_dist = float("inf")
    for cr in caption_regions:
        cx0, cy0, cx1, cy1 = cr
        if cx1 < bx0 or cx0 > bx1:
            continue
        if cy0 < by1:
            continue
        dist = cy0 - by1
        if dist < best_dist:
            best_dist = dist
            best = cr
    return best

def _find_caption_near_bbox(text_blocks: List[Dict[str, Any]], bbox: Tuple[float, float, float, float], search_distance: float = 100) -> Optional[str]:
    x0, y0, x1, y1 = bbox
    keywords = ("figure", "fig.", "fig ", "table", "scheme", "caption")
    candidates = []
    for block in text_blocks:
        bx0, by0, bx1, by1 = block.get("bbox", (0, 0, 0, 0))
        if by0 >= y1 and by0 < y1 + search_distance and not (bx1 < x0 or bx0 > x1):
            txt = block.get("text", "")
            if any(k in txt.lower() for k in keywords):
                candidates.append((by0, txt.strip()))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    return None

def _ocr_image(pil_img: Image.Image) -> Optional[str]:
    img_np = np.array(pil_img.convert("RGB"))
    try:
        res = _EASYOCR_READER.readtext(img_np, detail=1)
    except Exception:
        return None
    lines: List[str] = []
    for box, txt, conf in res:
        if txt and conf > 0.2:
            lines.append(txt.strip())
    if lines:
        return "\n".join(lines)
    return None

def cleanup_models():
    """Cleanup GPU resources by deleting models and clearing CUDA cache"""
    global _EASYOCR_READER, _LAYOUT_MODEL
    
    # Delete EasyOCR reader
    if _EASYOCR_READER is not None:
        del _EASYOCR_READER
        _EASYOCR_READER = None
    
    # Delete layout model
    if _LAYOUT_MODEL is not None:
        del _LAYOUT_MODEL
        _LAYOUT_MODEL = None
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("GPU models offloaded and CUDA cache cleared")
