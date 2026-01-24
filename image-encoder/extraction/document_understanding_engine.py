"""
Document Understanding Engine for DocLayout-YOLO Model

This module provides a unified, self-contained interface for document understanding tasks
including layout detection, figure/table extraction, caption association, and OCR integration.
"""

from typing import Dict, List, Tuple, Optional, Union
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import fitz  # PyMuPDF
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10


# ==================== VISUALIZATION UTILITIES ====================

COLORMAP = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
            'lime', 'indigo', 'violet', 'aqua', 'magenta', 'coral', 'gold', 'tan', 'skyblue']


def plot_bbox(image, data):
    """Plot bounding boxes on an image."""
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)
    
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1-5, label, color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.7))
    
    ax.axis('off')
    plt.tight_layout()
    return fig


def draw_detections(image, detections_df):
    """Draw all detections on an image with colored bounding boxes."""
    import random
    
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    
    for _, detection in detections_df.iterrows():
        bbox = detection['bbox']
        class_name = detection['class_name']
        
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        color = random.choice(COLORMAP)
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        draw.text((x1 + 5, y1 + 5), class_name, fill=color)
    
    return image_copy


# ==================== PDF UTILITIES ====================

def pdf_to_image(pdf_path: Union[str, Path], page_number: int = 0, dpi: int = 200) -> np.ndarray:
    """
    Convert a PDF page to an image array.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to convert (0-indexed)
        dpi: Resolution for conversion
        
    Returns:
        NumPy array representing the image
    """
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(page_number)
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return np.array(img)


def pdf_page_count(pdf_path: Union[str, Path]) -> int:
    """Get the total number of pages in a PDF."""
    doc = fitz.open(str(pdf_path))
    count = doc.page_count
    doc.close()
    return count


# ==================== SPATIAL ASSOCIATION UTILITIES ====================

def calculate_distance(main_bbox: List[float], caption_bbox: List[float]) -> Tuple[float, float, str]:
    """
    Calculate spatial distance between a main element (figure/table) and its caption.
    
    Args:
        main_bbox: [x_min, y_min, x_max, y_max] of main element
        caption_bbox: [x_min, y_min, x_max, y_max] of caption
        
    Returns:
        Tuple of (vertical_distance, horizontal_proximity, vertical_direction)
    """
    main_x_min, main_y_min, main_x_max, main_y_max = main_bbox
    cap_x_min, cap_y_min, cap_x_max, cap_y_max = caption_bbox
    
    # Vertical distance and direction
    if cap_y_min > main_y_max:  # Caption is below
        vertical_distance = cap_y_min - main_y_max
        vertical_direction = 'below'
    elif cap_y_max < main_y_min:  # Caption is above
        vertical_distance = main_y_min - cap_y_max
        vertical_direction = 'above'
    else:  # Overlapping vertically
        return float('inf'), float('inf'), 'overlap'
    
    # Horizontal overlap/distance
    x_overlap = max(0, min(main_x_max, cap_x_max) - max(main_x_min, cap_x_min))
    
    if x_overlap > 0:  # There is horizontal overlap
        horizontal_proximity = 0 - x_overlap  # Negative value, larger overlap is better
    else:  # No horizontal overlap
        dist1 = abs(main_x_max - cap_x_min)
        dist2 = abs(main_x_min - cap_x_max)
        horizontal_proximity = min(dist1, dist2)
    
    return vertical_distance, horizontal_proximity, vertical_direction


def crop_and_combine(main_bbox: List[float], caption_bbox: List[float], 
                     page_image: np.ndarray, pair_type: str, 
                     page_number: int) -> Dict:
    """
    Crop and combine a main element with its caption.
    
    Args:
        main_bbox: Bounding box of main element
        caption_bbox: Bounding box of caption
        page_image: Full page image
        pair_type: Type of pair ('figure-caption' or 'table-caption')
        page_number: Page number
        
    Returns:
        Dictionary containing combined image and metadata
    """
    # Ensure bounding box coordinates are integers
    main_bbox = [int(coord) for coord in main_bbox]
    caption_bbox = [int(coord) for coord in caption_bbox]
    
    # Crop the main image
    main_x_min, main_y_min, main_x_max, main_y_max = main_bbox
    main_image = page_image[main_y_min:main_y_max, main_x_min:main_x_max]
    
    # Crop the caption image
    cap_x_min, cap_y_min, cap_x_max, cap_y_max = caption_bbox
    caption_image = page_image[cap_y_min:cap_y_max, cap_x_min:cap_x_max]
    
    # Get max width for padding
    max_width = max(main_image.shape[1], caption_image.shape[1])
    
    # Pad main image if needed
    if main_image.shape[1] < max_width:
        pad_width = max_width - main_image.shape[1]
        main_image = np.pad(main_image, ((0, 0), (0, pad_width), (0, 0)), 
                           mode='constant', constant_values=255)
    
    # Pad caption image if needed
    if caption_image.shape[1] < max_width:
        pad_width = max_width - caption_image.shape[1]
        caption_image = np.pad(caption_image, ((0, 0), (0, pad_width), (0, 0)), 
                              mode='constant', constant_values=255)
    
    # Combine vertically with white separator (ensure uint8 type)
    separator = np.full((20, max_width, 3), 255, dtype=np.uint8)
    
    # Ensure inputs are uint8 before stacking to avoid upcasting
    main_image = main_image.astype(np.uint8)
    caption_image = caption_image.astype(np.uint8)
    
    combined_image = np.vstack((main_image, separator, caption_image))
    
    return {
        'combined_image': combined_image,
        'pair_type': pair_type,
        'page_number': page_number,
        'main_bbox': main_bbox,
        'caption_bbox': caption_bbox
    }


# ==================== CORE ENGINE ====================

class DocumentUnderstandingEngine:
    """
    Unified engine for document understanding tasks using DocLayout-YOLO model.
    
    Supports:
    - Document layout detection (figures, tables, captions, text blocks, etc.)
    - Figure-caption association and extraction
    - Table-caption association and extraction
    - Multi-page PDF processing
    - Spatial analysis and region extraction
    - Visualization of detected elements
    """
    
    def __init__(self, model_id: str = "juliozhao/DocLayout-YOLO-DocStructBench", 
                 model_filename: str = "doclayout_yolo_docstructbench_imgsz1024.pt"):
        """
        Initialize the Document Understanding Engine.
        
        Args:
            model_id: HuggingFace repository ID for DocLayout-YOLO
            model_filename: Model filename in the repository
        """
        self.model_id = model_id
        self.model_filename = model_filename
        self.model = None
        self._is_loaded = False
    
    def load(self):
        """Load the model from HuggingFace Hub."""
        if not self._is_loaded:
            print(f"Loading {self.model_id}...")
            model_path = hf_hub_download(repo_id=self.model_id, filename=self.model_filename)
            self.model = YOLOv10(model=model_path)
            self._is_loaded = True
            print("Model loaded successfully!")
        return self
    
    def _ensure_loaded(self):
        """Ensure model is loaded before inference."""
        if not self._is_loaded:
            self.load()
    
    # ==================== DETECTION TASKS ====================
    
    def detect_layout(self, image: Union[Image.Image, np.ndarray]) -> pd.DataFrame:
        """
        Detect document layout elements in an image.
        
        Args:
            image: PIL Image or NumPy array
            
        Returns:
            DataFrame with columns: bbox, class_id, class_name
        """
        self._ensure_loaded()
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Run inference
        results = self.model(image)
        result = results[0]
        
        # Extract detections
        detections_data = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            bboxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            class_names_map = result.names
            
            for i in range(len(bboxes)):
                bbox = bboxes[i].tolist()
                class_id = int(class_ids[i])
                class_name = class_names_map[class_id]
                
                detections_data.append({
                    'bbox': bbox,
                    'class_id': class_id,
                    'class_name': class_name
                })
        
        return pd.DataFrame(detections_data)
    
    def detect_layout_with_page(self, image: Union[Image.Image, np.ndarray], 
                                 page_number: int = 0) -> pd.DataFrame:
        """
        Detect document layout with page number information.
        
        Args:
            image: PIL Image or NumPy array
            page_number: Page number for tracking
            
        Returns:
            DataFrame with columns: page_number, bbox, class_id, class_name
        """
        detections_df = self.detect_layout(image)
        detections_df.insert(0, 'page_number', page_number)
        return detections_df
    
    def visualize_detections(self, image: Union[Image.Image, np.ndarray], 
                            detections_df: pd.DataFrame = None,
                            show_labels: bool = True, 
                            show_conf: bool = False) -> np.ndarray:
        """
        Visualize detections on an image.
        
        Args:
            image: PIL Image or NumPy array
            detections_df: Optional pre-computed detections DataFrame
            show_labels: Whether to show class labels
            show_conf: Whether to show confidence scores
            
        Returns:
            Annotated image as NumPy array (RGB)
        """
        self._ensure_loaded()
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Run inference
        results = self.model(image)
        result = results[0]
        
        # Plot results
        annotated_image = result.plot(labels=show_labels, conf=show_conf)
        
        # Convert BGR to RGB
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        return annotated_image_rgb
    
    # ==================== PDF PROCESSING TASKS ====================
    
    def process_pdf_page(self, pdf_path: Union[str, Path], page_number: int = 0, 
                         dpi: int = 200) -> Dict:
        """
        Process a single PDF page.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number to process (0-indexed)
            dpi: Resolution for conversion
            
        Returns:
            Dictionary with 'image' and 'detections' keys
        """
        image = pdf_to_image(pdf_path, page_number, dpi)
        detections = self.detect_layout_with_page(image, page_number)
        
        return {
            'image': image,
            'detections': detections,
            'page_number': page_number
        }
    
    def process_pdf(self, pdf_path: Union[str, Path], dpi: int = 200) -> Dict:
        """
        Process all pages of a PDF document.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            Dictionary with 'images', 'all_detections', and 'page_count' keys
        """
        self._ensure_loaded()
        
        total_pages = pdf_page_count(pdf_path)
        print(f"Processing {total_pages} pages from {pdf_path}...")
        
        all_images = []
        all_detections_data = []
        
        for page_number in range(total_pages):
            print(f"Processing page {page_number + 1}/{total_pages}...")
            
            # Convert page to image
            image = pdf_to_image(pdf_path, page_number, dpi)
            all_images.append(image)
            
            # Detect layout
            detections = self.detect_layout_with_page(image, page_number)
            
            # Append to all detections
            if not detections.empty:
                all_detections_data.append(detections)
            else:
                print(f"No detections found on page {page_number + 1}.")
        
        # Combine all detections
        all_detections_df = pd.concat(all_detections_data, ignore_index=True) if all_detections_data else pd.DataFrame()
        
        print(f"Successfully processed {total_pages} pages with {len(all_detections_df)} total detections.")
        
        return {
            'images': all_images,
            'all_detections': all_detections_df,
            'page_count': total_pages
        }
    
    # ==================== ASSOCIATION TASKS ====================
    
    def associate_figures_with_captions(self, detections_df: pd.DataFrame, 
                                       prefer_below: bool = True) -> List[Dict]:
        """
        Associate figure detections with their captions.
        
        Args:
            detections_df: DataFrame with all detections
            prefer_below: Whether to prefer captions below figures
            
        Returns:
            List of dictionaries containing association information
        """
        figures_df = detections_df[detections_df['class_name'] == 'figure'].copy()
        captions_df = detections_df[detections_df['class_name'] == 'figure_caption'].copy()
        
        print(f"Associating {len(figures_df)} figures with {len(captions_df)} captions...")
        
        associated_pairs = []
        available_captions = captions_df.copy()
        
        for fig_index, figure in figures_df.iterrows():
            figure_bbox = figure['bbox']
            figure_page = figure.get('page_number', 0)
            
            # Filter captions on the same page
            same_page_captions = available_captions[
                available_captions.get('page_number', 0) == figure_page
            ] if 'page_number' in available_captions.columns else available_captions
            
            best_caption = None
            min_vertical_dist = float('inf')
            min_horizontal_proximity = float('inf')
            best_caption_index = -1
            
            for cap_index, caption in same_page_captions.iterrows():
                caption_bbox = caption['bbox']
                
                vertical_dist, horizontal_prox, vertical_direction = calculate_distance(
                    figure_bbox, caption_bbox
                )
                
                # Apply preference for captions below figures
                if prefer_below and vertical_direction != 'below':
                    continue
                
                if vertical_dist != float('inf'):
                    if vertical_dist < min_vertical_dist:
                        min_vertical_dist = vertical_dist
                        min_horizontal_proximity = horizontal_prox
                        best_caption = caption
                        best_caption_index = cap_index
                    elif vertical_dist == min_vertical_dist and horizontal_prox < min_horizontal_proximity:
                        min_horizontal_proximity = horizontal_prox
                        best_caption = caption
                        best_caption_index = cap_index
            
            if best_caption is not None:
                associated_pairs.append({
                    'figure_page_number': figure_page,
                    'figure_bbox': figure_bbox,
                    'figure_class_name': figure['class_name'],
                    'caption_page_number': best_caption.get('page_number', 0),
                    'caption_bbox': best_caption['bbox'],
                    'caption_class_name': best_caption['class_name'],
                    'vertical_distance': min_vertical_dist,
                    'horizontal_proximity': min_horizontal_proximity
                })
                available_captions = available_captions.drop(best_caption_index)
        
        print(f"Successfully associated {len(associated_pairs)} figure-caption pairs.")
        return associated_pairs
    
    def associate_tables_with_captions(self, detections_df: pd.DataFrame) -> List[Dict]:
        """
        Associate table detections with their captions.
        
        Args:
            detections_df: DataFrame with all detections
            
        Returns:
            List of dictionaries containing association information
        """
        tables_df = detections_df[detections_df['class_name'] == 'table'].copy()
        captions_df = detections_df[detections_df['class_name'] == 'table_caption'].copy()
        
        print(f"Associating {len(tables_df)} tables with {len(captions_df)} captions...")
        
        associated_pairs = []
        available_captions = captions_df.copy()
        
        for table_index, table in tables_df.iterrows():
            table_bbox = table['bbox']
            table_page = table.get('page_number', 0)
            
            # Filter captions on the same page
            same_page_captions = available_captions[
                available_captions.get('page_number', 0) == table_page
            ] if 'page_number' in available_captions.columns else available_captions
            
            best_caption = None
            min_total_proximity = float('inf')
            best_caption_index = -1
            
            for cap_index, caption in same_page_captions.iterrows():
                caption_bbox = caption['bbox']
                
                vertical_dist, horizontal_prox, vertical_direction = calculate_distance(
                    table_bbox, caption_bbox
                )
                
                if vertical_dist != float('inf'):
                    # Combine distances for total proximity metric
                    current_total_proximity = vertical_dist + abs(horizontal_prox)
                    
                    if current_total_proximity < min_total_proximity:
                        min_total_proximity = current_total_proximity
                        best_caption = caption
                        best_caption_index = cap_index
            
            if best_caption is not None:
                final_vertical_dist, final_horizontal_prox, final_vertical_direction = calculate_distance(
                    table_bbox, best_caption['bbox']
                )
                
                associated_pairs.append({
                    'table_page_number': table_page,
                    'table_bbox': table_bbox,
                    'table_class_name': table['class_name'],
                    'caption_page_number': best_caption.get('page_number', 0),
                    'caption_bbox': best_caption['bbox'],
                    'caption_class_name': best_caption['class_name'],
                    'vertical_distance': final_vertical_dist,
                    'horizontal_proximity': final_horizontal_prox,
                    'vertical_direction': final_vertical_direction
                })
                available_captions = available_captions.drop(best_caption_index)
        
        print(f"Successfully associated {len(associated_pairs)} table-caption pairs.")
        return associated_pairs
    
    # ==================== EXTRACTION TASKS ====================
    
    def extract_figure_caption_pairs(self, pdf_path: Union[str, Path], 
                                     dpi: int = 200) -> List[Dict]:
        """
        Extract all figure-caption pairs from a PDF.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            List of extracted pairs with combined images
        """
        # Process PDF
        pdf_data = self.process_pdf(pdf_path, dpi)
        images = pdf_data['images']
        detections = pdf_data['all_detections']
        
        # Associate figures with captions
        associations = self.associate_figures_with_captions(detections)
        
        # Extract and combine images
        extracted_pairs = []
        for pair in associations:
            page_number = pair['figure_page_number']
            page_image = images[page_number]
            
            extracted_pair = crop_and_combine(
                pair['figure_bbox'],
                pair['caption_bbox'],
                page_image,
                'figure-caption',
                page_number
            )
            extracted_pairs.append(extracted_pair)
        
        print(f"Extracted {len(extracted_pairs)} figure-caption pairs.")
        return extracted_pairs
    
    def extract_table_caption_pairs(self, pdf_path: Union[str, Path], 
                                    dpi: int = 200) -> List[Dict]:
        """
        Extract all table-caption pairs from a PDF.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            List of extracted pairs with combined images
        """
        # Process PDF
        pdf_data = self.process_pdf(pdf_path, dpi)
        images = pdf_data['images']
        detections = pdf_data['all_detections']
        
        # Associate tables with captions
        associations = self.associate_tables_with_captions(detections)
        
        # Extract and combine images
        extracted_pairs = []
        for pair in associations:
            page_number = pair['table_page_number']
            page_image = images[page_number]
            
            extracted_pair = crop_and_combine(
                pair['table_bbox'],
                pair['caption_bbox'],
                page_image,
                'table-caption',
                page_number
            )
            extracted_pairs.append(extracted_pair)
        
        print(f"Extracted {len(extracted_pairs)} table-caption pairs.")
        return extracted_pairs
    
    def extract_all_pairs(self, pdf_path: Union[str, Path], 
                         dpi: int = 200) -> Dict[str, List[Dict]]:
        """
        Extract all figure-caption and table-caption pairs from a PDF.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            Dictionary with 'figures' and 'tables' keys containing extracted pairs
        """
        # Process PDF once
        pdf_data = self.process_pdf(pdf_path, dpi)
        images = pdf_data['images']
        detections = pdf_data['all_detections']
        
        # Associate figures and tables
        figure_associations = self.associate_figures_with_captions(detections)
        table_associations = self.associate_tables_with_captions(detections)
        
        # Extract figure pairs
        figure_pairs = []
        for pair in figure_associations:
            page_number = pair['figure_page_number']
            page_image = images[page_number]
            extracted_pair = crop_and_combine(
                pair['figure_bbox'], pair['caption_bbox'],
                page_image, 'figure-caption', page_number
            )
            figure_pairs.append(extracted_pair)
        
        # Extract table pairs
        table_pairs = []
        for pair in table_associations:
            page_number = pair['table_page_number']
            page_image = images[page_number]
            extracted_pair = crop_and_combine(
                pair['table_bbox'], pair['caption_bbox'],
                page_image, 'table-caption', page_number
            )
            table_pairs.append(extracted_pair)
        
        print(f"Extracted {len(figure_pairs)} figure pairs and {len(table_pairs)} table pairs.")
        
        return {
            'figures': figure_pairs,
            'tables': table_pairs,
            'total_pairs': len(figure_pairs) + len(table_pairs)
        }
    
    # ==================== UTILITY METHODS ====================
    
    def analyze_document(self, pdf_path: Union[str, Path], dpi: int = 200) -> Dict:
        """
        Perform comprehensive document analysis.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            Dictionary with complete analysis results
        """
        pdf_data = self.process_pdf(pdf_path, dpi)
        detections = pdf_data['all_detections']
        
        # Get statistics
        class_counts = detections['class_name'].value_counts().to_dict() if not detections.empty else {}
        
        # Extract pairs
        extracted_pairs = self.extract_all_pairs(pdf_path, dpi)
        
        return {
            'page_count': pdf_data['page_count'],
            'total_detections': len(detections),
            'class_counts': class_counts,
            'figure_caption_pairs': len(extracted_pairs['figures']),
            'table_caption_pairs': len(extracted_pairs['tables']),
            'extracted_pairs': extracted_pairs,
            'all_detections': detections
        }
    
    def display_extracted_pairs(self, extracted_pairs: List[Dict], 
                               max_display: int = 3):
        """
        Display extracted pairs.
        
        Args:
            extracted_pairs: List of extracted pair dictionaries
            max_display: Maximum number of pairs to display
        """
        num_to_display = min(max_display, len(extracted_pairs))
        
        for i in range(num_to_display):
            pair_data = extracted_pairs[i]
            plt.figure(figsize=(10, 10))
            plt.imshow(pair_data['combined_image'])
            plt.axis('off')
            plt.title(f"Extracted {pair_data['pair_type']} Pair {i+1} (Page {pair_data['page_number'] + 1})")
            plt.show()


# Convenience function for quick usage
def create_engine(model_id: str = "juliozhao/DocLayout-YOLO-DocStructBench",
                  model_filename: str = "doclayout_yolo_docstructbench_imgsz1024.pt") -> DocumentUnderstandingEngine:
    """
    Create and load a Document Understanding Engine.
    
    Args:
        model_id: HuggingFace repository ID
        model_filename: Model filename
        
    Returns:
        Loaded DocumentUnderstandingEngine instance
    """
    engine = DocumentUnderstandingEngine(model_id, model_filename)
    engine.load()
    return engine
