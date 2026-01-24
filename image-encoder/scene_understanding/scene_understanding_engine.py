"""
Scene Understanding Engine for Florence-2 Model

This module provides a unified, self-contained interface for all scene understanding tasks
including captioning, object detection, segmentation, OCR, and VQA.
"""

from typing import Dict, List, Tuple, Optional, Union
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


# ==================== VISUALIZATION UTILITIES ====================

COLORMAP = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red',
            'lime', 'indigo', 'violet', 'aqua', 'magenta', 'coral', 'gold', 'tan', 'skyblue']


def plot_bbox(image, data):
    """Plot bounding boxes on an image."""
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    
    ax.axis('off')
    plt.show()


def draw_polygons(image, prediction, fill_mask=False):
    """
    Draws segmentation masks with polygons on an image.
    
    Parameters:
    - image: PIL Image
    - prediction: Dictionary containing 'polygons' and 'labels' keys.
    - fill_mask: Boolean indicating whether to fill the polygons with color.
    
    Returns:
    - Modified image
    """
    draw = ImageDraw.Draw(image)
    scale = 1
    
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(COLORMAP)
        fill_color = random.choice(COLORMAP) if fill_mask else None
        
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue
            
            _polygon = (_polygon * scale).reshape(-1).tolist()
            
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
    
    return image


def draw_ocr_bboxes(image, prediction, scale=1):
    """Draw OCR bounding boxes on an image."""
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    
    for box, label in zip(bboxes, labels):
        color = random.choice(COLORMAP)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2), "{}".format(label), align="right", fill=color)
    
    return image


def convert_to_od_format(data):
    """
    Converts a dictionary with 'bboxes' and 'bboxes_labels' into standard object detection format.
    
    Parameters:
    - data: Dictionary with 'bboxes', 'bboxes_labels', 'polygons', 'polygons_labels' keys.
    
    Returns:
    - Dictionary with 'bboxes' and 'labels' keys.
    """
    return {
        'bboxes': data.get('bboxes', []),
        'labels': data.get('bboxes_labels', [])
    }


# ==================== CORE ENGINE ====================

class SceneUnderstandingEngine:
    """
    Unified engine for scene understanding tasks using Florence-2 model.
    
    Supports:
    - Captioning (basic, detailed, more detailed)
    - Object Detection
    - Dense Region Captioning
    - Region Proposals
    - Phrase Grounding
    - Referring Expression Segmentation
    - Region to Segmentation
    - Open Vocabulary Detection
    - Region to Category/Description
    - OCR and OCR with Regions
    - VQA (Visual Question Answering) with prompt-grounded captions
    - Cascaded tasks (Caption + Phrase Grounding)
    """
    
    def __init__(self, model_id: str = 'microsoft/Florence-2-base', device: str = 'cuda'):
        """
        Initialize the Scene Understanding Engine.
        
        Args:
            model_id: HuggingFace model ID for Florence-2
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._is_loaded = False
    
    def load(self):
        """Load the model and processor."""
        if not self._is_loaded:
            print(f"Loading {self.model_id}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                dtype='auto', 
                attn_implementation='eager'
            ).eval().to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self._is_loaded = True
            print("Model loaded successfully!")
        return self
    
    def _ensure_loaded(self):
        """Ensure model is loaded before inference."""
        if not self._is_loaded:
            self.load()
    
    def _run_task(self, image: Image.Image, task_prompt: str, text_input: Optional[str] = None) -> Dict:
        """
        Run a single task on the image.
        
        Args:
            image: PIL Image
            task_prompt: Task prompt (e.g., '<CAPTION>')
            text_input: Optional text input for the task
            
        Returns:
            Dictionary containing task results
        """
        self._ensure_loaded()
        
        # Prepare prompt
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        
        # Run inference
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
            use_cache=False,
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        
        return parsed_answer
    
    # ==================== CAPTIONING TASKS ====================
    
    def caption(self, image: Image.Image) -> str:
        """
        Generate a basic caption for the image.
        
        Args:
            image: PIL Image
            
        Returns:
            Caption string
        """
        result = self._run_task(image, '<CAPTION>')
        return result.get('<CAPTION>', '')
    
    def detailed_caption(self, image: Image.Image) -> str:
        """
        Generate a detailed caption for the image.
        
        Args:
            image: PIL Image
            
        Returns:
            Detailed caption string
        """
        result = self._run_task(image, '<DETAILED_CAPTION>')
        return result.get('<DETAILED_CAPTION>', '')
    
    def more_detailed_caption(self, image: Image.Image) -> str:
        """
        Generate a more detailed caption for the image.
        
        Args:
            image: PIL Image
            
        Returns:
            More detailed caption string
        """
        result = self._run_task(image, '<MORE_DETAILED_CAPTION>')
        return result.get('<MORE_DETAILED_CAPTION>', '')
    
    def all_captions(self, image: Image.Image) -> Dict[str, str]:
        """
        Generate all levels of captions for the image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with 'basic', 'detailed', 'more_detailed' keys
        """
        return {
            'basic': self.caption(image),
            'detailed': self.detailed_caption(image),
            'more_detailed': self.more_detailed_caption(image)
        }
    
    # ==================== OBJECT DETECTION TASKS ====================
    
    def object_detection(self, image: Image.Image) -> Dict:
        """
        Perform object detection on the image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with 'bboxes' and 'labels' keys
            Format: {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['label1', ...]}
        """
        result = self._run_task(image, '<OD>')
        return result.get('<OD>', {'bboxes': [], 'labels': []})
    
    def dense_region_caption(self, image: Image.Image) -> Dict:
        """
        Generate captions for dense regions in the image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with 'bboxes' and 'labels' keys
        """
        result = self._run_task(image, '<DENSE_REGION_CAPTION>')
        return result.get('<DENSE_REGION_CAPTION>', {'bboxes': [], 'labels': []})
    
    def region_proposal(self, image: Image.Image) -> Dict:
        """
        Generate region proposals for the image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with 'bboxes' and 'labels' keys
        """
        result = self._run_task(image, '<REGION_PROPOSAL>')
        return result.get('<REGION_PROPOSAL>', {'bboxes': [], 'labels': []})
    
    def phrase_grounding(self, image: Image.Image, phrase: str) -> Dict:
        """
        Ground a phrase in the image (locate objects matching the phrase).
        
        Args:
            image: PIL Image
            phrase: Text phrase to ground
            
        Returns:
            Dictionary with 'bboxes' and 'labels' keys
        """
        result = self._run_task(image, '<CAPTION_TO_PHRASE_GROUNDING>', text_input=phrase)
        return result.get('<CAPTION_TO_PHRASE_GROUNDING>', {'bboxes': [], 'labels': []})
    
    def open_vocabulary_detection(self, image: Image.Image, query: str) -> Dict:
        """
        Detect objects using open vocabulary detection.
        
        Args:
            image: PIL Image
            query: Search query
            
        Returns:
            Dictionary with 'bboxes', 'bboxes_labels', 'polygons', 'polygons_labels' keys
        """
        result = self._run_task(image, '<OPEN_VOCABULARY_DETECTION>', text_input=query)
        return result.get('<OPEN_VOCABULARY_DETECTION>', {})
    
    # ==================== SEGMENTATION TASKS ====================
    
    def referring_expression_segmentation(self, image: Image.Image, expression: str) -> Dict:
        """
        Segment objects based on a referring expression.
        
        Args:
            image: PIL Image
            expression: Referring expression (e.g., "a green car")
            
        Returns:
            Dictionary with 'polygons' and 'labels' keys
        """
        result = self._run_task(image, '<REFERRING_EXPRESSION_SEGMENTATION>', text_input=expression)
        return result.get('<REFERRING_EXPRESSION_SEGMENTATION>', {'polygons': [], 'labels': []})
    
    def region_to_segmentation(self, image: Image.Image, region: str) -> Dict:
        """
        Convert a region to segmentation.
        
        Args:
            image: PIL Image
            region: Region specification in format '<loc_x1><loc_y1><loc_x2><loc_y2>'
                   where coordinates are quantized to [0, 999]
            
        Returns:
            Dictionary with 'polygons' and 'labels' keys
        """
        result = self._run_task(image, '<REGION_TO_SEGMENTATION>', text_input=region)
        return result.get('<REGION_TO_SEGMENTATION>', {'polygons': [], 'labels': []})
    
    # ==================== REGION UNDERSTANDING TASKS ====================
    
    def region_to_category(self, image: Image.Image, region: str) -> str:
        """
        Get the category of a region.
        
        Args:
            image: PIL Image
            region: Region specification in format '<loc_x1><loc_y1><loc_x2><loc_y2>'
            
        Returns:
            Category string
        """
        result = self._run_task(image, '<REGION_TO_CATEGORY>', text_input=region)
        return result.get('<REGION_TO_CATEGORY>', '')
    
    def region_to_description(self, image: Image.Image, region: str) -> str:
        """
        Get a description of a region.
        
        Args:
            image: PIL Image
            region: Region specification in format '<loc_x1><loc_y1><loc_x2><loc_y2>'
            
        Returns:
            Description string
        """
        result = self._run_task(image, '<REGION_TO_DESCRIPTION>', text_input=region)
        return result.get('<REGION_TO_DESCRIPTION>', '')
    
    # ==================== OCR TASKS ====================
    
    def ocr(self, image: Image.Image) -> str:
        """
        Perform OCR on the image.
        
        Args:
            image: PIL Image
            
        Returns:
            Extracted text
        """
        result = self._run_task(image, '<OCR>')
        return result.get('<OCR>', '')
    
    def ocr_with_region(self, image: Image.Image) -> Dict:
        """
        Perform OCR with region information.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with 'quad_boxes' and 'labels' keys
            Format: {'quad_boxes': [[x1,y1,x2,y2,x3,y3,x4,y4], ...], 'labels': ['text1', ...]}
        """
        result = self._run_task(image, '<OCR_WITH_REGION>')
        return result.get('<OCR_WITH_REGION>', {'quad_boxes': [], 'labels': []})
    
    # ==================== VQA (Visual Question Answering) TASKS ====================
    
    def prompt_grounded_caption(self, image: Image.Image, prompt: str) -> Dict:
        """
        Generate a caption or answer based on a user prompt/question.
        This uses caption-to-phrase-grounding with a custom prompt.
        
        Args:
            image: PIL Image
            prompt: User prompt or question about the image
            
        Returns:
            Dictionary with 'prompt', 'caption', 'grounding' keys
            - prompt: The original user prompt
            - caption: Generated response/caption
            - grounding: Bounding boxes for grounded phrases if applicable
        """
        # First try direct caption-to-phrase-grounding with the prompt
        grounding_result = self._run_task(image, '<CAPTION_TO_PHRASE_GROUNDING>', text_input=prompt)
        grounding = grounding_result.get('<CAPTION_TO_PHRASE_GROUNDING>', {'bboxes': [], 'labels': []})
        
        # If grounding worked, return it with the prompt as caption
        if grounding['bboxes']:
            return {
                'prompt': prompt,
                'caption': prompt,
                'grounding': grounding
            }
        
        # Otherwise, generate a detailed caption and try to answer based on it
        caption = self.more_detailed_caption(image)
        
        # Try to ground the generated caption
        caption_grounding = self._run_task(image, '<CAPTION_TO_PHRASE_GROUNDING>', text_input=caption)
        
        return {
            'prompt': prompt,
            'caption': caption,
            'grounding': caption_grounding.get('<CAPTION_TO_PHRASE_GROUNDING>', {'bboxes': [], 'labels': []})
        }
    
    def visual_question_answering(self, image: Image.Image, question: str, 
                                   return_grounding: bool = True) -> Union[str, Dict]:
        """
        Answer a question about the image using visual understanding.
        
        Note: Florence-2 may not have native VQA support. This method uses
        detailed captioning and phrase grounding to provide context-based answers.
        
        Args:
            image: PIL Image
            question: Question about the image
            return_grounding: If True, returns dict with caption and grounding.
                            If False, returns only the caption text.
            
        Returns:
            If return_grounding=True: Dictionary with 'question', 'answer', 'grounding'
            If return_grounding=False: Answer string
            
        Examples:
            >>> answer = engine.visual_question_answering(image, "What is in this image?")
            >>> answer = engine.visual_question_answering(image, "Where is the car?", return_grounding=True)
        """
        result = self.prompt_grounded_caption(image, question)
        
        if return_grounding:
            return {
                'question': question,
                'answer': result['caption'],
                'grounding': result['grounding']
            }
        else:
            return result['caption']
    
    # ==================== CASCADED TASKS ====================

    
    def caption_and_ground(self, image: Image.Image, caption_level: str = 'basic') -> Dict:
        """
        Generate a caption and ground all phrases in it.
        
        Args:
            image: PIL Image
            caption_level: 'basic', 'detailed', or 'more_detailed'
            
        Returns:
            Dictionary with caption and grounding results
        """
        # Generate caption
        if caption_level == 'detailed':
            task_prompt = '<DETAILED_CAPTION>'
        elif caption_level == 'more_detailed':
            task_prompt = '<MORE_DETAILED_CAPTION>'
        else:
            task_prompt = '<CAPTION>'
        
        caption_result = self._run_task(image, task_prompt)
        caption = caption_result.get(task_prompt, '')
        
        # Ground the caption
        grounding_result = self._run_task(image, '<CAPTION_TO_PHRASE_GROUNDING>', text_input=caption)
        
        return {
            'caption': caption,
            'caption_level': caption_level,
            'grounding': grounding_result.get('<CAPTION_TO_PHRASE_GROUNDING>', {'bboxes': [], 'labels': []})
        }
    
    # ==================== UTILITY METHODS ====================
    
    def analyze_image(self, image: Image.Image, tasks: Optional[List[str]] = None) -> Dict:
        """
        Run multiple analysis tasks on an image.
        
        Args:
            image: PIL Image
            tasks: List of task names. If None, runs common tasks.
                   Options: 'caption', 'detailed_caption', 'object_detection', 
                           'dense_caption', 'ocr'
            
        Returns:
            Dictionary with results for each task
        """
        if tasks is None:
            tasks = ['caption', 'detailed_caption', 'object_detection', 'ocr']
        
        results = {}
        
        for task in tasks:
            if task == 'caption':
                results['caption'] = self.caption(image)
            elif task == 'detailed_caption':
                results['detailed_caption'] = self.detailed_caption(image)
            elif task == 'more_detailed_caption':
                results['more_detailed_caption'] = self.more_detailed_caption(image)
            elif task == 'object_detection':
                results['object_detection'] = self.object_detection(image)
            elif task == 'dense_caption':
                results['dense_caption'] = self.dense_region_caption(image)
            elif task == 'ocr':
                results['ocr'] = self.ocr(image)
            elif task == 'ocr_with_region':
                results['ocr_with_region'] = self.ocr_with_region(image)
        
        return results
    
    def process_image_path(self, image_path: Union[str, Path], tasks: Optional[List[str]] = None) -> Dict:
        """
        Load an image from path and analyze it.
        
        Args:
            image_path: Path to the image file
            tasks: List of task names to run
            
        Returns:
            Dictionary with results for each task
        """
        image = Image.open(image_path)
        return self.analyze_image(image, tasks)


# Convenience function for quick usage
def create_engine(model_id: str = 'microsoft/Florence-2-base', device: str = 'cuda') -> SceneUnderstandingEngine:
    """
    Create and load a Scene Understanding Engine.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to run on
        
    Returns:
        Loaded SceneUnderstandingEngine instance
    """
    engine = SceneUnderstandingEngine(model_id, device)
    engine.load()
    return engine
