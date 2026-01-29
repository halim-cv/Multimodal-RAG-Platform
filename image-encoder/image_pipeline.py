"""
Image Processing Pipeline

This pipeline processes documents and images through two stages:
1. Document Understanding: Extracts figures from PDFs using DocLayout-YOLO
2. Scene Understanding: Generates detailed captions and extracts text from all images

The pipeline manages memory by offloading models between stages.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import numpy as np
import gc
import torch
<<<<<<< HEAD
import multiprocessing
=======
>>>>>>> ea8f64683de8f2a7696545445eada79d060d26d9

# Import engines
from document_understanding.document_understanding_engine import DocumentUnderstandingEngine
from scene_understanding.scene_understanding_engine import SceneUnderstandingEngine


<<<<<<< HEAD
def configure_cpu_constraints():
    """
    Configure CPU threading constraints when CUDA is not available.
    
    This function detects the number of available CPU cores and sets
    appropriate threading limits for PyTorch and other libraries to
    prevent oversubscription of CPU resources.
    
    Returns:
        dict: Configuration information including device type and thread count
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        # Get the number of available CPU cores
        cpu_count = multiprocessing.cpu_count()
        
        # Use a conservative thread count (typically 50-75% of available cores)
        # to avoid oversubscription and leave resources for system processes
        optimal_threads = max(1, int(cpu_count * 0.75))
        
        # Set PyTorch threading constraints
        torch.set_num_threads(optimal_threads)
        torch.set_num_interop_threads(optimal_threads)
        
        # Set OpenMP threads if available
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
        
        print(f"\n{'='*60}")
        print(f"CPU MODE DETECTED")
        print(f"{'='*60}")
        print(f"Total CPU cores available: {cpu_count}")
        print(f"Configured thread count: {optimal_threads}")
        print(f"PyTorch intra-op threads: {torch.get_num_threads()}")
        print(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")
        print(f"{'='*60}\n")
        
        return {
            'device': device,
            'cpu_count': cpu_count,
            'thread_count': optimal_threads
        }
    else:
        print(f"\n{'='*60}")
        print(f"CUDA MODE DETECTED")
        print(f"{'='*60}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"{'='*60}\n")
        
        return {
            'device': device,
            'gpu_name': torch.cuda.get_device_name(0)
        }


=======
>>>>>>> ea8f64683de8f2a7696545445eada79d060d26d9
class ImagePipeline:
    """
    Pipeline for processing documents and images through document and scene understanding engines.
    """
    
    def __init__(self, input_dir: str = r"C:\Users\Sam-tech\Desktop\Github\Multimodal-RAG-Platform\image-encoder\input", output_dir: str = r"C:\Users\Sam-tech\Desktop\Github\Multimodal-RAG-Platform\image-encoder\output"):
        """
        Initialize the pipeline.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory to save output results
        """
<<<<<<< HEAD
        # Configure CPU/GPU constraints before initializing anything else
        self.device_config = configure_cpu_constraints()
        
=======
>>>>>>> ea8f64683de8f2a7696545445eada79d060d26d9
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized output
        self.figures_dir = self.output_dir / "extracted_figures"
        self.metadata_dir = self.output_dir / "metadata"
        
        self.figures_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Storage for all image metadata
        self.all_metadata: List[Dict[str, Any]] = []
        
        # Supported file extensions
        self.document_extensions = {'.pdf'}
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    def _offload_model(self, model_name: str):
        """
        Offload a model from memory and clear GPU cache.
        
        Args:
            model_name: Name of the model being offloaded (for logging)
        """
        print(f"\n{'='*60}")
        print(f"Offloading {model_name}...")
        print(f"{'='*60}")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"{model_name} offloaded successfully.\n")
    
    def phase1_document_processing(self) -> List[Dict[str, Any]]:
        """
        Phase 1: Process documents to extract figures using Document Understanding Engine.
        
        Returns:
            List of dictionaries containing extracted figure information
        """
        print("\n" + "="*60)
        print("PHASE 1: DOCUMENT UNDERSTANDING - FIGURE EXTRACTION")
        print("="*60 + "\n")
        
        # Find all PDF files
        pdf_files = [f for f in self.input_dir.iterdir() 
                     if f.suffix.lower() in self.document_extensions]
        
        if not pdf_files:
            print("No PDF files found in input directory.")
            return []
        
        print(f"Found {len(pdf_files)} PDF file(s) to process.\n")
        
        # Initialize document engine
        doc_engine = DocumentUnderstandingEngine()
        doc_engine.load()
        
        extracted_figures = []
        
        # Process each PDF
        for pdf_idx, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{pdf_idx}/{len(pdf_files)}] Processing: {pdf_path.name}")
            print("-" * 60)
            
            try:
                # Extract figure-caption pairs
                figure_pairs = doc_engine.extract_figure_caption_pairs(str(pdf_path))
                
                print(f"Extracted {len(figure_pairs)} figure(s) from {pdf_path.name}")
                
                # Save each extracted figure
                for fig_idx, pair in enumerate(figure_pairs):
                    # Generate filename
                    figure_filename = f"{pdf_path.stem}_page{pair['page_number']}_fig{fig_idx}.png"
                    figure_path = self.figures_dir / figure_filename
                    
                    # Save figure image
                    combined_image = pair['combined_image']
                    Image.fromarray(combined_image).save(figure_path)
                    
                    # Store metadata
                    figure_info = {
                        'path': str(figure_path),
                        'filename': figure_filename,
                        'source': str(pdf_path),
                        'source_type': 'document',
                        'page_number': pair['page_number'],
                        'pair_type': pair['pair_type'],
                        'main_bbox': pair['main_bbox'],
                        'caption_bbox': pair['caption_bbox']
                    }
                    
                    extracted_figures.append(figure_info)
                    print(f"  ✓ Saved: {figure_filename}")
            
            except Exception as e:
                print(f"  ✗ Error processing {pdf_path.name}: {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Phase 1 Complete: Extracted {len(extracted_figures)} total figures")
        print(f"{'='*60}\n")
        
        # Offload document engine
        del doc_engine
        self._offload_model("Document Understanding Engine")
        
        return extracted_figures
    
    def phase2_scene_understanding(self, extracted_figures: List[Dict[str, Any]]):
        """
        Phase 2: Process all images (extracted figures + original images) using Scene Understanding Engine.
        
        Args:
            extracted_figures: List of extracted figure metadata from Phase 1
        """
        print("\n" + "="*60)
        print("PHASE 2: SCENE UNDERSTANDING - CAPTION & TEXT EXTRACTION")
        print("="*60 + "\n")
        
        # Collect all images to process
        images_to_process = []
        
        # Add extracted figures
        for fig_info in extracted_figures:
            images_to_process.append({
                'path': fig_info['path'],
                'filename': fig_info['filename'],
                'source': fig_info['source'],
                'source_type': fig_info['source_type'],
                'page_number': fig_info.get('page_number'),
                'pair_type': fig_info.get('pair_type')
            })
        
        # Add original image files from input directory
        original_images = [f for f in self.input_dir.iterdir() 
                          if f.suffix.lower() in self.image_extensions]
        
        for img_path in original_images:
            images_to_process.append({
                'path': str(img_path),
                'filename': img_path.name,
                'source': str(img_path),
                'source_type': 'original_image'
            })
        
        if not images_to_process:
            print("No images to process.")
            return
        
        print(f"Found {len(images_to_process)} image(s) to process:")
        print(f"  - {len(extracted_figures)} extracted figure(s)")
        print(f"  - {len(original_images)} original image(s)\n")
        
<<<<<<< HEAD
        # Initialize scene understanding engine with configured device
        device = self.device_config['device']
=======
        # Initialize scene understanding engine
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
>>>>>>> ea8f64683de8f2a7696545445eada79d060d26d9
        scene_engine = SceneUnderstandingEngine(device=device)
        scene_engine.load()
        
        # Process each image
        for img_idx, img_info in enumerate(images_to_process, 1):
            print(f"\n[{img_idx}/{len(images_to_process)}] Processing: {img_info['filename']}")
            print("-" * 60)
            
            try:
                # Load image
                image = Image.open(img_info['path']).convert('RGB')
                
                # Generate detailed caption
                print("  → Generating detailed caption...")
                detailed_caption = scene_engine.more_detailed_caption(image)
                
                # Extract text using OCR
                print("  → Extracting text (OCR)...")
                extracted_text = scene_engine.ocr(image)
                
                # Create complete metadata
                metadata = {
                    'path': img_info['path'],
                    'filename': img_info['filename'],
                    'file_source': img_info['source'],
                    'source_type': img_info['source_type'],
                    'detailed_caption': detailed_caption,
                    'extracted_text': extracted_text
                }
                
                # Add optional fields if they exist
                if 'page_number' in img_info and img_info['page_number'] is not None:
                    metadata['page_number'] = img_info['page_number']
                if 'pair_type' in img_info and img_info['pair_type'] is not None:
                    metadata['pair_type'] = img_info['pair_type']
                
                # Store metadata
                self.all_metadata.append(metadata)
                
                # Save individual metadata file
                metadata_filename = f"{Path(img_info['filename']).stem}_metadata.json"
                metadata_path = self.metadata_dir / metadata_filename
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                print(f"  ✓ Caption: {detailed_caption[:100]}...")
                print(f"  ✓ Text: {extracted_text[:100] if extracted_text else 'No text detected'}...")
                print(f"  ✓ Metadata saved: {metadata_filename}")
            
            except Exception as e:
                print(f"  ✗ Error processing {img_info['filename']}: {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Phase 2 Complete: Processed {len(self.all_metadata)} images")
        print(f"{'='*60}\n")
        
        # Offload scene engine
        del scene_engine
        self._offload_model("Scene Understanding Engine")
    
    def save_combined_metadata(self):
        """
        Save all metadata to a single JSON file.
        """
        print("\n" + "="*60)
        print("SAVING COMBINED METADATA")
        print("="*60 + "\n")
        
        combined_metadata_path = self.output_dir / "all_metadata.json"
        
        with open(combined_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Combined metadata saved: {combined_metadata_path}")
        print(f"  Total images processed: {len(self.all_metadata)}")
        print(f"\n{'='*60}\n")
    
    def run(self):
        """
        Execute the complete pipeline.
        """
        print("\n" + "="*60)
        print("IMAGE PROCESSING PIPELINE - START")
        print("="*60)
        print(f"Input Directory: {self.input_dir.absolute()}")
        print(f"Output Directory: {self.output_dir.absolute()}")
<<<<<<< HEAD
        print(f"Device: {self.device_config['device'].upper()}")
        if self.device_config['device'] == 'cpu':
            print(f"CPU Cores: {self.device_config['cpu_count']}")
            print(f"Thread Count: {self.device_config['thread_count']}")
        else:
            print(f"GPU: {self.device_config.get('gpu_name', 'Unknown')}")
=======
>>>>>>> ea8f64683de8f2a7696545445eada79d060d26d9
        print("="*60 + "\n")
        
        # Phase 1: Document Processing
        extracted_figures = self.phase1_document_processing()
        
        # Phase 2: Scene Understanding
        self.phase2_scene_understanding(extracted_figures)
        
        # Save combined metadata
        self.save_combined_metadata()
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETE - SUMMARY")
        print("="*60)
        print(f"✓ Extracted Figures: {len(extracted_figures)}")
        print(f"✓ Total Images Processed: {len(self.all_metadata)}")
        print(f"✓ Output Directory: {self.output_dir.absolute()}")
        print(f"  - Figures: {self.figures_dir.absolute()}")
        print(f"  - Metadata: {self.metadata_dir.absolute()}")
        print("="*60 + "\n")


def main():
    """
    Main entry point for the pipeline.
    """
    # Initialize and run pipeline
    pipeline = ImagePipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
