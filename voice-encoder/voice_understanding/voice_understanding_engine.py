"""
Voice Understanding Engine using OpenAI Whisper-tiny model.
This module provides timestamped text extraction from audio files.
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import warnings
from typing import Dict, List, Optional, Union
import numpy as np

warnings.filterwarnings("ignore")


class VoiceUnderstandingEngine:
    """
    A class for loading and using the Whisper-tiny model for timestamped transcription.
    
    Attributes:
        model_name (str): The Hugging Face model identifier
        device (str): The device to run the model on ('cuda' or 'cpu')
        processor (WhisperProcessor): The Whisper processor for audio preprocessing
        model (WhisperForConditionalGeneration): The Whisper model
    """
    
    def __init__(
        self, 
        model_name: str = "openai/whisper-tiny",
        device: Optional[str] = None
    ):
        """
        Initialize the Voice Understanding Engine.
        
        Args:
            model_name (str): Hugging Face model identifier (default: "openai/whisper-tiny")
            device (str, optional): Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if self.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available. Falling back to CPU.")
            self.device = "cpu"
            
        print(f"Initializing Voice Understanding Engine on {self.device.upper()}...")
        
        # Load processor and model
        self._load_model()
        
        print(f"Model loaded successfully on {self.device.upper()}!")
        
    def _load_model(self):
        """Load the Whisper processor and model."""
        try:
            # Load processor
            print(f"Loading processor from {self.model_name}...")
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            
            # Load model
            print(f"Loading model from {self.model_name}...")
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def transcribe_with_timestamps(
        self,
        audio_input: Union[np.ndarray, str],
        sampling_rate: int = 16000,
        return_timestamps: bool = True,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict:
        """
        Transcribe audio with word-level or segment-level timestamps.
        
        Args:
            audio_input (Union[np.ndarray, str]): Audio array or path to audio file
            sampling_rate (int): Sampling rate of the audio (default: 16000)
            return_timestamps (bool): Whether to return timestamps (default: True)
            language (str, optional): Language code (e.g., 'en', 'fr'). Auto-detects if None.
            task (str): Task type - 'transcribe' or 'translate' (default: 'transcribe')
            
        Returns:
            Dict: Dictionary containing:
                - 'text': Full transcription text
                - 'chunks': List of timestamped segments (if return_timestamps=True)
                - 'language': Detected or specified language
        """
        try:
            # Process audio input
            if isinstance(audio_input, str):
                # If path is provided, load using processor
                import librosa
                audio_array, _ = librosa.load(audio_input, sr=sampling_rate)
            else:
                audio_array = audio_input
                
            # Prepare inputs
            input_features = self.processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).input_features
            
            # Move to device
            input_features = input_features.to(self.device)
            
            # Generate token ids with timestamps
            with torch.no_grad():
                if return_timestamps:
                    # Generate with timestamp tokens
                    predicted_ids = self.model.generate(
                        input_features,
                        return_timestamps=True,
                        language=language,
                        task=task
                    )
                else:
                    predicted_ids = self.model.generate(
                        input_features,
                        language=language,
                        task=task
                    )
            
            # Decode the transcription
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=False if return_timestamps else True
            )
            
            result = {
                'text': transcription[0] if transcription else "",
                'language': language if language else "auto-detected"
            }
            
            if return_timestamps:
                # Extract timestamps from the output
                chunks = self._extract_timestamps(predicted_ids[0], transcription[0])
                result['chunks'] = chunks
                
            return result
            
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def _extract_timestamps(
        self,
        token_ids: torch.Tensor,
        transcription: str
    ) -> List[Dict]:
        """
        Extract timestamp information from token IDs.
        
        Args:
            token_ids (torch.Tensor): Generated token IDs
            transcription (str): Decoded transcription text
            
        Returns:
            List[Dict]: List of chunks with timestamps
        """
        # Decode with timestamps
        output = self.processor.decode(
            token_ids,
            skip_special_tokens=False,
            output_offsets=True
        )
        
        chunks = []
        
        # Parse the output to extract timestamp information
        # Whisper uses special timestamp tokens like <|0.00|>, <|0.50|>, etc.
        import re
        
        # Find all timestamp patterns
        timestamp_pattern = r'<\|(\d+\.\d+)\|>'
        timestamps = re.findall(timestamp_pattern, output)
        
        # Split text by timestamp tokens
        text_segments = re.split(timestamp_pattern, output)
        
        # Clean up segments and pair with timestamps
        current_time = 0.0
        for i in range(1, len(text_segments), 2):
            if i < len(text_segments):
                timestamp = float(text_segments[i])
                text = text_segments[i + 1] if i + 1 < len(text_segments) else ""
                
                # Clean special tokens from text
                text = re.sub(r'<\|.*?\|>', '', text).strip()
                
                if text:
                    chunks.append({
                        'timestamp': [current_time, timestamp],
                        'text': text
                    })
                    current_time = timestamp
        
        return chunks
    
    def transcribe_batch(
        self,
        audio_inputs: List[Union[np.ndarray, str]],
        sampling_rate: int = 16000,
        **kwargs
    ) -> List[Dict]:
        """
        Transcribe multiple audio files in batch.
        
        Args:
            audio_inputs (List): List of audio arrays or file paths
            sampling_rate (int): Sampling rate of the audio
            **kwargs: Additional arguments passed to transcribe_with_timestamps
            
        Returns:
            List[Dict]: List of transcription results
        """
        results = []
        for audio_input in audio_inputs:
            result = self.transcribe_with_timestamps(
                audio_input,
                sampling_rate=sampling_rate,
                **kwargs
            )
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Model information including name, device, and parameters
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        return {
            'model_name': self.model_name,
            'device': self.device,
            'total_parameters': num_params,
            'trainable_parameters': num_trainable_params,
            'model_size_mb': num_params * 4 / (1024 ** 2)  # Assuming float32
        }

