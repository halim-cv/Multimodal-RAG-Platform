import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

audio_file = r"C:\Users\Sam-tech\Desktop\Multimodal-RAG-Platform\Text-encoding\sessions\45802bd9-0712-4513-8d95-81db8544fa5a\_tmp\your_audio_file" # We don't have the audio file path, but we can generate dummy audio
import numpy as np

audio_array = np.random.randn(16000 * 5) # 5 seconds of noise

input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
predicted_ids = model.generate(input_features, return_timestamps=True)

output = processor.decode(predicted_ids[0], skip_special_tokens=False, output_offsets=True)
print("Type of output:", type(output))
print("Output:", output)
