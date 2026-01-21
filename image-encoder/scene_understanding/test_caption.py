"""Test captioning functionality on test.jpg"""

from PIL import Image
from model_loader import load_model
from inference import run_inference

# Load the test image
image_path = r"c:\Users\Sam-tech\Desktop\Github\image-encoder\examples\test.jpg"
image = Image.open(image_path)

print(f"Loading image from: {image_path}")
print(f"Image size: {image.size}")
print("-" * 80)

# Load model and processor
print("Loading Florence-2 model...")
model, processor = load_model()
print("Model loaded successfully!")
print("-" * 80)

# Test different caption types
caption_tasks = [
    ('<CAPTION>', 'Basic Caption'),
    ('<DETAILED_CAPTION>', 'Detailed Caption'),
    ('<MORE_DETAILED_CAPTION>', 'More Detailed Caption')
]

for task_prompt, task_name in caption_tasks:
    print(f"\n{task_name}:")
    result = run_inference(model, processor, image, task_prompt)
    caption = result.get(task_prompt, 'No caption generated')
    print(f"  {caption}")
    print("-" * 80)

print("\nCaptioning test completed!")
