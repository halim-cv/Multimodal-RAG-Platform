"""Test VQA (Visual Question Answering) functionality on test.jpg"""

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

# Define test questions
test_questions = [
    "What type of medical scan is this?",
    "What part of the body is shown?",
    "What is the orientation of the scan?",
    "Are there any abnormalities visible?",
    "What imaging technique was used?",
]

# Test VQA with predefined questions
print("\nVisual Question Answering Tests:")
print("=" * 80)

for i, question in enumerate(test_questions, 1):
    print(f"\nQuestion {i}: {question}")
    
    # Florence-2 uses a specific format for VQA
    # The task prompt format might vary, trying common patterns
    try:
        # Try the standard VQA format
        result = run_inference(model, processor, image, '<VQA>', text_input=question)
        answer = result.get('<VQA>', 'No answer generated')
    except Exception as e:
        # If VQA doesn't work, try using MORE_DETAILED_CAPTION as fallback
        print(f"  Note: Direct VQA not available, using caption-based approach")
        result = run_inference(model, processor, image, '<MORE_DETAILED_CAPTION>')
        answer = result.get('<MORE_DETAILED_CAPTION>', 'No answer generated')
        print(f"  [Context-based answer from caption]: {answer[:200]}...")
        continue
    
    print(f"  Answer: {answer}")
    print("-" * 80)

print("\n" + "=" * 80)
print("VQA test completed!")
print("\nNote: You can modify the 'test_questions' list to ask your own questions.")
