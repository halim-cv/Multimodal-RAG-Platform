"""Interactive VQA - Ask your own questions about the image"""

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

print("\nInteractive Visual Question Answering")
print("=" * 80)
print("Ask questions about the image. Type 'quit' or 'exit' to stop.")
print("-" * 80)

# Get initial caption for context
print("\nGenerating image caption for context...")
caption_result = run_inference(model, processor, image, '<DETAILED_CAPTION>')
caption = caption_result.get('<DETAILED_CAPTION>', '')
print(f"\nImage Description: {caption}")
print("-" * 80)

while True:
    question = input("\nYour question: ").strip()
    
    if question.lower() in ['quit', 'exit', 'q']:
        print("\nExiting VQA session. Goodbye!")
        break
    
    if not question:
        print("Please enter a question.")
        continue
    
    try:
        # Try VQA task
        result = run_inference(model, processor, image, '<VQA>', text_input=question)
        answer = result.get('<VQA>', 'No answer generated')
        print(f"\nAnswer: {answer}")
    except Exception as e:
        print(f"\nNote: VQA task may not be directly supported.")
        print(f"Error: {str(e)}")
        print("\nYou can use the detailed caption above to answer your question,")
        print("or try other specific tasks like <OD> for object detection,")
        print("<CAPTION_TO_PHRASE_GROUNDING> for locating specific objects.")
    
    print("-" * 80)
