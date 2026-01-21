"""Test VQA (Visual Question Answering) functionality with the Scene Understanding Engine"""

from PIL import Image
from scene_understanding_engine import create_engine

# Load the test image
image_path = r"c:\Users\Sam-tech\Desktop\Github\image-encoder\examples\test.jpg"
image = Image.open(image_path)

print(f"Loading image from: {image_path}")
print(f"Image size: {image.size}")
print("=" * 80)

# Create and load the engine
print("\nLoading Scene Understanding Engine...")
engine = create_engine()

print("\n" + "=" * 80)
print("VQA - VISUAL QUESTION ANSWERING TESTS")
print("=" * 80)

# Test 1: Simple questions with text-only answers
print("\n### TEST 1: SIMPLE QUESTIONS (Text Answers) ###")
print("-" * 80)

questions = [
    "What type of image is this?",
    "What medical scan is shown?",
    "What body part is visible?",
]

for i, question in enumerate(questions, 1):
    print(f"\nQuestion {i}: {question}")
    answer = engine.visual_question_answering(image, question, return_grounding=False)
    print(f"Answer: {answer[:200]}...")
    print("-" * 40)

# Test 2: Prompt-grounded captions with bounding boxes
print("\n\n### TEST 2: PROMPT-GROUNDED CAPTIONS (With Grounding) ###")
print("-" * 80)

prompts = [
    "cervical spine",
    "MRI scan",
    "text and labels",
]

for i, prompt in enumerate(prompts, 1):
    print(f"\nPrompt {i}: '{prompt}'")
    result = engine.prompt_grounded_caption(image, prompt)
    print(f"Caption: {result['caption'][:150]}...")
    print(f"Grounded regions: {len(result['grounding']['bboxes'])} bounding boxes")
    
    if result['grounding']['bboxes']:
        print("  Locations:")
        for j, bbox in enumerate(result['grounding']['bboxes'][:3], 1):
            print(f"    {j}. {bbox}")
    print("-" * 40)

# Test 3: Full VQA with grounding
print("\n\n### TEST 3: FULL VQA WITH GROUNDING ###")
print("-" * 80)

vqa_questions = [
    "Where is the spine located?",
    "What text is visible in the image?",
    "Show me the cervical vertebrae",
]

for i, question in enumerate(vqa_questions, 1):
    print(f"\nQuestion {i}: {question}")
    result = engine.visual_question_answering(image, question, return_grounding=True)
    
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Grounded elements: {len(result['grounding']['bboxes'])}")
    
    if result['grounding']['labels']:
        print("  Detected elements:")
        for j, label in enumerate(result['grounding']['labels'][:5], 1):
            print(f"    {j}. {label}")
    print("-" * 40)

# Test 4: Comparison - Different caption levels
print("\n\n### TEST 4: CAPTION AS ANSWER (Comparison) ###")
print("-" * 80)

question = "Describe what you see in detail"
print(f"Question: {question}\n")

# Basic caption approach
basic = engine.caption(image)
print(f"Basic Caption:\n  {basic}\n")

# Detailed caption approach
detailed = engine.detailed_caption(image)
print(f"Detailed Caption:\n  {detailed}\n")

# VQA approach (uses more_detailed_caption internally)
vqa_answer = engine.visual_question_answering(image, question, return_grounding=False)
print(f"VQA Answer:\n  {vqa_answer}\n")

print("=" * 80)
print("VQA TESTS COMPLETED!")
print("=" * 80)
print("\nAvailable VQA Methods:")
print("  1. prompt_grounded_caption(image, prompt)")
print("     - Returns: {'prompt', 'caption', 'grounding'}")
print("  2. visual_question_answering(image, question, return_grounding=True/False)")
print("     - Returns: dict with 'question', 'answer', 'grounding' OR just answer string")
