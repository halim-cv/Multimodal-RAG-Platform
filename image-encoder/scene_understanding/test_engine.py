"""Test the Scene Understanding Engine with examples"""

from PIL import Image
from scene_understanding_engine import create_engine

# Load the test image
image_path = r"c:\Users\Sam-tech\Desktop\Github\image-encoder\examples\test.jpg"
image = Image.open(image_path)

print(f"Loading image from: {image_path}")
print(f"Image size: {image.size}")
print("=" * 80)

# Create and load the engine
engine = create_engine()

print("\n" + "=" * 80)
print("SCENE UNDERSTANDING ENGINE - COMPREHENSIVE TEST")
print("=" * 80)

# Test 1: All caption levels
print("\n### TEST 1: CAPTIONING ###")
print("-" * 80)
captions = engine.all_captions(image)
print(f"\nBasic Caption:\n  {captions['basic']}")
print(f"\nDetailed Caption:\n  {captions['detailed']}")
print(f"\nMore Detailed Caption:\n  {captions['more_detailed']}")

# Test 2: Object Detection
print("\n\n### TEST 2: OBJECT DETECTION ###")
print("-" * 80)
od_result = engine.object_detection(image)
print(f"\nDetected {len(od_result['labels'])} objects:")
for i, (bbox, label) in enumerate(zip(od_result['bboxes'], od_result['labels']), 1):
    print(f"  {i}. {label} at {bbox}")

# Test 3: Dense Region Captioning
print("\n\n### TEST 3: DENSE REGION CAPTIONING ###")
print("-" * 80)
dense_result = engine.dense_region_caption(image)
print(f"\nFound {len(dense_result['labels'])} regions:")
for i, (bbox, label) in enumerate(zip(dense_result['bboxes'], dense_result['labels']), 1):
    print(f"  {i}. '{label}' at {bbox}")

# Test 4: OCR
print("\n\n### TEST 4: OCR ###")
print("-" * 80)
ocr_text = engine.ocr(image)
print(f"\nExtracted Text:\n  {ocr_text}")

print("\n\n### TEST 5: OCR WITH REGIONS ###")
print("-" * 80)
ocr_regions = engine.ocr_with_region(image)
print(f"\nFound {len(ocr_regions['labels'])} text regions:")
for i, (box, text) in enumerate(zip(ocr_regions['quad_boxes'], ocr_regions['labels']), 1):
    print(f"  {i}. '{text}' at {box[:4]}...")  # Show first 4 coords

# Test 6: Phrase Grounding
print("\n\n### TEST 6: PHRASE GROUNDING ###")
print("-" * 80)
test_phrases = [
    "MRI scan",
    "cervical spine",
    "text"
]
for phrase in test_phrases:
    grounding = engine.phrase_grounding(image, phrase)
    print(f"\nPhrase: '{phrase}'")
    print(f"  Found {len(grounding['bboxes'])} instances")
    if grounding['bboxes']:
        for bbox in grounding['bboxes'][:3]:  # Show first 3
            print(f"    - {bbox}")

# Test 7: Caption and Ground
print("\n\n### TEST 7: CASCADED - CAPTION & GROUND ###")
print("-" * 80)
cascaded = engine.caption_and_ground(image, caption_level='detailed')
print(f"\nCaption: {cascaded['caption']}")
print(f"\nGrounded phrases: {len(cascaded['grounding']['labels'])}")
for i, (bbox, label) in enumerate(zip(
    cascaded['grounding']['bboxes'], 
    cascaded['grounding']['labels']
)[:5], 1):  # Show first 5
    print(f"  {i}. '{label}' at {bbox}")

# Test 8: Batch Analysis
print("\n\n### TEST 8: BATCH ANALYSIS ###")
print("-" * 80)
analysis = engine.analyze_image(image, tasks=[
    'caption', 
    'detailed_caption', 
    'object_detection', 
    'ocr'
])
print("\nBatch analysis results:")
for task, result in analysis.items():
    if isinstance(result, str):
        print(f"\n{task}: {result[:100]}...")
    elif isinstance(result, dict):
        print(f"\n{task}: {len(result.get('labels', result.get('bboxes', [])))} items")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 80)
