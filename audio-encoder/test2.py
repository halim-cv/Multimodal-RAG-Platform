import sys
try:
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    ids = [50258, 50259, 50359, 50363]
    out = processor.decode(ids, skip_special_tokens=False, output_offsets=True)
    print("TYPE:", type(out))
    print("REPR:", repr(out))
except Exception as e:
    import traceback
    traceback.print_exc()
