from transformers import AutoProcessor, AutoModelForCausalLM

def load_model(model_id='microsoft/Florence-2-base', device='cuda'):
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto', attn_implementation='eager').eval().to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor
