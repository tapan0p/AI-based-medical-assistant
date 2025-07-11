from prompts import get_medical_prompt
from transformers import TextStreamer
import torch

def run_inference(model, tokenizer):
    model.eval()
    prompt = get_medical_prompt().format("what are marine toxins?", "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=1024)
