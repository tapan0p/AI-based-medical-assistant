from unsloth import FastLanguageModel
from prompts import get_medical_prompt
from transformers import TextStreamer
import torch


def run_inference(model, tokenizer):
    model.eval()
    prompt = get_medical_prompt().format("what are the signs and symptoms of rabies?", "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=1024)

def main():
    # ✅ Load your fine-tuned model from the local directory
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./llama3_2_3b_medical_assistant",  # <--- local path
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = False,
    )

    FastLanguageModel.for_inference(model)
    run_inference(model, tokenizer)

if __name__ == "__main__":
    main()
