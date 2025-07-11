# push_to_hub.py

from unsloth import FastLanguageModel
from config import MAX_SEQ_LENGTH, DTYPE
from huggingface_hub import login
import argparse

def push_merged_model(repo_id: str):
    # Login to Hugging Face Hub
    login()

    # Load the LoRA fine-tuned model
    print("Loading LoRA model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_adapter",           # Load from  saved LoRA adapter
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = False,
    )

    # Merge LoRA weights into base model
    print("Merging LoRA into base model...")
    model = model.merge_and_unload()

    # Save merged model locally
    save_path = "llama3_2_3b_medical_assistant"
    print(f"Saving merged model to: {save_path}")
    model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)

    # Push to Hugging Face Hub
    print(f"Pushing model to Hugging Face Hub: {repo_id}")
    model.push_to_hub(repo_id, safe_serialization=True)
    tokenizer.push_to_hub(repo_id)

    print("Upload complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push merged model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Target Hub repo (e.g., Tapan101/llama3_2_3b_medical_assistant)")
    args = parser.parse_args()

    push_merged_model(args.repo_id)