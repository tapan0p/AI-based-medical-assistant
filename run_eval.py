# evaluate.py

from transformers import pipeline
from unsloth import FastLanguageModel
from datasets import load_dataset
from bert_score import score as bertscore
import evaluate
import torch
from tqdm import tqdm

# --------------------------
# Load model and tokenizer
# --------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "llama3_2_3b_medical_assistant",
    max_seq_length = 1024,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# --------------------------
# Load dataset (first 64 samples)
# --------------------------
dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train[:64]")

# --------------------------
# Prompt template
# --------------------------
prompt_template = """System:
You are an AI medical assistant. Your task is to answer medical, biology, and health-related questions. Based on the user query, answer it in a small concise paragraph.

User:
{}

AI:
"""

# --------------------------
# Generate model responses
# --------------------------
generated_answers = []
reference_answers = []

print("Generating responses...")
for example in tqdm(dataset):
    question = example["Question"]
    reference = example["Answer"]

    prompt = prompt_template.format(question)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1024,
            do_sample=False
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the model-generated answer
    if "AI:" in decoded:
        generated = decoded.split("AI:")[-1].strip()
    else:
        generated = decoded.strip()

    generated_answers.append(generated)
    reference_answers.append(reference)

# --------------------------
# Evaluate with BERTScore
# --------------------------
print("\nCalculating BERTScore...")
P, R, F1 = bertscore(
    cands=generated_answers,
    refs=reference_answers,
    lang="en"
)
bertscore_f1 = F1.mean().item()

# --------------------------
# Evaluate with ROUGE-L using Hugging Face `evaluate`
# --------------------------
print("Calculating ROUGE-L...")
rouge = evaluate.load("rouge")
rouge_result = rouge.compute(predictions=generated_answers, references=reference_answers)
rouge_l = rouge_result["rougeL"]

# --------------------------
# Final Report
# --------------------------
print("\n Evaluation Results (on 64 samples):")
print(f"BERTScore F1: {bertscore_f1:.4f}")
print(f"ROUGE-L F1 : {rouge_l:.4f}")
