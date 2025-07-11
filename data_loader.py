from datasets import load_dataset
from prompts import formatting_prompts_func

def load_and_prepare_dataset(tokenizer):
    dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    dataset = dataset.map(lambda x: formatting_prompts_func(x, tokenizer), batched=True)
    return dataset
