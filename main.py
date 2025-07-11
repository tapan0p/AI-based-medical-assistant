from model_loader import load_model_and_tokenizer
from data_loader import load_and_prepare_dataset
from trainer import train_model
from generate import run_inference

def main():
    model, tokenizer = load_model_and_tokenizer()
    dataset = load_and_prepare_dataset(tokenizer)
    train_model(model, tokenizer, dataset)

    # Save model and tokenizer locally
    model.save_pretrained("lora_adapter")
    tokenizer.save_pretrained("tokenizer")


    # run a quick test
    run_inference(model, tokenizer)

if __name__ == "__main__":
    main()
