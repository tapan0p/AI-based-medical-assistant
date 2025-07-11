from unsloth import FastLanguageModel
from config import MAX_SEQ_LENGTH, DTYPE, LOAD_IN_4BIT

def load_model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-3B",
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )


    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
        use_rslora = False,
        loftq_config = None,
    )
    return model, tokenizer
