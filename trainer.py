from transformers import TrainingArguments
from trl import SFTTrainer
from config import TRAINING_ARGS, MAX_SEQ_LENGTH

def train_model(model, tokenizer, dataset):
    args = TrainingArguments(**TRAINING_ARGS)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        args = args,
    )

    trainer_stats = trainer.train()
    return trainer_stats
