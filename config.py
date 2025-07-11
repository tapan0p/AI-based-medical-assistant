from unsloth import is_bfloat16_supported

MAX_SEQ_LENGTH = 1024
DTYPE = None
LOAD_IN_4BIT = True
TRAINING_ARGS = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "fp16": not is_bfloat16_supported(),
    "bf16": is_bfloat16_supported(),
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 42,
    "output_dir": "outputs",
    "report_to": "none"
}
