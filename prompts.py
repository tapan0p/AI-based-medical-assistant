def get_medical_prompt():
    return """System:
You are an AI medical assistant. Your task is to answer medical, biology, and health-related questions. Based on the user query, answer it in a small concise paragraph.

User:
{}

AI:
{}"""

def formatting_prompts_func(examples, tokenizer):
    EOS_TOKEN = tokenizer.eos_token
    prompt_template = get_medical_prompt()

    inputs = examples["Question"]
    outputs = examples["Answer"]
    texts = []

    for user_input, answer in zip(inputs, outputs):
        prompt = prompt_template.format(user_input, answer)
        texts.append(prompt + EOS_TOKEN)

    return {"text": texts}
