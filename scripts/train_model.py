import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# Small model for fast CPU training
model_name = "distilgpt2"

# Load dataset
dataset = load_dataset(
    "json",
    data_files="data/processed/cleaned_dataset.json"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT models need a pad token
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenization function
def tokenize_function(example):

    text = example["instruction"] + " " + example["response"]

    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128
    )

    tokens["labels"] = tokens["input_ids"]

    return tokens


# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function)

# 🔹 Reduce dataset size (faster training)
tokenized_dataset["train"] = tokenized_dataset["train"].select(range(3000))

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training configuration (optimized for < 1 hour)
training_args = TrainingArguments(
    output_dir="outputs",
    num_train_epochs=1,
    max_steps=1500,
    per_device_train_batch_size=4,
    logging_steps=20,
    save_steps=300
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

# Start training
trainer.train()

# Save trained model
trainer.save_model("emotion_chatbot_model")

print("✅ Training complete. Model saved in emotion_chatbot_model/")