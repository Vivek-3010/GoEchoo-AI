import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "models/goechoo-v1"

TRAIN_FILE = Path("data/processed/v2/train.jsonl")
VALIDATION_FILE = Path("data/processed/v2/validation.jsonl")

OUTPUT_DIR = "models/goechoo-v2"

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128


# -----------------------------
# Load JSONL
# -----------------------------

def load_jsonl(path):

    examples = []

    with open(path, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if line:
                examples.append(json.loads(line))

    return examples


# -----------------------------
# Main
# -----------------------------

def main():

    print("Loading V2 dataset...")

    train_examples = load_jsonl(TRAIN_FILE)
    validation_examples = load_jsonl(VALIDATION_FILE)

    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(validation_examples)}")


    dataset = DatasetDict({
        "train": Dataset.from_list(train_examples),
        "validation": Dataset.from_list(validation_examples),
    })


    # -------------------------
    # Load V1 tokenizer
    # -------------------------

    print("\nLoading Go Echoo V1 tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


    # -------------------------
    # Load V1 model
    # -------------------------

    print("Loading Go Echoo V1 model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)


    # -------------------------
    # Tokenization
    # -------------------------

    def tokenize_function(examples):

        inputs = [
            "punctuate: " + text
            for text in examples["input"]
        ]

        targets = examples["target"]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        )

        labels = tokenizer(
            text_target=targets,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


    print("\nTokenizing V2 dataset...")

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["input", "target"],
    )


    # -------------------------
    # Data collator
    # -------------------------

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )


    # -------------------------
    # Training
    # -------------------------

    training_args = Seq2SeqTrainingArguments(

        output_dir=OUTPUT_DIR,

        num_train_epochs=3,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,

        learning_rate=1e-4,

        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=2,

        logging_strategy="steps",
        logging_steps=100,

        predict_with_generate=True,

        report_to="none",

        use_cpu=True,

        load_best_model_at_end=True,
    )


    trainer = Seq2SeqTrainer(

        model=model,

        args=training_args,

        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],

        processing_class=tokenizer,

        data_collator=data_collator,
    )


    print("\nStarting Go Echoo V2 training...")
    print("--------------------------------")

    trainer.train()


    # -------------------------
    # Save V2
    # -------------------------

    print("\nSaving Go Echoo V2...")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nV2 training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()