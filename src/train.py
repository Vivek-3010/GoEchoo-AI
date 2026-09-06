import json
import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


def load_jsonl(path):

    examples = []

    with open(path, "r", encoding="utf-8") as file:

        for line in file:

            if line.strip():
                examples.append(json.loads(line))

    return examples


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        required=True
    )

    args = parser.parse_args()

    version = args.version

    # --------------------------------
    # Determine model lineage
    # --------------------------------

    if version == "v1":

        model_path = "google-t5/t5-small"

    else:

        previous_version = f"v{int(version[1:]) - 1}"

        model_path = f"models/goechoo-{previous_version}"

    output_dir = f"models/goechoo-{version}"

    train_file = Path(
        f"data/processed/{version}/train.jsonl"
    )

    validation_file = Path(
        f"data/processed/{version}/validation.jsonl"
    )

    print(f"Training Go Echoo {version.upper()}")

    print(f"Starting model : {model_path}")
    print(f"Output model   : {output_dir}")


    # --------------------------------
    # Dataset
    # --------------------------------

    train_examples = load_jsonl(train_file)
    validation_examples = load_jsonl(validation_file)

    print(
        f"\nTraining examples: {len(train_examples)}"
    )

    print(
        f"Validation examples: {len(validation_examples)}"
    )


    dataset = DatasetDict({

        "train": Dataset.from_list(
            train_examples
        ),

        "validation": Dataset.from_list(
            validation_examples
        )

    })


    # --------------------------------
    # Tokenizer
    # --------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )


    # --------------------------------
    # Model
    # --------------------------------

    print("Loading model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path
    )


    # --------------------------------
    # Tokenization
    # --------------------------------

    def tokenize_function(examples):

        inputs = [
            "punctuate: " + text
            for text in examples["input"]
        ]

        targets = examples["target"]

        model_inputs = tokenizer(
            inputs,
            max_length=128,
            truncation=True
        )

        labels = tokenizer(
            text_target=targets,
            max_length=128,
            truncation=True
        )

        model_inputs["labels"] = labels[
            "input_ids"
        ]

        return model_inputs


    print("\nTokenizing dataset...")

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[
            "input",
            "target"
        ]
    )


    # --------------------------------
    # Data collator
    # --------------------------------

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )


    # --------------------------------
    # Training
    # --------------------------------

    training_args = Seq2SeqTrainingArguments(

        output_dir=output_dir,

        num_train_epochs=3,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,

        # Slightly smaller LR for continued
        # fine-tuning from V2

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

        eval_dataset=tokenized_dataset[
            "validation"
        ],

        processing_class=tokenizer,

        data_collator=data_collator,
    )


    print("\nStarting training...")
    print("--------------------------------")

    trainer.train()


    # --------------------------------
    # Save
    # --------------------------------

    print("\nSaving model...")

    trainer.save_model(output_dir)

    tokenizer.save_pretrained(
        output_dir
    )

    print("\nTraining complete!")

    print(
        f"Model saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()