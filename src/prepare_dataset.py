import json
import re
from pathlib import Path
from random import Random


INPUT_FILE = Path("data/raw/v1_dataset.jsonl")
OUTPUT_DIR = Path("data/processed/v1")

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"


def normalize_for_comparison(text):
    """
    Remove punctuation and normalize whitespace.

    This helps us check whether the target contains
    the same words as the input.
    """

    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_dataset():
    examples = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_number}")
                continue

            if "input" not in item or "target" not in item:
                print(f"Skipping missing fields on line {line_number}")
                continue

            input_text = item["input"].strip()
            target_text = item["target"].strip()

            if not input_text or not target_text:
                continue

            examples.append({
                "input": input_text,
                "target": target_text
            })

    return examples


def remove_duplicates(examples):
    seen = set()
    unique_examples = []

    for example in examples:

        key = (
            example["input"].strip().lower(),
            example["target"].strip().lower()
        )

        if key not in seen:
            seen.add(key)
            unique_examples.append(example)

    return unique_examples


def check_word_preservation(examples):
    valid = []
    invalid_count = 0

    for example in examples:

        input_words = normalize_for_comparison(example["input"])
        target_words = normalize_for_comparison(example["target"])

        if input_words == target_words:
            valid.append(example)
        else:
            invalid_count += 1

    print(f"Invalid word-preservation examples: {invalid_count}")

    return valid


def save_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as file:

        for example in examples:
            file.write(
                json.dumps(example, ensure_ascii=False)
                + "\n"
            )


def main():

    print("Loading dataset...")

    examples = load_dataset()

    print(f"Loaded examples: {len(examples)}")

    examples = remove_duplicates(examples)

    print(f"After removing duplicates: {len(examples)}")

    examples = check_word_preservation(examples)

    print(f"After validation: {len(examples)}")

    # Shuffle reproducibly
    random = Random(42)
    random.shuffle(examples)

    total = len(examples)

    train_end = int(total * 0.8)
    validation_end = int(total * 0.9)

    train_examples = examples[:train_end]
    validation_examples = examples[train_end:validation_end]
    test_examples = examples[validation_end:]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_jsonl(TRAIN_FILE, train_examples)
    save_jsonl(VALIDATION_FILE, validation_examples)
    save_jsonl(TEST_FILE, test_examples)

    print("\nDataset split:")
    print(f"Train:      {len(train_examples)}")
    print(f"Validation: {len(validation_examples)}")
    print(f"Test:       {len(test_examples)}")

    print("\nFiles created:")
    print(TRAIN_FILE)
    print(VALIDATION_FILE)
    print(TEST_FILE)


if __name__ == "__main__":
    main()