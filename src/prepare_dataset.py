import json
import random
import argparse
from pathlib import Path


def load_data(input_file):

    examples = []

    with open(input_file, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Invalid JSON at line {line_number}")
                continue

            if "input" not in item or "target" not in item:
                print(f"Missing input/target at line {line_number}")
                continue

            examples.append({
                "input": item["input"].strip(),
                "target": item["target"].strip()
            })

    return examples


def normalize_text(text):

    punctuation = ".,!?;:"

    for char in punctuation:
        text = text.replace(char, "")

    return " ".join(text.lower().split())


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        required=True,
        help="Dataset version, for example v1, v2, v3"
    )

    args = parser.parse_args()

    version = args.version

    input_file = Path(f"data/raw/{version}_dataset.jsonl")
    output_dir = Path(f"data/processed/{version}")

    print(f"Preparing {version.upper()} dataset...")

    examples = load_data(input_file)

    print(f"Loaded examples: {len(examples)}")

    # Remove duplicates

    unique_examples = []
    seen = set()

    for example in examples:

        key = (
            example["input"],
            example["target"]
        )

        if key not in seen:

            seen.add(key)
            unique_examples.append(example)

    examples = unique_examples

    print(f"After removing duplicates: {len(examples)}")

    # Validate that words are preserved.
    # This allows spelling corrections by checking
    # only that the number of words is unchanged.

    valid_examples = []

    for example in examples:

        input_words = normalize_text(
            example["input"]
        ).split()

        target_words = normalize_text(
            example["target"]
        ).split()

        if len(input_words) != len(target_words):

            print("\nWARNING - word count mismatch:")
            print("INPUT :", example["input"])
            print("TARGET:", example["target"])

            continue

        valid_examples.append(example)

    examples = valid_examples

    print(f"After validation: {len(examples)}")

    # Shuffle

    random.seed(42)
    random.shuffle(examples)

    # 80 / 10 / 10 split

    total = len(examples)

    train_end = int(total * 0.8)
    validation_end = int(total * 0.9)

    train_data = examples[:train_end]
    validation_data = examples[train_end:validation_end]
    test_data = examples[validation_end:]

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    def save(data, filename):

        path = output_dir / filename

        with open(path, "w", encoding="utf-8") as file:

            for item in data:

                file.write(
                    json.dumps(
                        item,
                        ensure_ascii=False
                    ) + "\n"
                )

    save(train_data, "train.jsonl")
    save(validation_data, "validation.jsonl")
    save(test_data, "test.jsonl")

    print("\nDataset prepared successfully!")

    print(f"Train      : {len(train_data)}")
    print(f"Validation : {len(validation_data)}")
    print(f"Test       : {len(test_data)}")


if __name__ == "__main__":
    main()