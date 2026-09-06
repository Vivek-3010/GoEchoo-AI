import json
import argparse
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


def load_test_data(test_file):

    examples = []

    with open(test_file, "r", encoding="utf-8") as file:

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

    model_path = f"models/goechoo-{version}"
    test_file = f"data/processed/{version}/test.jsonl"


    print(f"Loading Go Echoo {version.upper()}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path
    )

    model.eval()
    model.to("cpu")


    examples = load_test_data(test_file)

    print(f"\nTesting {len(examples)} examples...\n")


    correct = 0

    for example in examples:

        input_text = (
            "punctuate: "
            + example["input"]
        )

        expected = example["target"]


        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )


        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_length=128,
                num_beams=4
            )


        prediction = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )


        if prediction.strip() == expected.strip():

            correct += 1


    accuracy = (
        correct / len(examples)
    ) * 100


    print("--------------------------------")
    print(
        f"Correct: {correct}/{len(examples)}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print("--------------------------------")


if __name__ == "__main__":
    main()