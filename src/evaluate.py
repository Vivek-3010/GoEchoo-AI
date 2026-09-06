import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_PATH = "models/goechoo-v2"
TEST_FILE = "data/processed/v2/test.jsonl"


def load_test_data():
    examples = []

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                examples.append(json.loads(line))

    return examples


def main():

    print("Loading Go Echoo V2...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

    model.eval()
    model.to("cpu")

    examples = load_test_data()

    correct = 0

    print(f"\nTesting {len(examples)} examples...\n")

    for example in examples:

        input_text = "punctuate: " + example["input"]
        expected = example["target"]

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
            )

        prediction = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        if prediction.strip() == expected.strip():
            correct += 1

    accuracy = (correct / len(examples)) * 100

    print("--------------------------------")
    print(f"Correct: {correct}/{len(examples)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("--------------------------------")


if __name__ == "__main__":
    main()