import argparse
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


def load_model(model_path):

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path
    )

    model.eval()
    model.to("cpu")

    return tokenizer, model


def clean_text(
    text,
    tokenizer,
    model
):

    input_text = "punctuate: " + text


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


    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        required=True
    )

    args = parser.parse_args()

    version = args.version

    model_path = (
        f"models/goechoo-{version}"
    )


    print(
        f"Loading Go Echoo {version.upper()}..."
    )

    tokenizer, model = load_model(
        model_path
    )


    print(
        f"\nGo Echoo {version.upper()} is ready!"
    )

    print("Type 'exit' to quit.\n")


    while True:

        text = input("You: ").strip()


        if text.lower() == "exit":
            break


        if not text:
            continue


        result = clean_text(
            text,
            tokenizer,
            model
        )


        print(
            "Go Echoo:",
            result
        )

        print()


if __name__ == "__main__":
    main()