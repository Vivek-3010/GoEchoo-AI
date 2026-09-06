import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_PATH = "models/goechoo-v2"


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

    model.eval()
    model.to("cpu")

    return tokenizer, model


def clean_text(text, tokenizer, model):

    input_text = "punctuate: " + text

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

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )


def main():

    print("Loading Go Echoo V2...")

    tokenizer, model = load_model()

    print("\nGo Echoo V2 is ready!")
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
            model,
        )

        print("Go Echoo:", result)
        print()


if __name__ == "__main__":
    main()