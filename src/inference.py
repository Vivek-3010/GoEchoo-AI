import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_PATH = "models/goechoo-v1"


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

    model.eval()
    model.to("cpu")

    return tokenizer, model


def punctuate(text, tokenizer, model):

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

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    return result


def main():

    print("Loading Go Echoo V1...")
    tokenizer, model = load_model()

    print("\nGo Echoo V1 is ready!")
    print("Type 'exit' to quit.\n")

    while True:

        text = input("You: ").strip()

        if text.lower() == "exit":
            break

        if not text:
            continue

        result = punctuate(
            text,
            tokenizer,
            model,
        )

        print("Go Echoo:", result)
        print()


if __name__ == "__main__":
    main()