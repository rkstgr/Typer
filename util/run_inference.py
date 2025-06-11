from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

dtype = torch.float16

def main(model, input, max_new_tokens):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=2048,
        dtype=dtype,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference


    inputs = tokenizer(
        [
            input,
        ],
        return_tensors="pt",
    ).to("cuda")


    outputs = model.generate(**inputs, max_new_tokens = max_new_tokens, use_cache = True)
    result = tokenizer.batch_decode(outputs)
    print(result)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/<model_name>")
    parser.add_argument("--input", type=str, default="Hello, world!")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()
    main(args.model, args.input, args.max_new_tokens)
