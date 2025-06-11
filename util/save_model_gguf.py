from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
import os

dtype = torch.float16

def main(model, output):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model, # /outputs/<model_name>
        max_seq_length=2048,
        dtype=dtype,
        load_in_4bit=False,
    )

    model.save_pretrained_gguf(output, tokenizer,)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/<model_name>")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    main(args.model, args.output)
