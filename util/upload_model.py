"""
TODO: create custom upload script
Not super happy with this script as it personalizes the commit messsage and gguf file with unsloth
"""

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
        max_seq_length=4096,
        dtype=dtype,
        load_in_4bit=False,
    )

    model.push_to_hub_gguf(output, tokenizer, token=os.environ["HF_TOKEN"])



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/<model_name>")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    main(args.model, args.output)
