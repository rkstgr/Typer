#!/usr/bin/env python3
"""
Upload a trained model to HuggingFace Hub as GGUF format.
Usage: python upload_model.py --model outputs/my-model --repo username/my-model-gguf
"""

from unsloth import FastLanguageModel
import torch
import os
import argparse
from huggingface_hub import HfApi

dtype = torch.float16

def create_readme(model_name, base_model, gguf_filename, description=None):
    """Create a README.md with model card frontmatter"""
    readme_content = f"""---
library_name: transformers
base_model: {base_model}
tags:
- code
- unsloth
- gguf
- 4bit
model_type: qwen2
license: apache-2.0
quantized_by: {model_name}
---

# {model_name}

{description or f'This is a GGUF quantized version of {base_model}.'}

## Model Details

- **Base Model:** {base_model}
- **Quantization:** Q8_0 GGUF format
- **File:** `{gguf_filename}`

## Usage

This model can be used with llama.cpp or any other GGUF-compatible inference engine.

```bash
# Download and run with llama.cpp
wget https://huggingface.co/{model_name}/resolve/main/{gguf_filename}
./llama-cli -m {gguf_filename} -p "Your prompt here"
```

## Training

This model was fine-tuned using the Unsloth library for efficient training.
"""
    return readme_content

def main():
    parser = argparse.ArgumentParser(description='Upload model to HuggingFace Hub as GGUF')
    parser.add_argument('--model', required=True, help='Path to the trained model directory')
    parser.add_argument('--repo', required=True, help='HuggingFace repository name (e.g., username/model-name)')
    parser.add_argument('--gguf-name', help='GGUF filename (default: auto-generated from repo name)')
    parser.add_argument('--base-model', default='unsloth/Qwen2.5-Coder-1.5B', help='Base model name for README')
    parser.add_argument('--description', help='Model description for README')
    parser.add_argument('--commit-message', default='Upload GGUF model', help='Commit message for upload')
    parser.add_argument('--max-seq-length', type=int, default=4096, help='Maximum sequence length')
    parser.add_argument('--quantization', default='q8_0', help='Quantization method (default: q8_0)')
    parser.add_argument('--token', help='HuggingFace token (defaults to HF_TOKEN env var)')
    
    args = parser.parse_args()
    
    # Get HF token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace token required. Set HF_TOKEN env var or use --token")
    
    # Generate GGUF filename if not provided
    if args.gguf_name:
        gguf_filename = args.gguf_name
    else:
        repo_name = args.repo.split('/')[-1]  # Get just the model name part
        gguf_filename = f"{repo_name}.Q8_0.gguf"
    
    print(f"Loading model from: {args.model}")
    print(f"Target repository: {args.repo}")
    print(f"GGUF filename: {gguf_filename}")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
    )
    
    # Create README content
    readme_content = create_readme(
        model_name=args.repo,
        base_model=args.base_model,
        gguf_filename=gguf_filename,
        description=args.description
    )
    
    print("\nUploading model to HuggingFace Hub...")
    
    # Upload GGUF model with custom settings
    model.push_to_hub_merged(
        repo_id=args.repo,
        tokenizer=tokenizer,
        save_method="merged_16bit",
        token=token,
        commit_message=args.commit_message
    )
    
    # Upload GGUF version
    print("Converting and uploading GGUF version...")
    model.push_to_hub_gguf(
        repo_id=args.repo,
        tokenizer=tokenizer,
        quantization_method=args.quantization,
        token=token,
        commit_message=f"{args.commit_message} - GGUF {args.quantization.upper()}"
    )
    
    # Upload custom README
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        commit_message=f"{args.commit_message} - Update README"
    )
    
    print(f"\n✅ Model successfully uploaded to: https://huggingface.co/{args.repo}")
    print(f"GGUF file: {gguf_filename}")

if __name__ == "__main__":
    main()
