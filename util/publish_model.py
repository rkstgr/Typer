#!/usr/bin/env python3
"""
Publish a trained model to HuggingFace Hub.
- Uploads 16-bit merged model to main repository
- Uploads GGUF quantizations (Q8_0 and Q4_K_M) to separate -gguf repository
"""

import os
import argparse
import torch
from unsloth import FastLanguageModel
from huggingface_hub import HfApi

def create_model_card(repo_name, base_model, is_gguf=False):
    """Create appropriate model card for the repository"""

    if is_gguf:
        return f"""---
library_name: transformers
base_model: {repo_name.replace('-gguf', '')}
tags:
- typst
- code
- gguf
model_type: qwen2
license: apache-2.0
---

# {repo_name}

This repository contains GGUF quantized versions of [{repo_name.replace('-gguf', '')}](https://huggingface.co/{repo_name.replace('-gguf', '')}).

## Usage with Ollama

```bash
ollama run hf.co/{repo_name}
```

## Usage with llama.cpp

```bash
# Download the model
wget https://huggingface.co/{repo_name}/resolve/main/{{model_file}}.gguf

# Run with llama.cpp
./llama-cli -m {{model_file}}.gguf -p "Your prompt here"
```

## Model Details

- **Base Model:** [{repo_name.replace('-gguf', '')}](https://huggingface.co/{repo_name.replace('-gguf', '')})
- **Training:** Fine-tuned using Unsloth
"""
    else:
        return f"""---
library_name: transformers
base_model: {base_model}
tags:
- typst
- code
model_type: qwen2
license: apache-2.0
---

# {repo_name}

A Typst-focused language model fine-tuned from {base_model}.

## Model Description

This model has been fine-tuned to provide accurate and detailed answers about Typst, a modern markup language for document preparation.

## Usage

### With Ollama

For easier usage, you can use the GGUF version with Ollama:

```bash
ollama run hf.co/{repo_name}-gguf
```

## Training Details

- **Base Model:** {base_model}
- **Training Framework:** Unsloth
- **Optimization:** LoRA fine-tuning
"""

def main():
    parser = argparse.ArgumentParser(description='Publish model to HuggingFace Hub')
    parser.add_argument('--model', required=True, help='Path to the trained model directory')
    parser.add_argument('--repo', required=True, help='HuggingFace repository name (e.g., username/model-name)')
    parser.add_argument('--base-model', default='unsloth/Qwen2.5-Coder-1.5B', help='Base model name')
    parser.add_argument('--token', help='HuggingFace token (defaults to HF_TOKEN env var)')
    parser.add_argument('--max-seq-length', type=int, default=4096, help='Maximum sequence length')
    parser.add_argument('--outputs-dir', help='Directory to save intermediate files (optional)')

    args = parser.parse_args()

    # Get HF token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace token required. Set HF_TOKEN env var or use --token")

    # Repository names
    main_repo = args.repo
    gguf_repo = f"{args.repo}-gguf"

    print(f"Loading model from: {args.model}")
    print(f"Main repository: {main_repo}")
    print(f"GGUF repository: {gguf_repo}")

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=torch.float16,
        load_in_4bit=False,
    )

    api = HfApi(token=token)

    # Step 1: Upload 16-bit merged model to main repository
    print(f"\n1. Uploading 16-bit merged model to {main_repo}...")

    # Create main repo if it doesn't exist
    try:
        api.create_repo(repo_id=main_repo, exist_ok=True, private=False)
    except Exception as e:
        print(f"Note: {e}")

    # Upload model
    model.push_to_hub_merged(
        repo_id=main_repo,
        tokenizer=tokenizer,
        save_method="merged_16bit",
        token=token,
        commit_message="Upload merged 16-bit model"
    )

    # Upload README for main repo
    readme_content = create_model_card(main_repo, args.base_model, is_gguf=False)
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=main_repo,
        token=token,
        commit_message="Add model card"
    )

    print(f"✅ 16-bit model uploaded to: https://huggingface.co/{main_repo}")

    # Step 2: Create GGUF repository and upload quantizations
    print(f"\n2. Creating GGUF quantizations and uploading to {gguf_repo}...")

    # Create GGUF repo if it doesn't exist
    try:
        api.create_repo(repo_id=gguf_repo, exist_ok=True, private=False)
    except Exception as e:
        print(f"Note: {e}")

    # Upload Q8_0 quantization
    print("   - Creating Q8_0 quantization...")
    model.push_to_hub_gguf(
        repo_id=gguf_repo,
        tokenizer=tokenizer,
        quantization_method="q8_0",
        token=token,
        commit_message="Add Q8_0 quantization"
    )

    # Upload Q4_K_M quantization
    print("   - Creating Q4_K_M quantization...")
    model.push_to_hub_gguf(
        repo_id=gguf_repo,
        tokenizer=tokenizer,
        quantization_method="q4_k_m",
        token=token,
        commit_message="Add Q4_K_M quantization"
    )

    # Upload README for GGUF repo
    readme_content = create_model_card(gguf_repo, args.base_model, is_gguf=True)
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=gguf_repo,
        token=token,
        commit_message="Add model card"
    )

    print(f"✅ GGUF models uploaded to: https://huggingface.co/{gguf_repo}")

    # Summary
    print("\n" + "="*60)
    print("🎉 Model publishing completed successfully!")
    print("="*60)
    print(f"16-bit model: https://huggingface.co/{main_repo}")
    print(f"GGUF models:  https://huggingface.co/{gguf_repo}")
    print("\nUsage with Ollama:")
    print(f"  ollama run hf.co/{gguf_repo}")

if __name__ == "__main__":
    main()
