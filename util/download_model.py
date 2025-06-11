#!/usr/bin/env python3
"""
Download a HuggingFace model to a local directory for offline use.
Usage: python download_model.py --model "unsloth/Qwen2.5-Coder-1.5B" --save local_models/qwen2.5-coder-1.5B
"""

import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace model for offline use")
    parser.add_argument("--model", required=True, help="HuggingFace model name (e.g., unsloth/Qwen2.5-Coder-1.5B)")
    parser.add_argument("--save", required=True, help="Local directory to save the model")

    args = parser.parse_args()

    model_name = args.model
    save_path = args.save

    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    print(f"Downloading model '{model_name}' to '{save_path}'...")

    try:
        # Download the complete model repository
        snapshot_download(
            repo_id=model_name,
            local_dir=save_path,
            local_dir_use_symlinks=False,  # Use actual files instead of symlinks
        )

        print(f"✅ Model '{model_name}' successfully downloaded to '{save_path}'")
        print(f"You can now use this model offline by pointing to: {os.path.abspath(save_path)}")

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
