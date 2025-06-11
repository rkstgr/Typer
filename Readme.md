# Typer: LLM for Typst

This repo contains the code for training (fine-tuning) a language model for Typst.

Currently, trained in two stages:
1. Continue pre-training (CPT) on official typst documentation and typst packages. Main goal of this stage is it to learn the syntax and structure of Typst. (`train-cpt.py`)

2. Instruction fine-tuning on Q&A dataset derived from Typst Forum Questions (`train-qa-instruct.py`)

Currently both stages use LoRa (stage 2 continues training the lora adapters)
