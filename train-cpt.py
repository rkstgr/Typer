from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth import UnslothTrainer, UnslothTrainingArguments
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Continual pre-training on text data')
    parser.add_argument('--model', default='unsloth/Qwen2.5-Coder-1.5B', help='Model name or path')
    parser.add_argument('--local-model', default='local-models/qwen2.5-coder-1.5B', help='Local model path to check first')
    parser.add_argument('--data-train', default='data/m2/train.jsonl', help='Training data path')
    parser.add_argument('--data-eval', default='data/m2/eval.jsonl', help='Evaluation data path')
    parser.add_argument('--output', default='outputs/slab-typer-1.5B-m2', help='Output directory')
    parser.add_argument('--max-seq-length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=8, help='Per device batch size')
    parser.add_argument('--grad-accum', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embedding-lr', type=float, default=1e-5, help='Embedding learning rate')
    parser.add_argument('--lora-r', type=int, default=128, help='LoRA r parameter')
    
    args = parser.parse_args()

    max_seq_length = args.max_seq_length
    dtype = torch.float16
    load_in_4bit = False

    # Use local model path if it exists, otherwise use specified model
    model_name = args.local_model if os.path.exists(args.local_model) else args.model

    print(f"Loading model from: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head",], # these two for continual pretraining
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = 3407,
        use_rslora = True,
        loftq_config = None,
    )

    _prompt = """{}"""

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        texts  = examples["text"]
        outputs = []
        for text in texts:
            text = _prompt.format(text) + EOS_TOKEN
            outputs.append(text)
        return { "text" : outputs, }


    data_files = {'train': args.data_train, 'test': args.data_eval}
    dataset = load_dataset('json', data_files=data_files)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,

        args = UnslothTrainingArguments(
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.grad_accum,
            warmup_ratio = 0.1,
            num_train_epochs = args.epochs,
            eval_strategy = "epoch",
            learning_rate = args.lr,
            embedding_learning_rate = args.embedding_lr,
            fp16 = True,
            bf16 = False,
            logging_steps = 1,
            weight_decay = 0.0,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = args.output,
            report_to = "none",
        ),
    )

    print("=" * 30)
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    print("=" * 30)

    trainer_stats = trainer.train()

    print(f"\nSaving model to: {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    
    print("Training completed successfully!")
    print(f"Final training loss: {trainer_stats.training_loss:.4f}")


if __name__ == "__main__":
    main()
