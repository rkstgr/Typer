from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
import os

def main():

    max_seq_length = 2048
    dtype = torch.float16
    load_in_4bit = False

    # Use local model path if it exists, otherwise use HF model name
    local_model_path = "local-models/qwen2.5-coder-1.5B"
    model_name = local_model_path if os.path.exists(local_model_path) else "unsloth/Qwen2.5-Coder-1.5B"

    print(f"Loading model from: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head",], # these two for continual pretraining
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True, # or unsloth
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


    data_files = {'train': 'data/m2/train.jsonl', 'test': 'data/m2/eval.jsonl'}
    dataset = load_dataset('json', data_files=data_files)
    dataset = dataset.map(formatting_prompts_func, batched=True) # Adjust num_proc as needed

    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,

        args = UnslothTrainingArguments(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 4,

            # max_steps = 120,
            # warmup_steps = 10,
            warmup_ratio = 0.1,
            num_train_epochs = 4,
            eval_strategy = "epoch",

            # Select a 2 to 10x smaller learning rate for the embedding matrices!
            learning_rate = 1e-4,
            embedding_learning_rate = 1e-5,

            fp16 = True,
            bf16 = False,
            logging_steps = 1,
            weight_decay = 0.0,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs/slab-typer-1.5B-m2",
            report_to = "none", # Use this for WandB etc
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

    model.save_pretrained("outputs/slab-typer-1.5B-m2")
    tokenizer.save_pretrained("outputs/slab-typer-1.5B-m2")


if __name__ == "__main__":
    main()
