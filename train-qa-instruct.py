from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth import UnslothTrainer, UnslothTrainingArguments
import argparse

def main():
    parser = argparse.ArgumentParser(description='Fine-tune model on Q&A dataset')
    parser.add_argument('--model', required=True, help='Path to the base model')
    parser.add_argument('--data', default='data/qa/train.jsonl', help='Path to training data')
    parser.add_argument('--output', default='outputs/typer-1.5B-instruct', help='Output directory')
    parser.add_argument('--max-seq-length', type=int, default=4096, help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=4, help='Per device batch size')
    parser.add_argument('--grad-accum', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--lora-r', type=int, default=128, help='LoRA r parameter')

    args = parser.parse_args()

    max_seq_length = args.max_seq_length
    dtype = torch.float16
    load_in_4bit = False

    print(f"Loading model from: {args.model}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=True,
        loftq_config=None,
    )

    # Set the Jinja2 chat template on the tokenizer
    # This will be embedded in the GGUF file automatically
    tokenizer.chat_template = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}
    {% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

    # Python format string for training data formatting
    chat_template_format = """<|im_start|>system
You are a helpful assistant that provides accurate and detailed answers about Typst, a modern markup language for document preparation.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        conversations = examples["messages"]
        texts = []

        for conversation in conversations:
            # Extract user question and assistant answer
            user_content = ""
            assistant_content = ""

            for message in conversation:
                if message["role"] == "user":
                    user_content = message["content"]
                elif message["role"] == "assistant":
                    assistant_content = message["content"]

            # Format as chat template
            text = chat_template_format.format(user_content, assistant_content) + EOS_TOKEN
            texts.append(text)

        return {"text": texts}

    # Load Q&A dataset
    data_files = {'train': args.data}

    print(f"Loading dataset from: {data_files}")
    dataset = load_dataset('json', data_files=data_files)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Print a sample to verify formatting
    print("\n" + "="*50)
    print("Sample formatted conversation:")
    print(dataset["train"][0]["text"][:500] + "...")
    print("="*50 + "\n")

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.03,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=True,
            bf16=False,
            logging_steps=1,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output,
            report_to="none",
            save_strategy="epoch",
            save_total_limit=2,
        ),
    )

    print("=" * 50)
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    print("=" * 50)

    # Train the model
    trainer_stats = trainer.train()

    # Save the model with the chat template
    print(f"\nSaving model to: {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Verify the chat template was saved
    print("\nChat template saved to tokenizer:")
    try:
        print(tokenizer.chat_template[:200] + "...")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTraining completed successfully!")
    print(f"Final training loss: {trainer_stats.training_loss:.4f}")

if __name__ == "__main__":
    main()
