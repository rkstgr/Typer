from huggingface_hub import hf_hub_download
import gguf

# Download the GGUF file
model_path = hf_hub_download(repo_id="rkstgr/typer-1.5b-instruct-gguf", filename="typer-1.5b-instruct.Q8_0.gguf")

# Read the GGUF file
reader = gguf.GGUFReader(model_path)

# Check for chat template
for key, value in reader.fields.items():
    if "chat_template" in key.lower() or "template" in key.lower():
        print(f"Key: {key}")
        print(f"Value: {value}")
        print("-" * 50)
