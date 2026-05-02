from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "RogerTzeng/Sparse-MERIT"

PRETRAINED_DIR = Path("./pretrained_models")
MODEL_DIR = Path("./model")

WEIGHT_EXTENSIONS = (".pt", ".pth", ".pth.tar", ".safetensors", ".ckpt")

api = HfApi()
files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")

PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

for file_path in files:
    # Skip non-weight files
    if not file_path.endswith(WEIGHT_EXTENSIONS):
        continue

    # Root-level files: WavLM-Large.pt, pretrained_pool.pt, etc.
    if "/" not in file_path:
        target_dir = PRETRAINED_DIR
    else:
        target_dir = MODEL_DIR

    print(f"Downloading {file_path} -> {target_dir}")

    hf_hub_download(
        repo_id=REPO_ID,
        filename=file_path,
        repo_type="model",
        local_dir=target_dir,
    )

print("Done.")