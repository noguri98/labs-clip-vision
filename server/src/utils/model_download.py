import os
from pathlib import Path

from huggingface_hub import snapshot_download


def download_clip_onnx():
    # Define the repository ID
    repo_id = "Xenova/clip-vit-base-patch32"

    # Define the target directory relative to this script
    # Current script: project_root/server/src/utils/model_download.py
    # Target directory: project_root/models/onnx/
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]
    target_dir = project_root / "models" / "onnx" / repo_id.split("/")[-1]

    print(f"Downloading {repo_id} to {target_dir}...")

    # Ensure the target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download the model files
    # We filter for .onnx and configuration files
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        allow_patterns=["*.onnx", "*.json", "*.txt"],
        local_dir_use_symlinks=False,
    )

    print("Download completed successfully.")


if __name__ == "__main__":
    download_clip_onnx()
