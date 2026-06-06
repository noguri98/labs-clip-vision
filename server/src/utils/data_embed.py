import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

# Add parent directory to sys.path to allow imports from modules
current_file = Path(__file__).resolve()
src_root = current_file.parents[1]
if str(src_root) not in sys.path:
    sys.path.append(str(src_root))

from modules.device import get_device


class ClipEmbedder:
    def __init__(self, model_path: Path):
        device = get_device()
        provider = device.get_execution_provider()

        # Fallback to CPU if provider not available in installed onnxruntime
        available_providers = ort.get_available_providers()
        if provider not in available_providers:
            print(f"Warning: {provider} not available. Falling back to CPU.")
            provider = "CPUExecutionProvider"

        print(f"Using {provider} for ONNX Runtime")
        self.session = ort.InferenceSession(str(model_path), providers=[provider])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image: Image.Image):
        # Very basic preprocessing for CLIP (224x224, normalized)
        # In a real scenario, use AutoProcessor from transformers
        image = image.convert("RGB").resize((224, 224))
        img_data = np.array(image).astype(np.float32) / 255.0
        # Normalize
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        img_data = (img_data - mean) / std
        img_data = img_data.transpose(2, 0, 1)  # HWC to CHW
        return img_data[np.newaxis, :].astype(
            np.float32
        )  # Ensure float32 and add batch dimension

    def embed_image(self, image_path: Path):
        with Image.open(image_path) as img:
            input_tensor = self.preprocess(img)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            return outputs[0]  # Assuming the first output is the embedding


def main():
    # Setup paths
    project_root = Path(__file__).resolve().parents[3]
    base_data_path = project_root / "data"
    model_path = (
        project_root
        / "models"
        / "onnx"
        / "clip-vit-base-patch32"
        / "onnx"
        / "vision_model.onnx"
    )

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run 'uv run src/utils/model_download.py' first.")
        return

    # Initialize embedder
    embedder = ClipEmbedder(model_path)

    # Process all subdirectories in data/imgs
    imgs_root = base_data_path / "imgs"
    if not imgs_root.exists():
        print(f"No images found in {imgs_root}")
        return

    for sub_dir_path in imgs_root.iterdir():
        if not sub_dir_path.is_dir():
            continue

        sub_dir = sub_dir_path.name
        print(f"Processing directory: {sub_dir}")

        embeddings = []
        image_files = (
            list(sub_dir_path.glob("*.jpg"))
            + list(sub_dir_path.glob("*.png"))
            + list(sub_dir_path.glob("*.jpeg"))
        )

        if not image_files:
            continue

        for img_file in tqdm(image_files, desc=f"Embedding {sub_dir}"):
            try:
                emb = embedder.embed_image(img_file)
                embeddings.append(emb)
            except Exception as e:
                print(f"Failed to process {img_file}: {e}")

        if embeddings:
            # Aggregate embeddings: Calculate the MEAN to compress all images into ONE vector
            # Result shape will be (1, embedding_dim)
            all_embeddings = np.vstack(embeddings)
            compressed_embedding = np.mean(all_embeddings, axis=0, keepdims=True)

            # Save to data/embed/<sub_dir>/timestamp.npy
            embed_dir = base_data_path / "embed" / sub_dir
            embed_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = embed_dir / f"{timestamp}.npy"

            np.save(save_path, compressed_embedding)
            print(
                f"Compressed {len(embeddings)} images into one embedding at {save_path}"
            )
            print(f"Embedding shape: {compressed_embedding.shape}")
            # Cleanup: Remove processed images and the sub-directory
            print(f"Cleaning up processed images in {sub_dir_path}...")
            for img_file in image_files:
                try:
                    img_file.unlink()
                except Exception as e:
                    print(f"Failed to delete {img_file}: {e}")

            try:
                # Remove the directory if it's empty
                sub_dir_path.rmdir()
                print(f"Removed directory: {sub_dir_path}")
            except Exception as e:
                print(
                    f"Failed to remove directory {sub_dir_path} (it might not be empty): {e}"
                )


if __name__ == "__main__":
    main()
