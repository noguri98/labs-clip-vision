import argparse
import sys
from pathlib import Path

from modules.inference import Detector


def main():
    parser = argparse.ArgumentParser(description="CLIP Vision Server")
    parser.add_argument(
        "--model", type=str, default="clip-vit-base-patch32", help="Model name to use"
    )
    args = parser.parse_args()

    # Define model path
    project_root = Path(__file__).resolve().parents[1]
    model_path = (
        project_root / "models" / "onnx" / args.model / "onnx" / "vision_model.onnx"
    )

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run 'uv run src/utils/model_download.py' first.")
        sys.exit(1)

    print(f"Starting server with model: {args.model}")

    try:
        # Now Detector handles everything: OS detection, model loading, and execution provider setup
        detector = Detector(str(model_path))
        print("Detector initialized successfully.")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
