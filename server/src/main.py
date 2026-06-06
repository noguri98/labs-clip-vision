import argparse
import sys

from modules.device import get_device


def main():
    parser = argparse.ArgumentParser(description="CLIP Vision Server")
    parser.add_argument(
        "--model", type=str, default="clip-vit-base-patch32", help="Model name to use"
    )
    args = parser.parse_args()

    print(f"Starting server with model: {args.model}")

    try:
        device = get_device()
        print(f"Detected OS: {device.get_info()}")
        print(f"Recommended Execution Provider: {device.get_execution_provider()}")
    except Exception as e:
        print(f"Error initializing device: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
