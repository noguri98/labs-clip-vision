import platform
from typing import Any, Dict, List

import onnxruntime as ort

from modules.device import Device


class LinuxDevice(Device):
    def __init__(self):
        self.session = None

    def get_info(self) -> str:
        return f"Linux Device (kernel: {platform.release()})"

    def load_model(self, model_path: str) -> None:
        # For ONNX Runtime on Linux, CUDA is preferred if available
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Check if CUDAExecutionProvider is available
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available_providers:
            print("Warning: CUDAExecutionProvider not found, falling back to CPU.")
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        print(f"Model loaded on Linux with providers: {self.session.get_providers()}")

    def infer(self, input_feed: Dict[str, Any]) -> List[Any]:
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.session.run(None, input_feed)

    def get_input_names(self) -> List[str]:
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return [i.name for i in self.session.get_inputs()]
