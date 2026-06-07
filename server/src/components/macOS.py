import platform
from typing import Any, Dict, List

import onnxruntime as ort

from modules.device import Device


class MacOSDevice(Device):
    def __init__(self):
        self.session = None

    def get_info(self) -> str:
        return f"macOS Device (version: {platform.mac_ver()[0]})"

    def load_model(self, model_path: str) -> None:
        # For ONNX Runtime on macOS, CoreMLExecutionProvider is used for acceleration
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        # Check if CoreMLExecutionProvider is available
        available_providers = ort.get_available_providers()
        if "CoreMLExecutionProvider" not in available_providers:
            print("Warning: CoreMLExecutionProvider not found, falling back to CPU.")
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        print(f"Model loaded on macOS with providers: {self.session.get_providers()}")

    def infer(self, input_feed: Dict[str, Any]) -> List[Any]:
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.session.run(None, input_feed)

    def get_input_names(self) -> List[str]:
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return [i.name for i in self.session.get_inputs()]
