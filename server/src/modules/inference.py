from typing import Any, List

import numpy as np
from PIL import Image

from modules.device import get_device


class Detector:
    def __init__(self, model_path: str):
        # The factory method now handles both platform branching and model loading
        self.device = get_device(model_path)
        self.input_names = self.device.get_input_names()
        print(f"Initialized detector on: {self.device.get_info()}")
        print(f"Model input names: {self.input_names}")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        # Basic CLIP preprocessing (224x224, normalized)
        image = image.convert("RGB").resize((224, 224))
        img_data = np.array(image).astype(np.float32) / 255.0
        # Normalization constants for CLIP
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        img_data = (img_data - mean) / std
        img_data = img_data.transpose(2, 0, 1)  # HWC to CHW
        return img_data[np.newaxis, :].astype(np.float32)

    def predict(self, image: Image.Image) -> List[Any]:
        input_tensor = self.preprocess(image)
        # Use the first input name for the preprocessed image
        input_feed = {self.input_names[0]: input_tensor}
        return self.device.infer(input_feed)
