import platform
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Device(ABC):
    @abstractmethod
    def get_info(self) -> str:
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load the model into the device with platform-specific optimizations."""
        pass

    @abstractmethod
    def infer(self, input_feed: Dict[str, Any]) -> List[Any]:
        """Execute inference using the loaded model."""
        pass

    @abstractmethod
    def get_input_names(self) -> List[str]:
        """Get the names of the model's inputs."""
        pass


def get_device(model_path: str) -> Device:
    os_name = platform.system()
    if os_name == "Darwin":
        from components.macOS import MacOSDevice

        device = MacOSDevice()
    elif os_name == "Linux":
        from components.linux import LinuxDevice

        device = LinuxDevice()
    else:
        raise NotImplementedError(f"Unsupported OS: {os_name}")

    device.load_model(model_path)
    return device
