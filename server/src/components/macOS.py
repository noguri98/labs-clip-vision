import platform

from modules.device import Device


class MacOSDevice(Device):
    def get_info(self) -> str:
        return f"macOS Device (version: {platform.mac_ver()[0]})"

    def get_execution_provider(self) -> str:
        # For ONNX Runtime on macOS, CoreMLExecutionProvider is often used for acceleration
        return "CoreMLExecutionProvider"
