import platform

from modules.device import Device


class LinuxDevice(Device):
    def get_info(self) -> str:
        return f"Linux Device (kernel: {platform.release()})"

    def get_execution_provider(self) -> str:
        # For ONNX Runtime on Linux, CUDA is preferred if available, otherwise CPU
        return "CUDAExecutionProvider"
