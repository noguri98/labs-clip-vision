import platform
from abc import ABC, abstractmethod


class Device(ABC):
    @abstractmethod
    def get_info(self) -> str:
        pass

    @abstractmethod
    def get_execution_provider(self) -> str:
        pass


def get_device() -> Device:
    os_name = platform.system()
    if os_name == "Darwin":
        from components.macOS import MacOSDevice

        return MacOSDevice()
    elif os_name == "Linux":
        from components.linux import LinuxDevice

        return LinuxDevice()
    else:
        raise NotImplementedError(f"Unsupported OS: {os_name}")
