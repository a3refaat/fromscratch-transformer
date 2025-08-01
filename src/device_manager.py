# device_manager.py
class DeviceManager:
    def __init__(self, device="cpu"):
        self.set_device(device)

    def set_device(self, device:str):
        if device == "cuda":
            try:
                import cupy as cp
                self.xp = cp
            except ImportError:
                raise ImportError("CuPy not installed.")
        elif device == "cpu":
            import numpy as np
            self.xp = np
        else:
            raise ValueError(f"Unsupported device: {device}")
        
        self.device = device

    def get_device(self):
        return self.device