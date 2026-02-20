from abc import ABC, abstractmethod
import numpy as np

class BaseEnhancer(ABC):
    def __init__(self, scale: int, device: str = "cpu"):
        self.scale = scale
        self.device = device

    @abstractmethod
    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        img_bgr: uint8 BGR image (OpenCV)
        return: uint8 BGR image
        """
        pass
