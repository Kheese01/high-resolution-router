from abc import ABC, abstractmethod
import numpy as np

class BaseSelector(ABC):
    @abstractmethod
    def select(self, img_bgr: np.ndarray) -> str:
        """
        return enhancer name (e.g. 'realesrgan', 'swinir')
        """
        pass
