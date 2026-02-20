import cv2
import numpy as np
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from models.base import BaseEnhancer


class RealESRGANEnhancer(BaseEnhancer):
    def __init__(self,
                 weight_path="model_weights/RealESRGAN_x4plus.pth",
                 device="cpu"):
        super().__init__(scale=4, device=device)

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self.scale
        )

        self.upsampler = RealESRGANer(
            scale=self.scale,
            model_path=weight_path,
            model=model,
            tile=128,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=device
        )

    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        out, _ = self.upsampler.enhance(img_bgr, outscale=self.scale)
        return out
