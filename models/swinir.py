import torch
import torch.nn.functional as F
import numpy as np
from models.network_swinir import SwinIR
from models.base import BaseEnhancer


class SwinIREnhancer(BaseEnhancer):
    def __init__(
        self,
        weight_path="model_weights/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
        device="cpu",
    ):
        super().__init__(scale=4, device=device)

        self.window_size = 8

        self.model = SwinIR(
            upscale=self.scale,
            in_chans=3,
            img_size=48,
            window_size=self.window_size,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        ).to(self.device)

        ckpt = torch.load(weight_path, map_location="cpu")
        state_dict = ckpt["params"] if "params" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.no_grad()
    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        # BGR -> RGB -> Tensor
        img = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
        img = (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        # padding
        _, _, h, w = img.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")

        # inference
        out = self.model(img)

        # crop
        out = out[:, :, : h * self.scale, : w * self.scale]

        # Tensor -> BGR uint8
        out = (
            out.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 1)
            .cpu()
            .numpy()
        )
        out = (out * 255.0).round().astype(np.uint8)
        return out[:, :, ::-1]
