from models.swinir import SwinIREnhancer
from models.realesrgan import RealESRGANEnhancer

def create_enhancer(name, device="cpu"):
    name = name.lower()
    if name == "swinir":
        return SwinIREnhancer(device=device)
    elif name in ["rrdb", "realesrgan"]:
        return RealESRGANEnhancer(device=device)
    else:
        raise ValueError(f"Unknown enhancer: {name}")
