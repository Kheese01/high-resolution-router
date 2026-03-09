import numpy as np
from skimage.metrics import structural_similarity

def psnr(img1, img2):

    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)

    if mse == 0:
        return 100

    return 20 * np.log10(255.0 / np.sqrt(mse))


def ssim(img1, img2):

    score, _ = structural_similarity(
        img1,
        img2,
        channel_axis=2,
        full=True
    )

    return score