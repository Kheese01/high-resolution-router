from selector.features import extract_features
from datasets.metrics import psnr, ssim

def build_sample(lr, hr, enhancers):
    scores = {}
    for name, enh in enhancers.items():
        sr = enh.enhance(lr)
        scores[name] = {
            "psnr": psnr(sr, hr),
            "ssim": ssim(sr, hr)
        }

    best_model = max(scores, key=lambda k: scores[k]["psnr"])
    return extract_features(lr), best_model, scores
