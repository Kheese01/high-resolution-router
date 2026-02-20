from selector.features import extract_features
from datasets.metrics import psnr

def build_sample(lr, hr, enhancers):
    scores = {}
    for name, enh in enhancers.items():
        sr = enh.enhance(lr)
        scores[name] = psnr(sr, hr)

    label = max(scores, key=scores.get)
    return extract_features(lr), label
