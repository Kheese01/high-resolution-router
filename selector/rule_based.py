import cv2
from .features import extract_features
from .base import BaseSelector

class RuleBasedSelector(BaseSelector):
    def select(self, img):
        feats = extract_features(img)
        e, c, h = feats

        if e > 0.12 and c < 40:
            return "realesrgan"
        else:
            return "swinir"
