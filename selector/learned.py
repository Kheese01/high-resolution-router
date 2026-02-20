import joblib
from .features import extract_features
from .base import BaseSelector

class LearnedSelector(BaseSelector):
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.clf = joblib.load(model_path)  

    def select(self, img):
        x = extract_features(img)[None, :]
        pred = self.clf.predict(x)[0]
        return pred
    
    def __call__(self, img):
        return self.select(img)
