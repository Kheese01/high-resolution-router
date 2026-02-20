import joblib
from .features import extract_features
from .base import BaseSelector

class LearnedSelector(BaseSelector):
    def __init__(self, model_path):
        self.clf = joblib.load(model_path)

    def select(self, img):
        x = extract_features(img)[None, :]
        pred = self.clf.predict(x)[0]
        return pred
