import cv2
import numpy as np

def edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return edges.mean()

def color_variance(img):
    img_small = cv2.resize(img, (64, 64))
    pixels = img_small.reshape(-1, 3)
    return pixels.std(axis=0).mean()

def high_freq_energy(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.array([
        edge_density(gray),
        color_variance(img),
        high_freq_energy(gray),
    ], dtype=np.float32)

