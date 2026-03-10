import os
import cv2
import csv

from datasets.builder import build_sample
from models.factory import create_enhancer


LR_DIR = "datasets/DIV2K/LR_x4"
HR_DIR = "datasets/DIV2K/HR"

OUTPUT_FILE = "datasets/router_dataset.csv"


def load_image(path):
    return cv2.imread(path)


def main():

<<<<<<< HEAD
    enhancers = {
    "swinir": create_enhancer("swinir", device="cpu"),
    "realesrgan": create_enhancer("realesrgan", device="cpu"),
    }
=======
    enhancers = create_enhancers()
>>>>>>> a6c0c6c (feat: implement SR benchmark dataset pipeline (#16))

    rows = []

    for name in os.listdir(LR_DIR):
<<<<<<< HEAD
        
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
=======
>>>>>>> a6c0c6c (feat: implement SR benchmark dataset pipeline (#16))

        lr_path = os.path.join(LR_DIR, name)
        hr_path = os.path.join(HR_DIR, name)

        if not os.path.exists(hr_path):
            continue

        lr = load_image(lr_path)
        hr = load_image(hr_path)

<<<<<<< HEAD
        if lr is None or hr is None:
            print("failed to load:", name)
            continue

=======
>>>>>>> a6c0c6c (feat: implement SR benchmark dataset pipeline (#16))
        features, label, scores = build_sample(lr, hr, enhancers)

        row = list(features) + [label]

        rows.append(row)

        print("processed:", name)

    with open(OUTPUT_FILE, "w", newline="") as f:

        writer = csv.writer(f)
<<<<<<< HEAD
        writer.writerow([
            "edge_density",
            "color_variance",
            "high_freq_energy",
            "best_model"
            ])
=======
>>>>>>> a6c0c6c (feat: implement SR benchmark dataset pipeline (#16))
        writer.writerows(rows)

    print("dataset saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()