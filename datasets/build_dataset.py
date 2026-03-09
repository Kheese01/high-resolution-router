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

    enhancers = create_enhancer()

    rows = []

    for name in os.listdir(LR_DIR):

        lr_path = os.path.join(LR_DIR, name)
        hr_path = os.path.join(HR_DIR, name)

        if not os.path.exists(hr_path):
            continue

        lr = load_image(lr_path)
        hr = load_image(hr_path)

        features, label, scores = build_sample(lr, hr, enhancers)

        row = list(features) + [label]

        rows.append(row)

        print("processed:", name)

    with open(OUTPUT_FILE, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerows(rows)

    print("dataset saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()