import os
import cv2

HR_DIR = "datasets/DIV2K/HR"
LR_DIR = "datasets/DIV2K/LR_x4"

SCALE = 4


def main():

    os.makedirs(LR_DIR, exist_ok=True)

    for name in os.listdir(HR_DIR):

        hr_path = os.path.join(HR_DIR, name)
        lr_path = os.path.join(LR_DIR, name)

        img = cv2.imread(hr_path)

        if img is None:
            continue

        h, w = img.shape[:2]

        lr = cv2.resize(
            img,
            (w // SCALE, h // SCALE),
            interpolation=cv2.INTER_CUBIC
        )

        cv2.imwrite(lr_path, lr)

        print("created:", lr_path)


if __name__ == "__main__":
    main()