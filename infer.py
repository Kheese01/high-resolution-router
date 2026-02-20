from selector.factory import build_selector
from models.factory import create_enhancer
import cv2

def run(input_path, output_path):
    img = cv2.imread(input_path)

    selector = build_selector(
        mode="rule",
        model_path="model_weights/selector.pkl"
    )

    choice = selector(img)
    enhancer = create_enhancer(choice)

    out = enhancer.enhance(img)
    cv2.imwrite(output_path, out)

