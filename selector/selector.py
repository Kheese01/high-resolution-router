from .rule_based import choose_model

def select_model(img_bgr):
    return choose_model(img_bgr)
