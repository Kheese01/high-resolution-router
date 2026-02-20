# High-Resolution Router

Content-Aware Super Resolution via Dynamic Model Selection

This project implements a dynamic super-resolution system that automatically selects the most suitable enhancement model for a given image.

Instead of relying on a single super-resolution model, the system:

1. Extracts statistical image features
2. Predicts which model performs better
3. Applies the selected enhancer

Currently supported models:

- SwinIR (Transformer-based super-resolution)
- Real-ESRGAN (GAN-based perceptual super-resolution)

---

## Architecture

Input Image  
↓  
Feature Extraction  
↓  
Selector (rule-based / learned)  
↓  
SwinIR  or  Real-ESRGAN  
↓  
Enhanced Output  

---

## Project Structure

```

High-Resolution/
│
├── models/            # Super-resolution model implementations
├── selector/          # Model selection logic (rule-based & learned)
├── datasets/          # Dataset builder and evaluation metrics
├── model_weights/     # Model weights (NOT included in repo)
├── infer.py           # Core inference pipeline
├── app.py             # CLI entry point
├── requirements.txt
├── README.md
└── .gitignore

````

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourname/high-resolution-router.git
cd high-resolution-router
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Model Weights

Model weights are NOT included in this repository.

Download the following weights separately:

* SwinIR weights
* Real-ESRGAN weights

Place them inside:

```
model_weights/
```

If using the learned selector, also place:

```
model_weights/selector.pkl
```

---

## Usage

Basic usage:

```bash
python app.py --input input.png --output output.png
```

If running the default inference script:

```bash
python infer.py
```

---

## Learned Selector

The learned selector is implemented using Logistic Regression.

Training:

```bash
python selector/train_selector.py
```

The trained model is saved as:

```
model_weights/selector.pkl
```

The selector predicts which super-resolution model will produce better quality (e.g., higher PSNR) based on extracted image features.

---

## Feature Design

Example extracted features:

* Edge density
* Laplacian variance (high-frequency proxy)
* Brightness variance
* Texture-related statistics

These features are used as input to the classifier for model selection.

---

## Evaluation

Evaluation utilities are implemented in:

```
datasets/metrics.py
```

Supported metrics may include:

* PSNR
* SSIM

---

## Future Improvements

* Replace Logistic Regression with CNN-based selector
* Add perceptual metrics (LPIPS)
* Cost-aware routing (optimize inference time)
* Automated dataset labeling pipeline
* Expand feature engineering

---

## Research Perspective

This project can be viewed as:

Dynamic model selection for perceptual image restoration.

It combines:

* Deep learning-based super-resolution
* Statistical modeling
* Meta-model selection
* Image quality evaluation

---

## License

MIT License

```

---