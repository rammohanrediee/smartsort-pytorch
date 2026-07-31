# SmartSort

SmartSort classifies household waste images into ten categories using PyTorch.

The project compares a small CNN trained from scratch with a ResNet18 transfer-learning
model. Macro-F1 is used for model selection because the dataset is imbalanced.

## Results

The committed notebook contains a two-epoch subset run used to check the training and
evaluation pipeline. It is not presented as a final benchmark. Full runs use the same
saved split manifest so the model comparison remains fair.

## Models

- A custom convolutional neural network built from scratch
- A fine-tuned ResNet18 transfer-learning baseline

## Dataset

Kaggle: [`sumn2u/garbage-classification-v2`](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)

The downloaded version contains 12,259 original images and two folders of resized
copies. The notebook reads only `data/raw/original` so resized versions of an image do
not end up in different splits.

Download it after configuring your Kaggle credentials:

```bash
kaggle datasets download -d sumn2u/garbage-classification-v2 -p data/raw --unzip
```

## Baseline notebook

Open `notebooks/01_smartsort_baseline.ipynb`. It covers:

- image integrity and duplicate checks
- stratified train, validation, and test splits
- image augmentation and class-weighted loss
- custom CNN and ResNet18 model definitions
- checkpointing, early stopping, and learning-rate scheduling
- per-class metrics, a confusion matrix, and error analysis

For a full ResNet18 run, change the notebook parameters to:

```python
QUICK_RUN = False
MODEL_NAME = "resnet18"
USE_PRETRAINED_WEIGHTS = True
EPOCHS = 15
```

Restart the kernel and run all cells. Keep `data/processed/splits.csv` unchanged when
comparing models.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Project structure

```text
SmartSort/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── LICENSE
├── README.md
└── requirements.txt
```
