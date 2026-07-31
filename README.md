# SmartSort

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CNN-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

SmartSort is an image-classification project that identifies household waste across ten
categories. It combines a PyTorch training workflow with a Streamlit interface for
uploading an image and reviewing the model's prediction, confidence, and top-three
classes.

## Features

- Ten-class household waste classification
- Image validation and exact-duplicate detection
- Leakage-safe, stratified train, validation, and test splits
- Data augmentation and class-weighted cross-entropy loss
- Custom CNN baseline and ResNet18 transfer-learning option
- Checkpointing, early stopping, and learning-rate scheduling
- Macro precision, recall, F1, confusion matrix, and error analysis
- Streamlit interface with model confidence and sorting guidance
- CPU, CUDA, and Apple MPS inference support

## Classes

`battery` · `biological` · `cardboard` · `clothes` · `glass` · `metal` · `paper` ·
`plastic` · `shoes` · `trash`

## Project workflow

1. Audit the source images and remove invalid or exact-duplicate files.
2. Create a fixed 70/15/15 stratified split.
3. Apply training-time augmentation and ImageNet normalization.
4. Train the selected architecture with class-weighted loss.
5. Select the best checkpoint using validation macro-F1.
6. Evaluate once on the held-out test split and inspect confident mistakes.
7. Load the checkpoint in Streamlit for interactive inference.

## Dataset

The project uses the
[Garbage Classification V2 dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
from Kaggle. The downloaded release contains 12,259 original images and two folders of
resized copies.

Only `data/raw/original` is used. Mixing the resized folders with the originals would
place different versions of the same image across data splits and inflate evaluation
results.

After configuring your Kaggle credentials, download the dataset with:

```bash
kaggle datasets download \
  -d sumn2u/garbage-classification-v2 \
  -p data/raw \
  --unzip
```

The dataset files are intentionally excluded from Git.

## Installation

```bash
git clone https://github.com/rammohanrediee/smartsort-pytorch.git
cd smartsort-pytorch

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the Streamlit app

```bash
streamlit run app.py
```

The app loads `models/custom_cnn_best.pt` by default. To use another compatible custom
CNN checkpoint:

```bash
SMARTSORT_MODEL_PATH=/path/to/checkpoint.pt streamlit run app.py
```

The included checkpoint comes from a short baseline run used to verify the complete
training and inference pipeline. Its predictions should be treated as a demonstration,
not authoritative disposal advice.

## Run the notebook

Open `notebooks/01_smartsort_baseline.ipynb` in JupyterLab:

```bash
jupyter lab
```

For a full ResNet18 experiment, update the notebook configuration:

```python
QUICK_RUN = False
MODEL_NAME = "resnet18"
USE_PRETRAINED_WEIGHTS = True
EPOCHS = 15
```

Keep the generated `data/processed/splits.csv` unchanged when comparing architectures.

## Tests

```bash
pytest -q
```

The tests cover notebook execution state, preprocessing dimensions, checkpoint loading,
probability output, Streamlit startup, and the upload-to-prediction path.

## Repository structure

```text
smartsort-pytorch/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── custom_cnn_best.pt
├── notebooks/
│   └── 01_smartsort_baseline.ipynb
├── tests/
│   ├── test_app.py
│   ├── test_inference.py
│   └── test_notebook.py
├── app.py
├── inference.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Limitations

- The dataset contains web-sourced images and may not represent every real disposal
  environment.
- Similar materials, cluttered backgrounds, and objects containing multiple materials
  can reduce accuracy.
- Recycling rules vary by location; the app's sorting tips are general guidance only.
- The bundled model is a baseline checkpoint and should be retrained before production
  use.

## License

This project is available under the [MIT License](LICENSE).

Copyright © 2026 [rammohanrediee](https://github.com/rammohanrediee).
