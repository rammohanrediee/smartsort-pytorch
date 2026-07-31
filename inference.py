from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, 0.05),
            ConvBlock(32, 64, 0.10),
            ConvBlock(64, 128, 0.15),
            ConvBlock(128, 256, 0.20),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "custom_cnn":
        raise ValueError("This app currently supports custom CNN checkpoints only.")

    class_names = checkpoint["class_names"]
    model = CustomCNN(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names, int(checkpoint["image_size"])


def predict_image(
    image: Image.Image,
    model: nn.Module,
    class_names: list[str],
    image_size: int,
    device: torch.device,
) -> Prediction:
    image_tensor = build_transform(image_size)(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(image_tensor).softmax(dim=1).squeeze(0).cpu()

    confidence, predicted_index = probabilities.max(dim=0)
    scores = {
        label: float(probabilities[index]) for index, label in enumerate(class_names)
    }
    return Prediction(
        label=class_names[int(predicted_index)],
        confidence=float(confidence),
        probabilities=scores,
    )
