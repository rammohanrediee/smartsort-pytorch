from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm.auto import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class RunConfig:
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    image_size: int
    seed: int
    num_workers: int
    max_train_batches: int | None
    max_eval_batches: int | None


class WasteDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        class_to_index: dict[str, int],
        transform: transforms.Compose,
    ):
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.data_root / row["relative_path"]
        with Image.open(image_path) as image:
            image = self.transform(image.convert("RGB"))
        return image, self.class_to_index[row["label"]], str(row["relative_path"])


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
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
    def __init__(self, number_of_classes: int):
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
            nn.Linear(256, number_of_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a full SmartSort comparison model.")
    parser.add_argument("--model", choices=("custom_cnn", "resnet18"), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def create_loaders(
    split_manifest: pd.DataFrame,
    data_root: Path,
    class_to_index: dict[str, int],
    config: RunConfig,
    device: torch.device,
):
    train_transform, evaluation_transform = build_transforms(config.image_size)
    frames = {
        split: split_manifest.loc[split_manifest["split"] == split].copy()
        for split in ("train", "validation", "test")
    }
    datasets = {
        "train": WasteDataset(frames["train"], data_root, class_to_index, train_transform),
        "validation": WasteDataset(
            frames["validation"], data_root, class_to_index, evaluation_transform
        ),
        "test": WasteDataset(frames["test"], data_root, class_to_index, evaluation_transform),
    }
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
            generator=generator,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        ),
    }
    return frames, loaders


def build_model(model_name: str, number_of_classes: int) -> nn.Module:
    if model_name == "custom_cnn":
        return CustomCNN(number_of_classes)
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.30),
        nn.Linear(model.fc.in_features, number_of_classes),
    )
    return model


def calculate_metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    maximum_batches: int | None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        progress = tqdm(loader, leave=False, desc="train" if training else "evaluate")
        for batch_index, (images, targets, _) in enumerate(progress):
            if maximum_batches is not None and batch_index >= maximum_batches:
                break
            images = images.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            targets_all.extend(targets.detach().cpu().tolist())
            predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
            progress.set_postfix(loss=f"{total_loss / sample_count:.4f}")

    metrics = calculate_metrics(targets_all, predictions_all)
    metrics["loss"] = total_loss / sample_count
    metrics["samples"] = sample_count
    return metrics


def train(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    config: RunConfig,
    class_names: list[str],
    device: torch.device,
    output_directory: Path,
):
    checkpoint_path = output_directory / "best_checkpoint.pt"
    history: list[dict] = []
    best_validation_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        synchronize(device)
        started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
            config.max_train_batches,
        )
        validation_metrics = run_epoch(
            model,
            loaders["validation"],
            criterion,
            device,
            None,
            config.max_eval_batches,
        )
        scheduler.step(validation_metrics["macro_f1"])
        synchronize(device)

        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output_directory / "history.csv", index=False)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"validation_loss={validation_metrics['loss']:.4f} "
            f"validation_macro_f1={validation_metrics['macro_f1']:.4f} "
            f"seconds={record['seconds']:.1f}"
        )

        if validation_metrics["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": config.model_name,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": config.image_size,
                    "best_validation_macro_f1": best_validation_f1,
                    "epoch": epoch,
                    "config": asdict(config),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"early_stopping_epoch={epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return pd.DataFrame(history), checkpoint


def evaluate_test_set(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    class_names: list[str],
    device: torch.device,
    maximum_batches: int | None,
    output_directory: Path,
):
    model.eval()
    rows: list[dict] = []
    total_loss = 0.0
    sample_count = 0
    with torch.inference_mode():
        progress = tqdm(loader, leave=False, desc="test")
        for batch_index, (images, targets, paths) in enumerate(progress):
            if maximum_batches is not None and batch_index >= maximum_batches:
                break
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            probabilities = logits.softmax(dim=1).cpu()
            confidences, predictions = probabilities.max(dim=1)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size

            for path, target, prediction, confidence in zip(
                paths,
                targets.cpu().tolist(),
                predictions.tolist(),
                confidences.tolist(),
            ):
                rows.append(
                    {
                        "relative_path": path,
                        "target_index": target,
                        "prediction_index": prediction,
                        "target": class_names[target],
                        "prediction": class_names[prediction],
                        "score": confidence,
                        "correct": target == prediction,
                    }
                )

    predictions_frame = pd.DataFrame(rows)
    predictions_frame.to_csv(output_directory / "test_predictions.csv", index=False)
    metrics = calculate_metrics(
        predictions_frame["target_index"].tolist(),
        predictions_frame["prediction_index"].tolist(),
    )
    metrics["loss"] = total_loss / sample_count
    metrics["samples"] = sample_count

    report = pd.DataFrame(
        classification_report(
            predictions_frame["target"],
            predictions_frame["prediction"],
            labels=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report.to_csv(output_directory / "classification_report.csv")

    matrix = confusion_matrix(
        predictions_frame["target"],
        predictions_frame["prediction"],
        labels=class_names,
        normalize="true",
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Normalized test confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_directory / "confusion_matrix.png", dpi=180)
    plt.close()
    return metrics


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    default_settings = {
        "custom_cnn": {"epochs": 20, "learning_rate": 1e-3, "patience": 5},
        "resnet18": {"epochs": 12, "learning_rate": 3e-4, "patience": 4},
    }[args.model]
    config = RunConfig(
        model_name=args.model,
        epochs=args.epochs or default_settings["epochs"],
        batch_size=args.batch_size,
        learning_rate=default_settings["learning_rate"],
        weight_decay=1e-4,
        patience=args.patience or default_settings["patience"],
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    seed_everything(config.seed)
    device = choose_device()

    split_path = project_root / "data" / "processed" / "splits.csv"
    data_root = project_root / "data" / "raw" / "original"
    split_manifest = pd.read_csv(split_path)
    class_names = sorted(split_manifest["label"].unique())
    class_to_index = {label: index for index, label in enumerate(class_names)}
    frames, loaders = create_loaders(
        split_manifest,
        data_root,
        class_to_index,
        config,
        device,
    )
    split_counts = {split: len(frame) for split, frame in frames.items()}
    expected_counts = {"train": 8581, "validation": 1839, "test": 1839}
    if split_counts != expected_counts:
        raise ValueError(f"Unexpected split sizes: {split_counts}; expected {expected_counts}")

    suffix = "_smoke" if config.max_train_batches is not None else ""
    output_directory = project_root / "reports" / "full_comparison" / f"{args.model}{suffix}"
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "device": device.type,
                "split_counts": split_counts,
                "class_names": class_names,
            },
            indent=2,
        )
        + "\n"
    )

    model = build_model(config.model_name, len(class_names)).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_counts = frames["train"]["label"].value_counts()
    class_weights = torch.tensor(
        [len(frames["train"]) / (len(class_names) * train_counts[label]) for label in class_names],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    print(
        json.dumps(
            {
                "model": config.model_name,
                "device": device.type,
                "parameters": parameter_count,
                "split_counts": split_counts,
            }
        )
    )
    history, checkpoint = train(
        model,
        loaders,
        criterion,
        optimizer,
        scheduler,
        config,
        class_names,
        device,
        output_directory,
    )
    test_metrics = evaluate_test_set(
        model,
        loaders["test"],
        criterion,
        class_names,
        device,
        config.max_eval_batches,
        output_directory,
    )
    summary = {
        "model": config.model_name,
        "parameters": parameter_count,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_f1": float(checkpoint["best_validation_macro_f1"]),
        "epochs_completed": int(history["epoch"].max()),
        "test": test_metrics,
        "split_counts": split_counts,
    }
    (output_directory / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
