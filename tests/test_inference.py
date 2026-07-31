from pathlib import Path

import torch
from PIL import Image

from inference import CustomCNN, build_transform, load_model, predict_image


def test_preprocessing_matches_model_input_size():
    image = Image.new("RGB", (400, 300), color="white")
    tensor = build_transform(224)(image)
    assert tensor.shape == (3, 224, 224)


def test_checkpoint_loads_and_predicts():
    checkpoint_path = Path(__file__).parents[1] / "models" / "custom_cnn_best.pt"
    device = torch.device("cpu")
    model, class_names, image_size = load_model(checkpoint_path, device)

    assert isinstance(model, CustomCNN)
    assert len(class_names) == 10

    prediction = predict_image(
        Image.new("RGB", (320, 240), color="white"),
        model,
        class_names,
        image_size,
        device,
    )
    assert prediction.label in class_names
    assert 0.0 <= prediction.confidence <= 1.0
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-5
