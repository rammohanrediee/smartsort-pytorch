from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import choose_device, load_model, predict_image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "custom_cnn_best.pt"
CHECKPOINT_PATH = Path(os.getenv("SMARTSORT_MODEL_PATH", DEFAULT_CHECKPOINT))

SORTING_TIPS = {
    "battery": "Keep batteries separate from household waste and use a designated collection point.",
    "biological": "Compost it where local facilities accept food and garden waste.",
    "cardboard": "Flatten it and keep it clean and dry before recycling.",
    "clothes": "Reuse, donate, or take it to a textile recycling collection point.",
    "glass": "Empty and rinse it, then follow local rules for glass color separation.",
    "metal": "Rinse containers and place them in the accepted metal recycling stream.",
    "paper": "Keep it dry and remove food-contaminated sections before recycling.",
    "plastic": "Check the resin code and local collection rules before recycling.",
    "shoes": "Donate wearable pairs or use a footwear or textile recycling program.",
    "trash": "Use general waste when the item cannot be reused, composted, or recycled locally.",
}


st.set_page_config(
    page_title="SmartSort",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2.5rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #dbe7df;}
    .eyebrow {color: #2f7652; font-weight: 700; letter-spacing: .12em; font-size: .78rem;}
    .result-card {background: #f2f8f4; border: 1px solid #cfe3d6; border-radius: 16px;
                  padding: 1.25rem 1.4rem; margin: .5rem 0 1rem;}
    .result-label {font-size: 2rem; font-weight: 750; color: #174c34; margin: .15rem 0;}
    .muted {color: #5f6f66; font-size: .92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model…")
def get_model(path: str):
    device = choose_device()
    model, class_names, image_size = load_model(Path(path), device)
    return model, class_names, image_size, device


with st.sidebar:
    st.header("About")
    st.write(
        "SmartSort uses a PyTorch convolutional neural network to classify common household waste."
    )
    st.divider()
    st.caption("MODEL")
    st.write("Custom CNN · 10 classes · 224 × 224 input")
    st.caption("STATUS")
    st.warning(
        "This is an early baseline trained on a short validation run. Treat predictions as a demo, not disposal advice.",
        icon="⚠️",
    )

st.markdown('<div class="eyebrow">HOUSEHOLD WASTE CLASSIFIER</div>', unsafe_allow_html=True)
st.title("Sort waste with a photo")
st.write(
    "Upload a clear image of one item. SmartSort will estimate its category and show the three most likely classes."
)

if not CHECKPOINT_PATH.exists():
    st.error(
        f"Model checkpoint not found at `{CHECKPOINT_PATH}`. "
        "Set `SMARTSORT_MODEL_PATH` or add the trained checkpoint to `models/`."
    )
    st.stop()

try:
    model, class_names, image_size, device = get_model(str(CHECKPOINT_PATH))
except (RuntimeError, KeyError, ValueError) as error:
    st.error(f"The model could not be loaded: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Waste image",
    type=("jpg", "jpeg", "png", "webp"),
    help="For better results, use one object, a plain background, and even lighting.",
)

if uploaded_file is None:
    st.info("Upload an image to begin.", icon="📷")
    st.stop()

try:
    uploaded_image = Image.open(uploaded_file).convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error("That file could not be read as an image. Try a JPG, PNG, or WebP file.")
    st.stop()

prediction = predict_image(uploaded_image, model, class_names, image_size, device)
top_scores = sorted(prediction.probabilities.items(), key=lambda item: item[1], reverse=True)[:3]

image_column, result_column = st.columns((1.05, 1), gap="large")
with image_column:
    st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

with result_column:
    st.caption("PREDICTION")
    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-label">{prediction.label.title()}</div>
          <div class="muted">Model confidence: {prediction.confidence:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Sorting tip")
    st.write(SORTING_TIPS[prediction.label])

st.subheader("Top predictions")
score_frame = pd.DataFrame(top_scores, columns=("Category", "Confidence")).set_index("Category")
score_frame["Confidence"] *= 100
st.bar_chart(score_frame, horizontal=True, color="#2f7652", x_label="Confidence (%)")
st.caption(f"Inference device: {device.type.upper()} · Scores are model probabilities, not guarantees.")
