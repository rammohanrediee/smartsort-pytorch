from io import BytesIO

from PIL import Image
from streamlit.testing.v1 import AppTest


def test_app_starts_without_errors():
    app = AppTest.from_file("app.py").run(timeout=30)
    assert app.exception == []
    assert app.title[0].value == "Sort waste with a photo"
    assert len(app.get("file_uploader")) == 1


def test_uploaded_image_reaches_prediction_results():
    image_buffer = BytesIO()
    Image.new("RGB", (320, 240), color="white").save(image_buffer, format="PNG")

    app = AppTest.from_file("app.py").run(timeout=30)
    app.get("file_uploader")[0].upload(
        "sample.png", image_buffer.getvalue(), "image/png"
    ).run(timeout=30)

    assert app.exception == []
    assert any(item.value == "Sorting tip" for item in app.subheader)
    assert any(item.value == "Top predictions" for item in app.subheader)
