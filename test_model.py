from model_utils import (
    load_model,
    DEVICE,
    CLASS_NAMES,
    CONVNEXT_TRANSFORM
)

import torch
from PIL import Image


# ============================================================
# LOAD MODEL
# ============================================================

model = load_model()

print("Model loaded successfully!")
print("Device:", DEVICE)


# ============================================================
# TEST IMAGE
# ============================================================

IMAGE_PATH = "test_image.jpg"

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ============================================================
# EXACT CONVNEXT PREPROCESSING
# ============================================================

image_tensor = CONVNEXT_TRANSFORM(
    image
).unsqueeze(0)

image_tensor = image_tensor.to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    logits = model(
        image_tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )[0]


predicted_index = torch.argmax(
    probabilities
).item()

predicted_class = CLASS_NAMES[
    predicted_index
]


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n==============================")
print("       MODEL PREDICTION")
print("==============================")

print(
    "Prediction:",
    predicted_class
)

print(
    "PCOS Probability:",
    f"{probabilities[0].item() * 100:.2f}%"
)

print(
    "Normal Probability:",
    f"{probabilities[1].item() * 100:.2f}%"
)

print("==============================")