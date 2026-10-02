import torch
import torch.nn as nn
from torchvision import models, transforms


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "models/convnext_pcos.pth"


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = {
    0: "PCOS",
    1: "Normal"
}


# ============================================================
# EXACT CONVNEXT PREPROCESSING
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CONVNEXT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = models.convnext_tiny(
        weights=None
    )

    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(DEVICE)
    model.eval()

    return model