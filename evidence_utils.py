import numpy as np
import cv2
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# GRAD-CAM TARGET LAYER
# ============================================================

def get_target_layer(model):
    """
    Target layer for ConvNeXt-Tiny Grad-CAM.
    """

    return model.features[-1]


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    input_tensor,
    target_class
):
    """
    Generate Grad-CAM activation map.
    """

    target_layer = get_target_layer(model)

    targets = [
        ClassifierOutputTarget(target_class)
    ]

    with GradCAM(
        model=model,
        target_layers=[target_layer]
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    grayscale_cam = np.maximum(
        grayscale_cam,
        0
    )

    if grayscale_cam.max() > 0:

        grayscale_cam = (
            grayscale_cam /
            grayscale_cam.max()
        )

    return grayscale_cam


# ============================================================
# HEATMAP
# ============================================================

def create_heatmap(
    grayscale_cam,
    image
):
    """
    Convert Grad-CAM values into a visual heatmap.
    """

    height, width = image.shape[:2]

    resized_cam = cv2.resize(
        grayscale_cam,
        (width, height)
    )

    heatmap_uint8 = np.uint8(
        255 * resized_cam
    )

    heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    return heatmap


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    image,
    heatmap,
    alpha=0.45
):
    """
    Overlay Grad-CAM heatmap on original image.
    """

    image = np.asarray(
        image,
        dtype=np.uint8
    )

    heatmap = np.asarray(
        heatmap,
        dtype=np.uint8
    )

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay


# ============================================================
# EVIDENCE MASK
# ============================================================

def create_evidence_mask(
    grayscale_cam,
    threshold=0.5
):
    """
    Create binary mask from high-importance Grad-CAM pixels.
    """

    mask = (
        grayscale_cam >= threshold
    ).astype(np.uint8)

    return mask


# ============================================================
# EVIDENCE STATISTICS
# ============================================================

def calculate_evidence_statistics(
    mask
):
    """
    Calculate image-derived evidence statistics.
    """

    total_pixels = mask.size

    important_pixels = int(
        np.sum(mask > 0)
    )

    important_percentage = (
        important_pixels /
        total_pixels
    ) * 100

    return {
        "total_pixels": total_pixels,
        "important_pixels": important_pixels,
        "important_percentage": important_percentage
    }


# ============================================================
# CREATE OCCLUDED IMAGE
# ============================================================

def create_occluded_image(
    image,
    mask,
    blur_kernel=31
):
    """
    Blur the selected evidence region.
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    blurred_image = cv2.GaussianBlur(
        image,
        (blur_kernel, blur_kernel),
        0
    )

    mask_bool = mask > 0

    occluded_image = image.copy()

    occluded_image[
        mask_bool
    ] = blurred_image[
        mask_bool
    ]

    return occluded_image


# ============================================================
# CREATE LOW-IMPORTANCE CONTROL MASK
# ============================================================

def create_control_mask(
    grayscale_cam,
    important_mask,
    random_state=42
):
    """
    Create a same-size low-importance control region.

    The control mask contains approximately the same
    number of pixels as the important region but is
    selected from low-attribution areas.
    """

    flat_cam = grayscale_cam.flatten()

    important_count = int(
        np.sum(important_mask > 0)
    )

    if important_count == 0:

        return np.zeros_like(
            important_mask,
            dtype=np.uint8
        )

    # Pixels not belonging to the important region
    candidate_indices = np.where(
        important_mask.flatten() == 0
    )[0]

    if len(candidate_indices) < important_count:

        important_count = len(
            candidate_indices
        )

    # Sort candidates by lowest attribution
    candidate_values = flat_cam[
        candidate_indices
    ]

    sorted_order = np.argsort(
        candidate_values
    )

    selected_indices = candidate_indices[
        sorted_order[:important_count]
    ]

    control_mask = np.zeros(
        grayscale_cam.size,
        dtype=np.uint8
    )

    control_mask[
        selected_indices
    ] = 1

    control_mask = control_mask.reshape(
        grayscale_cam.shape
    )

    return control_mask


# ============================================================
# MODEL PROBABILITY
# ============================================================

def get_target_probability(
    model,
    image,
    transform,
    device,
    target_class
):
    """
    Run model on an image and return target-class probability.
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    image_uint8 = (
        image * 255.0
    ).astype(np.uint8)

    tensor_image = torch.from_numpy(
        image_uint8
    ).permute(
        2, 0, 1
    ).float() / 255.0

    # Convert tensor back to PIL-compatible image
    from PIL import Image

    pil_image = Image.fromarray(
        image_uint8
    ).convert("RGB")

    input_tensor = transform(
        pil_image
    ).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():

        logits = model(
            input_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    return probabilities[
        target_class
    ].item()


# ============================================================
# EVIDENCE INFLUENCE
# ============================================================

def calculate_evidence_influence(
    original_probability,
    important_probability,
    control_probability
):
    """
    Calculate probability changes caused by
    important-region and control-region masking.
    """

    important_drop = (
        original_probability -
        important_probability
    )

    control_drop = (
        original_probability -
        control_probability
    )

    evidence_influence = (
        important_drop -
        control_drop
    )

    return {
        "important_drop": important_drop,
        "control_drop": control_drop,
        "evidence_influence": evidence_influence
    }