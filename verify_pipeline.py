"""
verify_pipeline.py
Verification script to ensure end-to-end functionality matches requirements.
"""

import os
import torch
import numpy as np
from PIL import Image

from model_utils import (
    load_model,
    DEVICE,
    CLASS_NAMES,
    CONVNEXT_TRANSFORM,
    MODEL_PATH
)
from evidence_utils import (
    generate_gradcam,
    create_heatmap,
    create_overlay,
    create_evidence_mask,
    calculate_evidence_statistics,
    create_occluded_image,
    create_control_mask,
    get_target_probability,
    calculate_evidence_influence
)
from report_utils import generate_medical_report


def run_full_verification():
    print("=== 1. Check Model Checkpoint ===")
    assert os.path.exists(MODEL_PATH), f"Missing {MODEL_PATH}"
    print(f"Verified checkpoint exists at {MODEL_PATH}")

    print("\n=== 2. Load Model ===")
    model = load_model()
    print("Model loaded successfully onto", DEVICE)

    print("\n=== 3. Load test_image.jpg ===")
    img_path = "test_image.jpg"
    assert os.path.exists(img_path), f"Missing {img_path}"
    image = Image.open(img_path).convert("RGB")
    print(f"Image loaded: size={image.size}, mode={image.mode}")

    print("\n=== 4. Test Preprocessing & Prediction ===")
    input_tensor = CONVNEXT_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    assert input_tensor.shape == (1, 3, 224, 224), f"Unexpected tensor shape {input_tensor.shape}"

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = torch.argmax(probabilities).item()
    predicted_class = CLASS_NAMES[predicted_index]
    pcos_prob = probabilities[0].item() * 100.0
    normal_prob = probabilities[1].item() * 100.0

    print(f"Prediction: {predicted_class} (Index: {predicted_index})")
    print(f"PCOS Probability: {pcos_prob:.4f}%")
    print(f"Normal Probability: {normal_prob:.4f}%")
    assert predicted_class in ["PCOS", "Normal"]

    print("\n=== 5. Verify Grad-CAM ===")
    grayscale_cam = generate_gradcam(model, input_tensor, target_class=predicted_index)
    print(f"Grad-CAM generated: shape={grayscale_cam.shape}, min={grayscale_cam.min():.4f}, max={grayscale_cam.max():.4f}")
    assert grayscale_cam.shape == (224, 224)
    assert grayscale_cam.max() > 0

    display_img = np.array(image.resize((224, 224)))
    heatmap = create_heatmap(grayscale_cam, display_img)
    overlay = create_overlay(display_img, heatmap)
    assert heatmap.shape == (224, 224, 3)
    assert overlay.shape == (224, 224, 3)
    print("Heatmap and overlay generated successfully.")

    print("\n=== 6. Verify Evidence Mask & Statistics ===")
    evidence_mask = create_evidence_mask(grayscale_cam, threshold=0.5)
    evidence_stats = calculate_evidence_statistics(evidence_mask)
    print(f"Evidence Mask: {evidence_stats['important_pixels']} / {evidence_stats['total_pixels']} pixels ({evidence_stats['important_percentage']:.2f}%)")
    assert evidence_stats["total_pixels"] == 224 * 224

    print("\n=== 7. Verify Control Mask ===")
    control_mask = create_control_mask(grayscale_cam, evidence_mask)
    ctrl_pixels = int(np.sum(control_mask > 0))
    print(f"Control Mask: {ctrl_pixels} pixels (matches important: {ctrl_pixels == evidence_stats['important_pixels']})")
    assert ctrl_pixels == evidence_stats["important_pixels"]

    print("\n=== 8. Verify Occlusion & Dynamic Perturbation Probabilities ===")
    original_arr = np.array(display_img).astype(np.float32) / 255.0
    important_occluded = create_occluded_image(original_arr, evidence_mask)
    control_occluded = create_occluded_image(original_arr, control_mask)

    p_orig = get_target_probability(model, original_arr, CONVNEXT_TRANSFORM, DEVICE, predicted_index)
    p_imp = get_target_probability(model, important_occluded, CONVNEXT_TRANSFORM, DEVICE, predicted_index)
    p_ctrl = get_target_probability(model, control_occluded, CONVNEXT_TRANSFORM, DEVICE, predicted_index)

    influence_results = calculate_evidence_influence(p_orig, p_imp, p_ctrl)
    print(f"P_orig: {p_orig * 100:.6f}%")
    print(f"P_imp:  {p_imp * 100:.6f}% (Drop: {influence_results['important_drop'] * 100:+.6f} pp)")
    print(f"P_ctrl: {p_ctrl * 100:.6f}% (Drop: {influence_results['control_drop'] * 100:+.6f} pp)")
    print(f"Evidence Influence Diff: {influence_results['evidence_influence'] * 100:+.6f} pp")

    print("\n=== 9. Verify Medical Report Generation ===")
    report = generate_medical_report(
        prediction_class=predicted_class,
        pcos_probability=pcos_prob,
        normal_probability=normal_prob,
        evidence_stats=evidence_stats,
        influence_results=influence_results,
        image_shape=image.size,
        patient_id="TEST-VERIFICATION-01"
    )
    assert len(report) > 500
    assert "MED-VISION-GPT" in report
    assert "ROTTERDAM" in report
    print("Report generated successfully.")
    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_full_verification()
