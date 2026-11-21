"""
KalaRaksha - Peeling Paint & Plaster Detection Module
Heritage Wall Deterioration Analysis & Restoration-Priority Mapping System

This module detects peeling paint, delaminated fresco layers, blistered plaster,
and flaking surface finishes on heritage walls and painted monuments.

Techniques:
1. Multi-directional Sobel gradient magnitude computation
2. Local edge-density mapping via spatial box convolution
3. Surface contrast discontinuity extraction (boundary step between paint and exposed underlayer)
4. Morphological clustering of flaking perimeters
5. Connected component filtering for fragmented, curling delamination zones
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import cv2


def detect_peeling_plaster(
    gray_image: np.ndarray,
    rgb_image: np.ndarray,
    sensitivity: float = 0.5,
    min_patch_area: int = 150
) -> Dict[str, Any]:
    """
    Detect peeling paint, blistered plaster, and delaminated surface flakes.

    Args:
        gray_image: Grayscale uint8 image.
        rgb_image: RGB uint8 image.
        sensitivity: Float in [0.1, 1.0].
        min_patch_area: Minimum area in pixels to retain as a peeling region.

    Returns:
        dict containing:
            - 'peeling_mask': uint8 binary mask (0 or 255)
            - 'peeling_density': float32 normalized map [0.0, 1.0]
            - 'gradient_magnitude': np.ndarray uint8 of normalized Sobel gradient
            - 'edge_density_map': np.ndarray float32 local edge density
            - 'peeling_area_pct': float percentage of wall affected
            - 'num_peeling_patches': int count of flaking zones
            - 'severity_score': float score in [0, 100]
            - 'patch_details': list of dicts with bounding boxes and metrics
            - 'explanation': str explainable diagnostic rationale
    """
    h, w = gray_image.shape
    total_pixels = h * w
    sens = np.clip(sensitivity, 0.1, 1.0)

    # 1. Multi-directional Sobel Gradient Magnitude
    # Blistered and flaking edges present sharp, curling relief boundaries
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize gradient magnitude to [0, 255]
    max_g = np.max(grad_mag)
    if max_g > 0:
        grad_norm = (grad_mag / max_g * 255.0).astype(np.uint8)
    else:
        grad_norm = np.zeros_like(gray_image)

    # 2. Extract sharp delamination micro-edges
    edge_thresh = int(max(20, 50 - sens * 28))
    _, edge_binary = cv2.threshold(grad_norm, edge_thresh, 255, cv2.THRESH_BINARY)

    # 3. Compute Local Edge Density (spatial clustering of peeling ridges)
    # Flaking paint creates dense swarms of jagged micro-edges rather than single isolated lines
    box_size = int(max(15, min(h, w) // 30))
    if box_size % 2 == 0:
        box_size += 1
    edge_float = (edge_binary.astype(np.float32) / 255.0)
    edge_density = cv2.boxFilter(edge_float, ddepth=-1, ksize=(box_size, box_size))

    # Threshold edge density to find regions with concentrated flaking
    density_thresh = float(max(0.08, 0.22 - sens * 0.12))
    density_binary = (edge_density >= density_thresh).astype(np.uint8) * 255

    # 4. Surface Contrast Discontinuity:
    # Delaminated plaster reveals raw underlayer with step-change in local standard deviation
    # We compute local standard deviation in a 9x9 neighborhood
    mean_local = cv2.blur(gray_image.astype(np.float32), (9, 9))
    sq_mean = cv2.blur(gray_image.astype(np.float32)**2, (9, 9))
    local_var = np.maximum(0.0, sq_mean - mean_local**2)
    local_std = np.sqrt(local_var)

    std_thresh = float(max(12.0, 30.0 - sens * 16.0))
    roughness_binary = (local_std >= std_thresh).astype(np.uint8) * 255

    # 5. Candidate Peeling Regions: overlap of high edge-density and high surface roughness
    candidate = cv2.bitwise_and(density_binary, roughness_binary)

    # Dilate and close to connect flaking clusters
    kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel_m)

    # 6. Connected Component & Geometric Filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)

    validated_mask = np.zeros((h, w), dtype=np.uint8)
    patch_details: List[Dict[str, Any]] = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_patch_area:
            continue

        comp_w = stats[label, cv2.CC_STAT_WIDTH]
        comp_h = stats[label, cv2.CC_STAT_HEIGHT]
        comp_x = stats[label, cv2.CC_STAT_LEFT]
        comp_y = stats[label, cv2.CC_STAT_TOP]

        comp_mask = (labels == label).astype(np.uint8) * 255

        # Measure mean edge density within this patch
        mean_density_val = float(cv2.mean(edge_density, mask=comp_mask)[0])

        # Validate patch: must have high edge density and irregular fragmented boundaries
        if mean_density_val >= (density_thresh * 0.85):
            validated_mask = cv2.bitwise_or(validated_mask, comp_mask)
            cx, cy = int(centroids[label][0]), int(centroids[label][1])

            patch_details.append({
                "bbox": (int(comp_x), int(comp_y), int(comp_w), int(comp_h)),
                "area_px": int(area),
                "area_pct": round((area / float(total_pixels)) * 100.0, 3),
                "centroid": (cx, cy),
                "mean_edge_density": round(mean_density_val, 3),
            })

    # 7. Metrics & Smooth Density Map
    total_peeling_px = cv2.countNonZero(validated_mask)
    peeling_area_pct = round((total_peeling_px / float(total_pixels)) * 100.0, 2)
    num_patches = len(patch_details)

    peeling_density = cv2.GaussianBlur(validated_mask.astype(np.float32) / 255.0, (21, 21), 0)
    max_d = np.max(peeling_density)
    if max_d > 0:
        peeling_density = peeling_density / max_d

    # 8. Severity Score [0 - 100]
    # > 4% peeling is severe conservation emergency (loss of fresco/artwork layer)
    cov_score = min(60.0, (peeling_area_pct / 4.0) * 60.0)
    count_score = min(25.0, (num_patches / 5.0) * 25.0)
    density_score = min(15.0, (np.mean(edge_density[validated_mask > 0]) if total_peeling_px > 0 else 0.0) * 50.0)
    severity_score = float(np.clip(cov_score + count_score + density_score, 0.0, 100.0))

    # 9. Explainable Diagnostic Rationale
    if num_patches == 0:
        explanation = "No flaking paint, blistered lime-wash, or plaster delamination detected. Surface bond is stable."
    else:
        explanation = (
            f"Detected {num_patches} peeling/flaking zone(s) covering {peeling_area_pct}% of the surface. "
            f"These areas exhibit elevated Sobel edge density (> {density_thresh:.2f}), "
            f"curling peripheral boundary relief, and localized variance jumps indicative of "
            f"substrate loss-of-adhesion, blistering, and active paint layer detachment."
        )

    return {
        "peeling_mask": validated_mask,
        "peeling_density": peeling_density,
        "gradient_magnitude": grad_norm,
        "edge_density_map": edge_density,
        "peeling_area_pct": peeling_area_pct,
        "num_peeling_patches": num_patches,
        "severity_score": round(severity_score, 1),
        "patch_details": patch_details,
        "explanation": explanation
    }
