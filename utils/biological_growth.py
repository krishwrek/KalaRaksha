"""
KalaRaksha - Biological Growth Detection Module
Heritage Wall Deterioration Analysis & Restoration-Priority Mapping System

This module identifies biological colonization on heritage stonework and masonry:
- Moss colonies (Bryophytes)
- Algal biofilms (Cyanobacteria / Chlorophyta)
- Crustose and foliose lichens

Techniques:
1. Dual-band HSV chlorophyll chromaticity segmentation
2. Normalized Excess Green Index (ExG = 2g - r - b)
3. Morphological opening (noise reduction) and closing (colony consolidation)
4. Connected component colony clustering and bio-coverage quantification
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import cv2


def detect_biological_growth(
    rgb_image: np.ndarray,
    hsv_image: np.ndarray,
    sensitivity: float = 0.5,
    min_colony_area: int = 100
) -> Dict[str, Any]:
    """
    Detect biological growth (moss, algae, lichen) on heritage surfaces.

    Args:
        rgb_image: RGB uint8 image.
        hsv_image: HSV uint8 image.
        sensitivity: Float in [0.1, 1.0].
        min_colony_area: Minimum area in pixels to register as a biological colony.

    Returns:
        dict containing:
            - 'biological_mask': uint8 binary mask (0 or 255)
            - 'biological_density': float32 normalized map [0.0, 1.0]
            - 'exg_map': np.ndarray float32 normalized Excess Green map
            - 'coverage_pct': float percentage of wall covered by biological growth
            - 'colony_count': int number of distinct colonies
            - 'severity_score': float score in [0, 100]
            - 'colony_details': list of dicts with bounding boxes and metrics
            - 'explanation': str explainable diagnostic rationale
    """
    h, w = rgb_image.shape[:2]
    total_pixels = h * w
    sens = np.clip(sensitivity, 0.1, 1.0)

    # 1. Normalized RGB & Excess Green Index (ExG)
    # ExG = 2*g - r - b, a gold-standard agronomical and heritage bio-index
    rgb_float = rgb_image.astype(np.float32)
    sum_rgb = np.sum(rgb_float, axis=2, keepdims=True) + 1e-6
    norm_r = rgb_float[:, :, 0:1] / sum_rgb
    norm_g = rgb_float[:, :, 1:2] / sum_rgb
    norm_b = rgb_float[:, :, 2:3] / sum_rgb

    exg = (2.0 * norm_g - norm_r - norm_b).squeeze()

    # Threshold ExG with sensitivity
    exg_thresh = float(max(0.06, 0.15 - sens * 0.09))
    exg_binary = (exg >= exg_thresh).astype(np.uint8) * 255

    # 2. HSV Color Segmentation for Chlorophyll & Lichen
    # Green moss & algae: Hue in [26, 88], Saturation in [40, 255] (rejects pale desaturated paints)
    h_min = int(max(24, 34 - sens * 8))
    h_max = int(min(88, 80 + sens * 8))
    s_min = int(max(38, 65 - sens * 25))
    v_min = 25

    green_mask = cv2.inRange(
        hsv_image,
        np.array([h_min, s_min, v_min]),
        np.array([h_max, 255, 230])
    )

    # Yellow-green / olive crustose lichen: Hue in [18, 32], Saturation in [45, 255]
    lichen_mask = cv2.inRange(
        hsv_image,
        np.array([int(max(18, 22 - sens * 4)), int(max(40, 65 - sens * 25)), 30]),
        np.array([32, 230, 220])
    )

    hsv_bio = cv2.bitwise_or(green_mask, lichen_mask)

    # 3. Combine ExG vegetation index with HSV chromatic verification
    # Require both ExG vegetation elevation and HSV chromatic confirmation
    # Also verify G channel > R channel + 10 to ensure true green dominance over earth tones
    r_chan = rgb_image[:, :, 0]
    g_chan = rgb_image[:, :, 1]
    g_dominance = (g_chan.astype(np.int16) - r_chan.astype(np.int16) > 8).astype(np.uint8) * 255

    candidate_bio = cv2.bitwise_and(
        cv2.bitwise_and(exg_binary, hsv_bio),
        g_dominance
    )

    # 4. Morphological Opening & Closing
    # Opening removes isolated color noise specks; Closing bridges colonial clusters
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    opened = cv2.morphologyEx(candidate_bio, cv2.MORPH_OPEN, kernel_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    # 5. Connected Component Analysis & Colony Extraction
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    validated_mask = np.zeros((h, w), dtype=np.uint8)
    colony_details: List[Dict[str, Any]] = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_colony_area:
            continue

        comp_w = stats[label, cv2.CC_STAT_WIDTH]
        comp_h = stats[label, cv2.CC_STAT_HEIGHT]
        comp_x = stats[label, cv2.CC_STAT_LEFT]
        comp_y = stats[label, cv2.CC_STAT_TOP]

        comp_mask = (labels == label).astype(np.uint8) * 255

        # Check mean ExG in colony
        mean_exg_val = float(np.mean(exg[comp_mask > 0]))

        validated_mask = cv2.bitwise_or(validated_mask, comp_mask)
        cx, cy = int(centroids[label][0]), int(centroids[label][1])

        colony_details.append({
            "bbox": (int(comp_x), int(comp_y), int(comp_w), int(comp_h)),
            "area_px": int(area),
            "area_pct": round((area / float(total_pixels)) * 100.0, 3),
            "centroid": (cx, cy),
            "mean_exg": round(mean_exg_val, 3)
        })

    # 6. Overall Metrics & Smooth Density Map
    total_bio_px = cv2.countNonZero(validated_mask)
    coverage_pct = round((total_bio_px / float(total_pixels)) * 100.0, 2)
    colony_count = len(colony_details)

    bio_density = cv2.GaussianBlur(validated_mask.astype(np.float32) / 255.0, (21, 21), 0)
    max_d = np.max(bio_density)
    if max_d > 0:
        bio_density = bio_density / max_d

    # 7. Severity Score [0 - 100]
    # Biological growth retains moisture and secretes oxalic/carbonic acids that dissolve calcium carbonate
    cov_score = min(60.0, (coverage_pct / 5.0) * 60.0)
    colony_score = min(25.0, (colony_count / 6.0) * 25.0)
    exg_score = min(15.0, (np.mean(exg[validated_mask > 0]) if total_bio_px > 0 else 0.0) * 50.0)
    severity_score = float(np.clip(cov_score + colony_score + exg_score, 0.0, 100.0))

    # 8. Explainable Diagnostic Rationale
    if colony_count == 0:
        explanation = "No biological growth (moss, algae, or lichen) detected. Surface is biocide-free and clean."
    else:
        explanation = (
            f"Detected {colony_count} biological growth colon(ies) covering {coverage_pct}% of the surface. "
            f"Spectroscopic Excess Green Index (ExG > {exg_thresh:.2f}) and HSV chlorophyll resonance "
            f"confirm active microbial/bryophyte colonization. Microflora rhizoids generate physical wedge "
            f"pressures while releasing biogenic acids that accelerate stone dissolution."
        )

    # Normalize ExG map for display [0, 1]
    exg_display = np.clip((exg + 0.1) / 0.5, 0.0, 1.0)

    return {
        "biological_mask": validated_mask,
        "biological_density": bio_density,
        "exg_map": exg_display,
        "coverage_pct": coverage_pct,
        "colony_count": colony_count,
        "severity_score": round(severity_score, 1),
        "colony_details": colony_details,
        "explanation": explanation
    }
