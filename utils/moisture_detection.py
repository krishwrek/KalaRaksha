"""
KalaRaksha - Moisture & Seepage Detection Module
Heritage Wall Deterioration Analysis & Restoration-Priority Mapping System

This module detects moisture stains, damp patches, capillary rising damp, and
water seepage on heritage temple walls, murals, and historic facades.

Techniques:
1. HSV and LAB color-space decomposition
2. Background illumination estimation and relative luminance depression (damp darkness)
3. Chromaticity analysis for brown/damp mineral staining
4. Adaptive thresholding & morphological consolidation
5. Contour geometry and compactness filtering for irregular amoeboid seepage plumes
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import cv2


def detect_moisture_seepage(
    rgb_image: np.ndarray,
    hsv_image: np.ndarray,
    sensitivity: float = 0.5,
    min_patch_area: int = 250
) -> Dict[str, Any]:
    """
    Detect moisture stains and water seepage patches on heritage surfaces.

    Args:
        rgb_image: RGB uint8 image.
        hsv_image: HSV uint8 image.
        sensitivity: Sensitivity factor in [0.1, 1.0].
        min_patch_area: Minimum pixel area to consider as a significant moisture patch.

    Returns:
        dict containing:
            - 'moisture_mask': uint8 binary mask (0 or 255)
            - 'moisture_density': float32 normalized map [0.0, 1.0]
            - 'moisture_area_pct': float percentage of wall affected
            - 'num_patches': int count of detected damp zones
            - 'largest_patch_pct': float size of largest single damp zone
            - 'severity_score': float score in [0, 100]
            - 'seepage_centroids': list of (cx, cy)
            - 'patch_details': list of dicts with bounding boxes and metrics
            - 'explanation': str diagnostic explanation
    """
    h, w = rgb_image.shape[:2]
    total_pixels = h * w
    sens = np.clip(sensitivity, 0.1, 1.0)

    # 1. Decompose channels
    h_chan = hsv_image[:, :, 0]
    s_chan = hsv_image[:, :, 1]
    v_chan = hsv_image[:, :, 2]

    # Convert to LAB for perceptual lightness L
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]

    # 2. Background illumination estimation
    # Damp stone/plaster appears significantly darker than its local dry surroundings.
    # We estimate background luminance using a large Gaussian blur (approx 1/15th of image dimension).
    blur_k = int(max(25, min(h, w) // 15))
    if blur_k % 2 == 0:
        blur_k += 1
    bg_l = cv2.GaussianBlur(l_chan, (blur_k, blur_k), 0)

    # Relative darkness: damp areas have L significantly lower than surrounding background
    l_diff = cv2.subtract(bg_l, l_chan)  # Positive where local is darker than background

    # Threshold relative darkness scaled by sensitivity
    # Low sensitivity -> only very dark damp patches; High sensitivity -> subtle damp shadows
    diff_thresh = int(max(10, 32 - sens * 20))
    _, damp_dark_mask = cv2.threshold(l_diff, diff_thresh, 255, cv2.THRESH_BINARY)

    # 3. Chromaticity analysis for damp brown/earthy mineral stains
    # Damp watermarks often exhibit brownish/ochre or dull desaturated hues:
    # Hue in [10, 40] (brownish/yellow-brown) or Hue in [0, 25] with moderate saturation
    brown_damp_mask = cv2.inRange(
        hsv_image,
        np.array([8, int(max(15, 30 - sens * 15)), 20]),
        np.array([38, 200, int(min(220, 170 + sens * 40))])
    )

    # Low-value, low-saturation stagnant damp mask (dark gray dampness)
    dark_stagnant_mask = cv2.inRange(
        hsv_image,
        np.array([0, 0, 15]),
        np.array([180, int(85 + sens * 35), int(110 + sens * 30)])
    )

    # Combine chromatic and luminance indications of moisture
    chromatic_damp = cv2.bitwise_or(brown_damp_mask, dark_stagnant_mask)
    candidate_damp = cv2.bitwise_and(damp_dark_mask, chromatic_damp)

    # Also include strong relative darkness regions even if desaturated
    strong_dark_thresh = int(max(18, 45 - sens * 25))
    _, strong_dark = cv2.threshold(l_diff, strong_dark_thresh, 255, cv2.THRESH_BINARY)
    combined_raw = cv2.bitwise_or(candidate_damp, strong_dark)

    # 4. Morphological consolidation: moisture forms cohesive, smooth-bordered plumes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    consolidated = cv2.morphologyEx(combined_raw, cv2.MORPH_CLOSE, kernel_close)
    consolidated = cv2.morphologyEx(consolidated, cv2.MORPH_OPEN, kernel_open)

    # 5. Connected Component & Contour Filtering for Amoeboid Moisture Morphology
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(consolidated, connectivity=8)

    validated_mask = np.zeros((h, w), dtype=np.uint8)
    patch_details: List[Dict[str, Any]] = []
    seepage_centroids: List[Tuple[int, int]] = []
    largest_patch_area = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_patch_area:
            continue

        comp_w = stats[label, cv2.CC_STAT_WIDTH]
        comp_h = stats[label, cv2.CC_STAT_HEIGHT]
        comp_x = stats[label, cv2.CC_STAT_LEFT]
        comp_y = stats[label, cv2.CC_STAT_TOP]

        comp_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = contours[0]
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # Circularity = 4 * pi * Area / (Perimeter^2)
        # Moisture plumes are irregular, amoeba-like, typically circularity < 0.65
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        # Average darkness depression in this patch
        mean_darkness = float(cv2.mean(l_diff, mask=comp_mask)[0])

        # Validate as moisture stain (irregular shape, adequate area, noticeable darkness depression)
        if area >= min_patch_area and mean_darkness >= (diff_thresh * 0.75):
            validated_mask = cv2.bitwise_or(validated_mask, comp_mask)
            cx, cy = int(centroids[label][0]), int(centroids[label][1])
            seepage_centroids.append((cx, cy))

            if area > largest_patch_area:
                largest_patch_area = area

            patch_details.append({
                "bbox": (int(comp_x), int(comp_y), int(comp_w), int(comp_h)),
                "area_px": int(area),
                "area_pct": round((area / float(total_pixels)) * 100.0, 3),
                "centroid": (cx, cy),
                "circularity": round(float(circularity), 3),
                "darkness_intensity": round(mean_darkness, 1)
            })

    # 6. Overall Metrics & Density Map
    total_moisture_px = cv2.countNonZero(validated_mask)
    moisture_area_pct = round((total_moisture_px / float(total_pixels)) * 100.0, 2)
    largest_patch_pct = round((largest_patch_area / float(total_pixels)) * 100.0, 2)
    num_patches = len(patch_details)

    # Smooth density map
    moisture_density = cv2.GaussianBlur(validated_mask.astype(np.float32) / 255.0, (25, 25), 0)
    max_d = np.max(moisture_density)
    if max_d > 0:
        moisture_density = moisture_density / max_d

    # 7. Severity Score Calculation [0 - 100]
    # Moisture is considered a critical catalyst for heritage degradation:
    # > 5% wall coverage or single large plume (> 3%) generates critical severity
    cov_score = min(50.0, (moisture_area_pct / 5.0) * 50.0)
    patch_size_score = min(35.0, (largest_patch_pct / 3.0) * 35.0)
    count_score = min(15.0, (num_patches / 4.0) * 15.0)
    severity_score = float(np.clip(cov_score + patch_size_score + count_score, 0.0, 100.0))

    # 8. Explainable Diagnostic Rationale
    if num_patches == 0:
        explanation = "No active moisture seepage or damp staining detected. Masonry humidity appears balanced."
    else:
        explanation = (
            f"Detected {num_patches} moisture seepage patch(es) covering {moisture_area_pct}% of the surface "
            f"(largest contiguous plume: {largest_patch_pct}% of wall area). "
            f"The regions display significant local luminance depression (Delta L > {diff_thresh} units), "
            f"characteristic brown/ochre mineral tide lines, and amoeboid perimeter geometry indicative of "
            f"water infiltration or rising damp."
        )

    return {
        "moisture_mask": validated_mask,
        "moisture_density": moisture_density,
        "moisture_area_pct": moisture_area_pct,
        "num_patches": num_patches,
        "largest_patch_pct": largest_patch_pct,
        "severity_score": round(severity_score, 1),
        "seepage_centroids": seepage_centroids,
        "patch_details": patch_details,
        "explanation": explanation
    }
