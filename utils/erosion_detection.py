"""
KalaRaksha - Surface Erosion & Weathering Detection Module
Heritage Wall Deterioration Analysis & Restoration-Priority Mapping System

This module quantifies stone weathering, granular disintegration, pitting, and
surface erosion on historic sandstone, limestone, and granite monuments.

Techniques:
1. Fast local variance computation via moving-window expectation: E[I^2] - (E[I])^2
2. Local Shannon entropy mapping to quantify surface disorder and mineral disintegration
3. High-frequency Laplacian texture roughness
4. Multi-metric statistical thresholding for granular pitting and weathering loss
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import cv2

# Optional skimage import for rank-based entropy
try:
    from skimage.filters.rank import entropy as ski_entropy
    from skimage.morphology import disk as ski_disk
    HAS_SKIMAGE_ENTROPY = True
except Exception:
    HAS_SKIMAGE_ENTROPY = False


def _compute_local_entropy(gray_image: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    Compute local Shannon entropy image.
    Uses skimage if available, otherwise an efficient NumPy histogram window approximation.
    """
    if HAS_SKIMAGE_ENTROPY:
        try:
            footprint = ski_disk(radius)
            ent = ski_entropy(gray_image, footprint=footprint)
            # Normalize to [0.0, 1.0]
            max_e = np.max(ent)
            return (ent / max_e) if max_e > 0 else np.zeros_like(ent, dtype=np.float32)
        except Exception:
            pass

    # Efficient fallback approximation: local entropy correlates with local standard deviation
    # and gradient dispersion
    gray_f = gray_image.astype(np.float32)
    ksize = radius * 2 + 1
    mean_val = cv2.blur(gray_f, (ksize, ksize))
    sq_mean = cv2.blur(gray_f**2, (ksize, ksize))
    var = np.maximum(0.0, sq_mean - mean_val**2)
    std = np.sqrt(var)

    # Normalize std to [0.0, 1.0]
    max_s = np.max(std)
    return (std / max_s) if max_s > 0 else np.zeros_like(gray_image, dtype=np.float32)


def detect_surface_erosion(
    gray_image: np.ndarray,
    sensitivity: float = 0.5,
    min_patch_area: int = 200
) -> Dict[str, Any]:
    """
    Detect surface erosion, stone crumbling, and weathering roughness.

    Args:
        gray_image: Grayscale uint8 image.
        sensitivity: Float in [0.1, 1.0].
        min_patch_area: Minimum area in pixels to register an eroded zone.

    Returns:
        dict containing:
            - 'erosion_mask': uint8 binary mask (0 or 255)
            - 'erosion_roughness': float32 normalized map [0.0, 1.0]
            - 'local_variance_map': float32 normalized variance map
            - 'entropy_map': float32 normalized entropy map
            - 'eroded_area_pct': float percentage of wall affected
            - 'num_eroded_patches': int count of erosion zones
            - 'mean_roughness_score': float score in [0, 100]
            - 'severity_score': float score in [0, 100]
            - 'patch_details': list of dicts with bounding boxes and metrics
            - 'explanation': str explainable diagnostic rationale
    """
    h, w = gray_image.shape
    total_pixels = h * w
    sens = np.clip(sensitivity, 0.1, 1.0)

    # 1. Local Variance & Standard Deviation (Granular Roughness)
    gray_f = gray_image.astype(np.float32)
    k_var = 11
    mean_local = cv2.blur(gray_f, (k_var, k_var))
    sq_mean = cv2.blur(gray_f**2, (k_var, k_var))
    local_var = np.maximum(0.0, sq_mean - mean_local**2)
    local_std = np.sqrt(local_var)

    max_std = np.max(local_std)
    norm_std = (local_std / max_std) if max_std > 0 else np.zeros_like(local_std)

    # 2. Local Shannon Entropy (Micro-Texture Randomness)
    norm_entropy = _compute_local_entropy(gray_image, radius=5)

    # 3. High-Frequency Micro-Roughness via Laplacian
    laplacian = cv2.Laplacian(gray_image, cv2.CV_32F, ksize=3)
    abs_lap = np.abs(laplacian)
    smooth_lap = cv2.blur(abs_lap, (7, 7))
    max_lap = np.max(smooth_lap)
    norm_lap = (smooth_lap / max_lap) if max_lap > 0 else np.zeros_like(smooth_lap)

    # 4. Fused Roughness Index
    # Granular erosion combines high local standard deviation, high entropy, and high Laplacian texture
    roughness_fused = (norm_std * 0.4) + (norm_entropy * 0.35) + (norm_lap * 0.25)

    # Threshold roughness with sensitivity
    # Pristine polished stone is smooth (low roughness); weathered/eroded stone has high pitting
    rough_thresh = float(max(0.20, 0.48 - sens * 0.25))
    candidate_mask = (roughness_fused >= rough_thresh).astype(np.uint8) * 255

    # 5. Morphological Refinement: consolidate weathered patches & prune single-pixel noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    refined = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel_open)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close)

    # 6. Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined, connectivity=8)

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
        mean_rough_val = float(np.mean(roughness_fused[comp_mask > 0]))

        validated_mask = cv2.bitwise_or(validated_mask, comp_mask)
        cx, cy = int(centroids[label][0]), int(centroids[label][1])

        patch_details.append({
            "bbox": (int(comp_x), int(comp_y), int(comp_w), int(comp_h)),
            "area_px": int(area),
            "area_pct": round((area / float(total_pixels)) * 100.0, 3),
            "centroid": (cx, cy),
            "mean_roughness": round(mean_rough_val, 3)
        })

    # 7. Metrics & Smooth Density Map
    total_eroded_px = cv2.countNonZero(validated_mask)
    eroded_area_pct = round((total_eroded_px / float(total_pixels)) * 100.0, 2)
    num_patches = len(patch_details)

    erosion_density = cv2.GaussianBlur(validated_mask.astype(np.float32) / 255.0, (21, 21), 0)
    max_d = np.max(erosion_density)
    if max_d > 0:
        erosion_density = erosion_density / max_d

    mean_roughness_score = float(np.mean(roughness_fused[validated_mask > 0])) * 100.0 if total_eroded_px > 0 else 0.0

    # 8. Severity Score [0 - 100]
    cov_score = min(50.0, (eroded_area_pct / 6.0) * 50.0)
    rough_score = min(35.0, (mean_roughness_score / 60.0) * 35.0)
    count_score = min(15.0, (num_patches / 5.0) * 15.0)
    severity_score = float(np.clip(cov_score + rough_score + count_score, 0.0, 100.0))

    # 9. Explainable Diagnostic Rationale
    if num_patches == 0:
        explanation = "No severe surface erosion, pitting, or granular weathering detected. Stone dressing is well-preserved."
    else:
        explanation = (
            f"Detected {num_patches} eroded zone(s) covering {eroded_area_pct}% of the surface "
            f"(mean roughness index: {mean_roughness_score:.1f}/100). "
            f"Elevated local variance (sigma > {rough_thresh:.2f}) and Shannon entropy gradients indicate "
            f"granular fretting, loss of protective patina, and micro-alveolization from wind/rain weathering."
        )

    return {
        "erosion_mask": validated_mask,
        "erosion_roughness": erosion_density,
        "local_variance_map": norm_std,
        "entropy_map": norm_entropy,
        "eroded_area_pct": eroded_area_pct,
        "num_eroded_patches": num_patches,
        "mean_roughness_score": round(mean_roughness_score, 1),
        "severity_score": round(severity_score, 1),
        "patch_details": patch_details,
        "explanation": explanation
    }
