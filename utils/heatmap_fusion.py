"""
KalaRaksha - Deterioration Heatmap Fusion Module
Heritage Wall Deterioration Analysis & Restoration-Priority Mapping System

This module performs multi-modal weighted fusion of the 5 individual deterioration
channels (Cracks, Moisture, Peeling, Biological Growth, and Erosion) to generate:
1. Continuous fused deterioration density map
2. Smooth Gaussian thermal-styled heatmaps (JET, TURBO, INFERNO, etc.)
3. Alpha-blended composite overlays on enhanced wall photography
4. Multi-hazard composite damage mask and relative channel contribution breakdown
"""

from typing import Dict, Any, Optional
import numpy as np
import cv2


DEFAULT_WEIGHTS = {
    "cracks": 0.30,       # High weight due to structural integrity risk
    "moisture": 0.25,     # High weight due to active decay acceleration
    "peeling": 0.15,      # High weight for loss of decorative fresco/plaster
    "biological": 0.15,   # Medium weight for biochemical stone dissolution
    "erosion": 0.15       # Medium weight for loss of patina and carved relief
}

COLORMAP_REGISTRY = {
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "PLASMA": cv2.COLORMAP_PLASMA,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "HOT": cv2.COLORMAP_HOT
}


def fuse_deterioration_channels(
    crack_density: np.ndarray,
    moisture_density: np.ndarray,
    peeling_density: np.ndarray,
    biological_density: np.ndarray,
    erosion_density: np.ndarray,
    channel_toggles: Optional[Dict[str, bool]] = None,
    channel_weights: Optional[Dict[str, float]] = None,
    enhanced_rgb: Optional[np.ndarray] = None,
    colormap_name: str = "JET",
    alpha_blend: float = 0.55,
    blur_ksize: int = 15,
    composite_thresh: float = 0.12
) -> Dict[str, Any]:
    """
    Fuse individual deterioration maps into a unified heatmap and composite mask.

    Args:
        crack_density: Float32 array [0, 1].
        moisture_density: Float32 array [0, 1].
        peeling_density: Float32 array [0, 1].
        biological_density: Float32 array [0, 1].
        erosion_density: Float32 array [0, 1].
        channel_toggles: Dict indicating enabled/disabled channels.
        channel_weights: Dict of weights for enabled channels.
        enhanced_rgb: RGB uint8 image for background blending.
        colormap_name: Name of OpenCV colormap.
        alpha_blend: Transparency factor for heatmap overlay [0.0 - 1.0].
        blur_ksize: Gaussian blur kernel size for smooth thermal dissipation.
        composite_thresh: Cutoff for composite binary damage mask.

    Returns:
        dict containing:
            - 'fused_density': float32 array [0.0, 1.0]
            - 'heatmap_rgb': uint8 RGB image of raw colorized heatmap
            - 'blended_rgb': uint8 RGB image of heatmap overlaid on wall
            - 'composite_mask': uint8 binary mask (0 or 255)
            - 'total_damaged_pct': float percentage of wall with active deterioration
            - 'channel_contributions': dict of percentage contribution per channel
    """
    h, w = crack_density.shape[:2]
    total_pixels = h * w

    # Default toggles if omitted
    toggles = {
        "cracks": True,
        "moisture": True,
        "peeling": True,
        "biological": True,
        "erosion": True
    }
    if channel_toggles:
        toggles.update(channel_toggles)

    # Weights
    weights = DEFAULT_WEIGHTS.copy()
    if channel_weights:
        weights.update(channel_weights)

    # Filter by active toggles
    active_weights = {k: (weights[k] if toggles.get(k, True) else 0.0) for k in weights}
    weight_sum = sum(active_weights.values())

    if weight_sum > 0:
        normalized_weights = {k: v / weight_sum for k, v in active_weights.items()}
    else:
        # All disabled
        normalized_weights = {k: 0.0 for k in weights}

    # 1. Weighted linear combination of density fields
    fused = (
        normalized_weights["cracks"] * crack_density +
        normalized_weights["moisture"] * moisture_density +
        normalized_weights["peeling"] * peeling_density +
        normalized_weights["biological"] * biological_density +
        normalized_weights["erosion"] * erosion_density
    )

    # 2. Gaussian Smoothing for professional continuous thermal dissipation
    if blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        fused_smooth = cv2.GaussianBlur(fused, (k, k), 0)
    else:
        fused_smooth = fused.copy()

    # Clip to valid [0.0, 1.0] range
    fused_norm = np.clip(fused_smooth, 0.0, 1.0)

    # 3. Create composite binary damage mask
    composite_mask = (fused_norm >= composite_thresh).astype(np.uint8) * 255

    # 4. Colorize Heatmap using chosen colormap
    fused_uint8 = (fused_norm * 255.0).astype(np.uint8)
    cmap_code = COLORMAP_REGISTRY.get(colormap_name.upper(), cv2.COLORMAP_JET)
    heatmap_bgr = cv2.applyColorMap(fused_uint8, cmap_code)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 5. Alpha Blending with Enhanced Wall Photography
    if enhanced_rgb is not None:
        bg = enhanced_rgb.copy()
        alpha = np.clip(alpha_blend, 0.0, 1.0)

        # Create localized alpha mask: where fused_norm is near 0, show bare wall
        # Smooth transition at low values so undamaged stone looks natural
        weight_mask = np.clip((fused_norm - 0.03) / 0.15, 0.0, 1.0)[:, :, np.newaxis]
        effective_alpha = alpha * weight_mask

        blended = (bg * (1.0 - effective_alpha) + heatmap_rgb * effective_alpha).astype(np.uint8)
    else:
        blended = heatmap_rgb.copy()

    # 6. Channel Contributions Quantification
    total_damaged_px = cv2.countNonZero(composite_mask)
    total_damaged_pct = round((total_damaged_px / float(total_pixels)) * 100.0, 2)

    contributions: Dict[str, float] = {}
    channel_arrays = {
        "cracks": crack_density,
        "moisture": moisture_density,
        "peeling": peeling_density,
        "biological": biological_density,
        "erosion": erosion_density
    }

    sum_all_damage = sum(
        float(np.sum(channel_arrays[ch])) for ch in channel_arrays if toggles.get(ch, True)
    )

    for ch, arr in channel_arrays.items():
        if not toggles.get(ch, True) or sum_all_damage == 0:
            contributions[ch] = 0.0
        else:
            ch_sum = float(np.sum(arr))
            contributions[ch] = round((ch_sum / sum_all_damage) * 100.0, 1)

    return {
        "fused_density": fused_norm,
        "heatmap_rgb": heatmap_rgb,
        "blended_rgb": blended,
        "composite_mask": composite_mask,
        "total_damaged_pct": total_damaged_pct,
        "channel_contributions": contributions,
        "active_weights": normalized_weights
    }
